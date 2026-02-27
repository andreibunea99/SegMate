"""
Train Standard U-Net Baseline (2D) — Dice loss only, single seg head.
======================================================================
Pure 2D training with standard U-Net decoder (no ASPP, no dense skips,
no auxiliary heads, no 2.5D wrapper, no class weights).

Uses the same 3 timm backbones as SegMate for fair ablation:
- tf_efficientnetv2_m
- mambaout_tiny
- fastvit_t12

Examples
--------
# Single GPU
python train/train_unet_baseline_2D.py \
  --root processed_dataset --encoder tf_efficientnetv2_m \
  --epochs 100 --batch_size 32 --lr 2e-4 --num_workers 12

# Multi-GPU (DDP)
torchrun --nproc_per_node=3 train/train_unet_baseline_2D.py \
  --root processed_dataset --encoder mambaout_tiny \
  --epochs 100 --batch_size 32 --lr 2e-4 --num_workers 12

# Resume
python train/train_unet_baseline_2D.py \
  --root processed_dataset --encoder fastvit_t12 \
  --resume checkpoints/unet_baseline_fastvit_t12_2D_latest.pth
"""

from __future__ import annotations

import os
import sys
import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler
from tqdm import tqdm

sys.path.append(".")

from models.unet_baseline import UNetBaseline
from dataloaders.balanced_dataloader import get_dataloaders_balanced as get_dataloaders

ORGANS = [
    "Left Lung", "Right Lung", "Cord", "Esophagus",
    "Liver", "Left Kidney", "Right Kidney", "Aorta", "Trachea",
]


# ────────────────────── Multi-class Dice loss (no weights) ──────────────────────

class MultiClassDiceLoss(nn.Module):
    """Sigmoid → per-class soft Dice → 1 - mean. No class weights."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: [B, C, H, W], targets: [B, C, H, W] one-hot
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)  # reduce over batch and spatial
        intersection = (probs * targets).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


# ────────────────────── Distributed helpers ──────────────────────

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


# ────────────────────── Dice metric (hard threshold) ──────────────────────

class HardDice(nn.Module):
    def __init__(self, num_classes, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.num_classes = num_classes

    def forward(self, preds, target):
        preds = (preds >= self.threshold).float()
        preds = preds.view(preds.size(0), preds.size(1), -1)
        target = target.view(target.size(0), target.size(1), -1)
        intersection = (preds * target).sum(dim=2)
        cardinality = preds.sum(dim=2) + target.sum(dim=2)
        dice = 2 * intersection / (cardinality + 1e-8)
        return dice.mean(dim=0)  # per-class


# ────────────────────── Average-meter ──────────────────────

class AverageMeter:
    def __init__(self):
        self.sum, self.count = 0.0, 0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


# ────────────────────── CLI ──────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="Train Standard U-Net Baseline (2D)")
    p.add_argument("--root", default="processed_dataset")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--logdir", default="runs/unet_baseline")
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--log_interval", type=int, default=5)
    p.add_argument("--save_interval", type=int, default=5)
    p.add_argument("--resume", default=None)
    p.add_argument("--encoder", default="tf_efficientnetv2_m",
                    help="timm encoder name (tf_efficientnetv2_m, mambaout_tiny, fastvit_t12)")
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    return p


# ────────────────────── Checkpoint helpers ──────────────────────

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step,
                    best_dice, cfg, filename):
    if not is_main():
        return
    Path("checkpoints").mkdir(exist_ok=True)
    path = Path("checkpoints") / filename

    if isinstance(model, (DDP, nn.DataParallel)):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "best_dice": best_dice,
        "config": {
            "encoder": cfg.encoder,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "epochs": cfg.epochs,
        },
    }, path)
    print(f"  -> Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu")

    if isinstance(model, (DDP, nn.DataParallel)):
        model.module.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt["model_state_dict"])

    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    return ckpt["epoch"], ckpt["global_step"], ckpt["best_dice"]


# ────────────────────── Main ──────────────────────

def main(cfg):
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main():
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(local_rank)}")
        print(f"Encoder: {cfg.encoder}")

    # ── DataLoaders (2D, single channel) ──
    train_ld, val_ld, _ = get_dataloaders(cfg.root, cfg.batch_size, num_workers=cfg.num_workers)

    if world_size > 1:
        train_sampler = DistributedSampler(train_ld.dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_ld.dataset, num_replicas=world_size, rank=rank)
        train_ld = torch.utils.data.DataLoader(
            train_ld.dataset, batch_size=cfg.batch_size,
            sampler=train_sampler, num_workers=cfg.num_workers, pin_memory=True,
        )
        val_ld = torch.utils.data.DataLoader(
            val_ld.dataset, batch_size=cfg.batch_size,
            sampler=val_sampler, num_workers=cfg.num_workers, pin_memory=True,
        )
    else:
        train_sampler = None

    # ── Model ──
    num_classes = len(ORGANS)
    model = UNetBaseline(
        num_classes=num_classes,
        in_channels=1,
        encoder_name=cfg.encoder,
        pretrained=not cfg.no_pretrained,
    ).to(device)

    if is_main():
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Parameters: {total_params:.2f}M")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                     gradient_as_bucket_view=True, broadcast_buffers=False,
                     find_unused_parameters=False)
    elif torch.cuda.device_count() > 1 and not dist.is_initialized():
        model = nn.DataParallel(model)

    # ── Loss, optimizer, scheduler ──
    loss_fn = MultiClassDiceLoss()
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    steps_per_epoch = len(train_ld)
    total_steps = cfg.epochs * steps_per_epoch

    scheduler = OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=total_steps,
        pct_start=0.05, anneal_strategy="cos",
    )

    use_amp = (not cfg.no_amp) and device.type == "cuda"
    scaler = GradScaler("cuda") if use_amp else None

    global_step, best_dice, start_epoch = 0, 0.0, 1

    # ── Resume ──
    if cfg.resume and os.path.exists(cfg.resume):
        start_epoch_loaded, global_step, best_dice = load_checkpoint(
            cfg.resume, model, optimizer, scheduler, scaler)
        start_epoch = start_epoch_loaded + 1
        if is_main():
            print(f"Resumed from epoch {start_epoch_loaded}, best_dice={best_dice:.4f}")

    # ── TensorBoard ──
    writer = SummaryWriter(cfg.logdir) if is_main() else None

    # ── Metrics ──
    dice_hard = HardDice(num_classes).to(device)

    if is_main():
        print(f"\n{'='*60}")
        print(f"Training U-Net Baseline ({cfg.encoder}) for {cfg.epochs} epochs")
        print(f"Batch size: {cfg.batch_size}, LR: {cfg.lr}, AMP: {use_amp}")
        print(f"{'='*60}\n")

    # ═══════════════════ Training loop ═══════════════════
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # ── Train ──
        model.train()
        meter = defaultdict(AverageMeter)

        pbar = tqdm(train_ld, desc=f"Epoch {epoch}/{cfg.epochs} [train]",
                     ncols=100) if is_main() else train_ld

        for img, lbl in pbar:
            img, lbl = img.to(device), lbl.to(device)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(img)
                    loss = loss_fn(logits, lbl)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(img)
                loss = loss_fn(logits, lbl)
                loss.backward()
                optimizer.step()

            scheduler.step()
            meter["loss"].update(loss.item(), img.size(0))
            global_step += 1

            if is_main():
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            if is_main() and writer and global_step % 50 == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        train_loss = meter["loss"].avg
        train_time = time.time() - t0

        # ── Validation ──
        model.eval()
        vmeter = defaultdict(AverageMeter)
        val_t0 = time.time()

        with torch.no_grad():
            pbar_v = tqdm(val_ld, desc=f"Epoch {epoch}/{cfg.epochs} [val]",
                          ncols=100) if is_main() else val_ld

            for img, lbl in pbar_v:
                img, lbl = img.to(device), lbl.to(device)

                if use_amp:
                    with torch.autocast("cuda", dtype=torch.float16):
                        logits = model(img)
                        loss = loss_fn(logits, lbl)
                else:
                    logits = model(img)
                    loss = loss_fn(logits, lbl)

                vmeter["loss"].update(loss.item(), img.size(0))

                prob = torch.sigmoid(logits)
                hd = dice_hard(prob, lbl)
                for c in range(num_classes):
                    vmeter[f"hard_c{c}"].update(hd[c].item(), img.size(0))

        mean_dice = sum(vmeter[f"hard_c{c}"].avg for c in range(num_classes)) / num_classes
        val_time = time.time() - val_t0

        if is_main():
            print(f"\n[Epoch {epoch}/{cfg.epochs}] "
                  f"train_loss={train_loss:.4f} val_loss={vmeter['loss'].avg:.4f} "
                  f"dice={mean_dice:.4f} "
                  f"lr={optimizer.param_groups[0]['lr']:.2e} "
                  f"time={train_time + val_time:.0f}s")

            if epoch % cfg.log_interval == 0 or epoch == cfg.epochs:
                print("  Per-organ Dice:")
                for c, organ in enumerate(ORGANS):
                    print(f"    {organ:<15s}: {vmeter[f'hard_c{c}'].avg:.4f}")

            if writer:
                writer.add_scalar("val/loss", vmeter["loss"].avg, epoch)
                writer.add_scalar("val/dice_mean", mean_dice, epoch)
                for c, organ in enumerate(ORGANS):
                    writer.add_scalar(f"val/dice_{organ}", vmeter[f"hard_c{c}"].avg, epoch)

        # ── Save periodic checkpoint ──
        if epoch % cfg.save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step,
                            best_dice, cfg,
                            f"unet_baseline_{cfg.encoder}_2D_latest.pth")

        # ── Save best ──
        if mean_dice > best_dice:
            best_dice = mean_dice
            if is_main():
                Path("checkpoints").mkdir(exist_ok=True)
                best_path = f"checkpoints/unet_baseline_{cfg.encoder}_2D_best.pth"

                if isinstance(model, (DDP, nn.DataParallel)):
                    state = model.module.state_dict()
                else:
                    state = model.state_dict()

                torch.save(state, best_path)
                print(f"  * New best model saved: {best_path} (dice={best_dice:.4f})")

                save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step,
                                best_dice, cfg,
                                f"unet_baseline_{cfg.encoder}_2D_best_resumable.pth")

    # ── Done ──
    if is_main():
        print(f"\n{'='*60}")
        print(f"Training complete. Best dice: {best_dice:.4f}")
        print(f"Best model: checkpoints/unet_baseline_{cfg.encoder}_2D_best.pth")
        print(f"{'='*60}")

        save_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step,
                        best_dice, cfg,
                        f"unet_baseline_{cfg.encoder}_2D_latest.pth")
        if writer:
            writer.close()

    cleanup_distributed()


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
