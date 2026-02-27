"""
Train SegMate MambaOut + FiLM (Thorax Segmentation) - 2.5D input
================================================================================
Identical training setup to train_segmatev2_film_25D.py but uses
SegMateMambaOutFiLM (MambaOut encoder + FiLM conditioning).

Examples
--------
# Single GPU
python train/train_segmate_mambaout_film_25D.py \\
  --root processed_dataset --deep_supervision \\
  --logdir runs/segmate_mambaout_film

# Multi-GPU (DDP)
torchrun --nproc_per_node=3 train/train_segmate_mambaout_film_25D.py \\
  --root processed_dataset --deep_supervision \\
  --encoder mambaout_tiny \\
  --logdir runs/segmate_mambaout_film

# Resume
torchrun --nproc_per_node=3 train/train_segmate_mambaout_film_25D.py \\
  --root processed_dataset --deep_supervision \\
  --logdir runs/segmate_mambaout_film \\
  --resume checkpoints/segmate_mambaout_film_mambaout_tiny_25D_latest.pth
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from tqdm import tqdm

sys.path.append(".")

from models.segmate_mambaout_film import SegMateMambaOutFiLM
from losses.custom_unetpp_loss import CustomUNetPlusPlusLoss
from dataloaders.balanced_dataloader_25D_film import get_dataloaders_balanced_film as get_dataloaders

ORGANS = [
    "Left Lung", "Right Lung", "Cord", "Esophagus",
    "Liver", "Left Kidney", "Right Kidney", "Aorta", "Trachea",
]


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


def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="processed_dataset")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--logdir", default="runs/segmate_mambaout_film")
    p.add_argument("--deep_supervision", action="store_true")
    p.add_argument("--log_interval", type=int, default=5)
    p.add_argument("--save_interval", type=int, default=5)
    p.add_argument("--resume", default=None)
    p.add_argument("--encoder", type=str, default="mambaout_tiny",
                   help="MambaOut encoder variant (mambaout_tiny, mambaout_small, mambaout_base)")
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--film_hidden", type=int, default=128)
    p.add_argument("--film_layers", type=int, default=3)
    p.add_argument("--film_dropout", type=float, default=0.1)
    p.add_argument("--film_init_scale", type=float, default=0.01)
    return p


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum, self.count = 0.0, 0
    def update(self, val, n=1): self.sum += float(val) * int(n); self.count += int(n)
    @property
    def avg(self): return self.sum / max(self.count, 1)


class CustomDice(nn.Module):
    def __init__(self, num_classes, threshold=None):
        super().__init__()
        self.num_classes = num_classes
        self.threshold = threshold

    def forward(self, preds, target):
        if self.threshold is not None:
            preds = (preds >= self.threshold).float()
        preds = preds.view(preds.size(0), preds.size(1), -1)
        target = target.view(target.size(0), target.size(1), -1)
        intersection = (preds * target).sum(dim=2)
        cardinality = preds.sum(dim=2) + target.sum(dim=2)
        dice = 2 * intersection / (cardinality + 1e-8)
        return dice.mean(dim=0)


def save_best_checkpoint(model, epoch, best_hd, encoder_name):
    if not is_main_process():
        return
    Path("checkpoints").mkdir(exist_ok=True)
    ckpt_path = f"checkpoints/segmate_mambaout_film_{encoder_name}_25D_best.pth"
    state = model.module.state_dict() if isinstance(model, (DDP, nn.DataParallel)) else model.state_dict()
    torch.save(state, ckpt_path)
    print(f"\n* New best saved → {ckpt_path}  (dice={best_hd:.4f})")


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, best_hd, cfg):
    if not is_main_process():
        return
    Path("checkpoints").mkdir(exist_ok=True)
    state = model.module.state_dict() if isinstance(model, (DDP, nn.DataParallel)) else model.state_dict()
    torch.save({
        "epoch": epoch, "global_step": global_step, "best_hd": best_hd,
        "model_state_dict": state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": vars(cfg),
    }, f"checkpoints/segmate_mambaout_film_{cfg.encoder}_25D_latest.pth")


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    state = model.module if isinstance(model, (DDP, nn.DataParallel)) else model
    state.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt["global_step"], ckpt["best_hd"]


def main(cfg):
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    train_ld, val_ld, _ = get_dataloaders(cfg.root, cfg.batch_size, num_workers=12)

    if world_size > 1:
        train_sampler = DistributedSampler(train_ld.dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_ld.dataset, num_replicas=world_size, rank=rank)
        train_ld = torch.utils.data.DataLoader(train_ld.dataset, batch_size=cfg.batch_size,
                                               sampler=train_sampler, num_workers=12, pin_memory=True)
        val_ld = torch.utils.data.DataLoader(val_ld.dataset, batch_size=cfg.batch_size,
                                             sampler=val_sampler, num_workers=12, pin_memory=True)
        train_sampler_ref = train_sampler
    else:
        train_sampler_ref = None

    class SliceFusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(16), nn.SiLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1, bias=True),
            )
        def forward(self, x): return self.net(x)

    class TwoPointFiveDWrapperFiLM(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.fusion = SliceFusion()
            self.model = backbone
        def forward(self, x25d, z_norm=None):
            return self.model(self.fusion(x25d), z_norm=z_norm)

    base_model = SegMateMambaOutFiLM(
        num_classes=len(ORGANS), in_channels=1,
        deep_supervision=cfg.deep_supervision,
        encoder_name=cfg.encoder,
        pretrained=(not cfg.no_pretrained),
        film_hidden_dim=cfg.film_hidden, film_num_layers=cfg.film_layers,
        film_dropout=cfg.film_dropout, film_init_scale=cfg.film_init_scale,
    ).to(device)

    model = TwoPointFiveDWrapperFiLM(base_model).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    gradient_as_bucket_view=True, broadcast_buffers=False,
                    find_unused_parameters=True)
    elif torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    class_w = torch.tensor([1, 1, 5, 5, 1, 3, 3, 6, 4], device=device, dtype=torch.float32)
    loss_fn = CustomUNetPlusPlusLoss(
        class_weights=class_w,
        weights={"dice": 2.0, "ce": 1.0, "boundary": 0.1, "presence": 0.2},
        deep_supervision_weight=0.2,
    )

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    global_step, best_hd, start_epoch = 0, 0.0, 1

    if cfg.resume and os.path.exists(cfg.resume):
        start_epoch, global_step, best_hd = load_checkpoint(cfg.resume, model, opt, sched)
        start_epoch += 1
        if is_main_process():
            print(f"Resumed from epoch {start_epoch - 1}, best_hd={best_hd:.4f}")

    writer = SummaryWriter(cfg.logdir) if is_main_process() else None

    dice_hard = CustomDice(len(ORGANS), threshold=0.5).to(device)
    pres_acc = torchmetrics.classification.MultilabelAccuracy(num_labels=len(ORGANS)).to(device)
    pres_auc = torchmetrics.classification.MultilabelAUROC(num_labels=len(ORGANS)).to(device)

    if is_main_process():
        print(f"\n{'='*60}")
        print(f"SegMate MambaOut FiLM Training")
        print(f"Encoder: {cfg.encoder}  |  Epochs: {cfg.epochs}  |  BS: {cfg.batch_size}")
        print(f"Deep supervision: {cfg.deep_supervision}  |  LR: {cfg.lr}")
        print(f"{'='*60}\n")

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        if train_sampler_ref is not None:
            train_sampler_ref.set_epoch(epoch)

        # ── Train ──
        model.train()
        meter = defaultdict(AverageMeter)
        pbar = tqdm(train_ld, desc=f"Epoch {epoch}/{cfg.epochs} [train]", ncols=110) if is_main_process() else train_ld

        for img, lbl, z_norm in pbar:
            img, lbl, z_norm = img.to(device), lbl.to(device), z_norm.to(device)
            opt.zero_grad(set_to_none=True)
            outputs = model(img, z_norm=z_norm)
            loss, components = loss_fn(outputs, lbl, cfg.deep_supervision)
            loss.backward()
            opt.step()
            meter["loss"].update(loss.item(), img.size(0))
            for k, v in components.items():
                meter[k].update(v, img.size(0))
            if is_main_process() and writer and global_step % 50 == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1
            if is_main_process():
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ── Validate ──
        model.eval()
        vmeter = defaultdict(AverageMeter)
        pres_acc.reset(); pres_auc.reset()
        pbar = tqdm(val_ld, desc=f"Epoch {epoch}/{cfg.epochs} [val]", ncols=110) if is_main_process() else val_ld

        with torch.no_grad():
            for img, lbl, z_norm in pbar:
                img, lbl, z_norm = img.to(device), lbl.to(device), z_norm.to(device)
                outputs = model(img, z_norm=z_norm)
                loss, components = loss_fn(outputs, lbl, cfg.deep_supervision)
                vmeter["loss"].update(loss.item(), img.size(0))
                for k, v in components.items():
                    vmeter[k].update(v, img.size(0))

                seg_logits = outputs[0] if isinstance(outputs, tuple) else outputs
                prob = torch.sigmoid(seg_logits)
                hard = (prob >= 0.5).float()
                h_d = dice_hard(hard, lbl)
                for c in range(len(ORGANS)):
                    vmeter[f"hard_c{c}"].update(h_d[c].item(), img.size(0))

                gt_presence = (lbl.sum((2, 3)) > 0).float()
                pres_acc.update(prob.mean((2, 3)), gt_presence)
                pres_auc.update(prob.mean((2, 3)), gt_presence.long())
                if is_main_process():
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

        mean_hd = sum(vmeter[f"hard_c{c}"].avg for c in range(len(ORGANS))) / len(ORGANS)
        sched.step(vmeter["loss"].avg)

        if is_main_process():
            if writer:
                writer.add_scalar("val/loss", vmeter["loss"].avg, epoch)
                writer.add_scalar("val/hard_dice_mean", mean_hd, epoch)
                writer.add_scalar("train/lr", opt.param_groups[0]["lr"], epoch)
            elapsed = time.time() - t0
            print(f"[{epoch}/{cfg.epochs}] train_loss={meter['loss'].avg:.4f}  "
                  f"val_loss={vmeter['loss'].avg:.4f}  dice={mean_hd:.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.1e}  t={elapsed:.0f}s")

            if epoch % cfg.log_interval == 0:
                print("  Per-organ Dice: " + "  ".join(
                    f"{ORGANS[c][:6]}={vmeter[f'hard_c{c}'].avg:.3f}" for c in range(len(ORGANS))
                ))

        if epoch % cfg.save_interval == 0:
            save_checkpoint(model, opt, sched, epoch, global_step, best_hd, cfg)

        if mean_hd > best_hd:
            best_hd = mean_hd
            save_best_checkpoint(model, epoch, best_hd, cfg.encoder)
            save_checkpoint(model, opt, sched, epoch, global_step, best_hd, cfg)

    if is_main_process():
        print(f"\nTraining complete. Best Dice: {best_hd:.4f}")
        print(f"Checkpoint: checkpoints/segmate_mambaout_film_{cfg.encoder}_25D_best.pth")
        if writer:
            writer.close()

    cleanup_distributed()


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
