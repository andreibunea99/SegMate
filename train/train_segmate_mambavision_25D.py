"""
Train SegMateMambaVision (Thorax Segmentation) – fully instrumented -> 2.5D input
================================================================================
* End-to-end training with deep-supervision and BalancedBatchSampler (2.5D loader).
* Comprehensive terminal logs + TensorBoard visualization.
* Detailed per-organ metrics and loss breakdown for each head.
* DistributedDataParallel (DDP) support for multi-GPU training.

This script uses MambaVision encoder (NVIDIA, CVPR 2025) - a hybrid Mamba-Transformer
architecture that achieves state-of-the-art results.

IMPORTANT: Requires mamba-ssm which needs GLIBC >= 2.32.
           Run inside Singularity container on systems with older GLIBC.

Available MambaVision variants:
- T (Tiny):  31.8M params, 82.3% ImageNet Top-1 (RECOMMENDED)
- S (Small): 50.1M params, 83.3% ImageNet Top-1
- B (Base):  97.7M params, 84.2% ImageNet Top-1
- L (Large): 227.9M params, 85.0% ImageNet Top-1

Examples
--------
# Inside Singularity container:
singularity exec --nv containers/pytorch_24.01.sif python train/train_segmate_mambavision_25D.py \\
  --root processed_dataset --deep_supervision --variant T \\
  --logdir runs/segmate_mambavision_T

# Multi-GPU (DDP) inside container:
singularity exec --nv containers/pytorch_24.01.sif \\
  torchrun --nproc_per_node=3 train/train_segmate_mambavision_25D.py \\
  --root processed_dataset --deep_supervision --variant T \\
  --logdir runs/segmate_mambavision_T
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

from models.segmate_mambavision import SegMateMambaVision
from losses.custom_unetpp_loss import CustomUNetPlusPlusLoss
from dataloaders.balanced_dataloader_25D import get_dataloaders_balanced as get_dataloaders

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
    else:
        return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="processed_dataset", help="Dataset root folder")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--logdir", default="runs/default")
    p.add_argument("--deep_supervision", action="store_true")
    p.add_argument("--log_interval", type=int, default=5)
    p.add_argument("--save_interval", type=int, default=5)
    p.add_argument("--resume", default=None)

    p.add_argument(
        "--variant",
        type=str,
        default="T",
        choices=["T", "S", "B", "L"],
        help="MambaVision variant: T(iny), S(mall), B(ase), L(arge)",
    )
    p.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights",
    )

    return p


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum, self.count = 0.0, 0

    def update(self, val: float, n: int = 1):
        self.sum += float(val) * int(n)
        self.count += int(n)

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def print_device_summary() -> None:
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if dist.is_initialized():
            print(f"DDP: Rank {dist.get_rank()}/{dist.get_world_size()}, Local GPU: {torch.cuda.current_device()}")
        else:
            print(f"Using {n} GPU(s):")
            for i in range(n):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("Using CPU only")


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


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, best_hd, cfg, filename):
    if not is_main_process():
        return None

    checkpoint_path = Path("checkpoints") / filename
    Path("checkpoints").mkdir(exist_ok=True)

    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    elif isinstance(model, nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_hd": best_hd,
        "config": {
            "deep_supervision": cfg.deep_supervision,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "logdir": cfg.logdir,
            "epochs": cfg.epochs,
            "variant": cfg.variant,
            "pretrained": (not cfg.no_pretrained),
        },
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"-> Checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    if is_main_process():
        print(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint["epoch"],
        "global_step": checkpoint["global_step"],
        "best_hd": checkpoint["best_hd"],
        "config": checkpoint.get("config", {}),
    }


def log_header(title: str, width: int = 80) -> None:
    if not is_main_process():
        return
    line = "=" * width
    print(f"\n{line}")
    print(f" {title} ".center(width, "="))
    print(f"{line}\n")


def log_metrics_summary(epoch, epochs, phase, metrics, lr=None, time_elapsed=None):
    if not is_main_process():
        return

    header = f"[Epoch {epoch}/{epochs}] {phase.upper()}"
    if lr is not None:
        header += f" (LR: {lr:.2e})"
    if time_elapsed is not None:
        header += f" - {time_elapsed:.1f}s"

    print(f"\n{header}")
    print("-" * len(header))
    print(f"Loss: {metrics['loss']:.4f}")

    if "dice_loss" in metrics:
        print(f"Component Losses: Dice={metrics['dice_loss']:.4f}, CE={metrics['ce_loss']:.4f}, "
              f"Boundary={metrics['boundary_loss']:.4f}, Presence={metrics['presence_loss']:.4f}")

    if "hard_dice_mean" in metrics:
        print(f"Mean Hard-Dice: {metrics['hard_dice_mean']:.4f}")

    if phase == "val" and "pres_acc" in metrics:
        print(f"Presence: Accuracy={metrics['pres_acc']:.4f}, AUC={metrics['pres_auc']:.4f}")


def log_organ_metrics(metrics):
    if not is_main_process():
        return
    print("\nPer-Organ Hard-Dice Scores:")
    print("-" * 50)
    max_len = max(len(organ) for organ in ORGANS)
    for i, organ in enumerate(ORGANS):
        dice_key = f"hard_c{i}"
        if dice_key in metrics:
            print(f"{organ:<{max_len}} : {metrics[dice_key]:.4f}")


def main(cfg):
    rank, world_size, local_rank = setup_distributed()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if is_main_process():
        print_device_summary()
        print("Getting the DataLoaders")

    train_ld, val_ld, _ = get_dataloaders(cfg.root, cfg.batch_size, num_workers=12)

    if world_size > 1:
        train_sampler = DistributedSampler(train_ld.dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_ld.dataset, num_replicas=world_size, rank=rank)
        train_ld = torch.utils.data.DataLoader(train_ld.dataset, batch_size=cfg.batch_size,
                                                sampler=train_sampler, num_workers=12, pin_memory=True)
        val_ld = torch.utils.data.DataLoader(val_ld.dataset, batch_size=cfg.batch_size,
                                              sampler=val_sampler, num_workers=12, pin_memory=True)
    else:
        train_sampler = None
        val_sampler = None

    if is_main_process():
        print("Getting the model")

    # 2.5D wrapper
    class SliceFusion(nn.Module):
        def __init__(self, in_ch=3, mid_ch=16, out_ch=1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=True),
            )
        def forward(self, x):
            return self.net(x)

    class TwoPointFiveDWrapper(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.fusion = SliceFusion(in_ch=3, mid_ch=16, out_ch=1)
            self.model = backbone
        def forward(self, x25d):
            x1 = self.fusion(x25d)
            return self.model(x1)

    base_model = SegMateMambaVision(
        num_classes=len(ORGANS),
        in_channels=1,
        deep_supervision=cfg.deep_supervision,
        variant=cfg.variant,
        pretrained=(not cfg.no_pretrained),
    ).to(device)

    if is_main_process():
        print(f"\n{'='*60}")
        print(f"ENCODER INFO (MambaVision):")
        print(f"  Variant: MambaVision-{cfg.variant}")
        total_params = sum(p.numel() for p in base_model.parameters()) / 1e6
        print(f"  Total model params: {total_params:.2f}M")
        print(f"  Pretrained: {not cfg.no_pretrained}")
        print(f"{'='*60}\n")

    model = TwoPointFiveDWrapper(base_model).to(device)

    if world_size > 1:
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
            find_unused_parameters=False
        )
    elif torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    class_w = torch.tensor([1, 1, 5, 5, 1, 3, 3, 6, 4], device=device, dtype=torch.float32)
    loss_weights = {"dice": 2.0, "ce": 1.0, "boundary": 0.1, "presence": 0.2}
    loss_fn = CustomUNetPlusPlusLoss(class_weights=class_w, weights=loss_weights, deep_supervision_weight=0.2)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    global_step, best_hd, start_epoch = 0, 0.0, 1

    if cfg.resume and os.path.exists(cfg.resume):
        checkpoint_data = load_checkpoint(cfg.resume, model, opt, sched)
        global_step = checkpoint_data["global_step"]
        best_hd = checkpoint_data["best_hd"]
        start_epoch = checkpoint_data["epoch"] + 1

    writer = None
    if is_main_process():
        writer = SummaryWriter(cfg.logdir)

    dice_hard = CustomDice(len(ORGANS), threshold=0.5).to(device)
    pres_acc = torchmetrics.classification.MultilabelAccuracy(num_labels=len(ORGANS)).to(device)
    pres_auc = torchmetrics.classification.MultilabelAUROC(num_labels=len(ORGANS)).to(device)

    if is_main_process():
        log_header("TRAINING START", 80)
        print(f"Encoder: MambaVision-{cfg.variant} (pretrained={'no' if cfg.no_pretrained else 'yes'})")
        print(f"Training for epochs {start_epoch} to {cfg.epochs}")
        print(f"Batch size: {cfg.batch_size}, LR: {cfg.lr}")
        print(f"Deep supervision: {'Enabled' if cfg.deep_supervision else 'Disabled'}\n")

    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_start_time = time.time()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # TRAIN
        model.train()
        meter = defaultdict(AverageMeter)
        pbar = tqdm(train_ld, desc=f"Epoch {epoch}/{cfg.epochs} [train]", ncols=110) if is_main_process() else train_ld

        for img, lbl in pbar:
            img, lbl = img.to(device), lbl.to(device)
            opt.zero_grad(set_to_none=True)
            outputs = model(img)
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

        current_lr = opt.param_groups[0]["lr"]
        train_metrics = {k: v.avg for k, v in meter.items()}
        train_time = time.time() - epoch_start_time

        # VALIDATION
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        val_start_time = time.time()
        model.eval()
        vmeter = defaultdict(AverageMeter)
        pres_acc.reset()
        pres_auc.reset()

        pbar = tqdm(val_ld, desc=f"Epoch {epoch}/{cfg.epochs} [val]", ncols=110) if is_main_process() else val_ld

        with torch.no_grad():
            for img, lbl in pbar:
                img, lbl = img.to(device), lbl.to(device)
                outputs = model(img)
                loss, components = loss_fn(outputs, lbl, cfg.deep_supervision)

                vmeter["loss"].update(loss.item(), img.size(0))
                for k, v in components.items():
                    vmeter[k].update(v, img.size(0))

                seg_logits = outputs[0] if not cfg.deep_supervision else outputs[0]
                prob = torch.sigmoid(seg_logits)
                hard = (prob >= 0.5).float()

                h_d = dice_hard(hard, lbl)
                for c in range(len(ORGANS)):
                    vmeter[f"hard_c{c}"].update(h_d[c].item(), img.size(0))

                gt_presence = (lbl.sum((2, 3)) > 0).float()
                pres_acc.update(prob.mean((2, 3)), gt_presence)
                pres_auc.update(prob.mean((2, 3)), gt_presence.long())

        val_time = time.time() - val_start_time
        mean_hd = sum(vmeter[f"hard_c{c}"].avg for c in range(len(ORGANS))) / len(ORGANS)

        val_metrics = {k: v.avg for k, v in vmeter.items()}
        val_metrics["hard_dice_mean"] = mean_hd
        val_metrics["pres_acc"] = pres_acc.compute().item()
        val_metrics["pres_auc"] = pres_auc.compute().item()

        if is_main_process() and writer:
            writer.add_scalar("val/loss", vmeter["loss"].avg, epoch)
            writer.add_scalar("val/hard_dice_mean", mean_hd, epoch)

        sched.step(vmeter["loss"].avg)

        log_metrics_summary(epoch, cfg.epochs, "train", train_metrics, current_lr, train_time)
        log_metrics_summary(epoch, cfg.epochs, "val", val_metrics, time_elapsed=val_time)

        if epoch % cfg.log_interval == 0:
            log_organ_metrics(val_metrics)

        if epoch % cfg.save_interval == 0:
            save_checkpoint(model, opt, sched, epoch, global_step, best_hd, cfg,
                            filename=f"segmate_mambavision_{cfg.variant}_25D_latest.pth")

        if mean_hd > best_hd:
            best_hd = mean_hd
            if is_main_process():
                Path("checkpoints").mkdir(exist_ok=True)
                checkpoint_path = f"checkpoints/segmate_mambavision_{cfg.variant}_25D_best.pth"

                if isinstance(model, DDP):
                    model_state = model.module.state_dict()
                elif isinstance(model, nn.DataParallel):
                    model_state = model.module.state_dict()
                else:
                    model_state = model.state_dict()

                torch.save(model_state, checkpoint_path)
                print(f"\n* New best model saved to {checkpoint_path} (mean hard-Dice {best_hd:.4f})")

    if is_main_process():
        log_header("TRAINING COMPLETE", 80)
        print(f"Best validation Dice: {best_hd:.4f}")
        if writer:
            writer.close()

    cleanup_distributed()


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
