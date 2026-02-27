"""
Train Vanilla U-Net++ Baseline (9-class Thorax Segmentation)
============================================================
Trains a standard U-Net++ model without custom enhancements for baseline comparison.

This training script is simplified compared to train_custom_unetpp.py:
- Uses standard U-Net++ architecture (ResNet34 backbone)
- No CBAM, SE, ASPP, or custom attention mechanisms
- Standard loss function (Dice + CrossEntropy + Boundary)
- Single-head output (no deep supervision for simplicity)

Purpose: Establish baseline performance to demonstrate the incremental value of
CustomUNetPlusPlus enhancements.

Example Usage:
--------------
# Single GPU training
python train/train_vanilla_unetpp.py --root processed_dataset --logdir runs/vanilla_unetpp

# Multi-GPU training with DDP
torchrun --nproc_per_node=4 train/train_vanilla_unetpp.py --root processed_dataset --logdir runs/vanilla_unetpp

# Resume training
torchrun --nproc_per_node=4 train/train_vanilla_unetpp.py --root processed_dataset --logdir runs/vanilla_unetpp --resume checkpoints/vanilla_unetpp_latest.pth
"""

from __future__ import annotations

# ───────────────────────────── imports ─────────────────────────────
import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Local modules
sys.path.append(".")
from models.simple_vanilla_unetpp import SimpleVanillaUNetPlusPlus, count_parameters
from losses.combined_loss import CorrectedAdvancedCombinedLoss, BoundaryLoss
from dataloaders.balanced_dataloader import get_dataloaders_balanced as get_dataloaders

# Organ names (channel order)
ORGANS = [
    "Left Lung", "Right Lung", "Cord", "Esophagus",
    "Liver", "Left Kidney", "Right Kidney", "Aorta", "Trachea",
]

# ────────────────────────── distributed setup ─────────────────────────

def setup_distributed():
    """Initialize distributed training if available"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    else:
        return 0, 1, 0

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    """Check if current process is the main process"""
    return not dist.is_initialized() or dist.get_rank() == 0

# ────────────────────────── arg-parser ─────────────────────────

def build_parser():
    """CLI arguments with sensible defaults"""
    p = argparse.ArgumentParser(description="Train Vanilla U-Net++ Baseline")
    p.add_argument("--root", default="processed_dataset", help="Dataset root folder")
    p.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--logdir", default="runs/vanilla_unetpp", help="TensorBoard log directory")
    p.add_argument("--log_interval", type=int, default=5, help="Log validation details every N epochs")
    p.add_argument("--save_interval", type=int, default=10, help="Save checkpoint every N epochs")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume training from")
    return p

# ───────────────────────── average-meter util ────────────────────────

class AverageMeter:
    """Keeps running average for any scalar metric"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum, self.count = 0.0, 0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)

# ─────────────────────── device info helper ──────────────────────────

def print_device_summary() -> None:
    """Pretty-print CPU/GPU information"""
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

# ────────────────── checkpoint save/load functions ──────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, best_dice, cfg,
                    filename='vanilla_unetpp_latest.pth'):
    """Save training checkpoint"""
    if not is_main_process():
        return None

    checkpoint_path = Path('checkpoints') / filename
    Path('checkpoints').mkdir(exist_ok=True)

    # Handle DDP wrapper
    if isinstance(model, (DDP, nn.DataParallel)):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_dice': best_dice,
        'config': {
            'batch_size': cfg.batch_size,
            'lr': cfg.lr,
            'logdir': cfg.logdir,
            'epochs': cfg.epochs
        }
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint saved: {checkpoint_path}")

    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """Load training checkpoint"""
    if is_main_process():
        print(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load model weights
    if isinstance(model, (DDP, nn.DataParallel)):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return {
        'epoch': checkpoint['epoch'],
        'best_dice': checkpoint['best_dice'],
        'config': checkpoint.get('config', {})
    }

# ─────────────────────── Dice metric ───────────────────────

def compute_dice_score(pred, target, smooth=1e-6):
    """
    Compute Dice coefficient per class

    Args:
        pred: [B, C, H, W] predictions (logits or probabilities)
        target: [B, C, H, W] ground truth (one-hot encoded)

    Returns:
        Tensor of Dice scores per class [C]
    """
    pred = torch.sigmoid(pred)  # Convert logits to probabilities
    pred = (pred > 0.5).float()  # Binarize

    # Flatten spatial dimensions
    pred_flat = pred.view(pred.size(0), pred.size(1), -1)
    target_flat = target.view(target.size(0), target.size(1), -1)

    # Compute intersection and cardinality
    intersection = (pred_flat * target_flat).sum(dim=2)
    cardinality = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

    # Dice = 2 * intersection / cardinality
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)

    # Average over batch, return per-class scores
    return dice.mean(dim=0)

# ─────────────────────── logging functions ───────────────────────

def log_header(title: str, width: int = 80) -> None:
    """Print section header"""
    if not is_main_process():
        return
    line = "=" * width
    print(f"\n{line}")
    print(f" {title} ".center(width, "="))
    print(f"{line}\n")

def log_epoch_summary(epoch, epochs, phase, metrics, lr=None, elapsed=None):
    """Print epoch summary"""
    if not is_main_process():
        return

    header = f"[Epoch {epoch}/{epochs}] {phase.upper()}"
    if lr is not None:
        header += f" (LR: {lr:.2e})"
    if elapsed is not None:
        header += f" - {elapsed:.1f}s"

    print(f"\n{header}")
    print("-" * len(header))
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Dice: {metrics['dice']:.4f}")

# ─────────────────────── training loop ───────────────────────

def train_one_epoch(model, loader, criterion_main, criterion_boundary,
                    optimizer, device, epoch, writer, boundary_weight=0.1):
    """Train for one epoch"""
    model.train()

    loss_meter = AverageMeter()
    dice_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"Epoch {epoch} [TRAIN]",
                disable=not is_main_process())

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        pred_masks, pred_boundaries = model(images)

        # Compute losses (criterion_main returns tuple: (loss, per_class_dice))
        loss_main, _ = criterion_main(pred_masks, masks)
        loss_boundary = criterion_boundary(pred_boundaries, masks)
        loss = loss_main + boundary_weight * loss_boundary

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            dice_scores = compute_dice_score(pred_masks, masks)
            dice_avg = dice_scores.mean().item()

        # Update meters
        loss_meter.update(loss.item(), images.size(0))
        dice_meter.update(dice_avg, images.size(0))

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}',
                         'dice': f'{dice_meter.avg:.4f}'})

    return {
        'loss': loss_meter.avg,
        'dice': dice_meter.avg
    }

def validate(model, loader, criterion_main, criterion_boundary, device,
             epoch, writer, boundary_weight=0.1):
    """Validate for one epoch"""
    model.eval()

    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    dice_per_organ = [AverageMeter() for _ in range(len(ORGANS))]

    pbar = tqdm(loader, desc=f"Epoch {epoch} [VAL]",
                disable=not is_main_process())

    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            pred_masks, pred_boundaries = model(images)

            # Compute losses (criterion_main returns tuple: (loss, per_class_dice))
            loss_main, _ = criterion_main(pred_masks, masks)
            loss_boundary = criterion_boundary(pred_boundaries, masks)
            loss = loss_main + boundary_weight * loss_boundary

            # Compute Dice scores
            dice_scores = compute_dice_score(pred_masks, masks)
            dice_avg = dice_scores.mean().item()

            # Update meters
            loss_meter.update(loss.item(), images.size(0))
            dice_meter.update(dice_avg, images.size(0))

            # Update per-organ Dice
            for i, score in enumerate(dice_scores):
                dice_per_organ[i].update(score.item(), images.size(0))

            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}',
                             'dice': f'{dice_meter.avg:.4f}'})

    # Log per-organ Dice
    if is_main_process():
        print("\nPer-Organ Dice Scores:")
        for i, organ in enumerate(ORGANS):
            print(f"  {organ:15s}: {dice_per_organ[i].avg:.4f}")

    return {
        'loss': loss_meter.avg,
        'dice': dice_meter.avg,
        'per_organ_dice': [m.avg for m in dice_per_organ]
    }

# ─────────────────────── main training function ───────────────────────

def main():
    """Main training function"""

    # Parse arguments
    cfg = build_parser().parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    # Device setup
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')

    # Print configuration
    if is_main_process():
        log_header("Vanilla U-Net++ Baseline Training")
        print_device_summary()
        print(f"\nConfiguration:")
        print(f"  Dataset: {cfg.root}")
        print(f"  Epochs: {cfg.epochs}")
        print(f"  Batch size: {cfg.batch_size}")
        print(f"  Learning rate: {cfg.lr}")
        print(f"  Log dir: {cfg.logdir}")
        if cfg.resume:
            print(f"  Resume from: {cfg.resume}")

    # Create model (Simple & Fast implementation)
    model = SimpleVanillaUNetPlusPlus(num_classes=9, in_channels=1)
    model = model.to(device)

    if is_main_process():
        params = count_parameters(model)
        print(f"\nModel parameters: {params:,} ({params/1e6:.2f}M)")

    # Wrap model for distributed training
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])

    # Loss functions (same as CustomUNet for fair comparison)
    class_weights = torch.tensor([1.0, 1.0, 6.0, 5.0, 1.0, 3.0, 3.0, 1.0, 3.0]).to(device)
    criterion_main = CorrectedAdvancedCombinedLoss(alpha=0.5, class_weights=class_weights)
    criterion_boundary = BoundaryLoss()

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    # TensorBoard writer (main process only)
    writer = None
    if is_main_process():
        writer = SummaryWriter(log_dir=cfg.logdir)

    # Load checkpoint if resuming
    start_epoch = 1
    best_dice = 0.0
    if cfg.resume:
        checkpoint_data = load_checkpoint(cfg.resume, model, optimizer, scheduler)
        start_epoch = checkpoint_data['epoch'] + 1
        best_dice = checkpoint_data['best_dice']
        if is_main_process():
            print(f"Resumed from epoch {checkpoint_data['epoch']}, best Dice: {best_dice:.4f}")

    # Data loaders (returns train, val, test)
    train_loader, val_loader, _ = get_dataloaders(
        root=cfg.root,
        batch_size=cfg.batch_size,
        num_workers=4
    )

    if is_main_process():
        print(f"\nDataset splits:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")

    # Training loop
    if is_main_process():
        log_header("Starting Training")

    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion_main, criterion_boundary,
            optimizer, device, epoch, writer
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion_main, criterion_boundary,
            device, epoch, writer
        )

        epoch_time = time.time() - epoch_start

        # Learning rate scheduler step
        scheduler.step(val_metrics['dice'])
        current_lr = optimizer.param_groups[0]['lr']

        # Log to terminal
        if is_main_process():
            log_epoch_summary(epoch, cfg.epochs, "TRAIN", train_metrics,
                            lr=current_lr, elapsed=epoch_time)
            log_epoch_summary(epoch, cfg.epochs, "VAL", val_metrics)

        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            writer.add_scalar('Dice/train', train_metrics['dice'], epoch)
            writer.add_scalar('Dice/val', val_metrics['dice'], epoch)
            writer.add_scalar('LR', current_lr, epoch)

            # Per-organ Dice
            for i, organ in enumerate(ORGANS):
                writer.add_scalar(f'Dice_Val/{organ}', val_metrics['per_organ_dice'][i], epoch)

        # Save checkpoint
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            save_checkpoint(model, optimizer, scheduler, epoch, best_dice, cfg,
                          filename='vanilla_unetpp_best.pth')
            if is_main_process():
                print(f"✓ New best model! Dice: {best_dice:.4f}")

        if epoch % cfg.save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_dice, cfg,
                          filename=f'vanilla_unetpp_epoch{epoch}.pth')

    # Final checkpoint
    save_checkpoint(model, optimizer, scheduler, cfg.epochs, best_dice, cfg,
                   filename='vanilla_unetpp_final.pth')

    # Cleanup
    if writer is not None:
        writer.close()

    cleanup_distributed()

    if is_main_process():
        log_header("Training Complete")
        print(f"Best validation Dice: {best_dice:.4f}")

if __name__ == "__main__":
    main()
