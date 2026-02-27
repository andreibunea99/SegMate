# training_segmate_v2_nose.py
"""
Train SegMateV2NoSE (Ablation Study - No SE on Skip Connections)
=================================================================
* Uses SegMateV2NoSE model with EfficientNetV2 encoder
* SE blocks on skip connections are replaced with Identity
* Main decoder blocks still use CBAM attention
* All other components (loss, dataloader, hyperparameters) remain identical

This ablation study tests the hypothesis that the combination of SE blocks
on skip pathways with CBAM on decoder blocks provides complementary benefits.

Example
-------
```bash
# Start new training (single GPU)
python train/training_segmate_v2_nose.py \
       --root processed_dataset --deep_supervision \
       --encoder tf_efficientnetv2_m --logdir runs/segmate_v2_nose

# Start new training (multi-GPU with DDP)
torchrun --nproc_per_node=4 train/training_segmate_v2_nose.py \
       --root processed_dataset --deep_supervision \
       --encoder tf_efficientnetv2_m --logdir runs/segmate_v2_nose

# Resume training from checkpoint
torchrun --nproc_per_node=4 train/training_segmate_v2_nose.py \
       --root processed_dataset --deep_supervision \
       --encoder tf_efficientnetv2_m --logdir runs/segmate_v2_nose --resume checkpoints/segmate_v2_nose_latest.pth
```
"""
from __future__ import annotations

# imports
import os, sys, argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import torchmetrics                # metrics lib
from tqdm import tqdm              # progress-bar

# local modules (repo-relative)
sys.path.append(".")
from models.segmate_v2_nose import SegMateV2NoSE
from losses.custom_unetpp_loss import CustomUNetPlusPlusLoss
from dataloaders.balanced_dataloader import get_dataloaders_balanced as get_dataloaders

# Label names (channel order) - keep in sync with dataset prep
ORGANS = [
    "Left Lung", "Right Lung", "Cord", "Esophagus",
    "Liver", "Left Kidney", "Right Kidney", "Aorta", "Trachea",
]

# distributed setup

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

# arg-parser helper

def build_parser():
    """CLI arguments with sensible defaults."""
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="processed_dataset", help="Dataset root folder")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--logdir", default="runs/segmate_v2_nose")
    p.add_argument("--deep_supervision", action="store_true")
    p.add_argument("--log_interval", type=int, default=5, help="Log validation details every N epochs")
    p.add_argument("--save_interval", type=int, default=5, help="Save checkpoint every N epochs")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume training from")
    
    # Encoder selection
    p.add_argument("--encoder", type=str, default="tf_efficientnetv2_m",
                   choices=["tf_efficientnetv2_s", "tf_efficientnetv2_m", "tf_efficientnetv2_l"],
                   help="EfficientNetV2 encoder name (default: tf_efficientnetv2_m)")
    p.add_argument("--no_pretrained", action="store_true",
                   help="Disable pretrained weights for encoder")

    return p

# average-meter util

class AverageMeter:
    """Keeps running average for any scalar metric."""
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum, self.count = 0.0, 0
    def update(self, val: float, n: int = 1):
        self.sum += val * n; self.count += n
    @property
    def avg(self):
        return self.sum / max(self.count, 1)

# device info helper

def print_device_summary() -> None:
    """Pretty-print CPU/GPU information."""
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

# Custom Dice implementation for torchmetrics 1.7.1

class CustomDice(nn.Module):
    """Custom Dice implementation specifically for torchmetrics 1.7.1"""
    def __init__(self, num_classes, threshold=None):
        super().__init__()
        self.num_classes = num_classes
        self.threshold = threshold

    def forward(self, preds, target):
        """
        Compute Dice scores for each class

        Args:
            preds: Predictions tensor of shape [B, C, H, W]
            target: Ground truth tensor of shape [B, C, H, W]

        Returns:
            Tensor of Dice scores for each class [C]
        """
        if self.threshold is not None:
            preds = (preds >= self.threshold).float()

        # Flatten spatial dimensions
        preds = preds.view(preds.size(0), preds.size(1), -1)
        target = target.view(target.size(0), target.size(1), -1)

        # Compute intersection and union
        intersection = (preds * target).sum(dim=2)
        cardinality = preds.sum(dim=2) + target.sum(dim=2)

        # Dice = 2 * intersection / cardinality
        dice = 2 * intersection / (cardinality + 1e-8)  # Add epsilon to avoid division by zero

        # Average over batch dimension, return per-class scores
        return dice.mean(dim=0)

# checkpoint save/load functions

def save_checkpoint(model, optimizer, scheduler, epoch, global_step, best_hd, cfg,
                    filename='segmate_v2_nose_latest.pth'):
    """
    Save a comprehensive checkpoint for training resumption

    Args:
        model: The model to save
        optimizer: The optimizer state
        scheduler: The LR scheduler state
        epoch: Current epoch number
        global_step: Global step counter
        best_hd: Best dice score so far
        cfg: Configuration arguments
        filename: Name of checkpoint file
    """
    # Only save on main process
    if not is_main_process():
        return None

    checkpoint_path = Path('checkpoints') / filename
    Path('checkpoints').mkdir(exist_ok=True)

    # Prepare model state (handle DDP wrapper)
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    elif isinstance(model, nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    # Build checkpoint dict with all necessary info
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_hd': best_hd,
        'config': {
            'deep_supervision': cfg.deep_supervision,
            'batch_size': cfg.batch_size,
            'lr': cfg.lr,
            'logdir': cfg.logdir,
            'epochs': cfg.epochs,
            'encoder': cfg.encoder
        },
        'model_name': 'SegMateV2NoSE'
    }

    # Save to disk
    torch.save(checkpoint, checkpoint_path)
    print(f"-> Checkpoint saved to {checkpoint_path}")

    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """
    Load a training checkpoint

    Args:
        checkpoint_path: Path to the checkpoint file
        model: The model to load weights into
        optimizer: The optimizer to load state into
        scheduler: The LR scheduler to load state into

    Returns:
        dict: Checkpoint data including epoch, global_step, best_hd, config
    """
    if is_main_process():
        print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load model weights (handle DDP wrapper)
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer and scheduler states
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Return metadata for resumption
    return {
        'epoch': checkpoint['epoch'],
        'global_step': checkpoint['global_step'],
        'best_hd': checkpoint['best_hd'],
        'config': checkpoint.get('config', {})
    }

# pretty logging functions

def log_header(title: str, width: int = 80) -> None:
    """Print a section header with nice formatting"""
    if not is_main_process():
        return
    line = "=" * width
    print(f"\n{line}")
    print(f" {title} ".center(width, "="))
    print(f"{line}\n")

def log_metrics_summary(epoch: int, epochs: int, phase: str, metrics: Dict[str, float],
                        lr: float = None, time_elapsed: float = None) -> None:
    """Print a summary of metrics for train/val phase"""
    if not is_main_process():
        return

    header = f"[Epoch {epoch}/{epochs}] {phase.upper()}"
    if lr is not None:
        header += f" (LR: {lr:.2e})"
    if time_elapsed is not None:
        header += f" - {time_elapsed:.1f}s"

    print(f"\n{header}")
    print("-" * len(header))

    # Main metrics
    print(f"Loss: {metrics['loss']:.4f}")
    if 'dice_loss' in metrics:
        print(f"Component Losses: Dice={metrics['dice_loss']:.4f}, CE={metrics['ce_loss']:.4f}, "
              f"Boundary={metrics['boundary_loss']:.4f}, Presence={metrics['presence_loss']:.4f}")

    if 'hard_dice_mean' in metrics:
        print(f"Mean Hard-Dice: {metrics['hard_dice_mean']:.4f}")

    if phase == 'val' and 'pres_acc' in metrics and 'pres_auc' in metrics:
        print(f"Presence: Accuracy={metrics['pres_acc']:.4f}, AUC={metrics['pres_auc']:.4f}")

def log_organ_metrics(metrics: Dict[str, float]) -> None:
    """Print detailed per-organ metrics"""
    if not is_main_process():
        return

    print("\nPer-Organ Hard-Dice Scores:")
    print("-" * 50)

    # Find the longest organ name for alignment
    max_len = max(len(organ) for organ in ORGANS)

    # Print metrics for each organ with aligned columns
    for i, organ in enumerate(ORGANS):
        dice_key = f'hard_c{i}'
        if dice_key in metrics:
            score = metrics[dice_key]
            print(f"{organ:<{max_len}} : {score:.4f}")

def log_deep_supervision_metrics(metrics: Dict[str, float]) -> None:
    """Print metrics for deep supervision outputs"""
    if not is_main_process():
        return

    if any(k.startswith('ds_') for k in metrics):
        print("\nDeep Supervision Metrics:")
        print("-" * 50)

        for k, v in sorted(metrics.items()):
            if k.startswith('ds_'):
                print(f"{k}: {v:.4f}")

def log_resumption_info(checkpoint_data):
    """Print information about the resumed training"""
    if not is_main_process():
        return

    log_header("RESUMING TRAINING", 80)
    print(f"Resuming from epoch {checkpoint_data['epoch']}")
    print(f"Global step: {checkpoint_data['global_step']}")
    print(f"Best hard-dice so far: {checkpoint_data['best_hd']:.4f}")

    # Print config if available
    if 'config' in checkpoint_data and checkpoint_data['config']:
        print("\nCheckpoint configuration:")
        for k, v in checkpoint_data['config'].items():
            print(f"  {k}: {v}")
    print()

# main loop

def main(cfg):
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if is_main_process():
        print_device_summary()
        print("=" * 80)
        print(f" ABLATION STUDY: SegMateV2NoSE with {cfg.encoder} ")
        print("=" * 80)
        print("Getting the DataLoaders")

    # DataLoaders (with distributed sampler if needed)
    train_ld, val_ld, _ = get_dataloaders(cfg.root, cfg.batch_size, num_workers=12)

    # Replace samplers with distributed versions if using DDP
    if world_size > 1:
        train_sampler = DistributedSampler(train_ld.dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_ld.dataset, num_replicas=world_size, rank=rank)

        # Create new dataloaders with distributed samplers
        train_ld = torch.utils.data.DataLoader(
            train_ld.dataset,
            batch_size=cfg.batch_size,
            sampler=train_sampler,
            num_workers=12,
            pin_memory=True
        )
        val_ld = torch.utils.data.DataLoader(
            val_ld.dataset,
            batch_size=cfg.batch_size,
            sampler=val_sampler,
            num_workers=12,
            pin_memory=True
        )
    else:
        train_sampler = None
        val_sampler = None

    if is_main_process():
        print(f"Getting the model: SegMateV2NoSE with encoder {cfg.encoder}")

    # Model + loss
    model = SegMateV2NoSE(
        num_classes=len(ORGANS),
        deep_supervision=cfg.deep_supervision,
        encoder_name=cfg.encoder,
        pretrained=not cfg.no_pretrained
    ).to(device)

    # Wrap with DDP if using distributed training
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
        model = nn.DataParallel(model)        # fallback to DataParallel

    class_w = torch.tensor([1,1,5,5,1,3,3,6,4], device=device, dtype=torch.float32)
    # Same loss weights as original: dice=2.0, ce=1.0, boundary=0.1, presence=0.0
    loss_weights = {'dice': 2.0, 'ce': 1.0, 'boundary': 0.1, 'presence': 0.0}
    loss_fn  = CustomUNetPlusPlusLoss(class_weights=class_w, weights=loss_weights, deep_supervision_weight=0.2)

    opt   = AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    # Initialize training state
    global_step, best_hd, start_epoch = 0, 0.0, 1

    # Resume from checkpoint if specified
    if cfg.resume:
        if os.path.exists(cfg.resume):
            checkpoint_data = load_checkpoint(cfg.resume, model, opt, sched)
            global_step = checkpoint_data['global_step']
            best_hd = checkpoint_data['best_hd']
            start_epoch = checkpoint_data['epoch'] + 1  # Start from next epoch
            log_resumption_info(checkpoint_data)
        else:
            if is_main_process():
                print(f"Warning: Checkpoint file {cfg.resume} not found. Starting from scratch.")

    # TensorBoard writer (only on main process)
    writer = None
    if is_main_process():
        if cfg.resume and os.path.exists(cfg.logdir):
            writer = SummaryWriter(cfg.logdir, purge_step=global_step)
            print(f"Continuing TensorBoard logs in: {cfg.logdir}")
        else:
            writer = SummaryWriter(cfg.logdir)
            print(f"Creating new TensorBoard logs in: {cfg.logdir}")

    # Metrics (soft/hard Dice, presence)
    dice_soft = CustomDice(len(ORGANS), threshold=None).to(device)
    dice_hard = CustomDice(len(ORGANS), threshold=0.5).to(device)

    # For torchmetrics 1.7.1, use num_labels instead of older versions' arguments
    pres_acc = torchmetrics.classification.MultilabelAccuracy(num_labels=len(ORGANS)).to(device)
    pres_auc = torchmetrics.classification.MultilabelAUROC(num_labels=len(ORGANS)).to(device)

    # Print training info (only on main process)
    if is_main_process():
        if start_epoch == 1:  # New training
            log_header("TRAINING START - SegMateV2NoSE Ablation", 80)
        print(f"Training for epochs {start_epoch} to {cfg.epochs}")
        print(f"Batch size: {cfg.batch_size}")
        print(f"Initial learning rate: {cfg.lr}")
        print(f"Deep supervision: {'Enabled' if cfg.deep_supervision else 'Disabled'}")
        print(f"Logs will be saved to: {cfg.logdir}")
        print(f"Checkpoints will be saved every {cfg.save_interval} epochs")
        print(f"Detailed validation every {cfg.log_interval} epochs\n")

    # training epochs
    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_start_time = time.time()

        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # TRAIN
        model.train(); meter = defaultdict(AverageMeter)

        # Create progress bar only on main process
        if is_main_process():
            pbar = tqdm(train_ld, desc=f"Epoch {epoch}/{cfg.epochs} [train]", ncols=110)
        else:
            pbar = train_ld

        for img, lbl in pbar:
            img, lbl = img.to(device), lbl.to(device)

            opt.zero_grad()
            outputs = model(img)
            loss, components = loss_fn(outputs, lbl, cfg.deep_supervision)
            loss.backward(); opt.step()

            # update meters
            meter['loss'].update(loss.item(), img.size(0))
            for k, v in components.items():
                meter[k].update(v, img.size(0))

            # Skip deep supervision metrics during training to prevent memory leak
            # Deep supervision metrics are only computed during validation

            # periodic TensorBoard write (only on main process)
            if is_main_process() and writer and global_step % 50 == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                for k in ['dice_loss','ce_loss','boundary_loss','presence_loss']:
                    writer.add_scalar(f'train/{k}', components[k], global_step)
            global_step += 1

            # Update progress bar with loss (only on main process)
            if is_main_process():
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        if is_main_process():
            print("Finished training")

        # Get current learning rate
        current_lr = opt.param_groups[0]['lr']

        if is_main_process():
            print(f"Current learning rate:\n{current_lr}")

        # Compute training metrics summary
        train_metrics = {k: v.avg for k, v in meter.items()}
        train_time = time.time() - epoch_start_time

        if is_main_process():
            print(f"Train metrics:\n{train_metrics}")
            print(train_time)

        # Log training metrics to TensorBoard (only on main process)
        if is_main_process() and writer:
            writer.add_scalar('train/loss_epoch', meter['loss'].avg, epoch)
            writer.add_scalar('train/lr', current_lr, epoch)

        # VALIDATION
        if is_main_process():
            print(f"\n--- Epoch {epoch}: Starting Validation ---")

        # Set epoch for distributed sampler
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        val_start_time = time.time()
        model.eval(); vmeter = defaultdict(AverageMeter)

        # Reset metrics at start of each epoch to prevent accumulation
        pres_acc.reset()
        pres_auc.reset()

        # Create progress bar only on main process
        if is_main_process():
            pbar = tqdm(val_ld, desc=f"Epoch {epoch}/{cfg.epochs} [val]", ncols=110)
        else:
            pbar = val_ld

        with torch.no_grad():
            for img, lbl in pbar:
                img, lbl = img.to(device), lbl.to(device)
                outputs = model(img)
                loss, components = loss_fn(outputs, lbl, cfg.deep_supervision)
                vmeter['loss'].update(loss.item(), img.size(0))
                for k, v in components.items():
                    vmeter[k].update(v, img.size(0))

                # metrics on main segmentation head
                if cfg.deep_supervision:
                    seg_logits, _, _, _ = outputs  # Unpack properly for deep supervision
                else:
                    seg_logits = outputs[0]
                prob  = torch.sigmoid(seg_logits)
                hard  = (prob >= 0.5).float()
                s_d   = dice_soft(prob, lbl)
                h_d   = dice_hard(hard, lbl)
                for c in range(len(ORGANS)):
                    vmeter[f'hard_c{c}'].update(h_d[c].item(), img.size(0))

                # Track deep supervision outputs if enabled
                if cfg.deep_supervision and len(outputs) == 4:  # seg, boundary, presence, deep_outs
                    deep_outs = outputs[3]  # Get deep supervision outputs
                    for i, ds_out in enumerate(deep_outs, 1):
                        prob = torch.sigmoid(ds_out)
                        hard = (prob >= 0.5).float()
                        ds_dice = dice_hard(hard, lbl)
                        ds_mean = ds_dice.mean().item()
                        vmeter[f'ds_{i}_dice'].update(ds_mean, img.size(0))

                # presence head metrics
                gt_presence = (lbl.sum((2,3)) > 0).float()
                pres_acc.update(prob.mean((2,3)), gt_presence)
                pres_auc.update(prob.mean((2,3)), gt_presence.long())

                # Update progress bar (only on main process)
                if is_main_process():
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_time = time.time() - val_start_time
        if is_main_process():
            print(f"--- Epoch {epoch}: Validation completed in {val_time:.1f} seconds ---")

        # mean hard-Dice over classes
        mean_hd = sum(vmeter[f'hard_c{c}'].avg for c in range(len(ORGANS))) / len(ORGANS)

        # Compute validation metrics summary
        val_metrics = {k: v.avg for k, v in vmeter.items()}
        val_metrics['hard_dice_mean'] = mean_hd
        val_metrics['pres_acc'] = pres_acc.compute().item()
        val_metrics['pres_auc'] = pres_auc.compute().item()
        val_time = time.time() - val_start_time
        total_time = time.time() - epoch_start_time

        # TensorBoard logging (epoch-level, only on main process)
        if is_main_process() and writer:
            writer.add_scalar('val/loss', vmeter['loss'].avg, epoch)
            writer.add_scalar('val/hard_dice_mean', mean_hd, epoch)
            writer.add_scalar('val/pres_acc', val_metrics['pres_acc'], epoch)
            writer.add_scalar('val/pres_auc', val_metrics['pres_auc'], epoch)
            for c, organ in enumerate(ORGANS):
                writer.add_scalar(f'val/hard_dice_{organ}', vmeter[f'hard_c{c}'].avg, epoch)

        # LR scheduler step
        sched.step(vmeter['loss'].avg)

        # Print epoch summary (only on main process)
        log_metrics_summary(epoch, cfg.epochs, "train", train_metrics, current_lr, train_time)
        log_metrics_summary(epoch, cfg.epochs, "val", val_metrics, time_elapsed=val_time)

        # Print detailed metrics every log_interval epochs or on last epoch
        if epoch % cfg.log_interval == 0 or epoch == cfg.epochs:
            log_organ_metrics(val_metrics)
            log_deep_supervision_metrics(val_metrics)
            if is_main_process():
                print(f"\nTotal epoch time: {total_time:.1f}s")

        # Print simple summary line for other epochs
        else:
            if is_main_process():
                print(f"\nSummary: val_loss={val_metrics['loss']:.4f}, dice={mean_hd:.4f}, time={total_time:.1f}s")

        # Save periodic checkpoint for resumption (only on main process)
        if epoch % cfg.save_interval == 0:
            save_checkpoint(model, opt, sched, epoch, global_step, best_hd, cfg)

        # save best checkpoint (based on mean hard-Dice, only on main process)
        if mean_hd > best_hd:
            best_hd = mean_hd
            if is_main_process():
                Path('checkpoints').mkdir(exist_ok=True)
                checkpoint_path = 'checkpoints/segmate_v2_nose_best.pth'

                # Save model state (handle DDP wrapper)
                if isinstance(model, DDP):
                    model_state = model.module.state_dict()
                elif isinstance(model, nn.DataParallel):
                    model_state = model.module.state_dict()
                else:
                    model_state = model.state_dict()

                torch.save(model_state, checkpoint_path)
                print(f"\n* New best model saved to {checkpoint_path} (mean hard-Dice {best_hd:.4f})")

                # Also save it as a resumable checkpoint
                save_checkpoint(model, opt, sched, epoch, global_step, best_hd, cfg,
                              filename='segmate_v2_nose_best_resumable.pth')

    if is_main_process():
        log_header("TRAINING COMPLETE - SegMateV2NoSE Ablation", 80)
        print(f"Best validation Dice: {best_hd:.4f}")
        print(f"Model saved to: checkpoints/segmate_v2_nose_best.pth")
        print(f"Final checkpoint saved to: checkpoints/segmate_v2_nose_latest.pth")
        print(f"Logs saved to: {cfg.logdir}")

        # Save final checkpoint
        save_checkpoint(model, opt, sched, epoch, global_step, best_hd, cfg)
        if writer:
            writer.close()

    # Cleanup distributed training
    cleanup_distributed()

# entry point
if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
