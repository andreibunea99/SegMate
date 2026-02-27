"""
Fine-tune SegMate models (2.5D) on target datasets (SegTHOR, AMOS22, etc.)
=============================================================================
Unified fine-tuning script supporting all 7 benchmark model types.
Loads pretrained weights from TotalSegmentator-trained checkpoints and
fine-tunes on a target dataset with a lower learning rate.

Supported model types:
  SegMateFiLM, SegMateV2, SegMate, SegMateMambaOut,
  VanillaSegMateMambaOut, SegMateFastViT, VanillaSegMateFastViT

Examples
--------
# Fine-tune SegMateFiLM on SegTHOR (single GPU, uses adaptive class weights)
python train/finetune_25D.py \
  --model_type SegMateFiLM --encoder_name tf_efficientnetv2_m \
  --pretrained_checkpoint archive/exp20/segmate_film_tf_efficientnetv2_m_25D_best.pth \
  --root processed_dataset_segthor --checkpoint_dir segthor_checkpoints --epochs 25

# Fine-tune SegMateMambaOut on AMOS22 (multi-GPU)
torchrun --nproc_per_node=3 train/finetune_25D.py \
  --model_type SegMateMambaOut --encoder_name mambaout_tiny \
  --pretrained_checkpoint archive/exp15/segmate_mambaout_tiny_25D_best.pth \
  --root processed_dataset_amos22 --checkpoint_dir amos_checkpoints --epochs 25
"""

from __future__ import annotations

# ----------------------------- imports -----------------------------
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from tqdm import tqdm

# local modules (repo-relative)
sys.path.append(".")

from models.segmate_v2_film import SegMateFiLM
from models.segmate_v2 import SegMate as SegMateV2
from models.vanilla_segmate_v2 import SegMate
from models.segmate_mambaout import SegMateMambaOut
from models.vanilla_segmate_mambaout import SegMateMambaOut as VanillaSegMateMambaOut
from models.segmate_fastvit import SegMateFastViT
from models.vanilla_segmate_fastvit import SegMateFastViT as VanillaSegMateFastViT
from models.segmate_fastvit_film import SegMateFastViTFiLM
from models.segmate_mambaout_film import SegMateMambaOutFiLM
from losses.custom_unetpp_loss import CustomUNetPlusPlusLoss

# Model registry
MODEL_CLASSES = {
    "SegMateFiLM": SegMateFiLM,
    "SegMateV2": SegMateV2,
    "SegMate": SegMate,
    "SegMateMambaOut": SegMateMambaOut,
    "VanillaSegMateMambaOut": VanillaSegMateMambaOut,
    "SegMateFastViT": SegMateFastViT,
    "VanillaSegMateFastViT": VanillaSegMateFastViT,
    "SegMateFastViTFiLM": SegMateFastViTFiLM,
    "SegMateMambaOutFiLM": SegMateMambaOutFiLM,
}

# Checkpoint name mapping (model_type -> prefix used in saved files)
MODEL_NAME_MAP = {
    "SegMateFiLM": "segmate_film",
    "SegMateV2": "segmatev2",
    "SegMate": "segmate",
    "SegMateMambaOut": "segmate_mambaout",
    "VanillaSegMateMambaOut": "vanilla_mambaout",
    "SegMateFastViT": "segmate_fastvit",
    "VanillaSegMateFastViT": "vanilla_fastvit",
    "SegMateFastViTFiLM": "segmate_fastvit_film",
    "SegMateMambaOutFiLM": "segmate_mambaout_film",
}

# Label names (channel order) - all 9 channels kept for compatibility
ORGANS = [
    "Left Lung", "Right Lung", "Cord", "Esophagus",
    "Liver", "Left Kidney", "Right Kidney", "Aorta", "Trachea",
]


# -------------------------- distributed setup -------------------------

def setup_distributed():
    """Initialize distributed training if available."""
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
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if current process is the main process."""
    return (not dist.is_initialized()) or dist.get_rank() == 0


# -------------------------- dataset detection -------------------------

def detect_dataset_type(root_path: str) -> str:
    """
    Detect dataset type from root path for adaptive class weighting.
    
    Returns:
        'segthor': SegTHOR dataset (3 organs: esophagus, aorta, trachea)
        'amos22': AMOS22 dataset
        'totalseg': TotalSegmentator or other (default)
    """
    root_lower = root_path.lower()
    if "segthor" in root_lower:
        return "segthor"
    elif "amos" in root_lower:
        return "amos22"
    else:
        return "totalseg"


# -------------------------- arg-parser helper -------------------------

def build_parser():
    """CLI arguments for fine-tuning."""
    p = argparse.ArgumentParser(description="Fine-tune SegMate models (2.5D)")

    # Required arguments
    p.add_argument("--model_type", type=str, required=True,
                   choices=list(MODEL_CLASSES.keys()),
                   help="Model class to fine-tune")
    p.add_argument("--encoder_name", type=str, required=True,
                   help="Encoder name (e.g., tf_efficientnetv2_m, mambaout_tiny, fastvit_t12)")
    p.add_argument("--pretrained_checkpoint", type=str, required=True,
                   help="Path to TotalSegmentator-trained .pth checkpoint")

    # Dataset / output
    p.add_argument("--root", type=str, default="processed_dataset_segthor",
                   help="Dataset root folder")
    p.add_argument("--checkpoint_dir", type=str, default="segthor_checkpoints",
                   help="Output directory for fine-tuned checkpoints")

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=25,
                   help="Number of epochs (default: 25, recommended for SegTHOR)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-6,
                   help="Learning rate (default: 5e-6, conservative to prevent catastrophic forgetting)")
    p.add_argument("--deep_supervision", action="store_true")
    
    # Fine-tuning specific
    p.add_argument("--freeze_encoder_epochs", type=int, default=5,
                   help="Freeze encoder for first N epochs to prevent catastrophic forgetting (default: 5)")
    p.add_argument("--early_stop_patience", type=int, default=10,
                   help="Stop training if no improvement for N epochs (default: 10, set 0 to disable)")
    p.add_argument("--num_workers", type=int, default=4,
                   help="Number of DataLoader workers (default: 4, reduce if experiencing DataLoader errors)")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to resumable checkpoint (.pth) to resume training from")

    # Logging
    p.add_argument("--logdir", default=None,
                   help="TensorBoard log directory (auto-generated if not specified)")
    p.add_argument("--log_interval", type=int, default=2,
                   help="Log validation details every N epochs")
    p.add_argument("--save_interval", type=int, default=2,
                   help="Save checkpoint every N epochs")

    # FiLM parameters (only used when model_type=SegMateFiLM)
    p.add_argument("--film_hidden", type=int, default=128)
    p.add_argument("--film_layers", type=int, default=3)
    p.add_argument("--film_dropout", type=float, default=0.1)
    p.add_argument("--film_init_scale", type=float, default=0.01)

    return p


# ------------------------- average-meter util ------------------------

class AverageMeter:
    """Keeps running average for any scalar metric."""

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


# ----------------------- device info helper -----------------------

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


# ------------------- Custom Dice implementation ------------------

class CustomDice(nn.Module):
    """Custom Dice implementation for per-organ metrics."""

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


# ------------------- 2.5D wrappers ------------------

class SliceFusion(nn.Module):
    def __init__(self, in_ch: int = 3, mid_ch: int = 16, out_ch: int = 1):
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
    """Standard 2.5D wrapper for non-FiLM models."""
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.fusion = SliceFusion(in_ch=3, mid_ch=16, out_ch=1)
        self.model = backbone

    def forward(self, x25d, z_norm=None):
        x1 = self.fusion(x25d)  # [B,1,H,W]
        if z_norm is not None and hasattr(self.model, "film"):
            return self.model(x1, z_norm=z_norm)
        return self.model(x1)


# --------------- checkpoint loading ---------------

def load_resumable_checkpoint(checkpoint_path, model, opt, sched, device):
    """Load a resumable checkpoint and restore training state.
    
    Returns:
        dict with keys: 'epoch', 'global_step', 'best_dice', 'config'
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Resumable checkpoint not found: {checkpoint_path}")
    
    print(f"Resuming from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid resumable checkpoint format. Expected dict with 'model_state_dict' key.")
    
    # Load model state
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys when loading model")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys when loading model")
    
    # Load optimizer and scheduler
    if "optimizer_state_dict" in checkpoint:
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        print("  Optimizer state restored")
    
    if "scheduler_state_dict" in checkpoint:
        sched.load_state_dict(checkpoint["scheduler_state_dict"])
        print("  Scheduler state restored")
    
    # Extract training progress
    epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0)
    best_dice = checkpoint.get("best_dice", 0.0)
    config = checkpoint.get("config", {})
    
    print(f"  Resuming from epoch {epoch}")
    print(f"  Global step: {global_step}")
    print(f"  Best dice so far: {best_dice:.4f}")
    
    return {
        "epoch": epoch,
        "global_step": global_step,
        "best_dice": best_dice,
        "config": config,
    }


def load_pretrained_weights(model, checkpoint_path, device):
    """Load pretrained weights from a TotalSegmentator-trained checkpoint.

    Handles:
    - Raw state_dict (just the weights)
    - Dict with 'model_state_dict' key (resumable checkpoint)
    - Dict with 'model_state' key (evaluation checkpoint)
    - 'module.' prefix from DDP training
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")

    print(f"Loading pretrained weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract state_dict from various checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        else:
            # Assume the dict IS the state_dict
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Strip 'module.' prefix from DDP-trained checkpoints
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys (expected for architecture changes)")
        for k in missing[:5]:
            print(f"    - {k}")
        if len(missing) > 5:
            print(f"    ... and {len(missing)-5} more")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys")
        for k in unexpected[:5]:
            print(f"    - {k}")
        if len(unexpected) > 5:
            print(f"    ... and {len(unexpected)-5} more")

    if not missing and not unexpected:
        print("  All weights loaded successfully (strict match)")

    return model


# --------------------- pretty logging functions ---------------------

def log_header(title: str, width: int = 80) -> None:
    if not is_main_process():
        return
    line = "=" * width
    print(f"\n{line}")
    print(f" {title} ".center(width, "="))
    print(f"{line}\n")


def log_metrics_summary(epoch: int, epochs: int, phase: str, metrics: Dict[str, float],
                        lr: float = None, time_elapsed: float = None) -> None:
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
        print(
            "Component Losses: "
            f"Dice={metrics['dice_loss']:.4f}, CE={metrics['ce_loss']:.4f}, "
            f"Boundary={metrics['boundary_loss']:.4f}, Presence={metrics['presence_loss']:.4f}"
        )

    if "hard_dice_mean" in metrics:
        print(f"Mean Hard-Dice: {metrics['hard_dice_mean']:.4f}")

    if phase == "val" and "pres_acc" in metrics and "pres_auc" in metrics:
        print(f"Presence: Accuracy={metrics['pres_acc']:.4f}, AUC={metrics['pres_auc']:.4f}")


def log_organ_metrics(metrics: Dict[str, float]) -> None:
    if not is_main_process():
        return

    print("\nPer-Organ Hard-Dice Scores:")
    print("-" * 50)

    max_len = max(len(organ) for organ in ORGANS)
    for i, organ in enumerate(ORGANS):
        dice_key = f"hard_c{i}"
        if dice_key in metrics:
            print(f"{organ:<{max_len}} : {metrics[dice_key]:.4f}")


# ----------------------------- main loop -----------------------------

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

    # ~~~~~~~~~~~~~ Determine model name for checkpoint filenames ~~~~~~~~~~~~
    model_prefix = MODEL_NAME_MAP[cfg.model_type]
    model_name = f"{model_prefix}_{cfg.encoder_name}"

    is_film = cfg.model_type in ("SegMateFiLM", "SegMateFastViTFiLM", "SegMateMambaOutFiLM")

    if is_main_process():
        log_header("FINE-TUNING SETUP", 80)
        print(f"Model type: {cfg.model_type}")
        print(f"Encoder: {cfg.encoder_name}")
        print(f"FiLM model: {is_film}")
        print(f"Pretrained checkpoint: {cfg.pretrained_checkpoint}")
        print(f"Dataset root: {cfg.root}")
        print(f"Output directory: {cfg.checkpoint_dir}")
        print(f"Model name: {model_name}")
        print(f"Epochs: {cfg.epochs}, LR: {cfg.lr}, Batch size: {cfg.batch_size}")
        print(f"Deep supervision: {cfg.deep_supervision}")

    # ~~~~~~~~~~~~~ DataLoaders ~~~~~~~~~~~~
    # Detect dataset type and import appropriate dataloader
    is_amos22 = "amos" in cfg.root.lower()
    
    if is_amos22:
        if is_film:
            from dataloaders.balanced_dataloader_25D_film_amos22 import get_dataloaders_balanced_film_amos22 as get_dataloaders
        else:
            from dataloaders.balanced_dataloader_25D_amos22 import get_dataloaders_balanced_amos22 as get_dataloaders
    else:
        if is_film:
            from dataloaders.balanced_dataloader_25D_film import get_dataloaders_balanced_film as get_dataloaders
        else:
            from dataloaders.balanced_dataloader_25D import get_dataloaders_balanced as get_dataloaders

    if is_main_process():
        print(f"\nLoading data from: {cfg.root} ({'AMOS22' if is_amos22 else 'SegTHOR/TotalSeg'} dataset)")

    train_ld, val_ld, _ = get_dataloaders(cfg.root, cfg.batch_size, num_workers=cfg.num_workers)

    if world_size > 1:
        train_sampler = DistributedSampler(train_ld.dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_ld.dataset, num_replicas=world_size, rank=rank)

        train_ld = torch.utils.data.DataLoader(
            train_ld.dataset,
            batch_size=cfg.batch_size,
            sampler=train_sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        val_ld = torch.utils.data.DataLoader(
            val_ld.dataset,
            batch_size=cfg.batch_size,
            sampler=val_sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
    else:
        train_sampler = None
        val_sampler = None

    # ~~~~~~~~~~~~~ Build model ~~~~~~~~~~~~~
    model_class = MODEL_CLASSES[cfg.model_type]

    if is_film:
        base_model = model_class(
            num_classes=len(ORGANS),
            in_channels=1,
            deep_supervision=cfg.deep_supervision,
            encoder_name=cfg.encoder_name,
            pretrained=False,  # We load our own weights
            film_hidden_dim=cfg.film_hidden,
            film_num_layers=cfg.film_layers,
            film_dropout=cfg.film_dropout,
            film_init_scale=cfg.film_init_scale,
        ).to(device)
    else:
        base_model = model_class(
            num_classes=len(ORGANS),
            in_channels=1,
            deep_supervision=cfg.deep_supervision,
            encoder_name=cfg.encoder_name,
            pretrained=False,  # We load our own weights
        ).to(device)

    model = TwoPointFiveDWrapper(base_model).to(device)

    # ~~~~~~~~~~~~~ Load pretrained weights ~~~~~~~~~~~~~
    model = load_pretrained_weights(model, cfg.pretrained_checkpoint, device)
    model = model.to(device)

    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"\nTotal model params: {total_params:.2f}M")

    # ~~~~~~~~~~~~~ DDP wrapping ~~~~~~~~~~~~~
    if world_size > 1:
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            gradient_as_bucket_view=True,  # More efficient gradient handling
            broadcast_buffers=False,        # Skip buffer broadcast if not needed
            find_unused_parameters=True     # Required: model has dead weights in skip connections
        )
    elif torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # ~~~~~~~~~~~~~ Loss, optimizer, scheduler ~~~~~~~~~~~~~
    # Adaptive class weights based on dataset
    dataset_type = detect_dataset_type(cfg.root)
    
    if dataset_type == "segthor":
        # SegTHOR: Prioritize esophagus (hardest), then aorta, then trachea
        # Channels: [LLung, RLung, Cord, Esophagus, Liver, LKidney, RKidney, Aorta, Trachea]
        class_w = torch.tensor([1, 1, 1, 6, 1, 1, 1, 5, 4], device=device, dtype=torch.float32)
        if is_main_process():
            print(f"\n[Class Weights] SegTHOR-optimized: Esophagus=6, Aorta=5, Trachea=4")
    else:
        # Default for TotalSegmentator / AMOS22
        class_w = torch.tensor([1, 1, 5, 5, 1, 3, 3, 6, 4], device=device, dtype=torch.float32)
        if is_main_process():
            print(f"\n[Class Weights] Default (TotalSegmentator): Esophagus=5, Aorta=6, Trachea=4")
    
    loss_weights = {"dice": 2.0, "ce": 1.0, "boundary": 0.1, "presence": 0.2}
    loss_fn = CustomUNetPlusPlusLoss(class_weights=class_w, weights=loss_weights, deep_supervision_weight=0.2)

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    
    # CosineAnnealingWarmRestarts: periodic LR increases help escape local minima
    # T_0=5: first restart at epoch 5, T_mult=2: periods increase (5→10→20)
    # Gives 3 restart cycles in 25 epochs for thorough fine-tuning
    sched = CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2, eta_min=1e-7)
    if is_main_process():
        print(f"[LR Schedule] CosineAnnealingWarmRestarts: T_0=5, T_mult=2, eta_min=1e-7")

    # Initialize training state
    global_step = 0
    best_dice = 0.0  # Track best mean dice for early stopping
    epochs_no_improve = 0  # Counter for early stopping
    start_epoch = 1
    
    # ~~~~~~~~~~~~~ Resume from checkpoint if specified ~~~~~~~~~~~~~
    if cfg.resume:
        resume_state = load_resumable_checkpoint(cfg.resume, model, opt, sched, device)
        start_epoch = resume_state["epoch"] + 1  # Start from next epoch
        global_step = resume_state["global_step"]
        best_dice = resume_state["best_dice"]
        # Calculate epochs_no_improve based on current and best epoch
        # This is approximate since we don't save it, but better than resetting to 0
        epochs_no_improve = 0  # Reset on resume for safety
        if is_main_process():
            print(f"\n{'='*80}")
            print(f" RESUMING TRAINING FROM EPOCH {start_epoch} ".center(80, "="))
            print(f"{'='*80}\n")

    # ~~~~~~~~~~~~~ TensorBoard writer ~~~~~~~~~~~~~
    writer = None
    if is_main_process():
        logdir = cfg.logdir or f"runs/finetune_{model_name}_{Path(cfg.root).name}"
        writer = SummaryWriter(logdir)
        print(f"TensorBoard logs: {logdir}")

    # ~~~~~~~~~~~~~ Metrics ~~~~~~~~~~~~~
    dice_hard = CustomDice(len(ORGANS), threshold=0.5).to(device)
    pres_acc = torchmetrics.classification.MultilabelAccuracy(num_labels=len(ORGANS)).to(device)
    pres_auc = torchmetrics.classification.MultilabelAUROC(num_labels=len(ORGANS)).to(device)

    # ~~~~~~~~~~~~~ Ensure checkpoint directory exists ~~~~~~~~~~~~~
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if is_main_process():
        if not cfg.resume:
            log_header("FINE-TUNING START", 80)
        print(f"Training from epoch {start_epoch} to {cfg.epochs}")
        if cfg.freeze_encoder_epochs > 0:
            print(f"Encoder will be frozen for first {cfg.freeze_encoder_epochs} epochs")
        if cfg.early_stop_patience > 0:
            print(f"Early stopping enabled: patience={cfg.early_stop_patience} epochs")
        print(f"Checkpoints will be saved to: {cfg.checkpoint_dir}/")
        print(f"  Best: {model_name}_finetuned_best.pth")
        print(f"  Latest: {model_name}_finetuned_latest.pth\n")

    # ===================== training epochs =====================
    for epoch in range(start_epoch, cfg.epochs + 1):
        # Freeze/unfreeze encoder based on epoch
        if cfg.freeze_encoder_epochs > 0:
            if epoch <= cfg.freeze_encoder_epochs:
                # Freeze encoder
                base_model = model.module if isinstance(model, (DDP, nn.DataParallel)) else model
                if hasattr(base_model, 'model') and hasattr(base_model.model, 'encoder'):
                    for param in base_model.model.encoder.parameters():
                        param.requires_grad = False
                    if epoch == 1 and is_main_process():
                        print(f"[Epoch {epoch}] Encoder FROZEN (training decoder only)")
            elif epoch == cfg.freeze_encoder_epochs + 1:
                # Unfreeze encoder
                base_model = model.module if isinstance(model, (DDP, nn.DataParallel)) else model
                if hasattr(base_model, 'model') and hasattr(base_model.model, 'encoder'):
                    for param in base_model.model.encoder.parameters():
                        param.requires_grad = True
                    if is_main_process():
                        print(f"[Epoch {epoch}] Encoder UNFROZEN (full model training)")
        
        epoch_start_time = time.time()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # -------------------- TRAIN --------------------
        model.train()
        meter = defaultdict(AverageMeter)

        pbar = tqdm(train_ld, desc=f"Epoch {epoch}/{cfg.epochs} [train]", ncols=110) if is_main_process() else train_ld

        for batch in pbar:
            if is_film:
                img, lbl, z_norm = batch
                img, lbl, z_norm = img.to(device), lbl.to(device), z_norm.to(device)
            else:
                img, lbl = batch
                img, lbl = img.to(device), lbl.to(device)
                z_norm = None

            opt.zero_grad(set_to_none=True)

            if is_film:
                outputs = model(img, z_norm=z_norm)
            else:
                outputs = model(img)

            loss, components = loss_fn(outputs, lbl, cfg.deep_supervision)
            loss.backward()
            opt.step()

            meter["loss"].update(loss.item(), img.size(0))
            for k, v in components.items():
                meter[k].update(v, img.size(0))

            if is_main_process() and writer and global_step % 50 == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                for k in ["dice_loss", "ce_loss", "boundary_loss", "presence_loss"]:
                    if k in components:
                        writer.add_scalar(f"train/{k}", components[k], global_step)

            global_step += 1
            if is_main_process():
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        current_lr = opt.param_groups[0]["lr"]
        train_metrics = {k: v.avg for k, v in meter.items()}
        train_time = time.time() - epoch_start_time

        if is_main_process() and writer:
            writer.add_scalar("train/loss_epoch", meter["loss"].avg, epoch)
            writer.add_scalar("train/lr", current_lr, epoch)

        # -------------------- VALIDATION --------------------
        if is_main_process():
            print(f"\n--- Epoch {epoch}: Starting Validation ---")

        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        val_start_time = time.time()
        model.eval()
        vmeter = defaultdict(AverageMeter)
        pres_acc.reset()
        pres_auc.reset()

        pbar = tqdm(val_ld, desc=f"Epoch {epoch}/{cfg.epochs} [val]", ncols=110) if is_main_process() else val_ld

        with torch.no_grad():
            for batch in pbar:
                if is_film:
                    img, lbl, z_norm = batch
                    img, lbl, z_norm = img.to(device), lbl.to(device), z_norm.to(device)
                else:
                    img, lbl = batch
                    img, lbl = img.to(device), lbl.to(device)
                    z_norm = None

                if is_film:
                    outputs = model(img, z_norm=z_norm)
                else:
                    outputs = model(img)

                loss, components = loss_fn(outputs, lbl, cfg.deep_supervision)

                vmeter["loss"].update(loss.item(), img.size(0))
                for k, v in components.items():
                    vmeter[k].update(v, img.size(0))

                # Metrics on main segmentation head
                if cfg.deep_supervision:
                    seg_logits, _, _, _ = outputs
                else:
                    seg_logits = outputs[0]

                prob = torch.sigmoid(seg_logits)
                hard = (prob >= 0.5).float()

                h_d = dice_hard(hard, lbl)
                for c in range(len(ORGANS)):
                    vmeter[f"hard_c{c}"].update(h_d[c].item(), img.size(0))

                # Presence head metrics
                gt_presence = (lbl.sum((2, 3)) > 0).float()
                pres_acc.update(prob.mean((2, 3)), gt_presence)
                pres_auc.update(prob.mean((2, 3)), gt_presence.long())

                if is_main_process():
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_time = time.time() - val_start_time

        mean_hd = sum(vmeter[f"hard_c{c}"].avg for c in range(len(ORGANS))) / len(ORGANS)

        val_metrics = {k: v.avg for k, v in vmeter.items()}
        val_metrics["hard_dice_mean"] = mean_hd
        val_metrics["pres_acc"] = pres_acc.compute().item()
        val_metrics["pres_auc"] = pres_auc.compute().item()

        # TensorBoard logging (epoch-level)
        if is_main_process() and writer:
            writer.add_scalar("val/loss", vmeter["loss"].avg, epoch)
            writer.add_scalar("val/hard_dice_mean", mean_hd, epoch)
            writer.add_scalar("val/pres_acc", val_metrics["pres_acc"], epoch)
            writer.add_scalar("val/pres_auc", val_metrics["pres_auc"], epoch)
            for c, organ in enumerate(ORGANS):
                writer.add_scalar(f"val/hard_dice_{organ}", vmeter[f"hard_c{c}"].avg, epoch)

        # LR scheduler step (CosineAnnealingWarmRestarts is epoch-based)
        sched.step()
        
        # Print epoch summary
        log_metrics_summary(epoch, cfg.epochs, "train", train_metrics, current_lr, train_time)
        log_metrics_summary(epoch, cfg.epochs, "val", val_metrics, time_elapsed=val_time)

        if epoch % cfg.log_interval == 0 or epoch == cfg.epochs:
            log_organ_metrics(val_metrics)
            if is_main_process():
                print(f"\nTotal epoch time: {time.time() - epoch_start_time:.1f}s")
        else:
            if is_main_process():
                print(f"\nSummary: val_loss={val_metrics['loss']:.4f}, dice={mean_hd:.4f}, time={time.time() - epoch_start_time:.1f}s")
        
        # Early stopping trigger
        if cfg.early_stop_patience > 0 and epochs_no_improve >= cfg.early_stop_patience:
            if is_main_process():
                print(f"\n⚠️  Early stopping triggered! No improvement for {cfg.early_stop_patience} epochs.")
                print(f"Best mean dice: {best_dice:.4f}")
            break

        # Save periodic checkpoint (resumable)
        if epoch % cfg.save_interval == 0:
            if is_main_process():
                if isinstance(model, (DDP, nn.DataParallel)):
                    model_state = model.module.state_dict()
                else:
                    model_state = model.state_dict()

                resumable_path = Path(cfg.checkpoint_dir) / f"{model_name}_finetuned_latest.pth"
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model_state,
                    "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": sched.state_dict(),
                    "best_dice": best_dice,
                    "config": {
                        "model_type": cfg.model_type,
                        "encoder_name": cfg.encoder_name,
                        "deep_supervision": cfg.deep_supervision,
                        "batch_size": cfg.batch_size,
                        "lr": cfg.lr,
                        "epochs": cfg.epochs,
                        "root": cfg.root,
                    },
                }, resumable_path)
                print(f"-> Periodic checkpoint saved to {resumable_path}")

        # Save best checkpoint (based on mean hard-Dice)
        if mean_hd > best_dice:
            best_dice = mean_hd
            epochs_no_improve = 0  # Reset counter on improvement
            if is_main_process():
                if isinstance(model, (DDP, nn.DataParallel)):
                    model_state = model.module.state_dict()
                else:
                    model_state = model.state_dict()

                # Save lightweight best (just weights, for evaluation)
                best_path = Path(cfg.checkpoint_dir) / f"{model_name}_finetuned_best.pth"
                torch.save(model_state, best_path)
                print(f"\n* New best model saved to {best_path} (mean hard-Dice {best_dice:.4f})")

                # Save resumable best
                resumable_best = Path(cfg.checkpoint_dir) / f"{model_name}_finetuned_best_resumable.pth"
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model_state,
                    "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": sched.state_dict(),
                    "best_dice": best_dice,
                    "config": {
                        "model_type": cfg.model_type,
                        "encoder_name": cfg.encoder_name,
                        "deep_supervision": cfg.deep_supervision,
                        "batch_size": cfg.batch_size,
                        "lr": cfg.lr,
                        "epochs": cfg.epochs,
                        "root": cfg.root,
                    },
                }, resumable_best)
        else:
            # No improvement
            epochs_no_improve += 1
            if is_main_process() and cfg.early_stop_patience > 0:
                print(f"Early stopping: {epochs_no_improve}/{cfg.early_stop_patience} epochs without improvement")

    # ~~~~~~~~~~~~~ Final summary ~~~~~~~~~~~~~
    if is_main_process():
        # Save final latest checkpoint
        if isinstance(model, (DDP, nn.DataParallel)):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()

        final_path = Path(cfg.checkpoint_dir) / f"{model_name}_finetuned_latest.pth"
        torch.save({
            "epoch": cfg.epochs,
            "global_step": global_step,
            "model_state_dict": model_state,
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "best_dice": best_dice,
            "config": {
                "model_type": cfg.model_type,
                "encoder_name": cfg.encoder_name,
                "deep_supervision": cfg.deep_supervision,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "epochs": cfg.epochs,
                "root": cfg.root,
            },
        }, final_path)

        log_header("FINE-TUNING COMPLETE", 80)
        print(f"Best validation Dice: {best_dice:.4f}")
        print(f"Best model: {cfg.checkpoint_dir}/{model_name}_finetuned_best.pth")
        print(f"Latest checkpoint: {cfg.checkpoint_dir}/{model_name}_finetuned_latest.pth")
        if writer:
            writer.close()

    cleanup_distributed()


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
