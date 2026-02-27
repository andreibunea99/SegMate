# evaluation/metrics_evaluation_segthor_25D.py
"""
SegTHOR-specific metrics evaluation (2.5D).
Adapted from metrics_evaluation_25D.py for SegTHOR dataset.

SegTHOR Note:
-------------
- Dataset contains 3 overlapping organs: esophagus, trachea, aorta
- Other 6 channels (lungs, kidneys, liver, cord) are zero-filled
- Metrics computed only for organs with GT presence
- Uses processed_dataset_segthor/ as default dataset path

Usage:
------
# Evaluate on validation set
python evaluation/metrics_evaluation_segthor_25D.py \
  --split val \
  --model_path checkpoints/segmate_film_tf_efficientnetv2_s_25D_best.pth

# Evaluate on test set
python evaluation/metrics_evaluation_segthor_25D.py \
  --split test \
  --model_path checkpoints/segmate_segthor_finetune_best.pth \
  --deep_supervision
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from medpy.metric.binary import dc, hd95, precision, recall
from sklearn.metrics import jaccard_score, accuracy_score, roc_auc_score
import argparse
import sys

sys.path.append(".")

ORGANS = [
    "left_lung", "right_lung", "cord",
    "esophagus", "liver", "left_kidney", "right_kidney",
    "aorta", "trachea"
]

# Default paths (override via CLI)
MODEL_PATH = "checkpoints/segmate_film_tf_efficientnetv2_s_25D_best.pth"
ROOT_DIR = "processed_dataset_segthor"  # SegTHOR-specific
CSV_PATH = "evaluation/metrics_summary_segthor.csv"  # SegTHOR-specific
os.makedirs("evaluation", exist_ok=True)


# ----------------------------
# 2.5D wrapper: 3 slices -> 1 pseudo-slice -> backbone (expects 1ch)
# ----------------------------
class SliceFusion(nn.Module):
    """
    Learned fusion: (t-1, t, t+1) stacked as 3 channels -> 1 pseudo-slice.
    """
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
    """
    Wrap a 2D backbone (expects 1-channel) with a fusion module that converts 3-channel 2.5D input to 1-channel.
    Input:  [B, 3, H, W]
    Output: whatever backbone returns (same as before)
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.fusion = SliceFusion(in_ch=3, mid_ch=16, out_ch=1)
        self.model = backbone

    def forward(self, x25d, z_norm=None):
        x1 = self.fusion(x25d)   # [B, 1, H, W]
        if z_norm is not None and hasattr(self.model, "film"):
            return self.model(x1, z_norm=z_norm)
        return self.model(x1)


def _load_state_dict_safely(model: nn.Module, ckpt_path: str, device: torch.device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Strip DDP/DataParallel prefix if present
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    return model


def evaluate_model(
    split="test",
    threshold=0.5,
    model_path=None,
    model_class=None,
    deep_supervision=False,
    root_dir=None,
    batch_size=1,
    num_workers=2,
    encoder_name=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert model_class is not None, "model_class must be provided"

    # Reset peak stats for full-eval VRAM measurement (optional)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # ---- Build model wrapper ----
    # Pass encoder_name only if model supports it (e.g., SegMateFastViT, SegMateMambaOut)
    models_with_encoder = ["SegMateFastViT", "SegMateMambaOut", "SegMate", "SegMateFiLM"]
    if encoder_name is not None and model_class.__name__ in models_with_encoder:
        print(f"[metrics_evaluation_segthor_25D] Using encoder: {encoder_name}")
        base_model = model_class(num_classes=9, deep_supervision=deep_supervision, encoder_name=encoder_name).to(device)
    else:
        if encoder_name is not None:
            print(f"[metrics_evaluation_segthor_25D] Warning: encoder_name '{encoder_name}' ignored for {model_class.__name__}")
        base_model = model_class(num_classes=9, deep_supervision=deep_supervision).to(device)
    model = TwoPointFiveDWrapper(base_model).to(device)

    # ---- Load checkpoint into wrapper ----
    model_path = model_path or MODEL_PATH
    model = _load_state_dict_safely(model, model_path, device)
    model.eval()

    # ---- Data ----
    rd = root_dir or ROOT_DIR
    if model_class.__name__ == "SegMateFiLM":
        from dataloaders.balanced_dataloader_25D_film import get_dataloaders_balanced_film
        train_loader, val_loader, test_loader = get_dataloaders_balanced_film(rd, batch_size=batch_size, num_workers=num_workers)
    else:
        from dataloaders.balanced_dataloader_25D import get_dataloaders_balanced as get_dataloaders
        train_loader, val_loader, test_loader = get_dataloaders(rd, batch_size=batch_size, num_workers=num_workers)
    loader_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    dataloader = loader_map.get(split)
    if dataloader is None:
        raise ValueError(f"Invalid split '{split}'. Choose from 'train', 'val', or 'test'.")

    print(f"\n[EVAL] Evaluating SegTHOR 2.5D on {split.upper()} set with threshold={threshold}")
    print(f"   Model: {model_path}")
    print(f"   Root:  {rd}")

    metrics = {organ: {"dice": [], "hd95": [], "iou": [], "precision": [], "recall": []} for organ in ORGANS}
    presence_metrics = {organ: {"y_true": [], "y_pred": [], "y_prob": []} for organ in ORGANS}

    with torch.no_grad():
        for batch in tqdm(dataloader):
            if len(batch) == 3:
                images, labels, z_norm = batch
                z_norm = z_norm.to(device)
            else:
                images, labels = batch
                z_norm = None
            # images expected: [B, 3, H, W]
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images, z_norm=z_norm)

            # CustomUNetPlusPlus returns tuple: (seg_logits, boundary, presence, deep_outs) or similar
            if isinstance(outputs, tuple):
                pred_masks = outputs[0]
            else:
                pred_masks = outputs

            pred = torch.sigmoid(pred_masks).detach().cpu().numpy()
            gt = labels.detach().cpu().numpy()

            # Use first item (batch_size=1 typical here)
            pred = pred[0]  # [C,H,W]
            gt = gt[0]      # [C,H,W]

            for i, organ in enumerate(ORGANS):
                p_prob = pred[i]
                p = (p_prob > threshold).astype(np.uint8)
                g = gt[i].astype(np.uint8)

                # Presence metrics
                gt_present = int(np.sum(g) > 0)
                pred_present = int(np.sum(p) > 0)
                pred_prob_presence = float(np.max(p_prob))

                presence_metrics[organ]["y_true"].append(gt_present)
                presence_metrics[organ]["y_pred"].append(pred_present)
                presence_metrics[organ]["y_prob"].append(pred_prob_presence)

                # Segmentation metrics only when organ is present in GT
                if np.sum(g) == 0:
                    continue

                try:
                    metrics[organ]["dice"].append(dc(p, g))
                
                    # hd95 undefined if prediction is empty
                    if np.sum(p) > 0:
                        metrics[organ]["hd95"].append(hd95(p, g))
                    # else: skip hd95 for this sample (or set a penalty value)
                
                    metrics[organ]["iou"].append(jaccard_score(g.flatten(), p.flatten()))
                    metrics[organ]["precision"].append(precision(p, g))
                    metrics[organ]["recall"].append(recall(p, g))
                
                except Exception as e:
                    print(f"[WARNING] {organ.upper()} failed: {e}")
                    continue

    # Optional: report full-evaluation peak VRAM
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"\n[MEMORY] Peak VRAM usage during evaluation: {peak_mem_mb:.2f} MB")

    # ------------------- build CSV -------------------
    rows = []
    total_metrics = {"dice": [], "hd95": [], "iou": [], "precision": [], "recall": []}

    for organ in ORGANS:
        row = {"organ": organ}

        for metric_name in metrics[organ]:
            values = metrics[organ][metric_name]
            score = np.mean(values) if values else None
            row[metric_name] = round(score, 4) if score is not None else "n/a"
            if values:
                total_metrics[metric_name].extend(values)

        y_true = np.array(presence_metrics[organ]["y_true"])
        y_pred = np.array(presence_metrics[organ]["y_pred"])
        y_prob = np.array(presence_metrics[organ]["y_prob"])

        if len(np.unique(y_true)) > 1:
            presence_acc = accuracy_score(y_true, y_pred)
            presence_auc = roc_auc_score(y_true, y_prob)

            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))

            presence_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            presence_recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            row["presence_acc"] = round(float(presence_acc), 4)
            row["presence_precision"] = round(float(presence_precision), 4)
            row["presence_recall"] = round(float(presence_recall), 4)
            row["presence_auc"] = round(float(presence_auc), 4)
        else:
            row["presence_acc"] = "n/a"
            row["presence_precision"] = "n/a"
            row["presence_recall"] = "n/a"
            row["presence_auc"] = "n/a"

        rows.append(row)

    total_row = {"organ": "TOTAL"}
    for metric_name in total_metrics:
        if total_metrics[metric_name]:
            score = np.mean(total_metrics[metric_name])
            total_row[metric_name] = round(float(score), 4)
        else:
            total_row[metric_name] = "n/a"
    rows.append(total_row)

    df = pd.DataFrame(rows)
    csv_final = CSV_PATH.replace(".csv", f"_25D_{split}.csv")
    df.to_csv(csv_final, index=False)
    print(f"\n[SAVED] SegTHOR summary to {csv_final}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate SegTHOR dataset metrics (2.5D)"
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Which split to evaluate")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary mask")
    parser.add_argument("--deep_supervision", action="store_true", help="Enable if model uses deep supervision")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to the 2.5D checkpoint (.pth)")
    parser.add_argument("--root_dir", type=str, default=ROOT_DIR, help="Dataset root dir (processed_dataset_segthor)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default=1)")
    parser.add_argument("--num_workers", type=int, default=2, help="Num workers (default=2)")
    args = parser.parse_args()

    from models.custom_unet_plus_plus import CustomUNetPlusPlus

    evaluate_model(
        split=args.split,
        threshold=args.threshold,
        model_path=args.model_path,
        model_class=CustomUNetPlusPlus,
        deep_supervision=args.deep_supervision,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
