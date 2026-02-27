# evaluation/metrics_evaluation_3d.py
"""
3D Volume-Level Evaluation Pipeline.

Computes Dice, HD95, IoU, Precision, Recall on full 3D patient volumes
using medpy, then reports mean +/- std across patients (paper convention).

Supports incremental saving and resume from partial runs.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from medpy.metric.binary import dc, hd95, precision, recall

sys.path.append(".")

from evaluation.volume_dataloader_3d import VolumeIterator
from evaluation.benchmark_models import (
    ModelSpec, build_and_load_model,
    _load_state_dict, _ckpt_has_deep_supervision,
)

# Use the TwoPointFiveDWrapper from metrics_evaluation_25D (supports z_norm)
from evaluation.metrics_evaluation_25D import TwoPointFiveDWrapper
from evaluation.center_slice_wrapper import CenterSliceWrapper

ORGANS = [
    "left_lung", "right_lung", "cord",
    "esophagus", "liver", "left_kidney", "right_kidney",
    "aorta", "trachea",
]

# Active organs per dataset (others are zero-filled in GT)
DATASET_ACTIVE_ORGANS = {
    "totalseg": ORGANS,  # all 9
    "segthor": ["esophagus", "trachea", "aorta"],
    "amos22": ["esophagus", "liver", "left_kidney", "right_kidney", "aorta"],
}


def _build_model(spec: ModelSpec, device: torch.device, num_classes: int = 9):
    """Build model using benchmark_models infrastructure but with z_norm-aware wrapper."""
    state = _load_state_dict(spec.ckpt_path, device)

    deep_sup = spec.force_deep_supervision
    if deep_sup is None:
        deep_sup = _ckpt_has_deep_supervision(state)

    kwargs = dict(spec.model_kwargs or {})
    kwargs["num_classes"] = num_classes

    try:
        backbone = spec.model_class(**kwargs, deep_supervision=deep_sup).to(device)
    except TypeError:
        backbone = spec.model_class(**kwargs).to(device)

    if spec.model_type == "2D":
        # Pure 2D model: load weights into backbone, then wrap for 2.5D eval input
        backbone.load_state_dict(state, strict=True)
        model = CenterSliceWrapper(backbone).to(device)
    else:
        # Existing 2.5D path
        model = TwoPointFiveDWrapper(backbone).to(device)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            from evaluation.benchmark_models import _filter_state_dict_for_model
            filtered = _filter_state_dict_for_model(model, state)
            model.load_state_dict(filtered, strict=False)

    model.eval()
    return model


def _compute_patient_metrics(model, patient_id, imgs, lbls, z_norms,
                             active_indices, device, batch_size, use_amp, threshold,
                             spacing=(1.0, 1.0, 1.0)):
    """Run inference + metrics for one patient. Returns list of row dicts.

    Args:
        spacing: Voxel spacing (spacing_z, spacing_y, spacing_x) in mm.
                 Used for HD95 computation. Defaults to isotropic 1mm (voxel units).
                 Pass actual per-patient spacing for physically meaningful HD95 values.
    """
    Z = imgs.shape[0]

    # Batched inference
    pred_list = []
    for start in range(0, Z, batch_size):
        end = min(start + batch_size, Z)
        x_batch = imgs[start:end].to(device)
        z_batch = z_norms[start:end].to(device)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(use_amp and device.type == "cuda"),
        ):
            out = model(x_batch, z_norm=z_batch)
            logits = out[0] if isinstance(out, tuple) else out

        pred_binary = (torch.sigmoid(logits) > threshold).cpu().to(torch.uint8)
        pred_list.append(pred_binary)

    pred_vol = torch.cat(pred_list, dim=0).numpy()
    gt_vol = lbls.numpy().astype(np.uint8)

    rows = []
    for organ_idx in active_indices:
        organ_name = ORGANS[organ_idx]
        gt_3d = gt_vol[:, organ_idx, :, :]
        pred_3d = pred_vol[:, organ_idx, :, :]

        gt_any = gt_3d.sum() > 0
        pred_any = pred_3d.sum() > 0

        if not gt_any:
            continue

        row = {"patient_id": patient_id, "organ": organ_name}

        try:
            row["dice"] = float(dc(pred_3d, gt_3d))
        except Exception:
            row["dice"] = 0.0

        if pred_any:
            try:
                row["hd95"] = float(hd95(pred_3d, gt_3d, voxelspacing=spacing))
            except Exception:
                row["hd95"] = np.nan
        else:
            row["hd95"] = np.nan

        intersection = np.logical_and(pred_3d, gt_3d).sum()
        union = np.logical_or(pred_3d, gt_3d).sum()
        row["iou"] = float(intersection / (union + 1e-8)) if union > 0 else 0.0

        try:
            row["precision"] = float(precision(pred_3d, gt_3d))
        except Exception:
            row["precision"] = 0.0

        try:
            row["recall"] = float(recall(pred_3d, gt_3d))
        except Exception:
            row["recall"] = 0.0

        rows.append(row)

    del pred_vol, gt_vol, pred_list
    return rows


@torch.inference_mode()
def evaluate_model_3d(
    spec: ModelSpec,
    data_root: str,
    split: str = "test",
    dataset_key: str = "totalseg",
    threshold: float = 0.5,
    batch_size: int = 16,
    use_amp: bool = True,
    num_classes: int = 9,
    save_path: str = None,
    gpu_id: int = None,
    patient_subset: set = None,
) -> pd.DataFrame:
    """
    Evaluate a model on full 3D patient volumes.

    Args:
        save_path: If provided, saves results incrementally per patient.
                   Also resumes from existing partial results.
        gpu_id: Specific GPU index to use (e.g. 0, 1, 2). None = auto.
        patient_subset: If provided, only evaluate these patient IDs.
                        Used for random subsampling (--max_patients).

    Returns:
        DataFrame with columns: patient_id, organ, dice, hd95, iou, precision, recall
    """
    if gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[3D Eval] Model: {spec.name}")
    print(f"[3D Eval] Dataset: {data_root} ({dataset_key}), split={split}")

    # Resume: load existing partial results
    done_patients = set()
    existing_rows = []
    if save_path and os.path.isfile(save_path):
        try:
            existing_df = pd.read_csv(save_path)
            if not existing_df.empty and "patient_id" in existing_df.columns:
                done_patients = set(existing_df["patient_id"].unique())
                existing_rows = existing_df.to_dict("records")
                print(f"[3D Eval] Resuming: {len(done_patients)} patients already done")
        except pd.errors.EmptyDataError:
            # Empty CSV file from previous run with no data
            pass

    vol_iter = VolumeIterator(data_root, split)

    # Filter to subset if requested (--max_patients)
    if patient_subset is not None:
        vol_iter.patient_ids = [p for p in vol_iter.patient_ids if p in patient_subset]
        print(f"[3D Eval] Subset: {len(vol_iter.patient_ids)} patients (of {len(patient_subset)} requested)")

    active_organs = DATASET_ACTIVE_ORGANS.get(dataset_key, ORGANS)
    active_indices = [i for i, o in enumerate(ORGANS) if o in active_organs]

    remaining = len(vol_iter) - len(done_patients)
    print(f"[3D Eval] Patients: {len(vol_iter)} total, {len(done_patients)} done, {remaining} remaining")

    # Skip model loading entirely if all patients are already done
    if remaining <= 0:
        print(f"[3D Eval] All patients already evaluated, skipping model load.")
        return pd.DataFrame(existing_rows)

    model = _build_model(spec, device, num_classes)

    all_rows = list(existing_rows)

    for patient_id, imgs, lbls, z_norms in tqdm(vol_iter, desc="Patients", total=len(vol_iter)):
        if patient_id in done_patients:
            continue
        if imgs.numel() == 0:
            continue

        spacing = vol_iter.get_spacing(patient_id)
        patient_rows = _compute_patient_metrics(
            model, patient_id, imgs, lbls, z_norms,
            active_indices, device, batch_size, use_amp, threshold,
            spacing=spacing,
        )
        all_rows.extend(patient_rows)

        # Incremental save after each patient
        if save_path:
            pd.DataFrame(all_rows).to_csv(save_path, index=False)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    print(f"[3D Eval] Done: {len(vol_iter)} patients, {len(df)} organ measurements")
    return df


def summarize_3d_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize per-patient per-organ results into mean +/- std per organ.

    Returns:
        DataFrame with columns: organ, dice_mean, dice_std, hd95_mean, hd95_std,
                                 iou_mean, iou_std, precision_mean, precision_std,
                                 recall_mean, recall_std, n_patients
    """
    metric_cols = ["dice", "hd95", "iou", "precision", "recall"]
    summary_rows = []

    # Handle empty dataframe
    if df.empty or "organ" not in df.columns:
        return pd.DataFrame(columns=["organ", "n_patients"] + 
                          [f"{m}_{s}" for m in metric_cols for s in ["mean", "std"]])

    for organ in df["organ"].unique():
        organ_df = df[df["organ"] == organ]
        row = {"organ": organ, "n_patients": len(organ_df)}

        for metric in metric_cols:
            values = organ_df[metric].dropna()
            row[f"{metric}_mean"] = round(float(values.mean()), 4) if len(values) > 0 else np.nan
            row[f"{metric}_std"] = round(float(values.std()), 4) if len(values) > 0 else np.nan

        summary_rows.append(row)

    # TOTAL row: average across all organs per patient, then mean/std across patients
    total_row = {"organ": "TOTAL", "n_patients": df["patient_id"].nunique()}
    for metric in metric_cols:
        # Mean per patient first (average across organs), then mean/std across patients
        patient_means = df.groupby("patient_id")[metric].mean().dropna()
        total_row[f"{metric}_mean"] = round(float(patient_means.mean()), 4) if len(patient_means) > 0 else np.nan
        total_row[f"{metric}_std"] = round(float(patient_means.std()), 4) if len(patient_means) > 0 else np.nan
    summary_rows.append(total_row)

    return pd.DataFrame(summary_rows)
