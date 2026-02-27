#!/usr/bin/env python3
"""
SegTHOR Dataset Pre-processing Script (Adapted for SegMate Pipeline)
=====================================================================

Processes SegTHOR dataset to match TotalSegmentator format:
- Extracts 3 overlapping organs (esophagus, trachea, aorta) from combined GT
- Maps to 9-channel format (channels 3, 7, 8; others zero-filled)
- Maintains compatibility with SegMateFiLM model (num_classes=9)
- Uses same CT normalization, chunking, and CSV structure

SegTHOR GT Label Encoding:
--------------------------
0: Background
1: Esophagus → Channel 3 (CORRECTED: docs incorrectly said label 2)
2: Heart (excluded - not in TotalSegmentator overlap)
3: Trachea → Channel 8
4: Aorta → Channel 7

Run example:
-----------
python scripts/prepare_segthor_dataset.py --root ../SegTHOR/SegTHOR
python scripts/prepare_segthor_dataset.py --root ../SegTHOR/SegTHOR --chunk_size 8 --min_foreground 0.00001
"""

import os
import random
import csv
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from tqdm import tqdm

# -------------------------- configuration --------------------------

# TotalSegmentator organ order (9 channels)
ORGAN_ORDER = [
    "left_lung",      # Channel 0 - empty for SegTHOR
    "right_lung",     # Channel 1 - empty for SegTHOR
    "cord",           # Channel 2 - empty for SegTHOR
    "esophagus",      # Channel 3 - GT label 2
    "liver",          # Channel 4 - empty for SegTHOR
    "left_kidney",    # Channel 5 - empty for SegTHOR
    "right_kidney",   # Channel 6 - empty for SegTHOR
    "aorta",          # Channel 7 - GT label 4
    "trachea",        # Channel 8 - GT label 3
]

# SegTHOR GT label → channel index mapping (only for present organs)
# NOTE: SegTHOR documentation was WRONG! Correct mapping:
# Label 1 = Esophagus, Label 2 = HEART, Label 3 = Trachea, Label 4 = Aorta
SEGTHOR_LABEL_TO_CHANNEL = {
    1: 3,  # Esophagus: GT label 1 → channel 3 (FIXED: was incorrectly 2)
    3: 8,  # Trachea: GT label 3 → channel 8
    4: 7,  # Aorta: GT label 4 → channel 7
}

SLICE_AXIS = 2  # axial slices
SEED = 42

OUTPUT_ROOT = "processed_dataset_segthor"
Path(OUTPUT_ROOT).mkdir(exist_ok=True)

# -------------------------- helpers --------------------------

def load_nifti(fp: str) -> np.ndarray:
    """Load NIfTI file and convert to canonical orientation."""
    img = nib.load(fp)
    img = nib.as_closest_canonical(img)
    return img.get_fdata()


def load_and_normalize_ct(fp: str) -> np.ndarray:
    """Load CT and normalize to [0, 1] using HU window [-1000, 400]."""
    data = load_nifti(fp)
    data = np.clip(data, -1000, 400)
    return ((data + 1000) / 1400).astype(np.float16)


def load_segthor_gt(fp: str) -> np.ndarray:
    """
    Load SegTHOR ground truth and convert to 9-channel format.
    
    Returns:
        mask: [C=9, H, W, D] where C is channel dimension
    """
    gt_vol = load_nifti(fp).astype(np.uint8)
    
    # Initialize 9-channel mask (all zeros)
    mask = np.zeros((len(ORGAN_ORDER), *gt_vol.shape), dtype=np.uint8)
    
    # Map SegTHOR labels to appropriate channels
    for gt_label, channel_idx in SEGTHOR_LABEL_TO_CHANNEL.items():
        mask[channel_idx] = (gt_vol == gt_label).astype(np.uint8)
    
    return mask


def resize_slice(slice_: np.ndarray, tgt_shape, order: int) -> np.ndarray:
    """Resize slice only if target shape differs from current shape."""
    if slice_.shape == tgt_shape:
        return slice_
    return resize(slice_, tgt_shape, order=order, preserve_range=True, anti_aliasing=order > 0)


def extract_slices(vol: np.ndarray) -> List[np.ndarray]:
    """Extract axial slices from volume."""
    return [vol[:, :, i] for i in range(vol.shape[SLICE_AXIS])]


# -------------------------- core save routine --------------------------

def save_chunked_slices(
    ct_vol: np.ndarray,
    mask_vol: np.ndarray,
    patient_id: str,
    split: str,
    csv_writer,
    chunk_size: int = 8,
    min_foreground_ratio: float = 0.00001,
    target_size: Optional[Tuple[int, int]] = None,
):
    """
    Save slices in chunks for efficiency.
    
    Args:
        ct_vol: [H, W, D] normalized CT volume
        mask_vol: [C=9, H, W, D] multi-channel mask
        patient_id: Patient identifier (e.g., 'Patient_01')
        split: 'train', 'val', or 'test'
        csv_writer: CSV writer for slice_index.csv
        chunk_size: Number of slices per chunk
        min_foreground_ratio: Minimum foreground to keep slice
        target_size: Optional (H, W) to resize slices
    """
    img_dir = Path(OUTPUT_ROOT) / f"images{split.capitalize()}"
    lbl_dir = Path(OUTPUT_ROOT) / f"labels{split.capitalize()}"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    ct_slices = extract_slices(ct_vol)
    valid_slices = []
    
    # First pass: identify valid slices (with sufficient foreground)
    for i, ct_slice in enumerate(ct_slices):
        masks = []
        for organ_idx in range(mask_vol.shape[0]):
            masks.append(mask_vol[organ_idx, :, :, i])
        
        mask_stack = np.stack(masks)
        # Check if there's enough foreground content
        if mask_stack.sum() / mask_stack.size >= min_foreground_ratio:
            valid_slices.append((i, ct_slice, mask_stack))
    
    if not valid_slices:
        print(f"Patient {patient_id}: No valid slices found")
        return
    
    # Process slices in chunks
    total_chunks = (len(valid_slices) + chunk_size - 1) // chunk_size
    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(valid_slices))
        chunk_slices = valid_slices[start_idx:end_idx]
        
        # Prepare data arrays
        imgs = []
        lbls = []
        slice_indices = []
        
        for idx, ct_slice, mask_stack in chunk_slices:
            slice_indices.append(idx)
            
            # Resize if target size is specified
            if target_size:
                img_res = resize_slice(ct_slice, target_size, 1)[None, ...]  # [1,H,W]
                lbl_res = np.stack([resize_slice(m, target_size, 0) for m in mask_stack])
            else:
                img_res = ct_slice[None, ...]  # [1,H,W]
                lbl_res = mask_stack
            
            imgs.append(img_res)
            lbls.append(lbl_res)
            
            # Add entry to CSV (9 columns for organ presence)
            presence = (lbl_res.reshape(len(ORGAN_ORDER), -1).sum(axis=1) > 0).astype(int)
            rel_path = f"images{split.capitalize()}/{patient_id}_chunk{chunk_idx:03d}_slice{idx:03d}.npy"
            csv_writer.writerow([rel_path, *presence.tolist()])
        
        # Stack and save the chunk
        img_stack = np.vstack(imgs)  # [N,H,W]
        lbl_stack = np.vstack(lbls)  # [N*C,H,W] where C=9
        
        img_path = img_dir / f"{patient_id}_chunk{chunk_idx:03d}.npz"
        lbl_path = lbl_dir / f"{patient_id}_chunk{chunk_idx:03d}.npz"
        
        # Use compression
        np.savez_compressed(img_path, data=img_stack, indices=slice_indices)
        np.savez_compressed(lbl_path, data=lbl_stack, indices=slice_indices)
    
    print(f"Patient {patient_id}: saved {len(valid_slices)} slices in {total_chunks} chunks")


# -------------------------- main entry --------------------------

def main(
    root_dir: str, 
    limit_patients: int = None, 
    chunk_size: int = 8,
    min_foreground: float = 0.00001,
    target_size: Tuple[int, int] = None
):
    """
    Main preprocessing pipeline for SegTHOR dataset.
    
    Args:
        root_dir: Path to SegTHOR/SegTHOR folder containing Patient_* directories
        limit_patients: Optional limit on number of patients (for debugging)
        chunk_size: Number of slices per chunk
        min_foreground: Minimum foreground ratio to keep a slice
        target_size: Optional (H, W) to resize slices
    """
    random.seed(SEED)

    # SegTHOR folder pattern: Patient_01, Patient_02, ..., Patient_40
    patients = sorted(glob(os.path.join(root_dir, "Patient_*/")))
    
    if not patients:
        raise ValueError(f"No Patient_* folders found in {root_dir}")
    
    print(f"Found {len(patients)} patients in {root_dir}")
    
    if limit_patients and limit_patients < len(patients):
        patients = random.sample(patients, limit_patients)

    # Split: 30 train, 5 val, 5 test (75% / 12.5% / 12.5%)
    # First split: 30 train vs 10 (val+test)
    train_ids, temp_ids = train_test_split(patients, train_size=30, random_state=SEED)
    # Second split: 5 val vs 5 test
    val_ids, test_ids = train_test_split(temp_ids, train_size=5, random_state=SEED)
    
    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    
    print(f"Split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")

    with open(Path(OUTPUT_ROOT) / "slice_index.csv", "w", newline="") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(["path", *ORGAN_ORDER])

        for split, dirs in splits.items():
            print(f"\nProcessing {split.upper()} – {len(dirs)} patients")
            for p in tqdm(dirs):
                pid = Path(p).stem  # e.g., 'Patient_01'
                ct_fp = os.path.join(p, f"{pid}.nii.gz")
                gt_fp = os.path.join(p, "GT.nii.gz")
                
                if not (os.path.exists(ct_fp) and os.path.exists(gt_fp)):
                    print(f"✘ Missing files for {pid}, skip")
                    continue

                try:
                    ct = load_and_normalize_ct(ct_fp)
                    mask = load_segthor_gt(gt_fp)
                    save_chunked_slices(
                        ct, mask, pid, split, writer, 
                        chunk_size=chunk_size,
                        min_foreground_ratio=min_foreground,
                        target_size=target_size
                    )
                except Exception as e:
                    print(f"Error {pid}: {e}")

    # Quick stats
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    total_size_mb = 0
    for root, _, files in os.walk(OUTPUT_ROOT):
        size = sum(os.path.getsize(os.path.join(root, name)) for name in files)
        total_size_mb += size / (1024 * 1024)
        if files:
            print(f"{root}: {len(files)} files, {size / (1024 * 1024):.2f} MB")
    
    print(f"\nTotal dataset size: {total_size_mb:.2f} MB")
    print(f"✓ Preprocessing complete! Output: {OUTPUT_ROOT}/")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Preprocess SegTHOR dataset for SegMate training/evaluation"
    )
    ap.add_argument("--root", required=True, 
                   help="Path to SegTHOR/SegTHOR folder (contains Patient_* directories)")
    ap.add_argument("--limit", type=int, default=None, 
                   help="Limit number of patients (for debugging)")
    ap.add_argument("--chunk_size", type=int, default=8, 
                   help="Number of slices per chunk (default: 8)")
    ap.add_argument("--min_foreground", type=float, default=0.00001, 
                    help="Minimum foreground ratio to keep slice (default: 0.00001, lowered to keep small trachea slices)")
    ap.add_argument("--resize", type=int, nargs=2, default=None, 
                   help="Target size (height width) or None to keep original size")
    args = ap.parse_args()

    target_size = tuple(args.resize) if args.resize else None
    
    main(
        args.root, 
        limit_patients=args.limit, 
        chunk_size=args.chunk_size,
        min_foreground=args.min_foreground,
        target_size=target_size
    )

