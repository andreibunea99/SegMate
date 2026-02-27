#!/usr/bin/env python3
"""
AMOS22 Dataset Pre-processing Script (Adapted for SegMate Pipeline)
====================================================================

Processes AMOS22 dataset to match TotalSegmentator format:
- Extracts 5 overlapping organs (esophagus, liver, left_kidney, right_kidney, aorta) from combined GT
- Maps to 9-channel format (channels 3, 4, 5, 6, 7; others zero-filled)
- Maintains compatibility with SegMateFiLM model (num_classes=9)
- Uses same CT normalization, chunking, and CSV structure
- Filters to CT scans only (patient IDs < 500)

AMOS22 GT Label Encoding:
--------------------------
0: Background
1: Spleen (excluded - not in TotalSegmentator overlap)
2: Right kidney → Channel 6
3: Left kidney → Channel 5
4: Gall bladder (excluded)
5: Esophagus → Channel 3
6: Liver → Channel 4
7: Stomach (excluded)
8: Aorta → Channel 7
9: Postcava (excluded)
10: Pancreas (excluded)
11: Right adrenal gland (excluded)
12: Left adrenal gland (excluded)
13: Duodenum (excluded)
14: Bladder (excluded)
15: Prostate/uterus (excluded)

Run example:
-----------
python scripts/prepare_amos22_dataset.py --root /path/to/amos22/data
python scripts/prepare_amos22_dataset.py --root /path/to/amos22/data --chunk_size 8 --min_foreground 0.00001
"""

import os
import random
import csv
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import nibabel as nib
from skimage.transform import resize
from tqdm import tqdm

# -------------------------- configuration --------------------------

# TotalSegmentator organ order (9 channels)
ORGAN_ORDER = [
    "left_lung",      # Channel 0 - empty for AMOS22
    "right_lung",     # Channel 1 - empty for AMOS22
    "cord",           # Channel 2 - empty for AMOS22
    "esophagus",      # Channel 3 - GT label 5
    "liver",          # Channel 4 - GT label 6
    "left_kidney",    # Channel 5 - GT label 3
    "right_kidney",   # Channel 6 - GT label 2
    "aorta",          # Channel 7 - GT label 8
    "trachea",        # Channel 8 - empty for AMOS22
]

# AMOS22 GT label → channel index mapping (only for present organs)
AMOS22_LABEL_TO_CHANNEL = {
    2: 6,  # Right kidney: GT label 2 → channel 6
    3: 5,  # Left kidney: GT label 3 → channel 5
    5: 3,  # Esophagus: GT label 5 → channel 3
    6: 4,  # Liver: GT label 6 → channel 4
    8: 7,  # Aorta: GT label 8 → channel 7
}

SLICE_AXIS = 2  # axial slices
SEED = 42

OUTPUT_ROOT = "processed_dataset_amos22"
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


def load_amos22_gt(fp: str) -> np.ndarray:
    """
    Load AMOS22 ground truth and convert to 9-channel format.
    
    Returns:
        mask: [C=9, H, W, D] where C is channel dimension
    """
    gt_vol = load_nifti(fp).astype(np.uint8)
    
    # Initialize 9-channel mask (all zeros)
    mask = np.zeros((len(ORGAN_ORDER), *gt_vol.shape), dtype=np.uint8)
    
    # Map AMOS22 labels to appropriate channels
    for gt_label, channel_idx in AMOS22_LABEL_TO_CHANNEL.items():
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
        patient_id: Patient identifier (e.g., 'amos_0001')
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
    Main preprocessing pipeline for AMOS22 dataset.
    
    Args:
        root_dir: Path to amos22 folder containing imagesTr/, imagesVa/, imagesTs/, etc.
        limit_patients: Optional limit on number of patients (for debugging)
        chunk_size: Number of slices per chunk
        min_foreground: Minimum foreground ratio to keep a slice
        target_size: Optional (H, W) to resize slices
    """
    random.seed(SEED)

    # AMOS22 structure: imagesTr/amos_XXXX.nii.gz, labelsTr/amos_XXXX.nii.gz
    # Filter CT only: patient IDs < 500
    splits_config = {
        "train": ("imagesTr", "labelsTr"),
        "val": ("imagesVa", "labelsVa"),
        "test": ("imagesTs", "labelsTs")
    }
    
    splits = {}
    
    for split_name, (img_dir, lbl_dir) in splits_config.items():
        img_path = os.path.join(root_dir, img_dir)
        lbl_path = os.path.join(root_dir, lbl_dir)
        
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping {split_name} split")
            continue
        
        # Get all CT scans
        all_images = sorted(glob(os.path.join(img_path, "amos_*.nii.gz")))
        
        # Filter CT only (IDs < 500)
        ct_images = []
        for img_fp in all_images:
            # Extract patient ID from filename (handle .nii.gz double extension)
            base_name = os.path.basename(img_fp)
            patient_id = base_name.split('.')[0]  # e.g., 'amos_0001' from 'amos_0001.nii.gz'
            try:
                patient_num = int(patient_id.split("_")[1])
                if patient_num < 500:  # CT only
                    # Check if corresponding label exists (for train/val splits)
                    if os.path.exists(lbl_path):
                        lbl_fp = os.path.join(lbl_path, f"{patient_id}.nii.gz")
                        if os.path.exists(lbl_fp):
                            ct_images.append((img_fp, lbl_fp, patient_id))
                        else:
                            print(f"Warning: Label not found for {patient_id}, skipping")
                    else:
                        # Test set has no labels
                        ct_images.append((img_fp, None, patient_id))
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not parse patient ID from {base_name}: {e}, skipping")
                continue
        
        splits[split_name] = ct_images
        print(f"Found {len(ct_images)} CT scans in {split_name} split")
    
    if not any(splits.values()):
        raise ValueError(f"No CT scans found in {root_dir}")
    
    # Apply limit if specified
    if limit_patients:
        for split_name in splits:
            if len(splits[split_name]) > limit_patients:
                splits[split_name] = random.sample(splits[split_name], limit_patients)
    
    print(f"\n{'='*60}")
    print(f"Total CT scans: {sum(len(v) for v in splits.values())}")
    for split_name, data in splits.items():
        print(f"  {split_name}: {len(data)} patients")
    print(f"{'='*60}\n")

    with open(Path(OUTPUT_ROOT) / "slice_index.csv", "w", newline="") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(["path", *ORGAN_ORDER])

        for split, data in splits.items():
            if not data:
                continue
                
            print(f"\nProcessing {split.upper()} – {len(data)} patients")
            for ct_fp, gt_fp, patient_id in tqdm(data):
                try:
                    ct = load_and_normalize_ct(ct_fp)
                    
                    # For test set without labels, create empty mask
                    if gt_fp is None:
                        print(f"Warning: No label for {patient_id} (test set), creating empty mask")
                        mask = np.zeros((len(ORGAN_ORDER), *ct.shape), dtype=np.uint8)
                    else:
                        mask = load_amos22_gt(gt_fp)
                    
                    save_chunked_slices(
                        ct, mask, patient_id, split, writer, 
                        chunk_size=chunk_size,
                        min_foreground_ratio=min_foreground,
                        target_size=target_size
                    )
                except Exception as e:
                    print(f"Error {patient_id}: {e}")

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
        description="Preprocess AMOS22 dataset for SegMate training/evaluation (CT scans only)"
    )
    ap.add_argument("--root", required=True, 
                   help="Path to amos22 folder (contains imagesTr/, imagesVa/, imagesTs/ directories)")
    ap.add_argument("--limit", type=int, default=None, 
                   help="Limit number of patients per split (for debugging)")
    ap.add_argument("--chunk_size", type=int, default=8, 
                   help="Number of slices per chunk (default: 8)")
    ap.add_argument("--min_foreground", type=float, default=0.00001, 
                    help="Minimum foreground ratio to keep slice (default: 0.00001)")
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
