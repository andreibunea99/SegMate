#!/usr/bin/env python3
"""
Optimized Dataset pre-processing script for TotalSegmentator v1.5 slices.

Key optimizations:
- Uses compression when saving numpy arrays
- Option to maintain original image dimensions (no resize)
- Combines multiple slices into data chunks
- Option to filter empty/minimal slices
- Keeps original bit depth when appropriate

Run example:
-----------
python improved_totalsegmentator_dataset.py --root /data/TotalSegmentator --chunk_size 16
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

ORGANS: Dict[str, List[str]] = {
    "left_lung": ["lung_upper_lobe_left.nii.gz", "lung_lower_lobe_left.nii.gz"],
    "right_lung": [
        "lung_upper_lobe_right.nii.gz",
        "lung_middle_lobe_right.nii.gz",
        "lung_lower_lobe_right.nii.gz",
    ],
    "cord": ["spinal_cord.nii.gz"],
    "esophagus": ["esophagus.nii.gz"],
    "liver": ["liver.nii.gz"],
    "left_kidney": ["kidney_left.nii.gz"],
    "right_kidney": ["kidney_right.nii.gz"],
    "aorta": ["aorta.nii.gz"],
    "trachea": ["trachea.nii.gz"],
}

ORGAN_ORDER = list(ORGANS.keys())  # fixed order for channels / CSV
SLICE_AXIS = 2  # axial slices
SEED = 42

OUTPUT_ROOT = "processed_dataset"  # sub-dirs will be created here
Path(OUTPUT_ROOT).mkdir(exist_ok=True)

# -------------------------- helpers --------------------------

def load_nifti(fp: str) -> np.ndarray:
    img = nib.load(fp)
    img = nib.as_closest_canonical(img)
    return img.get_fdata()


def load_and_normalize_ct(fp: str) -> np.ndarray:
    data = load_nifti(fp)
    data = np.clip(data, -1000, 400)
    return ((data + 1000) / 1400).astype(np.float16)


def load_segmentation(fp: str) -> np.ndarray:
    data = load_nifti(fp)
    return (data > 0).astype(np.uint8)


def resize_slice(slice_: np.ndarray, tgt_shape, order: int) -> np.ndarray:
    # Only resize if target shape is different from current shape
    if slice_.shape == tgt_shape:
        return slice_
    return resize(slice_, tgt_shape, order=order, preserve_range=True, anti_aliasing=order > 0)


def build_mask(seg_dir: str, ref_shape) -> np.ndarray:
    mask = np.zeros((len(ORGANS), *ref_shape), dtype=np.uint8)
    for idx, organ in enumerate(ORGAN_ORDER):
        for subfile in ORGANS[organ]:
            path = os.path.join(seg_dir, subfile)
            if not os.path.exists(path):
                continue
            try:
                vol = load_segmentation(path)
                mask[idx] |= vol
            except Exception as e:
                print(f"Warning reading {path}: {e}")
    return mask


def extract_slices(vol: np.ndarray) -> List[np.ndarray]:
    return [vol[:, :, i] for i in range(vol.shape[SLICE_AXIS])]


# -------------------------- core save routine --------------------------

def save_chunked_slices(
    ct_vol: np.ndarray,
    mask_vol: np.ndarray,
    patient_id: str,
    split: str,
    csv_writer,
    chunk_size: int = 8,
    min_foreground_ratio: float = 0.001,
    target_size: Optional[Tuple[int, int]] = None,
):
    """Save slices in chunks for greater efficiency"""
    img_dir = Path(OUTPUT_ROOT) / f"images{split.capitalize()}"
    lbl_dir = Path(OUTPUT_ROOT) / f"labels{split.capitalize()}"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    ct_slices = extract_slices(ct_vol)
    valid_slices = []
    
    # First pass: identify valid slices
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
            
            # Add entry to CSV
            presence = (lbl_res.reshape(len(ORGANS), -1).sum(axis=1) > 0).astype(int)
            rel_path = f"images{split.capitalize()}/{patient_id}_chunk{chunk_idx:03d}_slice{idx:03d}.npy"
            csv_writer.writerow([rel_path, *presence.tolist()])
        
        # Stack and save the chunk
        img_stack = np.vstack(imgs)  # [N,H,W]
        lbl_stack = np.vstack(lbls)  # [N*C,H,W]
        
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
    min_foreground: float = 0.001,
    target_size: Tuple[int, int] = None
):
    random.seed(SEED)

    # TotalSegmentator folder pattern: one folder per patient (e.g. Ts_0001)
    patients = sorted(glob(os.path.join(root_dir, "*/")))
    if limit_patients and limit_patients < len(patients):
        patients = random.sample(patients, limit_patients)

    train_ids, test_ids = train_test_split(patients, test_size=0.1, random_state=SEED)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.1, random_state=SEED)
    splits = {"train": train_ids, "val": val_ids, "test": test_ids}

    with open(Path(OUTPUT_ROOT) / "slice_index.csv", "w", newline="") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(["path", *ORGAN_ORDER])

        for split, dirs in splits.items():
            print(f"\nProcessing {split.upper()} – {len(dirs)} patients")
            for p in tqdm(dirs):
                pid = Path(p).stem
                ct_fp = os.path.join(p, "ct.nii.gz")
                seg_dir = os.path.join(p, "segmentations")
                if not (os.path.exists(ct_fp) and os.path.exists(seg_dir)):
                    print(f"✘ Missing modality for {pid}, skip")
                    continue

                try:
                    ct = load_and_normalize_ct(ct_fp)
                    mask = build_mask(seg_dir, ct.shape)
                    save_chunked_slices(
                        ct, mask, pid, split, writer, 
                        chunk_size=chunk_size,
                        min_foreground_ratio=min_foreground,
                        target_size=target_size
                    )
                except Exception as e:
                    print(f"Error {pid}: {e}")

    # quick stats
    total_size_mb = 0
    for root, _, files in os.walk(OUTPUT_ROOT):
        size = sum(os.path.getsize(os.path.join(root, name)) for name in files)
        total_size_mb += size / (1024 * 1024)
        if files:
            print(f"{root}: {len(files)} files, {size / (1024 * 1024):.2f} MB")
    
    print(f"Total dataset size: {total_size_mb:.2f} MB")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to TotalSegmentator root folder")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of patients (debug)")
    ap.add_argument("--chunk_size", type=int, default=8, help="Number of slices per chunk")
    ap.add_argument("--min_foreground", type=float, default=0.001, 
                   help="Minimum foreground ratio to keep a slice")
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