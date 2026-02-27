# prepare_totalsegmentator_dataset.py
"""
Dataset pre‑processing script for **TotalSegmentator v1.5** slices.

Adds two extra organs – *aorta* and *trachea* – and writes
`slice_index.csv` so the BalancedBatchSampler can balance batches.

Run example
-----------
```bash
python prepare_totalsegmentator_dataset.py --root /data/TotalSegmentator
```
"""

import os
import random
import csv
from glob import glob
from pathlib import Path
from typing import Dict, List

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
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
    # new organs
    "aorta": ["aorta.nii.gz"],
    "trachea": ["trachea.nii.gz"],
}

ORGAN_ORDER = list(ORGANS.keys())  # fixed order for channels / CSV
TARGET_SIZE = (512, 512)
SLICE_AXIS = 2  # axial slices
SEED = 42

OUTPUT_ROOT = "processed_dataset"  # sub‑dirs will be created here
Path(OUTPUT_ROOT).mkdir(exist_ok=True)

# -------------------------- helpers --------------------------

def load_nifti(fp: str) -> np.ndarray:
    img = nib.load(fp)
    img = nib.as_closest_canonical(img)
    return img.get_fdata()


def load_and_normalize_ct(fp: str) -> np.ndarray:
    data = load_nifti(fp)
    data = np.clip(data, -1000, 400)
    return ((data + 1000) / 1400).astype(np.float32)


def load_segmentation(fp: str) -> np.ndarray:
    data = load_nifti(fp)
    return (data > 0).astype(np.uint8)


def resize_slice(slice_: np.ndarray, tgt_shape, order: int) -> np.ndarray:
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

def save_slices(
    ct_vol: np.ndarray,
    mask_vol: np.ndarray,
    patient_id: str,
    split: str,
    csv_writer: csv.writer,
):
    img_dir = Path(OUTPUT_ROOT) / f"images{split.capitalize()}"
    lbl_dir = Path(OUTPUT_ROOT) / f"labels{split.capitalize()}"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    ct_slices = extract_slices(ct_vol)
    skipped = 0

    for i, ct_slice in enumerate(ct_slices):
        masks = []
        for organ_idx in range(mask_vol.shape[0]):
            masks.append(mask_vol[organ_idx, :, :, i])

        img_res = resize_slice(ct_slice, TARGET_SIZE, 1)[None, ...]  # [1,H,W]
        lbl_res = np.stack([resize_slice(m, TARGET_SIZE, 0) for m in masks])

        if lbl_res.sum() == 0:
            skipped += 1
            continue

        img_path = img_dir / f"{patient_id}_slice{i:03d}.npy"
        lbl_path = lbl_dir / f"{patient_id}_slice{i:03d}.npy"
        np.save(img_path, img_res.astype(np.float32))
        np.save(lbl_path, lbl_res.astype(np.uint8))

        presence = (lbl_res.reshape(len(ORGANS), -1).sum(axis=1) > 0).astype(int)
        csv_writer.writerow([str(img_path.relative_to(OUTPUT_ROOT)), *presence.tolist()])

    print(f"Patient {patient_id}: saved {len(ct_slices) - skipped} slices, skipped {skipped} empty")


# -------------------------- main entry --------------------------

def main(root_dir: str, limit_patients: int | None = None):
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
                    save_slices(ct, mask, pid, split, writer)
                except Exception as e:
                    print(f"Error {pid}: {e}")

    # quick stats
    for split in splits:
        n = len(glob(str(Path(OUTPUT_ROOT) / f"images{split.capitalize()}/*.npy")))
        print(f"{split.capitalize():>5}: {n} slices")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to TotalSegmentator root folder")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of patients (debug)")
    args = ap.parse_args()

    main(args.root, limit_patients=args.limit)
