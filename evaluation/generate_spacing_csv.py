#!/usr/bin/env python3
"""
Generate spacing.csv from original NIfTI files for each processed dataset.

The preprocessing pipeline discards voxel spacing metadata. This script
re-reads the original NIfTI files, extracts per-patient voxel spacing (mm),
and writes a spacing.csv into the processed dataset directory.

spacing.csv format:
    patient_id, spacing_z, spacing_y, spacing_x
    (z = slice/axial direction, y/x = in-plane)

Usage:
    # TotalSegmentator
    python evaluation/generate_spacing_csv.py \\
        --dataset totalseg \\
        --raw_root /data/TotalSegmentator \\
        --processed_root processed_dataset

    # SegTHOR
    python evaluation/generate_spacing_csv.py \\
        --dataset segthor \\
        --raw_root ../SegTHOR/SegTHOR \\
        --processed_root processed_dataset_segthor

    # AMOS22
    python evaluation/generate_spacing_csv.py \\
        --dataset amos22 \\
        --raw_root /data/amos22 \\
        --processed_root processed_dataset_amos22
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


def get_patient_ids(processed_root: str) -> list:
    """Extract unique patient IDs from slice_index.csv."""
    csv_path = os.path.join(processed_root, "slice_index.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"slice_index.csv not found in {processed_root}")

    patient_ids = set()
    with open(csv_path) as f:
        rdr = csv.reader(f)
        next(rdr)  # skip header
        for row in rdr:
            path = row[0]
            if "/" not in path:
                continue
            filename = path.split("/")[1]
            patient_id = filename.split("_chunk")[0]
            patient_ids.add(patient_id)
    return sorted(patient_ids)


def find_nifti_totalseg(raw_root: str, patient_id: str) -> str | None:
    """Find ct.nii.gz for a TotalSegmentator patient."""
    # TotalSegmentator: raw_root/{patient_id}/ct.nii.gz
    candidate = os.path.join(raw_root, patient_id, "ct.nii.gz")
    if os.path.exists(candidate):
        return candidate
    # Some TotalSeg releases use different naming
    for name in [f"{patient_id}.nii.gz", f"ct_{patient_id}.nii.gz"]:
        candidate = os.path.join(raw_root, patient_id, name)
        if os.path.exists(candidate):
            return candidate
    return None


def find_nifti_segthor(raw_root: str, patient_id: str) -> str | None:
    """Find NIfTI CT for a SegTHOR patient.
    SegTHOR: raw_root/Patient_{id}/Patient_{id}.nii.gz
    patient_id in processed_dataset is like 'Patient_01' or 'P01'.
    """
    # Try direct match
    candidate = os.path.join(raw_root, patient_id, f"{patient_id}.nii.gz")
    if os.path.exists(candidate):
        return candidate
    # SegTHOR Train folder structure
    for subfolder in ["train", "Train", "test", "Test", ""]:
        candidate = os.path.join(raw_root, subfolder, patient_id, f"{patient_id}.nii.gz")
        if os.path.exists(candidate):
            return candidate
    return None


def find_nifti_amos22(raw_root: str, patient_id: str) -> str | None:
    """Find NIfTI CT for an AMOS22 patient.
    AMOS22: raw_root/imagesTr/amos_{id:04d}.nii.gz
    patient_id in processed_dataset is like 'amos_0001'.
    """
    # Try imagesTr, imagesTs, imagesVa
    for subdir in ["imagesTr", "imagesTs", "imagesVa"]:
        candidate = os.path.join(raw_root, subdir, f"{patient_id}.nii.gz")
        if os.path.exists(candidate):
            return candidate
    # Direct match in root
    candidate = os.path.join(raw_root, f"{patient_id}.nii.gz")
    if os.path.exists(candidate):
        return candidate
    return None


def extract_spacing(nifti_path: str) -> tuple[float, float, float]:
    """Extract voxel spacing (z, y, x) in mm from a NIfTI file.

    Uses as_closest_canonical to match the preprocessing pipeline axis order.
    Returns (spacing_z, spacing_y, spacing_x).
    """
    img = nib.load(nifti_path)
    img = nib.as_closest_canonical(img)
    # Header zooms are (x, y, z) in canonical orientation
    zooms = img.header.get_zooms()  # (sx, sy, sz, ...) all in mm
    sx, sy, sz = float(zooms[0]), float(zooms[1]), float(zooms[2])
    # Preprocessing extracts slices along axis=2 (z after canonical), so:
    # volume shape is (H, W, Z) = (sx, sy, sz) indexed as [:,: ,z]
    # VolumeIterator builds (Z, H, W) so spacing = (sz, sy, sx)
    return sz, sy, sx


FIND_NIFTI_FNS = {
    "totalseg": find_nifti_totalseg,
    "segthor": find_nifti_segthor,
    "amos22": find_nifti_amos22,
}


def generate_spacing_csv(
    dataset: str,
    raw_root: str,
    processed_root: str,
    output_path: str | None = None,
) -> None:
    if dataset not in FIND_NIFTI_FNS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(FIND_NIFTI_FNS)}")

    find_fn = FIND_NIFTI_FNS[dataset]
    patient_ids = get_patient_ids(processed_root)
    print(f"[generate_spacing_csv] {len(patient_ids)} patients in {processed_root}")

    output_path = output_path or os.path.join(processed_root, "spacing.csv")

    missing = []
    rows = []

    for pid in tqdm(patient_ids, desc="Extracting spacing"):
        nifti_path = find_fn(raw_root, pid)
        if nifti_path is None:
            missing.append(pid)
            continue
        try:
            spacing_z, spacing_y, spacing_x = extract_spacing(nifti_path)
            rows.append({
                "patient_id": pid,
                "spacing_z": round(spacing_z, 6),
                "spacing_y": round(spacing_y, 6),
                "spacing_x": round(spacing_x, 6),
            })
        except Exception as e:
            print(f"  [WARN] {pid}: failed to read {nifti_path}: {e}")
            missing.append(pid)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "spacing_z", "spacing_y", "spacing_x"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {output_path} ({len(rows)} patients)")
    if missing:
        print(f"Missing ({len(missing)} patients): {missing[:10]}{'...' if len(missing) > 10 else ''}")


def main():
    parser = argparse.ArgumentParser(description="Generate spacing.csv from original NIfTI files")
    parser.add_argument("--dataset", required=True,
                        choices=["totalseg", "segthor", "amos22"],
                        help="Dataset type")
    parser.add_argument("--raw_root", required=True,
                        help="Root directory of original NIfTI files")
    parser.add_argument("--processed_root", required=True,
                        help="Root directory of processed dataset (contains slice_index.csv)")
    parser.add_argument("--output", default=None,
                        help="Output path for spacing.csv (default: processed_root/spacing.csv)")
    args = parser.parse_args()

    generate_spacing_csv(
        dataset=args.dataset,
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
