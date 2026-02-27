# evaluation/volume_dataloader_3d.py
"""
3D Volume Iterator — yields complete per-patient 3D volumes for volume-level evaluation.

Groups slices from slice_index.csv by patient ID, loads all slices in order,
builds 2.5D context (t-1, t, t+1) matching ChunkedSliceDataset.__getitem__,
and computes z_norm per slice using organ boundary logic from the FiLM dataloader.

Yields: (patient_id, imgs[Z,3,H,W], lbls[Z,C,H,W], z_norms[Z])
"""

import os
import csv
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Iterator

import numpy as np
import torch

# Default fallback spacing (isotropic 1mm) used when spacing.csv is absent.
# This preserves backward-compatible behaviour: HD95 is in voxel units.
_DEFAULT_SPACING = (1.0, 1.0, 1.0)


class VolumeIterator:
    """Iterate over complete 3D patient volumes from chunked .npz storage."""

    def __init__(self, root: str, split: str):
        """
        Args:
            root:  Dataset root (e.g. "processed_dataset", "processed_dataset_segthor")
            split: "Train", "Val", or "Test"
        """
        self.root = root
        self.split = split.capitalize()

        # Per-patient voxel spacing (z, y, x) in mm.
        # Populated from spacing.csv if present; otherwise _DEFAULT_SPACING is used.
        self.spacings: Dict[str, Tuple[float, float, float]] = {}
        spacing_csv = os.path.join(root, "spacing.csv")
        if os.path.exists(spacing_csv):
            with open(spacing_csv) as sf:
                rdr = csv.DictReader(sf)
                for row in rdr:
                    self.spacings[row["patient_id"]] = (
                        float(row["spacing_z"]),
                        float(row["spacing_y"]),
                        float(row["spacing_x"]),
                    )
            print(f"[VolumeIterator] Loaded spacing for {len(self.spacings)} patients"
                  f" from {spacing_csv}")

        csv_path = os.path.join(root, "slice_index.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"slice_index.csv not found in {root}")

        # Parse CSV and group by patient
        self.patients: Dict[str, List[dict]] = defaultdict(list)
        self.organ_names: List[str] = []
        # For z_norm: track per-patient organ presence
        patient_slice_presence: Dict[str, List[Tuple[int, List[int]]]] = defaultdict(list)

        prefix = f"images{self.split}"
        with open(csv_path) as f:
            rdr = csv.reader(f)
            header = next(rdr)
            self.organ_names = header[1:]

            for row in rdr:
                path = row[0]
                if not path.startswith(prefix):
                    continue

                presence = list(map(int, row[1:]))

                # Parse: "imagesTrain/s0910_chunk000_slice042.npz"
                filename = path.split("/")[1]  # "s0910_chunk000_slice042.npz"
                patient_id = filename.split("_chunk")[0]  # "s0910"
                chunk_id = filename.split("_slice")[0]  # "s0910_chunk000"
                slice_tag = filename.split("_slice")[1]  # "042.npz"
                slice_idx = int(slice_tag.split(".")[0])  # 42

                self.patients[patient_id].append({
                    "chunk_id": chunk_id,
                    "slice_idx": slice_idx,
                    "presence": presence,
                })
                patient_slice_presence[patient_id].append((slice_idx, presence))

        # Sort slices within each patient by slice_idx
        for pid in self.patients:
            self.patients[pid].sort(key=lambda x: x["slice_idx"])

        # Build organ boundaries for z_norm (same as FiLM dataloader)
        self.patient_bounds: Dict[str, Tuple[int, int]] = {}
        for patient_id, slices in patient_slice_presence.items():
            slices_with_organs = [
                sidx for sidx, pres in slices if any(pres)
            ]
            if slices_with_organs:
                self.patient_bounds[patient_id] = (
                    min(slices_with_organs), max(slices_with_organs)
                )
            else:
                all_indices = [s[0] for s in slices]
                if all_indices:
                    self.patient_bounds[patient_id] = (
                        min(all_indices), max(all_indices)
                    )

        self.patient_ids = sorted(self.patients.keys())
        print(f"[VolumeIterator] split={self.split}, patients={len(self.patient_ids)}, "
              f"total_slices={sum(len(v) for v in self.patients.values())}")

    def get_spacing(self, patient_id: str) -> Tuple[float, float, float]:
        """Return voxel spacing (z, y, x) in mm for a patient.

        Falls back to _DEFAULT_SPACING (1,1,1) when spacing.csv was not loaded
        or the patient is not listed — HD95 will then be in voxel units.
        """
        return self.spacings.get(patient_id, _DEFAULT_SPACING)

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __iter__(self) -> Iterator[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]:
        for pid in self.patient_ids:
            yield self._load_patient(pid)

    def _load_patient(self, patient_id: str):
        """Load all slices for one patient, returning 3D tensors.

        Returns:
            (patient_id, imgs[Z,3,H,W], lbls[Z,C,H,W], z_norms[Z])
        """
        entries = self.patients[patient_id]
        first_organ, last_organ = self.patient_bounds.get(
            patient_id, (entries[0]["slice_idx"], entries[-1]["slice_idx"])
        )
        organ_range = max(last_organ - first_organ, 1)

        # Cache loaded chunks to avoid re-reading the same .npz
        img_cache: Dict[str, dict] = {}
        lbl_cache: Dict[str, dict] = {}

        imgs_list = []
        lbls_list = []
        z_norms_list = []

        for entry in entries:
            chunk_id = entry["chunk_id"]
            slice_idx = entry["slice_idx"]

            # Load chunk data (cached)
            if chunk_id not in img_cache:
                img_npz = os.path.join(
                    self.root, f"images{self.split}", f"{chunk_id}.npz"
                )
                lbl_npz = os.path.join(
                    self.root, f"labels{self.split}", f"{chunk_id}.npz"
                )
                if not (os.path.exists(img_npz) and os.path.exists(lbl_npz)):
                    continue
                img_cache[chunk_id] = np.load(img_npz, allow_pickle=True)
                lbl_cache[chunk_id] = np.load(lbl_npz, allow_pickle=True)

            img_data = img_cache[chunk_id]
            lbl_data = lbl_cache[chunk_id]

            idx_arr = img_data["indices"]
            matches = np.where(idx_arr == slice_idx)[0]
            if len(matches) == 0:
                continue
            pos = matches[0]

            # 2.5D context: (t-1, t, t+1) clamped at chunk boundaries
            data = img_data["data"]  # [N, H, W]
            total_slices = len(idx_arr)

            pos_m1 = max(pos - 1, 0)
            pos_p1 = min(pos + 1, total_slices - 1)

            img_m1 = data[pos_m1].astype(np.float32)
            img_0 = data[pos].astype(np.float32)
            img_p1 = data[pos_p1].astype(np.float32)

            img = np.stack([img_m1, img_0, img_p1], axis=0)  # [3, H, W]

            # Apply same normalization as test augmentations:
            # A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0)
            # does: img = (img / 255.0 - 0.5) / 0.5
            img = (img / 255.0 - 0.5) / 0.5

            # Labels
            lbl = lbl_data["data"]
            C = lbl.shape[0] // total_slices
            lbl_slice = lbl[pos * C: (pos + 1) * C]  # [C, H, W]

            imgs_list.append(img)
            lbls_list.append(lbl_slice)

            # z_norm
            z_norm = np.clip(
                (slice_idx - first_organ) / organ_range, 0.0, 1.0
            ).astype(np.float32)
            z_norms_list.append(z_norm)

        if len(imgs_list) == 0:
            # Return empty tensors if no valid slices
            return patient_id, torch.empty(0), torch.empty(0), torch.empty(0)

        imgs = torch.from_numpy(np.stack(imgs_list, axis=0))     # [Z, 3, H, W]
        lbls = torch.from_numpy(np.stack(lbls_list, axis=0))     # [Z, C, H, W]
        z_norms = torch.tensor(z_norms_list, dtype=torch.float32)  # [Z]

        return patient_id, imgs, lbls, z_norms
