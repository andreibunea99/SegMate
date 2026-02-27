# balanced_dataloader.py
"""
Balanced slice DataLoader (TotalSegmentator + chunked .npz support)
===================================================================
✓ Compatible with compressed chunked slices from improved_totalsegmentator_dataset.py
✓ Supports balancing across organ presence
✓ Robust against corrupted or missing files
"""

import os, csv, random, math
from typing import List
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from utils.segthor_augmentations import get_augmentations

# ───────────────────────── CSV Index ──────────────────────────

class SliceIndex:
    def __init__(self, csv_path: str):
        self.entries: List[str] = []
        self.organ_names: List[str] = []
        self.indices_by_cls: List[List[int]] = []
        self.metadata = {}

        with open(csv_path) as f:
            rdr = csv.reader(f)
            header = next(rdr)
            self.organ_names = header[1:]
            self.indices_by_cls = [[] for _ in self.organ_names]

            for idx, row in enumerate(rdr):
                path = row[0]
                self.entries.append(path)
                presence = list(map(int, row[1:]))
                self.metadata[path] = presence
                for i, flag in enumerate(presence):
                    if flag:
                        self.indices_by_cls[i].append(idx)

# ───────────────────── Sampler ─────────────────────

class BalancedBatchSampler(Sampler[List[int]]):
    def __init__(self, valid_indices: List[int], batch_size: int, seed: int = 42):
        self.valid = valid_indices
        self.batch = batch_size
        self.rng = random.Random(seed)
        
        # Calculate the expected number of complete batches at initialization
        self.num_batches = len(valid_indices) // batch_size
        
        # Ensure we have at least one batch if there are any valid indices
        if len(valid_indices) > 0 and self.num_batches == 0:
            self.num_batches = 1

    def __iter__(self):
        indices = list(self.valid)
        if not indices:
            raise ValueError("Empty valid_indices list in sampler")
            
        self.rng.shuffle(indices)
        pos = 0
        batch_count = 0
        
        # Only yield up to self.num_batches batches
        while batch_count < self.num_batches:
            batch = set()
            attempts = 0
            
            while len(batch) < self.batch and attempts < self.batch * 3:
                if pos >= len(indices):
                    self.rng.shuffle(indices)
                    pos = 0
                    
                idx = indices[pos]
                pos += 1
                
                if idx not in batch:
                    batch.add(idx)
                    
                attempts += 1
                
            if len(batch) == self.batch:
                yield list(batch)
                batch_count += 1
            else:
                # If we can't form a complete batch, stop iteration
                break

    def __len__(self):
        return self.num_batches


# ───────────────────── Dataset ─────────────────────

class ChunkedSliceDataset(Dataset):
    def __init__(self, root: str, split: str, si: SliceIndex, transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.slice_index = si

        prefix = f"images{split.capitalize()}"
        self.paths = [p for p in si.entries if p.startswith(prefix)]
        self.valid_indices = []

        for idx, path in enumerate(self.paths):
            try:
                chunk_id = path.split("/")[1].split("_slice")[0]
                img_npz = os.path.join(root, f"images{split.capitalize()}", f"{chunk_id}.npz")
                lbl_npz = os.path.join(root, f"labels{split.capitalize()}", f"{chunk_id}.npz")
                if os.path.exists(img_npz) and os.path.exists(lbl_npz):
                    self.valid_indices.append(idx)
                else:
                    print(f"[Missing] {img_npz} or {lbl_npz} not found.")
            except Exception as e:
                print(f"[DatasetInitError] path={path}, error={e}")

        print(f"[ChunkedSliceDataset] Split='{split}' -> {len(self.valid_indices)} valid slices out of {len(self.paths)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        path = self.paths[real_idx]
        chunk_id, slice_tag = path.split("/")[1].split("_slice")
        slice_idx = int(slice_tag.split(".")[0])

        img_npz = os.path.join(self.root, f"images{self.split.capitalize()}", f"{chunk_id}.npz")
        lbl_npz = os.path.join(self.root, f"labels{self.split.capitalize()}", f"{chunk_id}.npz")

        img_data = np.load(img_npz, allow_pickle=True)
        lbl_data = np.load(lbl_npz, allow_pickle=True)

        idx_arr = img_data['indices']
        pos = np.where(idx_arr == slice_idx)[0][0]
        img = img_data['data'][pos].astype(np.float32)

        lbl = lbl_data['data']
        total_slices = len(idx_arr)
        C = lbl.shape[0] // total_slices
        if lbl.shape[0] % total_slices != 0:
            raise ValueError(f"[Slice shape mismatch] {chunk_id}: total_lbl={lbl.shape[0]}, slices={total_slices} -> Cannot divide.")

        lbl = lbl[pos*C:(pos+1)*C]
        if lbl.ndim == 3 and lbl.shape[0] != img.shape[0]:
            lbl = np.transpose(lbl, (1, 2, 0))

        if img.shape != lbl.shape[:2]:
            raise ValueError(f"[Shape mismatch before augment] IMG {img.shape}, LBL {lbl.shape} at {chunk_id}_slice{slice_idx}")

        if self.transform:
            aug = self.transform(image=img, mask=lbl)
            img = aug["image"]
            lbl = aug["mask"].permute(2, 0, 1)
        else:
            img = torch.tensor(img).unsqueeze(0)
            lbl = torch.tensor(lbl).permute(2, 0, 1)
        return img.float(), lbl.float()

# ───────────────────── Loader Factory ─────────────────────

def get_dataloaders_balanced(root: str, batch_size: int = 60, num_workers: int = 8):
    csv_path = os.path.join(root, "slice_index.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError("slice_index.csv missing – run prepare_dataset first")

    si = SliceIndex(csv_path)
    train_ds = ChunkedSliceDataset(root, "Train", si, get_augmentations(True))
    val_ds   = ChunkedSliceDataset(root, "Val",   si, get_augmentations(False))
    test_ds  = ChunkedSliceDataset(root, "Test",  si, get_augmentations(False))

    train_ld = DataLoader(train_ds, batch_sampler=BalancedBatchSampler(train_ds.valid_indices, batch_size), num_workers=num_workers, pin_memory=True)
    val_ld   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_ld  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers)
    return train_ld, val_ld, test_ld
