import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.segthor_augmentations import get_augmentations


class SegThorSliceDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".npy")])
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".npy")])
        assert len(self.image_files) == len(self.label_files), "Mismatch between images and labels"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        image = np.load(image_path).astype(np.float32)  # [1, 256, 256]
        label = np.load(label_path).astype(np.uint8)    # [8, 256, 256]

        # Medical preprocessing
        image = np.clip(image, -1000, 400)
        image = (image + 1000) / 1400  # → [0, 1]

        image = image.squeeze(0)  # [256, 256]
        label = np.transpose(label, (1, 2, 0))  # [256, 256, 8]

        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image, label = augmented["image"], augmented["mask"]
            label = label.permute(2, 0, 1)  # [8, 256, 256]
        else:
            image = torch.tensor(image).unsqueeze(0)
            label = torch.tensor(label).permute(2, 0, 1)

        return image.float(), label.float()


def get_dataloaders(processed_root, batch_size=4, num_workers=4):
    train_set = SegThorSliceDataset(
        os.path.join(processed_root, "imagesTrain"),
        os.path.join(processed_root, "labelsTrain"),
        transform=get_augmentations(is_train=True)
    )

    val_set = SegThorSliceDataset(
        os.path.join(processed_root, "imagesVal"),
        os.path.join(processed_root, "labelsVal"),
        transform=get_augmentations(is_train=False)
    )

    test_set = SegThorSliceDataset(
        os.path.join(processed_root, "imagesTest"),
        os.path.join(processed_root, "labelsTest"),
        transform=get_augmentations(is_train=False)
    )

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        DataLoader(test_set, batch_size=1, shuffle=False, num_workers=num_workers)
    )
