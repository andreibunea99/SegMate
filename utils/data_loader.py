import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from augmentations import get_augmentations



class ThoraxDataset(Dataset):
    def __init__(self, data_dir, transform=None, 
                 smooth_masks=False, kernel_size=5, sigma=1.0):
        self.data_dir = data_dir
        self.transform = transform
        self.smooth_masks = smooth_masks
        self.kernel = (kernel_size, kernel_size)
        self.sigma = sigma
        self.data = []

        # Load data paths
        for patient_folder in os.listdir(data_dir):
            patient_folder_path = os.path.join(data_dir, patient_folder)
            if os.path.isdir(patient_folder_path):
                for file in os.listdir(patient_folder_path):
                    if file.endswith('_image.npy'):
                        slice_index = file.split('_')[2]
                        self.data.append({
                            "image": os.path.join(patient_folder_path, 
                                f"Patient_{patient_folder.split('_')[-1]}_{slice_index}_image.npy"),
                            "label": os.path.join(patient_folder_path, 
                                f"Patient_{patient_folder.split('_')[-1]}_{slice_index}_label.npy")
                        })

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data[idx]["image"]
        label_path = self.data[idx]["label"]

        # Load data
        image = np.load(image_path).astype(np.float32)  # (H, W, 1)
        label = np.load(label_path).astype(np.float32)  # (C, H, W)

        # Smooth masks using Gaussian blur
        if self.smooth_masks:
            for channel in range(label.shape[0]):
                mask_channel = label[channel]
                blurred_mask = cv2.GaussianBlur(mask_channel, 
                                               self.kernel, 
                                               self.sigma)
                label[channel] = np.clip(blurred_mask, 0, 1)

        # Prepare for Albumentations
        image = image.squeeze()  # (H, W)
        label = np.transpose(label, (1, 2, 0))  # (H, W, C)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image, label = augmented["image"], augmented["mask"]

        # Convert to (C, H, W)
        label = label.permute(2, 0, 1)  # PyTorch tensor

        return image.float(), label.float()

def get_dataloaders(train_dir, val_dir, batch_size, 
                   augmentations, val_augmentations, 
                   num_workers=4):
    train_dataset = ThoraxDataset(
        train_dir, 
        transform=augmentations,
        smooth_masks=True,
        kernel_size=5,
        sigma=1.0
    ) if train_dir else None
    
    val_dataset = ThoraxDataset(
        val_dir,
        transform=val_augmentations,
        smooth_masks=False
    ) if val_dir else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    ) if train_dataset else None

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    ) if val_dataset else None

    return train_loader, val_loader

# Main function
if __name__ == "__main__":
    train_dir=r"../../Dataset/Training_2d"
    val_dir=r"../../Dataset/Validation_2d"
    batch_size = 4
    augmentations = get_augmentations()
    num_workers = 4

    train_loader, val_loader = get_dataloaders(
        train_dir, 
        val_dir, 
        batch_size=batch_size,
        augmentations=get_augmentations(is_train=True),  # Apply training augmentations
        val_augmentations=get_augmentations(is_train=False),  # Apply only normalization for validation
        num_workers=4
    )

    for images, labels in train_loader:
        print(f"Training Images shape: {images.shape}, Labels shape: {labels.shape}")
        break

    # for images, labels in val_loader:
    #     # print(f"Validation Images shape: {images.shape}, Labels shape: {labels.shape}")
    #     # if the shape of the images and labels is not as expected, print the shape
    #     if images.shape[1] != 1 or labels.shape[1] != 8:
    #         print(f"ERROR Validation Images shape: {images.shape}, {images.shape[1]}, Labels shape: {labels.shape}")


    print(f"Length of Train Loader: {len(train_loader)}")
    print(f"Length of Validation Loader: {len(val_loader)}")
