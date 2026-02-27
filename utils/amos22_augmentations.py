import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_augmentations(is_train=True):
    """
    Augmentation pipeline for AMOS22 dataset.
    Handles variable image sizes (512x512, 768x768, etc.) by resizing to 512x512.
    """
    if is_train:
        return A.Compose([
            A.Resize(512, 512, interpolation=cv2.INTER_LINEAR),  # Handle variable sizes
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Affine(
                scale=(0.85, 1.15),
                rotate=(-15, 15),
                translate_percent=(-0.1, 0.1),
                shear=(-10, 10),
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT
            ),
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.GridDistortion(p=0.2),
            A.GaussNoise(var_limit=(0.001, 0.01), p=0.2),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
        ], additional_targets={"mask": "mask"})
    else:
        return A.Compose([
            A.Resize(512, 512, interpolation=cv2.INTER_LINEAR),  # Handle variable sizes
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
        ], additional_targets={"mask": "mask"})
