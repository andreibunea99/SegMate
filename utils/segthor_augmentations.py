import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_augmentations(is_train=True):
    if is_train:
        return A.Compose([
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
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
        ], additional_targets={"mask": "mask"})
