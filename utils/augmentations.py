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
                shear=(-15, 15),
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT
            ),
            A.RandomResizedCrop(
                size=(256, 256),
                scale=(0.75, 1.0),
                ratio=(0.8, 1.2),
                p=0.3
            ),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                p=0.3,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST
            ),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})
    else:
        return A.Compose([
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})