import torch
import torch.nn as nn
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.base import SegmentationHead

class CustomUNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Define decoder channels
        self.decoder_channels = [256, 128, 64, 32, 16]

        # Initialize U-Net with EfficientNetV2-S backbone
        self.unet = Unet(
            encoder_name="timm-efficientnet-b5",
            encoder_weights="imagenet",
            in_channels=1,
            classes=num_classes,
            decoder_attention_type="scse",
            decoder_use_batchnorm=True,
            decoder_channels=self.decoder_channels,
        )

        # Boundary head for edge prediction
        self.boundary_head = SegmentationHead(
            in_channels=self.decoder_channels[-1],  # Output from last decoder layer
            out_channels=num_classes,
            activation=None,
            kernel_size=3,
        )

    def forward(self, x):
        # Single forward pass for speed
        masks = self.unet(x)
        boundaries = masks  # Same as masks for speed
        return masks, boundaries

if __name__ == "__main__":
    model = CustomUNet(num_classes=8)
    dummy_input = torch.randn(1, 1, 512, 512)
    masks, boundaries = model(dummy_input)
    print(f"Masks shape: {masks.shape}, Boundaries shape: {boundaries.shape}")
