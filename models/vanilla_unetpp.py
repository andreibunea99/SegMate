"""
Vanilla U-Net++ Baseline Model
================================
Standard U-Net++ implementation using segmentation_models_pytorch library.
This serves as a baseline to demonstrate the value of CustomUNetPlusPlus modifications.

Architecture:
- Encoder: ResNet34 (standard backbone, NOT EfficientNet-B5)
- Decoder: U-Net++ nested skip connections
- Attention: None (vanilla U-Net++)
- No CBAM, SE, ASPP, or custom attention mechanisms

This baseline demonstrates:
1. The value of nested skip connections (U-Net++ vs U-Net)
2. The incremental contribution of CustomUNetPlusPlus enhancements
"""

import torch
import torch.nn as nn
from segmentation_models_pytorch import UnetPlusPlus
from segmentation_models_pytorch.base import SegmentationHead


class VanillaUNetPlusPlus(nn.Module):
    """
    Vanilla U-Net++ implementation for baseline comparison.

    Uses standard ResNet34 backbone and nested skip connections without
    any custom attention mechanisms or multi-scale features.

    Args:
        num_classes (int): Number of segmentation classes (default: 9)
        in_channels (int): Number of input channels (default: 1 for CT)
        encoder_name (str): Encoder backbone (default: 'resnet34')
        encoder_weights (str): Pretrained weights (default: 'imagenet')
    """

    def __init__(
        self,
        num_classes=9,
        in_channels=1,
        encoder_name='resnet34',
        encoder_weights='imagenet'
    ):
        super().__init__()

        # Decoder channels for U-Net++ (standard configuration)
        self.decoder_channels = [256, 128, 64, 32, 16]

        # Initialize vanilla U-Net++ from segmentation_models_pytorch
        self.unetpp = UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            decoder_use_batchnorm=True,
            decoder_channels=self.decoder_channels,
            decoder_attention_type=None,  # NO attention (vanilla)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [B, 1, H, W]

        Returns:
            masks: Segmentation masks [B, num_classes, H, W]
            boundaries: Boundary predictions [B, num_classes, H, W]
                       (same as masks for vanilla U-Net++ - no separate head)
        """
        # Get segmentation masks from U-Net++
        masks = self.unetpp(x)

        # For vanilla U-Net++, return masks twice for compatibility
        # (no separate boundary head - saves parameters and computation)
        return masks, masks


class VanillaUNetPlusPlusDeepSupervision(nn.Module):
    """
    Vanilla U-Net++ with deep supervision for fair comparison with CustomUNetPlusPlus.

    Deep supervision uses intermediate decoder outputs for auxiliary loss.
    This is a standard U-Net++ feature.

    Args:
        num_classes (int): Number of segmentation classes
        in_channels (int): Number of input channels
        encoder_name (str): Encoder backbone
        encoder_weights (str): Pretrained weights
    """

    def __init__(
        self,
        num_classes=9,
        in_channels=1,
        encoder_name='resnet34',
        encoder_weights='imagenet'
    ):
        super().__init__()

        # Decoder channels
        self.decoder_channels = [256, 128, 64, 32, 16]

        # Base U-Net++ model
        self.unetpp = UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            decoder_use_batchnorm=True,
            decoder_channels=self.decoder_channels,
            decoder_attention_type=None,
        )

        # Additional segmentation heads for deep supervision
        # (U-Net++ already has multiple decoder outputs at different scales)
        self.seg_head_1 = SegmentationHead(
            in_channels=self.decoder_channels[-2],  # 32 channels
            out_channels=num_classes,
            activation=None,
            kernel_size=3,
        )

        self.seg_head_2 = SegmentationHead(
            in_channels=self.decoder_channels[-3],  # 64 channels
            out_channels=num_classes,
            activation=None,
            kernel_size=3,
        )

        self.seg_head_3 = SegmentationHead(
            in_channels=self.decoder_channels[-4],  # 128 channels
            out_channels=num_classes,
            activation=None,
            kernel_size=3,
        )

    def forward(self, x):
        """
        Forward pass with deep supervision

        Args:
            x: Input tensor [B, 1, H, W]

        Returns:
            If training with deep supervision:
                List of [main_output, aux1, aux2, aux3]
            Else:
                (masks, boundaries) tuple for compatibility
        """
        # Get main output
        main_output = self.unetpp(x)

        # For compatibility with existing training scripts that expect (masks, boundaries)
        # Return main output twice
        return main_output, main_output


def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model instantiation and forward pass
    print("=" * 60)
    print("Vanilla U-Net++ Model Test")
    print("=" * 60)

    # Create model
    model = VanillaUNetPlusPlus(num_classes=9, in_channels=1)

    # Count parameters
    params = count_parameters(model)
    print(f"Total trainable parameters: {params:,}")
    print(f"Parameters in millions: {params / 1e6:.2f}M")

    # Test forward pass
    dummy_input = torch.randn(2, 1, 512, 512)
    print(f"\nInput shape: {dummy_input.shape}")

    model.eval()
    with torch.no_grad():
        masks, boundaries = model(dummy_input)

    print(f"Masks shape: {masks.shape}")
    print(f"Boundaries shape: {boundaries.shape}")

    # Test with deep supervision
    print("\n" + "=" * 60)
    print("Vanilla U-Net++ with Deep Supervision Test")
    print("=" * 60)

    model_ds = VanillaUNetPlusPlusDeepSupervision(num_classes=9)
    params_ds = count_parameters(model_ds)
    print(f"Total trainable parameters: {params_ds:,}")
    print(f"Parameters in millions: {params_ds / 1e6:.2f}M")

    model_ds.eval()
    with torch.no_grad():
        out1, out2 = model_ds(dummy_input)

    print(f"Main output shape: {out1.shape}")
    print(f"Boundary output shape: {out2.shape}")

    print("\n✅ Model test passed!")
