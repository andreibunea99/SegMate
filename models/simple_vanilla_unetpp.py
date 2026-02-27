"""
Simple & Fast Vanilla U-Net++ Implementation
=============================================
Minimalist U-Net++ with nested skip connections - NO bloat, FAST training.

Key features:
- Standard ResNet34 encoder (torchvision, NOT segmentation_models_pytorch)
- Nested skip connections (U-Net++ core feature)
- NO attention mechanisms
- NO ASPP
- Minimal decoder with standard conv blocks

This is a TRUE baseline - simple, fast, and demonstrates the value of nested skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


class ConvBlock(nn.Module):
    """Basic conv block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class SimpleVanillaUNetPlusPlus(nn.Module):
    """
    Simple Vanilla U-Net++ - Fast baseline implementation

    Architecture:
    - Encoder: ResNet34 (5 levels)
    - Decoder: Nested skip connections (U-Net++ feature)
    - No attention, no ASPP, no fancy stuff

    Args:
        num_classes (int): Number of output classes (default: 9)
        in_channels (int): Number of input channels (default: 1)
    """

    def __init__(self, num_classes=9, in_channels=1):
        super().__init__()

        # Encoder: ResNet34 backbone
        resnet = resnet34(pretrained=True)

        # Modify first conv for grayscale input
        if in_channels != 3:
            self.encoder0 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Initialize from pretrained weights (average across channels)
            with torch.no_grad():
                self.encoder0.weight.copy_(resnet.conv1.weight.mean(dim=1, keepdim=True))
        else:
            self.encoder0 = resnet.conv1

        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Encoder blocks (ResNet34 layers)
        self.encoder1 = resnet.layer1  # 64 channels, stride 1 (relative to after maxpool)
        self.encoder2 = resnet.layer2  # 128 channels, stride 2
        self.encoder3 = resnet.layer3  # 256 channels, stride 2
        self.encoder4 = resnet.layer4  # 512 channels, stride 2

        # Decoder channels (actual encoder output channels)
        channels = [64, 64, 128, 256, 512]  # [x0_0, x1_0, x2_0, x3_0, x4_0]

        # U-Net++ nested skip connections
        # Notation: up_concat_ij means node at column i, row j

        # Column 1 (first upsampling path)
        self.up_concat31 = ConvBlock(channels[3] + channels[4], channels[3])  # 256 + 512 -> 256
        self.up_concat21 = ConvBlock(channels[2] + channels[3], channels[2])  # 128 + 256 -> 128
        self.up_concat11 = ConvBlock(channels[1] + channels[2], channels[1])  # 64 + 128 -> 64
        self.up_concat01 = ConvBlock(channels[0] + channels[1], channels[0])  # 64 + 64 -> 64

        # Column 2
        self.up_concat22 = ConvBlock(channels[2] + channels[2] + channels[3], channels[2])  # 128 + 128 + 256 -> 128
        self.up_concat12 = ConvBlock(channels[1] + channels[1] + channels[2], channels[1])  # 64 + 64 + 128 -> 64
        self.up_concat02 = ConvBlock(channels[0] + channels[0] + channels[1], channels[0])  # 64 + 64 + 64 -> 64

        # Column 3
        self.up_concat13 = ConvBlock(channels[1] + channels[1] + channels[1] + channels[2], channels[1])  # 64 + 64 + 64 + 128 -> 64
        self.up_concat03 = ConvBlock(channels[0] + channels[0] + channels[0] + channels[1], channels[0])  # 64*3 + 64 -> 64

        # Column 4
        self.up_concat04 = ConvBlock(channels[0] * 4 + channels[1], channels[0])  # 64*4 + 64 -> 64

        # Final output
        self.final = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input [B, 1, H, W]

        Returns:
            masks: [B, num_classes, H, W]
            boundaries: [B, num_classes, H, W] (same as masks)
        """
        # Encoder
        x0_0 = self.relu(self.bn0(self.encoder0(x)))  # 64, H/2, W/2
        x_pool = self.maxpool(x0_0)                    # 64, H/4, W/4

        x1_0 = self.encoder1(x_pool)                   # 64, H/4, W/4
        x2_0 = self.encoder2(x1_0)                     # 128, H/8, W/8
        x3_0 = self.encoder3(x2_0)                     # 256, H/16, W/16
        x4_0 = self.encoder4(x3_0)                     # 512, H/32, W/32

        # Decoder with nested skip connections (U-Net++)
        # Column 1 (first decoder layer)
        x3_1 = self.up_concat31(torch.cat([x3_0, self._upsample(x4_0, x3_0)], dim=1))
        x2_1 = self.up_concat21(torch.cat([x2_0, self._upsample(x3_1, x2_0)], dim=1))
        x1_1 = self.up_concat11(torch.cat([x1_0, self._upsample(x2_1, x1_0)], dim=1))
        x0_1 = self.up_concat01(torch.cat([x0_0, self._upsample(x1_1, x0_0)], dim=1))

        # Column 2
        x2_2 = self.up_concat22(torch.cat([x2_0, x2_1, self._upsample(x3_1, x2_0)], dim=1))
        x1_2 = self.up_concat12(torch.cat([x1_0, x1_1, self._upsample(x2_2, x1_0)], dim=1))
        x0_2 = self.up_concat02(torch.cat([x0_0, x0_1, self._upsample(x1_2, x0_0)], dim=1))

        # Column 3
        x1_3 = self.up_concat13(torch.cat([x1_0, x1_1, x1_2, self._upsample(x2_2, x1_0)], dim=1))
        x0_3 = self.up_concat03(torch.cat([x0_0, x0_1, x0_2, self._upsample(x1_3, x0_0)], dim=1))

        # Column 4
        x0_4 = self.up_concat04(torch.cat([x0_0, x0_1, x0_2, x0_3, self._upsample(x1_3, x0_0)], dim=1))

        # Final output
        output = self.final(x0_4)

        # Upsample to original size (H, W)
        output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)

        # Return masks twice for compatibility with training script
        return output, output

    def _upsample(self, x, target):
        """Upsample x to match target spatial size"""
        return F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import time

    print("=" * 70)
    print("Simple Vanilla U-Net++ - Speed Test")
    print("=" * 70)

    # Create model
    model = SimpleVanillaUNetPlusPlus(num_classes=9, in_channels=1)
    model.eval()

    # Count parameters
    params = count_parameters(model)
    print(f"\nParameters: {params:,} ({params/1e6:.2f}M)")

    # Test forward pass
    dummy = torch.randn(2, 1, 512, 512)
    print(f"\nInput shape: {dummy.shape}")

    with torch.no_grad():
        masks, boundaries = model(dummy)

    print(f"Output shape: {masks.shape}")
    print(f"Boundaries shape: {boundaries.shape}")

    # Speed test (CPU)
    print(f"\n{'='*70}")
    print("CPU Inference Speed")
    print("=" * 70)

    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(dummy)

        # Time 100 iterations
        start = time.time()
        for _ in range(100):
            _ = model(dummy)
        elapsed = time.time() - start

    print(f"Average time: {elapsed/100*1000:.2f}ms per batch")
    print(f"Throughput: {200/elapsed:.2f} images/sec")

    # GPU test if available
    if torch.cuda.is_available():
        print(f"\n{'='*70}")
        print("GPU Inference Speed")
        print("=" * 70)

        model = model.cuda()
        dummy = dummy.cuda()

        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(dummy)

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(100):
                _ = model(dummy)
            torch.cuda.synchronize()
            elapsed = time.time() - start

        print(f"Average time: {elapsed/100*1000:.2f}ms per batch")
        print(f"Throughput: {200/elapsed:.2f} images/sec")

    print(f"\n{'='*70}")
    print("✅ Model ready for training!")
    print("=" * 70)
