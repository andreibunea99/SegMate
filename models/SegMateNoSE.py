import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b5

"""
SegMateNoSE - Ablation Study Model
==================================
This is an exact copy of CustomUNetPlusPlus with SE blocks removed from
skip connections to study the effect of SE+CBAM attention combination.

Changes from original:
- Skip connections (skip_dec*) use 'none' instead of 'se'
- Main decoder blocks still use CBAM

This ablation tests the hypothesis that combining SE on skip pathways
with CBAM on decoder blocks provides complementary attention benefits.
"""

# --- CBAM Module ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

    def forward(self, x):
        avg_out = self.fc2(F.relu(self.fc1(F.adaptive_avg_pool2d(x, 1))))
        max_out = self.fc2(F.relu(self.fc1(F.adaptive_max_pool2d(x, 1))))
        out = avg_out + max_out
        return torch.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return torch.sigmoid(x)

class CBAMBlock(nn.Module):
    def __init__(self, channels):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# --- Squeeze and Excitation Block (kept for reference, not used in this ablation) ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# --- ASPP Module ---
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.atrous_block1 = nn.Conv2d(in_channels, out_channels, 1, padding=0, dilation=1, bias=False)
        self.atrous_block6 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.atrous_block12 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.atrous_block18 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn6 = nn.BatchNorm2d(out_channels)
        self.bn12 = nn.BatchNorm2d(out_channels)
        self.bn18 = nn.BatchNorm2d(out_channels)

        self.conv_1x1_output = nn.Conv2d(out_channels * 4, out_channels, 1, bias=False)
        self.bn_output = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.bn1(self.atrous_block1(x)))
        x6 = self.relu(self.bn6(self.atrous_block6(x)))
        x12 = self.relu(self.bn12(self.atrous_block12(x)))
        x18 = self.relu(self.bn18(self.atrous_block18(x)))

        x = torch.cat([x1, x6, x12, x18], dim=1)
        x = self.relu(self.bn_output(self.conv_1x1_output(x)))
        return x

# --- Feature Alignment Module ---
class AlignmentBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AlignmentBlock, self).__init__()
        self.align_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.align_conv(x)))

# --- Improved Decoder Block with Residual Connection ---
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention='cbam'):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Residual connection
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

        # Attention mechanism
        if use_attention == 'cbam':
            self.attention = CBAMBlock(out_channels)
        elif use_attention == 'se':
            self.attention = SEBlock(out_channels)
        else:
            self.attention = nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)  # Adding dropout for regularization

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.attention(x)
        x = x + residual  # Add residual connection
        return self.relu(x)

# --- Main SegMate without SE (Ablation Study) ---
class SegMateNoSE(nn.Module):
    """
    SegMate model with SE blocks removed from skip connections.

    This ablation model replaces SE attention on dense skip connections
    with Identity (no attention) while keeping CBAM on main decoder blocks.

    Used to evaluate the contribution of SE blocks on skip pathways
    when combined with CBAM on the decoder path.
    """
    def __init__(self, num_classes=9, in_channels=1, deep_supervision=False):
        super(SegMateNoSE, self).__init__()

        # Load EfficientNet-B5 backbone
        base_model = efficientnet_b5(weights="IMAGENET1K_V1")

        # Modify first layer for different input channels
        if in_channels != 3:
            weight = base_model.features[0][0].weight
            base_model.features[0][0] = nn.Conv2d(
                in_channels, 48, kernel_size=3, stride=2, padding=1, bias=False
            )
            with torch.no_grad():
                base_model.features[0][0].weight[:] = weight.mean(dim=1, keepdim=True)

        self.deep_supervision = deep_supervision

        # Encoder
        self.encoder = base_model.features

        # Feature alignment blocks
        self.align_e2 = AlignmentBlock(40, 40)  # feature[2]
        self.align_e3 = AlignmentBlock(64, 64)  # feature[3]
        self.align_e4 = AlignmentBlock(128, 128)  # feature[4]
        self.align_e5 = AlignmentBlock(176, 176)  # feature[5]

        # ASPP module
        self.aspp = ASPP(2048, 256)

        # Decoder path (using CBAM - same as original)
        self.up4 = nn.ConvTranspose2d(256, 160, kernel_size=2, stride=2)
        self.dec4 = DecoderBlock(160 + 176, 160, 'cbam')

        self.up3 = nn.ConvTranspose2d(160, 88, kernel_size=2, stride=2)
        self.dec3 = DecoderBlock(88 + 128, 88, 'cbam')

        self.up2 = nn.ConvTranspose2d(88, 48, kernel_size=2, stride=2)
        self.dec2 = DecoderBlock(48 + 64, 48, 'cbam')

        self.up1 = nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2)
        self.dec1 = DecoderBlock(24 + 40, 32, 'cbam')

        # Dense skip connections for UNet++ - ABLATION: 'none' instead of 'se'
        self.skip_dec3_4 = DecoderBlock(88 + 88, 88, 'none')
        self.skip_dec2_3 = DecoderBlock(48 + 48, 48, 'none')
        self.skip_dec1_2 = DecoderBlock(56, 32, 'none')  # Adjust in_channels to 56
        self.skip_dec2_4 = DecoderBlock(160 + 48, 48, 'none')
        self.skip_dec1_3 = DecoderBlock(88 + 32, 32, 'none')
        self.skip_dec1_4 = DecoderBlock(160 + 32, 32, 'none')

        # Output heads
        self.segmentation_head = nn.Conv2d(32, num_classes, 1)
        self.boundary_head = nn.Conv2d(32, num_classes, 1)
        self.presence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
            nn.Sigmoid()
        )

        # Deep supervision heads
        if deep_supervision:
            self.deep_head1 = nn.Conv2d(160, num_classes, 1)
            self.deep_head2 = nn.Conv2d(88, num_classes, 1)
            self.deep_head3 = nn.Conv2d(48, num_classes, 1)

    def forward(self, x):
        # Store encoder features
        enc_features = []
        for block in self.encoder:
            x = block(x)
            enc_features.append(x)

        # Apply feature alignment
        e2 = self.align_e2(enc_features[2])  # 40 channels
        e3 = self.align_e3(enc_features[3])  # 64 channels
        e4 = self.align_e4(enc_features[4])  # 128 channels
        e5 = self.align_e5(enc_features[5])  # 176 channels

        # ASPP on bottleneck features
        x = self.aspp(enc_features[-1])

        # Decoder path with skip connections
        x0_4 = self.up4(x)
        if x0_4.shape[2:] != e5.shape[2:]:
            x0_4 = F.interpolate(x0_4, size=e5.shape[2:], mode='bilinear', align_corners=True)
        x0_4 = self.dec4(torch.cat([x0_4, e5], dim=1))

        x0_3 = self.up3(x0_4)
        if x0_3.shape[2:] != e4.shape[2:]:
            x0_3 = F.interpolate(x0_3, size=e4.shape[2:], mode='bilinear', align_corners=True)
        x0_3 = self.dec3(torch.cat([x0_3, e4], dim=1))

        x0_2 = self.up2(x0_3)
        if x0_2.shape[2:] != e3.shape[2:]:
            x0_2 = F.interpolate(x0_2, size=e3.shape[2:], mode='bilinear', align_corners=True)
        x0_2 = self.dec2(torch.cat([x0_2, e3], dim=1))

        x0_1 = self.up1(x0_2)
        if x0_1.shape[2:] != e2.shape[2:]:
            x0_1 = F.interpolate(x0_1, size=e2.shape[2:], mode='bilinear', align_corners=True)
        x0_1 = self.dec1(torch.cat([x0_1, e2], dim=1))

        # Dense skip connections
        x1_3 = self.up3(x0_4)
        if x1_3.shape[2:] != x0_3.shape[2:]:
            x1_3 = F.interpolate(x1_3, size=x0_3.shape[2:], mode='bilinear', align_corners=True)
        x1_3 = self.skip_dec3_4(torch.cat([x1_3, x0_3], dim=1))

        x1_2 = self.up2(x1_3)
        if x1_2.shape[2:] != x0_2.shape[2:]:
            x1_2 = F.interpolate(x1_2, size=x0_2.shape[2:], mode='bilinear', align_corners=True)
        x1_2 = self.skip_dec2_3(torch.cat([x1_2, x0_2], dim=1))

        x1_1 = self.up1(x1_2)
        if x1_1.shape[2:] != x0_1.shape[2:]:
            x1_1 = F.interpolate(x1_1, size=x0_1.shape[2:], mode='bilinear', align_corners=True)
        x1_1 = self.skip_dec1_2(torch.cat([x1_1, x0_1], dim=1))

        # Output
        segmentation = self.segmentation_head(x1_1)
        boundary = self.boundary_head(x1_1)
        presence = self.presence_head(x1_1)

        # Resize outputs to match input size (512x512 from the data loader)
        input_size = (512, 512)  # Replace with the actual size from the data loader
        if segmentation.shape[2:] != input_size:
            segmentation = F.interpolate(segmentation, size=input_size, mode='bilinear', align_corners=True)
            boundary = F.interpolate(boundary, size=input_size, mode='bilinear', align_corners=True)

        if self.deep_supervision:
            deep_outs = [
                F.interpolate(self.deep_head1(x0_4), size=input_size, mode='bilinear', align_corners=True),
                F.interpolate(self.deep_head2(x0_3), size=input_size, mode='bilinear', align_corners=True),
                F.interpolate(self.deep_head3(x0_2), size=input_size, mode='bilinear', align_corners=True)
            ]
            return segmentation, boundary, presence, deep_outs

        return segmentation, boundary, presence
