# models/segmate_fastvit.py
"""
SegMate / CustomUNet++ variant with FastViT encoder (timm)
=========================================================
This is a COPY of your existing model (CBAM/SE/ASPP/Decoder/Dense skips/heads),
but replaces the EfficientNet-B5 encoder with a FastViT encoder.

Key design choice (to keep everything else identical):
- We keep your decoder widths EXACTLY the same (160/88/48/32) and heads identical.
- We add alignment blocks that project FastViT feature channels -> the SAME channel
  sizes your decoder expects (40, 64, 128, 176), so the rest of the network stays
  structurally consistent.

Requirements:
    pip install timm

Usage:
    from models.segmate_fastvit import SegMateFastViT
    model = SegMateFastViT(num_classes=9, in_channels=1, deep_supervision=True, encoder_name="fastvit_t12")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm


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


# --- Squeeze and Excitation Block ---
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

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

        if use_attention == 'cbam':
            self.attention = CBAMBlock(out_channels)
        elif use_attention == 'se':
            self.attention = SEBlock(out_channels)
        else:
            self.attention = nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.attention(x)
        x = x + residual
        return self.relu(x)


class SegMateFastViT(nn.Module):
    """
    Copy of your CustomUNetPlusPlus, but with FastViT encoder.

    We keep the decoder/head channel plan the same as your EfficientNet-B5 version by aligning
    FastViT feature maps to these target channels:
        e2 -> 40
        e3 -> 64
        e4 -> 128
        e5 -> 176
    Then everything downstream stays unchanged.
    """
    def __init__(
        self,
        num_classes=9,
        in_channels=1,
        deep_supervision=False,
        encoder_name: str = "fastvit_t12",
        pretrained: bool = True,
        # which feature stages to extract (4 stages)
        out_indices=(0, 1, 2, 3),
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder_name = encoder_name

        # ---- FastViT encoder from timm (features_only returns list of feature maps) ----
        # in_chans lets you use 1-channel CT directly.
        print(f"[SegMateFastViT] Initializing with encoder: {encoder_name}")
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels,
            out_indices=out_indices,
        )
        print(f"[SegMateFastViT] Encoder '{encoder_name}' loaded successfully")

        # Determine encoder feature channels dynamically
        feat_info = self.encoder.feature_info
        enc_chs = [feat_info[i]["num_chs"] for i in range(len(out_indices))]
        # We expect 4 feature maps
        if len(enc_chs) != 4:
            raise ValueError(
                f"{encoder_name} with out_indices={out_indices} returned {len(enc_chs)} features; expected 4."
            )

        ch2, ch3, ch4, ch5 = enc_chs  # e2,e3,e4,e5 from FastViT stages

        # ---- Align FastViT stages to your decoder-expected channels (keep rest identical) ----
        self.align_e2 = AlignmentBlock(ch2, 40)
        self.align_e3 = AlignmentBlock(ch3, 64)
        self.align_e4 = AlignmentBlock(ch4, 128)
        self.align_e5 = AlignmentBlock(ch5, 176)

        # ASPP on deepest feature (FastViT last stage channels -> 256)
        self.aspp = ASPP(ch5, 256)

        # ---- Decoder path (identical widths as your current model) ----
        self.up4 = nn.ConvTranspose2d(256, 160, kernel_size=2, stride=2)
        self.dec4 = DecoderBlock(160 + 176, 160, 'cbam')

        self.up3 = nn.ConvTranspose2d(160, 88, kernel_size=2, stride=2)
        self.dec3 = DecoderBlock(88 + 128, 88, 'cbam')

        self.up2 = nn.ConvTranspose2d(88, 48, kernel_size=2, stride=2)
        self.dec2 = DecoderBlock(48 + 64, 48, 'cbam')

        self.up1 = nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2)
        self.dec1 = DecoderBlock(24 + 40, 32, 'cbam')

        # ---- Dense skip connections (keep as in your code) ----
        self.skip_dec3_4 = DecoderBlock(88 + 88, 88, 'se')
        self.skip_dec2_3 = DecoderBlock(48 + 48, 48, 'se')
        self.skip_dec1_2 = DecoderBlock(56, 32, 'se')  # (32 + 24?) you used 56; keep identical
        self.skip_dec2_4 = DecoderBlock(160 + 48, 48, 'se')
        self.skip_dec1_3 = DecoderBlock(88 + 32, 32, 'se')
        self.skip_dec1_4 = DecoderBlock(160 + 32, 32, 'se')

        # ---- Output heads ----
        self.segmentation_head = nn.Conv2d(32, num_classes, 1)
        self.boundary_head = nn.Conv2d(32, num_classes, 1)
        self.presence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
            nn.Sigmoid()
        )

        # ---- Deep supervision heads (same as your model) ----
        if deep_supervision:
            self.deep_head1 = nn.Conv2d(160, num_classes, 1)
            self.deep_head2 = nn.Conv2d(88, num_classes, 1)
            self.deep_head3 = nn.Conv2d(48, num_classes, 1)

    def forward(self, x):
        input_size = x.shape[2:]  # DO NOT hardcode 512x512

        # FastViT features: list of 4 feature maps
        # f2, f3, f4, f5 correspond to increasing depth / decreasing resolution
        f2, f3, f4, f5 = self.encoder(x)

        # Align to expected channels
        e2 = self.align_e2(f2)  # 40
        e3 = self.align_e3(f3)  # 64
        e4 = self.align_e4(f4)  # 128
        e5 = self.align_e5(f5)  # 176

        # ASPP on deepest (use raw f5 channels)
        x = self.aspp(f5)  # -> 256

        # Decoder path with skip connections
        x0_4 = self.up4(x)
        if x0_4.shape[2:] != e5.shape[2:]:
            x0_4 = F.interpolate(x0_4, size=e5.shape[2:], mode='bilinear', align_corners=True)
        x0_4 = self.dec4(torch.cat([x0_4, e5], dim=1))  # -> 160

        x0_3 = self.up3(x0_4)
        if x0_3.shape[2:] != e4.shape[2:]:
            x0_3 = F.interpolate(x0_3, size=e4.shape[2:], mode='bilinear', align_corners=True)
        x0_3 = self.dec3(torch.cat([x0_3, e4], dim=1))  # -> 88

        x0_2 = self.up2(x0_3)
        if x0_2.shape[2:] != e3.shape[2:]:
            x0_2 = F.interpolate(x0_2, size=e3.shape[2:], mode='bilinear', align_corners=True)
        x0_2 = self.dec2(torch.cat([x0_2, e3], dim=1))  # -> 48

        x0_1 = self.up1(x0_2)
        if x0_1.shape[2:] != e2.shape[2:]:
            x0_1 = F.interpolate(x0_1, size=e2.shape[2:], mode='bilinear', align_corners=True)
        x0_1 = self.dec1(torch.cat([x0_1, e2], dim=1))  # -> 32

        # Dense skip connections (same pattern you used)
        x1_3 = self.up3(x0_4)
        if x1_3.shape[2:] != x0_3.shape[2:]:
            x1_3 = F.interpolate(x1_3, size=x0_3.shape[2:], mode='bilinear', align_corners=True)
        x1_3 = self.skip_dec3_4(torch.cat([x1_3, x0_3], dim=1))  # -> 88

        x1_2 = self.up2(x1_3)
        if x1_2.shape[2:] != x0_2.shape[2:]:
            x1_2 = F.interpolate(x1_2, size=x0_2.shape[2:], mode='bilinear', align_corners=True)
        x1_2 = self.skip_dec2_3(torch.cat([x1_2, x0_2], dim=1))  # -> 48

        x1_1 = self.up1(x1_2)
        if x1_1.shape[2:] != x0_1.shape[2:]:
            x1_1 = F.interpolate(x1_1, size=x0_1.shape[2:], mode='bilinear', align_corners=True)
        x1_1 = self.skip_dec1_2(torch.cat([x1_1, x0_1], dim=1))  # -> 32

        # Output heads
        segmentation = self.segmentation_head(x1_1)
        boundary = self.boundary_head(x1_1)
        presence = self.presence_head(x1_1)

        # Resize outputs to match input size
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
