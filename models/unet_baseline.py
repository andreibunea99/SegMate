# models/unet_baseline.py
"""
Standard U-Net Baseline with timm encoders.

Textbook symmetric U-Net: encoder (timm features_only) + symmetric decoder
with bilinear upsample + skip concat + double Conv3x3-BN-ReLU blocks.
Single segmentation head, no ASPP, no dense skips, no auxiliary heads.

Supports the same 3 backbones as SegMate:
- tf_efficientnetv2_m: out_indices=(1,2,3,4)
- mambaout_tiny: out_indices=(0,1,2,3), NHWC→NCHW permute
- fastvit_t12: out_indices=(0,1,2,3)

Usage:
    model = UNetBaseline(num_classes=9, in_channels=1, encoder_name="tf_efficientnetv2_m")
    out = model(x)  # returns tensor [B, C, H, W], NOT a tuple
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# Encoders that need NHWC → NCHW conversion
_NHWC_ENCODERS = {"mambaout_tiny", "mambaout_small", "mambaout_base"}

# Default out_indices per encoder family
_OUT_INDICES = {
    "tf_efficientnetv2_m": (1, 2, 3, 4),
}


class DecoderBlock(nn.Module):
    """Upsample + concat skip + double Conv3x3-BN-ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNetBaseline(nn.Module):
    def __init__(
        self,
        num_classes: int = 9,
        in_channels: int = 1,
        encoder_name: str = "tf_efficientnetv2_m",
        pretrained: bool = True,
        **kwargs,  # absorb deep_supervision etc. from eval
    ):
        super().__init__()
        self.is_nhwc = encoder_name in _NHWC_ENCODERS
        out_indices = _OUT_INDICES.get(encoder_name, (0, 1, 2, 3))

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels,
            out_indices=out_indices,
        )

        # feature_info contains ALL stages; select only the ones we requested
        all_info = self.encoder.feature_info
        enc_chs = [all_info[i]["num_chs"] for i in out_indices]
        assert len(enc_chs) == 4, (
            f"{encoder_name} with out_indices={out_indices} gave {len(enc_chs)} features; expected 4"
        )

        # Symmetric decoder: mirror encoder channels in reverse
        # dec[i] receives (dec[i-1]_ch + skip_ch) → outputs dec_ch
        # Stage 0 (deepest): enc_chs[3] + enc_chs[2] → enc_chs[2]
        # Stage 1:           enc_chs[2] + enc_chs[1] → enc_chs[1]
        # Stage 2:           enc_chs[1] + enc_chs[0] → enc_chs[0]
        self.dec3 = DecoderBlock(enc_chs[3] + enc_chs[2], enc_chs[2])
        self.dec2 = DecoderBlock(enc_chs[2] + enc_chs[1], enc_chs[1])
        self.dec1 = DecoderBlock(enc_chs[1] + enc_chs[0], enc_chs[0])

        # Final upsample to input resolution + head
        self.final_conv = nn.Sequential(
            nn.Conv2d(enc_chs[0], enc_chs[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(enc_chs[0]),
            nn.ReLU(inplace=True),
        )
        self.seg_head = nn.Conv2d(enc_chs[0], num_classes, 1)

    def forward(self, x, **kwargs):
        input_size = x.shape[2:]
        features = self.encoder(x)

        if self.is_nhwc:
            features = [f.permute(0, 3, 1, 2).contiguous() for f in features]

        f1, f2, f3, f4 = features  # shallow → deep

        x = self.dec3(f4, f3)
        x = self.dec2(x, f2)
        x = self.dec1(x, f1)

        # Upsample to input resolution (encoder may downsample more than 4 stages)
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

        x = self.final_conv(x)
        return self.seg_head(x)
