# models/segmate_fastvit_film.py
"""
SegMate with FastViT encoder + FiLM positional encoding.

Identical to SegMateFiLM (EfficientNetV2) but uses FastViT as the encoder.
FastViT has 4 stages (out_indices 0-3), vs EfficientNetV2's 5 stages (1-4).

Usage:
    from models.segmate_fastvit_film import SegMateFastViTFiLM
    model = SegMateFastViTFiLM(num_classes=9, in_channels=1,
                                encoder_name="fastvit_t12", deep_supervision=True)
    output = model(x, z_norm=z_norm)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# Reuse all shared building blocks from the EfficientNetV2 FiLM model
from models.segmate_v2_film import (
    ChannelAttention, SpatialAttention, CBAMBlock, SEBlock,
    ASPP, AlignmentBlock, DecoderBlock, FiLMLayer,
)


class SegMateFastViTFiLM(nn.Module):
    """SegMate FiLM with FastViT encoder (4-stage, NCHW output)."""

    def __init__(
        self,
        num_classes: int = 9,
        in_channels: int = 1,
        deep_supervision: bool = False,
        encoder_name: str = "fastvit_t12",
        pretrained: bool = True,
        film_hidden_dim: int = 128,
        film_num_layers: int = 3,
        film_dropout: float = 0.1,
        film_init_scale: float = 0.01,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder_name = encoder_name

        print(f"[SegMateFastViTFiLM] Initializing encoder: {encoder_name}")
        # FastViT has 4 stages → out_indices=(0,1,2,3)
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels,
            out_indices=(0, 1, 2, 3),
        )
        print(f"[SegMateFastViTFiLM] Encoder '{encoder_name}' loaded")

        feat_info = self.encoder.feature_info
        enc_chs = [feat_info[i]["num_chs"] for i in range(4)]
        print(f"[SegMateFastViTFiLM] Feature channels: {enc_chs}")
        ch2, ch3, ch4, ch5 = enc_chs

        self.align_e2 = AlignmentBlock(ch2, 40)
        self.align_e3 = AlignmentBlock(ch3, 64)
        self.align_e4 = AlignmentBlock(ch4, 128)
        self.align_e5 = AlignmentBlock(ch5, 176)

        self.aspp = ASPP(ch5, 256)

        self.film = FiLMLayer(
            channels=256,
            hidden_dim=film_hidden_dim,
            num_layers=film_num_layers,
            dropout_rate=film_dropout,
            init_scale=film_init_scale,
        )
        print(f"[SegMateFastViTFiLM] FiLM layer added (hidden={film_hidden_dim}, layers={film_num_layers})")

        # Decoder (identical widths to SegMateFiLM)
        self.up4 = nn.ConvTranspose2d(256, 160, kernel_size=2, stride=2)
        self.dec4 = DecoderBlock(160 + 176, 160, 'cbam')

        self.up3 = nn.ConvTranspose2d(160, 88, kernel_size=2, stride=2)
        self.dec3 = DecoderBlock(88 + 128, 88, 'cbam')

        self.up2 = nn.ConvTranspose2d(88, 48, kernel_size=2, stride=2)
        self.dec2 = DecoderBlock(48 + 64, 48, 'cbam')

        self.up1 = nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2)
        self.dec1 = DecoderBlock(24 + 40, 32, 'cbam')

        self.skip_dec3_4 = DecoderBlock(88 + 88, 88, 'se')
        self.skip_dec2_3 = DecoderBlock(48 + 48, 48, 'se')
        self.skip_dec1_2 = DecoderBlock(56, 32, 'se')
        self.skip_dec2_4 = DecoderBlock(160 + 48, 48, 'se')
        self.skip_dec1_3 = DecoderBlock(88 + 32, 32, 'se')
        self.skip_dec1_4 = DecoderBlock(160 + 32, 32, 'se')

        self.segmentation_head = nn.Conv2d(32, num_classes, 1)
        self.boundary_head = nn.Conv2d(32, num_classes, 1)
        self.presence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
            nn.Sigmoid(),
        )

        if deep_supervision:
            self.deep_head1 = nn.Conv2d(160, num_classes, 1)
            self.deep_head2 = nn.Conv2d(88, num_classes, 1)
            self.deep_head3 = nn.Conv2d(48, num_classes, 1)

    def forward(self, x, z_norm=None):
        input_size = x.shape[2:]

        f2, f3, f4, f5 = self.encoder(x)  # FastViT → NCHW already

        e2 = self.align_e2(f2)
        e3 = self.align_e3(f3)
        e4 = self.align_e4(f4)
        e5 = self.align_e5(f5)

        x = self.aspp(f5)
        x = self.film(x, z_norm)

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

        segmentation = self.segmentation_head(x1_1)
        boundary = self.boundary_head(x1_1)
        presence = self.presence_head(x1_1)

        if segmentation.shape[2:] != input_size:
            segmentation = F.interpolate(segmentation, size=input_size, mode='bilinear', align_corners=True)
            boundary = F.interpolate(boundary, size=input_size, mode='bilinear', align_corners=True)

        if self.deep_supervision:
            deep_outs = [
                F.interpolate(self.deep_head1(x0_4), size=input_size, mode='bilinear', align_corners=True),
                F.interpolate(self.deep_head2(x0_3), size=input_size, mode='bilinear', align_corners=True),
                F.interpolate(self.deep_head3(x0_2), size=input_size, mode='bilinear', align_corners=True),
            ]
            return segmentation, boundary, presence, deep_outs

        return segmentation, boundary, presence
