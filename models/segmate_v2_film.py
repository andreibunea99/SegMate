# models/segmate_v2_film.py
"""
SegMate with EfficientNetV2 encoder + FiLM positional encoding
================================================================
Same architecture as SegMate V2, but with Feature-wise Linear Modulation (FiLM)
to condition bottleneck features on normalized slice position (z_norm).

Key idea: y = gamma(z_norm) * x + beta(z_norm) applied after ASPP bottleneck.

Usage:
    from models.segmate_v2_film import SegMateFiLM
    model = SegMateFiLM(num_classes=9, in_channels=1, deep_supervision=True)
    output = model(x, z_norm=z_norm)  # z_norm: [B] tensor in [0, 1]
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


# --- FiLM Components ---
class PositionEncoder(nn.Module):
    """Maps z_norm scalar to gamma/beta vectors (256 dims each).

    Uses an MLP to transform normalized slice position into FiLM parameters.
    Initialized for near-identity transform: gamma=1, beta=0.
    """
    def __init__(self, out_channels=256, hidden_dim=128, num_layers=3,
                 dropout_rate=0.1, init_scale=0.01):
        super().__init__()
        self.out_channels = out_channels

        # Input dropout for robustness
        self.input_dropout = nn.Dropout(dropout_rate)

        # Build MLP layers
        layers = []
        in_dim = 1  # z_norm is a scalar
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Separate heads for gamma and beta
        self.gamma_head = nn.Linear(hidden_dim, out_channels)
        self.beta_head = nn.Linear(hidden_dim, out_channels)

        # Initialize for near-identity transform
        self._init_weights(init_scale)

    def _init_weights(self, init_scale):
        """Initialize weights for near-identity FiLM transform."""
        # Small weights for gradual learning
        nn.init.normal_(self.gamma_head.weight, mean=0.0, std=init_scale)
        nn.init.normal_(self.beta_head.weight, mean=0.0, std=init_scale)

        # Gamma bias = 1 (identity scaling)
        nn.init.ones_(self.gamma_head.bias)

        # Beta bias = 0 (no shift)
        nn.init.zeros_(self.beta_head.bias)

    def forward(self, z_norm):
        """
        Args:
            z_norm: Tensor of shape [B] or [B, 1] with values in [0, 1]

        Returns:
            gamma: [B, out_channels] scaling factors
            beta: [B, out_channels] shift factors
        """
        # Ensure shape is [B, 1]
        if z_norm.dim() == 1:
            z_norm = z_norm.unsqueeze(1)

        # Apply input dropout for robustness
        z = self.input_dropout(z_norm)

        # Forward through MLP
        z = self.mlp(z)

        # Generate gamma and beta
        gamma = self.gamma_head(z)  # [B, out_channels]
        beta = self.beta_head(z)    # [B, out_channels]

        return gamma, beta


class FiLMLayer(nn.Module):
    """Applies Feature-wise Linear Modulation: y = gamma * x + beta

    Conditions feature maps on slice position for position-aware representations.
    """
    def __init__(self, channels=256, hidden_dim=128, num_layers=3,
                 dropout_rate=0.1, init_scale=0.01):
        super().__init__()
        self.position_encoder = PositionEncoder(
            out_channels=channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            init_scale=init_scale
        )

    def forward(self, x, z_norm=None):
        """
        Args:
            x: Feature tensor of shape [B, C, H, W]
            z_norm: Optional position tensor [B] or [B, 1] with values in [0, 1]

        Returns:
            Modulated features of shape [B, C, H, W]
        """
        if z_norm is None:
            return x  # Backward compatible - no modulation

        gamma, beta = self.position_encoder(z_norm)

        # Reshape for broadcasting: [B, C] -> [B, C, 1, 1]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        # Apply FiLM: y = gamma * x + beta
        return gamma * x + beta


class SegMateFiLM(nn.Module):
    """
    SegMate with EfficientNetV2 encoder and FiLM positional encoding.

    Uses timm for the encoder, keeping decoder identical to SegMate V2.
    Adds FiLM layer after ASPP to condition on slice position.
    """
    def __init__(
        self,
        num_classes=9,
        in_channels=1,
        deep_supervision=False,
        encoder_name: str = "tf_efficientnetv2_s",
        pretrained: bool = True,
        # FiLM parameters
        film_hidden_dim: int = 128,
        film_num_layers: int = 3,
        film_dropout: float = 0.1,
        film_init_scale: float = 0.01,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder_name = encoder_name

        # ---- EfficientNetV2 encoder from timm ----
        print(f"[SegMateFiLM] Initializing with encoder: {encoder_name}")
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels,
            out_indices=(1, 2, 3, 4),  # Skip first stage, use stages 1-4
        )
        print(f"[SegMateFiLM] Encoder '{encoder_name}' loaded successfully")

        # Get encoder feature channels
        feat_info = self.encoder.feature_info
        enc_chs = [feat_info[i]["num_chs"] for i in range(len(feat_info))]
        enc_chs = enc_chs[-4:] if len(enc_chs) > 4 else enc_chs
        print(f"[SegMateFiLM] Encoder feature channels: {enc_chs}")

        if len(enc_chs) != 4:
            raise ValueError(f"Expected 4 feature maps, got {len(enc_chs)}")

        ch2, ch3, ch4, ch5 = enc_chs

        # ---- Align encoder stages to decoder-expected channels ----
        self.align_e2 = AlignmentBlock(ch2, 40)
        self.align_e3 = AlignmentBlock(ch3, 64)
        self.align_e4 = AlignmentBlock(ch4, 128)
        self.align_e5 = AlignmentBlock(ch5, 176)

        # ASPP on deepest feature
        self.aspp = ASPP(ch5, 256)

        # ---- FiLM layer after ASPP ----
        self.film = FiLMLayer(
            channels=256,
            hidden_dim=film_hidden_dim,
            num_layers=film_num_layers,
            dropout_rate=film_dropout,
            init_scale=film_init_scale,
        )
        print(f"[SegMateFiLM] FiLM layer added (hidden={film_hidden_dim}, layers={film_num_layers})")

        # ---- Decoder path (identical widths as SegMate V2) ----
        self.up4 = nn.ConvTranspose2d(256, 160, kernel_size=2, stride=2)
        self.dec4 = DecoderBlock(160 + 176, 160, 'cbam')

        self.up3 = nn.ConvTranspose2d(160, 88, kernel_size=2, stride=2)
        self.dec3 = DecoderBlock(88 + 128, 88, 'cbam')

        self.up2 = nn.ConvTranspose2d(88, 48, kernel_size=2, stride=2)
        self.dec2 = DecoderBlock(48 + 64, 48, 'cbam')

        self.up1 = nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2)
        self.dec1 = DecoderBlock(24 + 40, 32, 'cbam')

        # ---- Dense skip connections ----
        self.skip_dec3_4 = DecoderBlock(88 + 88, 88, 'se')
        self.skip_dec2_3 = DecoderBlock(48 + 48, 48, 'se')
        self.skip_dec1_2 = DecoderBlock(56, 32, 'se')
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

        # ---- Deep supervision heads ----
        if deep_supervision:
            self.deep_head1 = nn.Conv2d(160, num_classes, 1)
            self.deep_head2 = nn.Conv2d(88, num_classes, 1)
            self.deep_head3 = nn.Conv2d(48, num_classes, 1)

    def forward(self, x, z_norm=None):
        """
        Forward pass with optional position conditioning.

        Args:
            x: Input tensor [B, C, H, W]
            z_norm: Optional normalized slice position [B] or [B, 1], values in [0, 1]

        Returns:
            segmentation, boundary, presence outputs (and deep_outs if deep_supervision)
        """
        input_size = x.shape[2:]

        # EfficientNetV2 features (4 feature maps from stages 1-4)
        features = self.encoder(x)
        f2, f3, f4, f5 = features

        # Align to expected channels
        e2 = self.align_e2(f2)  # 40
        e3 = self.align_e3(f3)  # 64
        e4 = self.align_e4(f4)  # 128
        e5 = self.align_e5(f5)  # 176

        # ASPP on deepest
        x = self.aspp(f5)  # -> 256

        # FiLM modulation (NEW)
        x = self.film(x, z_norm)

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
