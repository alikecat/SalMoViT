import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.mobilevit import mobilevit_xxs
from torchvision.transforms import Compose, Normalize, Resize

logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = mobilevit_xxs(pretrained=pretrained)
        self.stem = backbone.stem
        self.stages = backbone.stages

    def forward(self, x):
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class Bottleneck(nn.Module):
    def __init__(self, channels, kernel_sizes=[3, 5, 7], expansion=6):
        super().__init__()
        expanded_channels = channels * expansion
        self.expand = nn.Sequential(
            nn.Conv2d(channels, expanded_channels, 1, bias=False),
            nn.GroupNorm(1, expanded_channels),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        expanded_channels,
                        expanded_channels,
                        kernel_size,
                        padding=kernel_size // 2,
                        groups=expanded_channels,
                        bias=False,
                    ),
                    nn.Conv2d(
                        expanded_channels,
                        expanded_channels,
                        1,
                        groups=expansion,
                        bias=False,
                    ),
                    nn.GroupNorm(1, expanded_channels),
                    nn.SiLU(),
                    nn.ChannelShuffle(expansion),
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels, channels, 1, bias=False),
            nn.GroupNorm(1, channels),
        )

    def forward(self, x):
        residual = x
        x = self.expand(x)
        for block in self.blocks:
            x = block(x)
        x = self.project(x)
        x = x + residual
        return x


class Upsample(nn.Module):
    def __init__(
        self, in_channels, out_channels, skip_channels=0, kernel_sizes=[7, 5, 3]
    ):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.GroupNorm(1, in_channels),
            nn.SiLU(),
        )
        self.channel_adjust = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size,
                        padding=kernel_size // 2,
                        groups=out_channels,
                        bias=False,
                    ),
                    nn.Conv2d(out_channels, out_channels, 1, bias=False),
                    nn.GroupNorm(1, out_channels),
                    nn.SiLU(),
                )
                for kernel_size in kernel_sizes
            ]
        )
        if skip_channels > 0:
            self.skip_conn = nn.Sequential(
                nn.Conv2d(skip_channels, skip_channels, 1, bias=False),
            )
            self.attention = nn.Sequential(
                nn.Conv2d(skip_channels, 1, 1, bias=False), nn.Sigmoid()
            )

    def forward(self, x, skip_x=None):
        x = self.upsample(x)
        if skip_x is not None:
            diff_y = skip_x.size()[2] - x.size()[2]
            diff_x = skip_x.size()[3] - x.size()[3]
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
            skip_x = self.skip_conn(skip_x) * self.attention(skip_x)
            x = torch.cat([x, skip_x], dim=1)
        x = self.channel_adjust(x)
        residual = x
        for block in self.blocks:
            x = block(x)
        x = x + residual
        return x


class CenterBias(nn.Module):
    def __init__(
        self,
        source=None,
        fusion="mul",
        strength=10.0,
        learn_strength=False,
        sigma=40.0,
        learn_sigma=False,
    ):
        super().__init__()
        self.source = source
        self.fusion = fusion.lower()
        self.strength = (
            nn.Parameter(torch.tensor(strength))
            if learn_strength
            else torch.tensor(strength)
        )
        self.sigma_base = torch.tensor(sigma)
        self.sigma_offset = (
            nn.Parameter(torch.tensor(0.0)) if learn_sigma else torch.tensor(0.0)
        )
        if isinstance(source, (torch.Tensor, np.ndarray)):
            if isinstance(source, np.ndarray):
                source = torch.from_numpy(source).float()
            self.register_buffer("predefined_centerbias", source)

    def forward(self, x):
        if self.source is None or self.fusion is None:
            return x
        batch_size, _, height, width = x.shape
        if hasattr(self, "predefined_centerbias"):
            centerbias = nn.functional.interpolate(
                self.predefined_centerbias.unsqueeze(0).unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).to(x.device)
        elif isinstance(self.source, str) and self.source.lower() == "gaussian":
            centerbias = self._generate_gaussian_centerbias(
                batch_size, height, width, x.device
            )
        else:
            logger.warning(
                f"Invalid centerbias source: {self.source}, skipping centerbias application"
            )
            return x
        centerbias = (centerbias - centerbias.min()) / (
            centerbias.max() - centerbias.min() + torch.finfo(torch.float32).eps
        )
        if self.fusion == "add":
            x = x + self.strength * centerbias
        elif self.fusion == "mul":
            x = x * (1 + self.strength * centerbias)
        return x

    def _generate_gaussian_centerbias(self, batch_size, height, width, device):
        x = torch.linspace(-1, 1, width, device=device)
        y = torch.linspace(-1, 1, height, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        d = torch.sqrt(xx**2 + yy**2)
        sigma = (
            self.sigma_base
            * F.softplus(self.sigma_offset)
            / torch.log(torch.tensor(2.0))
        )
        g = torch.exp(-(d**2) / (2 * sigma**2))
        g = g.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        return g


class GaussSmoothing(nn.Module):
    def __init__(self, sigma=20.0, kernel_size=41, learn_sigma=True):
        super().__init__()
        self.sigma_base = torch.tensor(sigma)
        self.sigma_offset = (
            nn.Parameter(torch.tensor(0.0)) if learn_sigma else torch.tensor(0.0)
        )
        self.kernel_size = kernel_size

    def forward(self, x):
        channel = x.size(1)
        sigma = (
            self.sigma_base
            * F.softplus(self.sigma_offset)
            / torch.log(torch.tensor(2.0))
        )
        kernel_1d = torch.linspace(
            -(self.kernel_size // 2),
            self.kernel_size // 2,
            self.kernel_size,
            device=x.device,
        )
        kernel_1d = torch.exp(-(kernel_1d**2) / sigma**2 / 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_h = kernel_1d.view(1, 1, 1, -1).repeat(channel, 1, 1, 1)
        kernel_v = kernel_1d.view(1, 1, -1, 1).repeat(channel, 1, 1, 1)
        x = F.conv2d(x, kernel_h, padding=(0, self.kernel_size // 2), groups=channel)
        x = F.conv2d(x, kernel_v, padding=(self.kernel_size // 2, 0), groups=channel)
        return x


class SalMoViT(nn.Module):
    def __init__(self, preprocess=True):
        super().__init__()
        self.preprocess = (
            Compose(
                [
                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    Resize(384, max_size=512),
                ]
            )
            if preprocess
            else nn.Identity()
        )
        self.encoder = Encoder()
        self.bottleneck = Bottleneck(80)
        self.up0 = Upsample(80, 64, 64)
        self.up1 = Upsample(64, 48, 48)
        self.up2 = Upsample(48, 24, 24)
        self.up3 = Upsample(24, 16, 16)
        self.up4 = Upsample(16, 8)
        self.out_conv = nn.Sequential(
            nn.Conv2d(8, 4, 3, padding=1, bias=False),
            nn.GroupNorm(1, 4),
            nn.SiLU(),
            nn.Conv2d(4, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.centerbias = CenterBias(
            "gaussian", "mul", learn_strength=True, learn_sigma=True
        )
        self.smoothing = GaussSmoothing(20.0, 41, True)

    def forward(self, x):
        target_size = x.size()[2:]
        x = self.preprocess(x)
        f2, f4, f8, f16, f32 = self.encoder(x)
        x = self.bottleneck(f32)
        x = self.up0(x, f16)
        x = self.up1(x, f8)
        x = self.up2(x, f4)
        x = self.up3(x, f2)
        x = self.up4(x)
        x = self.out_conv(x)
        x = self.centerbias(x)
        x = self.smoothing(x)
        if x.size()[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=True)
        x_sum = torch.sum(x.flatten(1), dim=1, keepdim=True).view(-1, 1, 1, 1)
        x = x / (x_sum + torch.finfo(torch.float32).eps)
        x = torch.log(x + torch.finfo(torch.float32).eps)
        return x
