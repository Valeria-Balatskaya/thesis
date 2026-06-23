import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, msg_len: int = 30, channels: int = 64):
        super().__init__()
        self.msg_len = msg_len
        self.features = nn.Sequential(
            ConvBNRelu(3, channels),
            ConvBNRelu(channels, channels),
            ConvBNRelu(channels, channels),
            ConvBNRelu(channels + 3 + msg_len, channels),
            ConvBNRelu(channels, channels),
            nn.Conv2d(channels, 3, kernel_size=1),
        )

    def forward(self, image: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        b, _, h, w = image.shape
        msg_map = message.view(b, self.msg_len, 1, 1).expand(-1, -1, h, w)
        x1 = self.features[0](image)
        x2 = self.features[1](x1)
        x3 = self.features[2](x2)
        x = torch.cat([x3, image, msg_map], dim=1)
        x = self.features[3](x)
        x = self.features[4](x)
        residual = torch.tanh(self.features[5](x))
        return torch.clamp(image + 0.05 * residual, 0.0, 1.0)


class Decoder(nn.Module):
    def __init__(self, msg_len: int = 30, channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNRelu(3, channels),
            ConvBNRelu(channels, channels),
            ConvBNRelu(channels, channels),
            ConvBNRelu(channels, channels),
        )
        self.head = nn.Linear(channels, msg_len)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.net(image)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


class Adversary(nn.Module):
    def __init__(self, channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels * 2, channels * 4, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * 4, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class IdentityNoise(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class GaussianNoise(nn.Module):
    def __init__(self, sigma: float = 0.03):
        super().__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x + torch.randn_like(x) * self.sigma, 0.0, 1.0)


class GaussianBlurNoise(nn.Module):
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)


class ResizeNoise(nn.Module):
    def __init__(self, min_scale: float = 0.6, max_scale: float = 0.9):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        scale = random.uniform(self.min_scale, self.max_scale)
        nh, nw = max(8, int(h * scale)), max(8, int(w * scale))
        y = F.interpolate(x, size=(nh, nw), mode='bilinear', align_corners=False)
        return F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)


class CropNoise(nn.Module):
    def __init__(self, keep_ratio: Tuple[float, float] = (0.8, 0.95)):
        super().__init__()
        self.keep_ratio = keep_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ratio = random.uniform(*self.keep_ratio)
        ch, cw = max(8, int(h * ratio)), max(8, int(w * ratio))
        top = random.randint(0, h - ch)
        left = random.randint(0, w - cw)
        crop = x[:, :, top:top + ch, left:left + cw]
        return F.interpolate(crop, size=(h, w), mode='bilinear', align_corners=False)


class ApproxJpegNoise(nn.Module):
    def __init__(self, levels: int = 64):
        super().__init__()
        self.levels = levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.round(x * (self.levels - 1)) / (self.levels - 1)
        return x + (y - x).detach()


class RandomNoiseLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            IdentityNoise(),
            GaussianNoise(0.02),
            GaussianBlurNoise(3),
            ResizeNoise(0.65, 0.9),
            CropNoise((0.8, 0.95)),
            ApproxJpegNoise(64),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer = random.choice(list(self.layers))
        return layer(x)


class HiDDeN(nn.Module):
    def __init__(self, msg_len: int = 30):
        super().__init__()
        self.msg_len = msg_len
        self.encoder = Encoder(msg_len)
        self.decoder = Decoder(msg_len)
        self.adversary = Adversary()
        self.noise = RandomNoiseLayer()

    def forward(self, image: torch.Tensor, message: torch.Tensor):
        encoded = self.encoder(image, message)
        attacked = self.noise(encoded)
        decoded_logits = self.decoder(attacked)
        return encoded, attacked, decoded_logits
