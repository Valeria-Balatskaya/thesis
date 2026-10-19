# src/hidden_v2.py
# HiDDeN v2 — retrained for the deployed (residual) pipeline and for the
# distortions stolen product images actually go through.
#
# Differences from src/hidden_watermark.py (v1, kept for reproducibility):
#   1. Residual-style training. The encoder sees a 128px copy of the image and
#      outputs only a watermark residual. The residual is upsampled and added
#      to the untouched full-resolution image — exactly what happens at
#      deployment — and the decoder is trained on that.
#   2. Fixed quality budget. The residual is rescaled so every watermarked
#      image has the same target PSNR. There is no image loss to balance, so
#      the network cannot "win" by hiding nothing (the v1 Phase 4C failure).
#   3. Differentiable distortions modelled on the e-commerce attack suite:
#      real JPEG quantisation, perspective, colour shifts, crop, rescale,
#      screenshot and print-photograph pipelines, and combinations.
#   4. Deeper decoder with downsampling; 63-bit payload (BCH codeword,
#      see src/payload.py).
#
# References:
#   Zhu et al. (2018). HiDDeN: Hiding Data with Deep Networks. ECCV.
#   Tancik et al. (2020). StegaStamp: Invisible Hyperlinks in Physical
#   Photographs. CVPR.  (perspective / colour / print-capture distortions)
#   Shin & Song (2017). JPEG-resistant adversarial images. (differentiable JPEG)

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.hidden_watermark import ConvBNRelu


class EncoderV2(nn.Module):
    """Image (low-res) + message -> unscaled watermark residual."""

    PROJ_CH, PROJ_SIZE = 3, 64

    def __init__(self, msg_len: int, channels: int = 64):
        super().__init__()
        self.msg_len = msg_len
        self.pre = nn.Sequential(
            ConvBNRelu(3, channels),
            ConvBNRelu(channels, channels),
            ConvBNRelu(channels, channels),
            ConvBNRelu(channels, channels),
        )
        # Each bit also gets its own learnable low-resolution spatial carrier
        # (a linear message projection, as in StegaStamp). The carrier is fed
        # to the conv layers AND added straight to the output, so from the
        # first step there is an additive, per-bit signal for the decoder to
        # lock onto. With only HiDDeN's spatially constant message map the
        # message reaches the residual through two random ReLU layers and
        # training stalls around 60% bit accuracy.
        self.msg_proj = nn.Linear(msg_len, self.PROJ_CH * self.PROJ_SIZE ** 2)
        self.post = nn.Sequential(
            ConvBNRelu(channels + 3 + msg_len + self.PROJ_CH, channels),
            ConvBNRelu(channels, channels),
            nn.Conv2d(channels, 3, kernel_size=1),
        )

    def forward(self, image: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        b, _, h, w = image.shape
        signed = message * 2 - 1
        msg_map = signed.view(b, self.msg_len, 1, 1).expand(-1, -1, h, w)
        pattern = self.msg_proj(signed).view(b, self.PROJ_CH, self.PROJ_SIZE, self.PROJ_SIZE)
        pattern = F.interpolate(pattern, size=(h, w), mode="bilinear", align_corners=False)
        feats = self.pre(image)
        return self.post(torch.cat([feats, image, msg_map, pattern], dim=1)) + pattern


class DecoderV2(nn.Module):
    """
    Two read-outs of the same features, concatenated:
      - global average pooling (as in HiDDeN): position-independent, so it
        tolerates crops and shifts;
      - a flattened 8x8 grid (as in StegaStamp): keeps *where* a pattern is,
        which is what lets training get off the ground quickly.
    """

    GRID = 8
    FINE_CH, FINE_SIZE = 4, 64

    def __init__(self, msg_len: int, channels: int = 64):
        super().__init__()
        c = channels
        self.net = nn.Sequential(
            ConvBNRelu(3, c),
            ConvBNRelu(c, c),
            nn.MaxPool2d(2),
            ConvBNRelu(c, 2 * c),
            ConvBNRelu(2 * c, 2 * c),
            nn.MaxPool2d(2),
            ConvBNRelu(2 * c, 4 * c),
            ConvBNRelu(4 * c, 4 * c),
            nn.MaxPool2d(2),
            ConvBNRelu(4 * c, 4 * c),
        )
        self.spatial = nn.Sequential(nn.MaxPool2d(2), ConvBNRelu(4 * c, c // 2))
        # Shallow full-resolution branch: suppress image content, keep the
        # watermark, then read it with a linear layer (a learned matched filter).
        self.fine = nn.Sequential(ConvBNRelu(3, c), ConvBNRelu(c, c),
                                  nn.Conv2d(c, self.FINE_CH, 3, padding=1))
        self.fine_head = nn.Linear(self.FINE_CH * self.FINE_SIZE ** 2, msg_len)
        self.head = nn.Sequential(
            nn.Linear(4 * c + (c // 2) * self.GRID ** 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, msg_len),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.net(image * 2 - 1)
        pooled = x.mean(dim=(2, 3))
        grid = self.spatial(x).flatten(1)
        fine = F.adaptive_avg_pool2d(self.fine(image * 2 - 1), self.FINE_SIZE).flatten(1)
        return self.head(torch.cat([pooled, grid], dim=1)) + self.fine_head(fine)


# ─── Differentiable JPEG ──────────────────────────────────────────

_Q_LUMA = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
]
_Q_CHROMA = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
] + [99] * 32


def _quant_table(base: list[int], quality: float, device) -> torch.Tensor:
    """libjpeg quality scaling of the standard quantisation tables."""
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    table = torch.tensor(base, dtype=torch.float32, device=device).view(8, 8)
    return torch.clamp(torch.floor((table * scale + 50) / 100), 1, 255)


def _dct_matrix(device) -> torch.Tensor:
    k = torch.arange(8, dtype=torch.float32, device=device).view(8, 1)
    n = torch.arange(8, dtype=torch.float32, device=device).view(1, 8)
    d = torch.cos((2 * n + 1) * k * math.pi / 16) * math.sqrt(2 / 8)
    d[0] = d[0] / math.sqrt(2)
    return d


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    """Round in the forward pass, identity gradient in the backward pass."""
    return x + (torch.round(x) - x).detach()


def _jpeg_channel(ch: torch.Tensor, table: torch.Tensor, dct: torch.Tensor) -> torch.Tensor:
    b, _, h, w = ch.shape
    blocks = (ch - 128).unfold(2, 8, 8).unfold(3, 8, 8)          # B,1,h/8,w/8,8,8
    coeffs = dct @ blocks @ dct.t()
    coeffs = _round_ste(coeffs / table) * table
    blocks = dct.t() @ coeffs @ dct
    return blocks.permute(0, 1, 2, 4, 3, 5).reshape(b, 1, h, w) + 128


def diff_jpeg(x: torch.Tensor, quality: float) -> torch.Tensor:
    """JPEG round-trip (YCbCr, 4:2:0 chroma, 8x8 DCT quantisation) on [0,1] RGB."""
    _, _, h, w = x.shape
    x = F.pad(x, (0, (-w) % 16, 0, (-h) % 16), mode="replicate") * 255
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128

    dct = _dct_matrix(x.device)
    y = _jpeg_channel(y, _quant_table(_Q_LUMA, quality, x.device), dct)
    q_c = _quant_table(_Q_CHROMA, quality, x.device)
    cb = _jpeg_channel(F.avg_pool2d(cb, 2), q_c, dct)
    cr = _jpeg_channel(F.avg_pool2d(cr, 2), q_c, dct)
    cb = F.interpolate(cb, scale_factor=2, mode="bilinear", align_corners=False) - 128
    cr = F.interpolate(cr, scale_factor=2, mode="bilinear", align_corners=False) - 128

    rgb = torch.cat([y + 1.402 * cr,
                     y - 0.344136 * cb - 0.714136 * cr,
                     y + 1.772 * cb], dim=1) / 255
    return torch.clamp(rgb[:, :, :h, :w], 0, 1)


# ─── Distortion layer ─────────────────────────────────────────────

def _u(lo: float, hi: float) -> float:
    return random.uniform(lo, hi)


class Distortions(nn.Module):
    """
    One randomly chosen distortion per batch. `s` in [0,1] scales how severe
    the distortions are (curriculum: 0 at the start of training, 1 at the end).
    Output may have a different spatial size than the input (crop / rescale),
    as a re-uploaded image would.
    """

    NAMES = ["identity", "jpeg", "noise", "blur", "resize", "crop", "perspective",
             "colour", "screenshot", "print_photo", "combo"]
    WEIGHTS = [1, 2, 1, 1, 1, 1.5, 1.5, 1.5, 2, 1.5, 1.5]

    def forward(self, x: torch.Tensor, s: float = 1.0, name: str | None = None):
        if name is None:
            name = random.choices(self.NAMES, weights=self.WEIGHTS)[0] if s > 0 else "identity"
        return getattr(self, name)(x, s), name

    # --- single distortions ---

    def identity(self, x, s):
        return x

    def jpeg(self, x, s):
        return diff_jpeg(x, _u(100 - 70 * s, 95))

    def noise(self, x, s):
        return torch.clamp(x + torch.randn_like(x) * _u(0, 0.06 * s), 0, 1)

    def blur(self, x, s):
        sigma = _u(0.1, 0.1 + 1.4 * s)
        k = torch.arange(-3, 4, dtype=torch.float32, device=x.device)
        k = torch.exp(-k ** 2 / (2 * sigma ** 2))
        k = (k / k.sum()).view(1, 1, 1, 7).repeat(3, 1, 1, 1)
        x = F.conv2d(F.pad(x, (3, 3, 0, 0), mode="replicate"), k, groups=3)
        return F.conv2d(F.pad(x, (0, 0, 3, 3), mode="replicate"),
                        k.transpose(2, 3), groups=3)

    def resize(self, x, s):
        _, _, h, w = x.shape
        scale = _u(1 - 0.6 * s, 1)
        size = (max(32, int(h * scale)), max(32, int(w * scale)))
        mode = random.choice(["bilinear", "bicubic"])
        y = F.interpolate(x, size=size, mode=mode, align_corners=False,
                          antialias=random.random() < 0.5)
        if random.random() < 0.5:   # re-uploaded at the original resolution
            y = F.interpolate(y, size=(h, w), mode="bicubic", align_corners=False)
        return torch.clamp(y, 0, 1)

    def crop(self, x, s):
        _, _, h, w = x.shape
        ch = max(32, int(h * _u(1 - 0.4 * s, 1)))
        cw = max(32, int(w * _u(1 - 0.4 * s, 1)))
        top, left = random.randint(0, h - ch), random.randint(0, w - cw)
        if random.random() < 0.5:   # region cut out (thumbnail, snip)
            return x[:, :, top:top + ch, left:left + cw]
        mask = torch.zeros_like(x)  # borders blanked, size kept
        mask[:, :, top:top + ch, left:left + cw] = 1
        return x * mask

    def perspective(self, x, s, max_shift: float = 0.1):
        b, _, h, w = x.shape
        src = torch.tensor([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]).expand(b, 4, 2)
        dst = src + (torch.rand(b, 4, 2) * 2 - 1) * 2 * max_shift * s
        # Homography H with H @ [x, y, 1] ~ [u, v, 1]  (output grid -> sample coords)
        rows, rhs = [], []
        for i in range(4):
            sx, sy, u, v = src[:, i, 0], src[:, i, 1], dst[:, i, 0], dst[:, i, 1]
            zero, one = torch.zeros(b), torch.ones(b)
            rows.append(torch.stack([sx, sy, one, zero, zero, zero, -u * sx, -u * sy], 1))
            rows.append(torch.stack([zero, zero, zero, sx, sy, one, -v * sx, -v * sy], 1))
            rhs += [u, v]
        coeffs = torch.linalg.solve(torch.stack(rows, 1), torch.stack(rhs, 1))
        hom = torch.cat([coeffs, torch.ones(b, 1)], 1).view(b, 3, 3).to(x.device)

        ys = torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=x.device)
        xs = torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=x.device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        pts = torch.stack([gx, gy, torch.ones_like(gx)], -1).view(1, h * w, 3)
        warped = pts @ hom.transpose(1, 2)
        grid = (warped[..., :2] / warped[..., 2:3]).view(b, h, w, 2)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border",
                             align_corners=False)

    def colour(self, x, s):
        b = x.size(0)

        def rand(spread, *shape):
            return 1 + (torch.rand(b, *shape, device=x.device) * 2 - 1) * spread * s

        x = x * rand(0.08, 3, 1, 1)                               # white balance
        gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
        x = gray + (x - gray) * rand(0.4, 1, 1, 1)                # saturation
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * rand(0.3, 1, 1, 1) + mean                # contrast
        x = x * rand(0.3, 1, 1, 1)                                # brightness
        x = torch.clamp(x, 1e-4, 1) ** rand(0.25, 1, 1, 1)        # gamma
        if random.random() < 0.3:                                 # vignette
            _, _, h, w = x.shape
            ys = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1)
            xs = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w)
            x = x * (1 - _u(0, 0.4 * s) * (xs ** 2 + ys ** 2) / 2)
        return torch.clamp(x, 0, 1)

    # --- pipelines ---

    def screenshot(self, x, s):
        """Display scaling -> capture at another size -> snip -> colour -> JPEG."""
        _, _, h, w = x.shape
        up = _u(1.25, 2.0)
        x = F.interpolate(x, size=(int(h * up), int(w * up)), mode="bicubic",
                          align_corners=False)
        down = _u(1 - 0.3 * s, 1)
        x = F.interpolate(x, size=(max(32, int(h * down)), max(32, int(w * down))),
                          mode="bilinear", align_corners=False, antialias=True)
        x = torch.clamp(x, 0, 1)
        _, _, h, w = x.shape
        m = [int(d * _u(0, 0.1 * s)) for d in (h, h, w, w)]
        x = x[:, :, m[0]:h - m[1], m[2]:w - m[3]]
        x = self.colour(x, 0.3 * s)
        x = self.noise(x, 0.25 * s)
        return diff_jpeg(x, _u(95 - 35 * s, 95))

    def print_photo(self, x, s):
        x = self.perspective(x, s, max_shift=0.06)
        x = self.colour(x, s)
        x = self.blur(x, 0.6 * s)
        return diff_jpeg(x, _u(95 - 35 * s, 95))

    def combo(self, x, s):
        for name in random.sample(["resize", "crop", "colour", "noise", "blur"], 2):
            x = getattr(self, name)(x, s)
        return diff_jpeg(x, _u(100 - 60 * s, 95))


# ─── Full model ───────────────────────────────────────────────────

def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(3 * sigma))
    k = torch.arange(-radius, radius + 1, dtype=torch.float32, device=x.device)
    k = torch.exp(-k ** 2 / (2 * sigma ** 2))
    k = (k / k.sum()).view(1, 1, 1, -1).repeat(x.size(1), 1, 1, 1)
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="replicate"), k, groups=x.size(1))
    return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="replicate"),
                    k.transpose(2, 3), groups=x.size(1))


def _luma(x: torch.Tensor) -> torch.Tensor:
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def texture_mask(image: torch.Tensor, floor: float, tau: float = 0.03) -> torch.Tensor:
    """
    Perceptual mask in [floor, 1]: 1 where the image is textured, `floor` where
    it is flat (white backgrounds, plain surfaces), where the eye notices a
    watermark most. Based on local luminance standard deviation.
    """
    unit = max(image.shape[-2:]) / 256          # blur radii scale with image size
    y = _luma(image)
    mean = _gaussian_blur(y, unit)
    std = (_gaussian_blur(y * y, unit) - mean * mean).clamp(min=0).sqrt()
    return _gaussian_blur(floor + (1 - floor) * (std / tau).clamp(0, 1), 2 * unit)


class HiDDeNv2(nn.Module):
    def __init__(self, msg_len: int = 63, image_size: int = 128,
                 target_psnr: float = 40.0, chroma_weight: float = 1.0,
                 mask_floor: float = 1.0):
        """
        chroma_weight: how much more a colour change counts against the quality
            budget than a brightness change. 1.0 = plain RGB budget. Trained at
            1.0 the network hides the message in broad colour patches, which are
            visible as a green/purple tint on white backgrounds.
        mask_floor: watermark amplitude on flat image areas relative to textured
            ones (1.0 = no masking).
        """
        super().__init__()
        self.msg_len = msg_len
        self.image_size = image_size
        self.target_psnr = target_psnr
        self.chroma_weight = chroma_weight
        self.mask_floor = mask_floor
        self.encoder = EncoderV2(msg_len)
        self.decoder = DecoderV2(msg_len)
        self.distort = Distortions()

    def _to_model_size(self, x: torch.Tensor) -> torch.Tensor:
        size = (self.image_size, self.image_size)
        if x.shape[-2:] == size:
            return x
        return F.interpolate(x, size=size, mode="bilinear",
                             align_corners=False, antialias=True)

    def watermark(self, image: torch.Tensor, message: torch.Tensor,
                  strength: float = 1.0, psnr: float | None = None,
                  mask_floor: float | None = None) -> torch.Tensor:
        """
        Full-resolution image [B,3,H,W] in [0,1] -> watermarked image, same size.
        The residual is zero-mean per channel (no global colour shift) and
        scaled to the quality budget: target_psnr at strength=1, with colour
        changes counted chroma_weight times. `psnr` overrides the target
        (training curriculum); `mask_floor` overrides the model's masking.
        """
        residual = self.encoder(self._to_model_size(image), message)
        residual = F.interpolate(residual, size=image.shape[-2:], mode="bicubic",
                                 align_corners=False)
        residual = residual - residual.mean(dim=(2, 3), keepdim=True)
        # Budget = plain RGB energy, plus an extra charge on the colour part of
        # the residual (chroma_weight = 1 is exactly the plain RGB budget).
        energy = residual.pow(2).mean(dim=(1, 2, 3), keepdim=True)
        if self.chroma_weight != 1.0:
            chroma = residual - _luma(residual)
            energy = energy + (self.chroma_weight - 1) * chroma.pow(2).mean(dim=(1, 2, 3), keepdim=True)
        target_rms = 10 ** (-(psnr or self.target_psnr) / 20)
        residual = residual * (target_rms / (energy.sqrt() + 1e-8)) * strength
        floor = self.mask_floor if mask_floor is None else mask_floor
        if floor < 1.0:
            residual = residual * texture_mask(image, floor)
        return torch.clamp(image + residual, 0, 1)

    def decode(self, image: torch.Tensor) -> torch.Tensor:
        """Any-size image -> message logits."""
        return self.decoder(self._to_model_size(image))

    def forward(self, image: torch.Tensor, message: torch.Tensor,
                severity: float = 1.0, distortion: str | None = None,
                psnr: float | None = None):
        watermarked = self.watermark(image, message, psnr=psnr)
        # Published files are 8-bit
        published = _round_ste(watermarked * 255) / 255
        attacked, name = self.distort(published, severity, distortion)
        return watermarked, self.decode(attacked), name
