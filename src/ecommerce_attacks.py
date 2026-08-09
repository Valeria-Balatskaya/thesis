# src/ecommerce_attacks.py
# E-commerce-specific attacks. These simulate what actually happens to product
# images when they're stolen and reposted — not the textbook attacks from
# WAVES or Cox et al., but the real distortions found on scraper sites,
# marketplaces, and social feeds.
#
# Each attack takes (input_path, output_path, **params) and writes a PNG.
# Signature matches src/attacks.py so they slot straight into the same
# benchmark runners.

import io
import math
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# ─── 1. Marketplace crop-to-square ─────────────────────────────────

def marketplace_square(image_path: str, output_path: str,
                       target_size: int = 1024) -> None:
    """
    Simulate Amazon/eBay/Etsy auto-crop-to-square + resize to a canonical size.

    Real marketplaces enforce 1:1 aspect ratio for grid thumbnails. Non-square
    photos get center-cropped, then resized to a fixed dimension (typically
    1000-2000px on a side).
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top  = (h - side) // 2
    cropped = img.crop((left, top, left + side, top + side))
    resized = cropped.resize((target_size, target_size), Image.BICUBIC)
    resized.save(output_path, format="PNG")


# ─── 2. Instagram-style filter chain ───────────────────────────────

def instagram_filter(image_path: str, output_path: str,
                     saturation: float = 1.35,
                     contrast: float = 1.20,
                     brightness: float = 1.05,
                     vignette_strength: float = 0.35) -> None:
    """
    Deterministic Instagram-style filter: saturation boost, contrast bump,
    slight brightness lift, radial vignette. No AI, no randomness — pure PIL
    so the attack is reproducible for benchmarks.
    """
    img = Image.open(image_path).convert("RGB")

    img = ImageEnhance.Color(img).enhance(saturation)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Brightness(img).enhance(brightness)

    # Radial vignette
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / math.sqrt(cx ** 2 + cy ** 2)
    mask = 1.0 - vignette_strength * (dist ** 2)
    mask = np.clip(mask, 0, 1)[:, :, None]  # [H, W, 1]

    arr = np.array(img, dtype=np.float32) * mask
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").save(output_path, format="PNG")


# ─── 3. Screenshot simulation ──────────────────────────────────────

def screenshot_simulation(image_path: str, output_path: str,
                          jpeg_quality: int = 80,
                          resize_scale: float = 0.85,
                          noise_sigma: float = 2.0) -> None:
    """
    Simulate what happens when someone views your product photo, takes a
    screenshot, and re-uploads. The pipeline is: browser rescales to fit,
    OS screenshots at slightly lower resolution, then the screenshot gets
    JPEG-compressed on save/share, plus a bit of sensor-like noise from
    display re-rasterisation.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # Resize down (browser fit)
    small = img.resize((int(w * resize_scale), int(h * resize_scale)), Image.BICUBIC)

    # Add noise (display rasterisation)
    arr = np.array(small, dtype=np.float32)
    noise = np.random.normal(0, noise_sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    noisy = Image.fromarray(arr, "RGB")

    # JPEG round-trip (share/save)
    buf = io.BytesIO()
    noisy.save(buf, format="JPEG", quality=jpeg_quality)
    buf.seek(0)

    # Resize back up (someone re-uploaded expecting full resolution)
    restored = Image.open(buf).convert("RGB").resize((w, h), Image.BICUBIC)
    restored.save(output_path, format="PNG")


# ─── 4. Print-then-photograph simulation ───────────────────────────

def print_photograph_simulation(image_path: str, output_path: str,
                                perspective_strength: float = 0.03,
                                color_shift: float = 0.08,
                                blur_radius: float = 0.8,
                                jpeg_quality: int = 85,
                                seed: int | None = 42) -> None:
    """
    Simulate: print your product photo on paper, photograph the print with
    a phone. Steps applied:
      - Mild perspective warp (camera not perfectly parallel to paper)
      - Warm color shift (indoor lighting, paper reflectance)
      - Slight blur (lens + printer dot pattern)
      - JPEG (phone camera output)

    This is StegaStamp's target attack (Tancik et al. 2019, CVPR). We do a
    simplified version without a physics model — good enough for a benchmark.
    """
    rng = random.Random(seed)
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # 1. Perspective warp
    dx = int(w * perspective_strength)
    dy = int(h * perspective_strength)
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [
        (rng.randint(0, dx),      rng.randint(0, dy)),
        (w - rng.randint(0, dx),  rng.randint(0, dy)),
        (w - rng.randint(0, dx),  h - rng.randint(0, dy)),
        (rng.randint(0, dx),      h - rng.randint(0, dy)),
    ]
    coeffs = _perspective_coeffs(dst, src)
    img = img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC,
                        fillcolor=(255, 255, 255))

    # 2. Warm color shift (add to R, subtract from B)
    arr = np.array(img, dtype=np.float32)
    shift = color_shift * 255
    arr[:, :, 0] = np.clip(arr[:, :, 0] + shift * 0.6, 0, 255)   # warmer red
    arr[:, :, 2] = np.clip(arr[:, :, 2] - shift * 0.4, 0, 255)   # cooler blue
    img = Image.fromarray(arr.astype(np.uint8), "RGB")

    # 3. Slight blur
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # 4. JPEG round-trip
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    buf.seek(0)
    Image.open(buf).convert("RGB").save(output_path, format="PNG")


def _perspective_coeffs(src, dst):
    """Standard 8-parameter perspective transform, PIL convention."""
    matrix = []
    for (x, y), (X, Y) in zip(src, dst):
        matrix.append([X, Y, 1, 0, 0, 0, -x * X, -x * Y])
        matrix.append([0, 0, 0, X, Y, 1, -y * X, -y * Y])
    A = np.array(matrix, dtype=np.float64)
    B = np.array(src, dtype=np.float64).flatten()
    return np.linalg.solve(A, B).tolist()