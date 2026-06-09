# src/attacks.py
# Attack suite for watermark robustness evaluation
# Reference: An et al. (2024) WAVES benchmark, arXiv:2401.08573

import numpy as np
from PIL import Image
import io
import os


def jpeg_compress(image_path: str, output_path: str, quality: int) -> None:
    """
    JPEG compression attack at given quality level (1-95).
    Introduces DCT quantisation noise that destroys LSB watermarks.
    Standard attack in watermark robustness benchmarks [An et al. 2024].
    """
    img = Image.open(image_path).convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    attacked = Image.open(buffer).convert("RGB")
    attacked.save(output_path, format="PNG")


def resize_attack(image_path: str, output_path: str, scale: float) -> None:
    """
    Resize image by scale factor then back to original dimensions.
    Interpolation destroys pixel-level LSB patterns.
    scale=0.5 means shrink to 50% then enlarge back.
    """
    img = Image.open(image_path).convert("RGB")
    orig_size = img.size
    small_size = (int(orig_size[0] * scale), int(orig_size[1] * scale))
    small = img.resize(small_size, Image.BICUBIC)
    restored = small.resize(orig_size, Image.BICUBIC)
    restored.save(output_path, format="PNG")


def gaussian_noise(image_path: str, output_path: str, sigma: float) -> None:
    """
    Add Gaussian noise with standard deviation sigma (0-255 scale).
    Simulates sensor noise and analogue transmission degradation.
    """
    img = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
    noise = np.random.normal(0, sigma, img.shape)
    noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
    Image.fromarray(noisy, "RGB").save(output_path, format="PNG")


def crop_attack(image_path: str, output_path: str, crop_fraction: float) -> None:
    """
    Crop a fraction of the image from each edge, then pad back to original size.
    crop_fraction=0.1 removes 10% from each side (total 20% area lost).
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    cy = int(h * crop_fraction)
    cx = int(w * crop_fraction)
    cropped = img[cy:h-cy, cx:w-cx]
    # Pad back to original size with zeros (black border)
    padded = np.zeros_like(img)
    padded[cy:h-cy, cx:w-cx] = cropped
    Image.fromarray(padded, "RGB").save(output_path, format="PNG")


def brightness_attack(image_path: str, output_path: str, factor: float) -> None:
    """
    Multiply pixel values by factor (e.g. 1.2 = 20% brighter).
    Simulates post-processing colour adjustments common in e-commerce.
    """
    img = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
    adjusted = np.clip(img * factor, 0, 255).astype(np.uint8)
    Image.fromarray(adjusted, "RGB").save(output_path, format="PNG")