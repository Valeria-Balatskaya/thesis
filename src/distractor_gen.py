# src/distractor_gen.py
# Generate content-unique distractor images for scale testing.
#
# Real marketplaces have thousands of sellers, each with different product
# photos. Our scale test needs many *different* images, not one image reused
# many times. This module procedurally generates diverse synthetic images
# so each distractor seller has content nobody else has.
#
# The generated images aren't realistic photos — they're colorful geometric
# patterns. That's fine for scale testing because we're measuring whether
# unique-content sellers interfere with each other, not testing image realism.

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import hashlib


def generate_distractor(seed: str, size: int = 256,
                        output_path: str | None = None) -> str:
    """
    Deterministically generate a unique-content 256×256 image from a seed string.
    Same seed → same image (idempotent). Different seeds → visually distinct images.

    The image has enough visual complexity (edges, gradients, textures) that
    watermarking algorithms can operate meaningfully on it.
    """
    # Hash the seed to get a reproducible int seed for numpy RNG
    hashed = int(hashlib.sha256(seed.encode()).hexdigest(), 16) % (2 ** 32)
    rng = np.random.default_rng(hashed)

    # Base: smooth color gradient
    h, w = size, size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) / size
    r_base = 0.5 + 0.5 * np.sin(xx * rng.uniform(3, 15) + rng.uniform(0, 6))
    g_base = 0.5 + 0.5 * np.sin(yy * rng.uniform(3, 15) + rng.uniform(0, 6))
    b_base = 0.5 + 0.5 * np.sin((xx + yy) * rng.uniform(3, 15) + rng.uniform(0, 6))
    arr = np.stack([r_base, g_base, b_base], axis=2) * 255

    # Overlay random geometric shapes for structural content
    img = Image.fromarray(arr.astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(img)
    n_shapes = rng.integers(4, 12)
    for _ in range(int(n_shapes)):
        shape_type = rng.choice(["rect", "ellipse", "line"])
        x1, y1 = rng.integers(0, size, size=2)
        x2, y2 = rng.integers(0, size, size=2)
        color = tuple(rng.integers(0, 256, size=3).tolist())
        if shape_type == "rect":
            draw.rectangle([min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)],
                          fill=color, outline=None)
        elif shape_type == "ellipse":
            draw.ellipse([min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)],
                        fill=color, outline=None)
        else:
            draw.line([x1, y1, x2, y2], fill=color,
                     width=int(rng.integers(2, 8)))

    # Add high-frequency noise so DCT coefficients aren't all zero anywhere
    noise = rng.normal(0, 8, (h, w, 3))
    final = np.clip(np.array(img, dtype=np.float32) + noise, 0, 255).astype(np.uint8)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(final, "RGB").save(output_path, format="PNG")
    return output_path