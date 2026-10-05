# src/modern_attacks.py
# What image thieves do today, beyond the classic JPEG / resize / noise suite:
# geometric re-use (off-centre crops, rotation, mirroring, screenshots of a
# whole page), content edits (banners, stickers, object removal) and
# generative-AI edits (background replacement, regeneration through a
# diffusion VAE).
#
# Same signature as src/attacks.py: (image_path, output_path, **params) -> PNG.
#
# The two AI attacks need optional packages and a one-time model download:
#   regenerate_vae      pip install diffusers   (stabilityai/sd-vae-ft-mse, ~335 MB)
#   replace_background  pip install rembg onnxruntime   (u2net, ~176 MB)
#
# References:
#   Zhao et al. (2023). Invisible Image Watermarks Are Provably Removable Using
#   Generative AI. arXiv:2306.01953.  (regeneration attack)
#   An et al. (2024). WAVES: Benchmarking the Robustness of Image Watermarks.

import io

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def _open(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _font(size: int):
    for name in ("/System/Library/Fonts/Helvetica.ttc", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ─── Geometric re-use ─────────────────────────────────────────────

def offcentre_crop(image_path: str, output_path: str, keep: float = 0.7,
                   dx: float = 1.0, dy: float = 0.0) -> None:
    """Keep `keep` of each side; dx/dy in [0,1] place the window (0 = left/top)."""
    img = _open(image_path)
    w, h = img.size
    cw, ch = int(w * keep), int(h * keep)
    left, top = int((w - cw) * dx), int((h - ch) * dy)
    img.crop((left, top, left + cw, top + ch)).save(output_path, format="PNG")


def rotate(image_path: str, output_path: str, degrees: float = 8.0) -> None:
    _open(image_path).rotate(degrees, resample=Image.BICUBIC, expand=True,
                             fillcolor=(255, 255, 255)).save(output_path, format="PNG")


def mirror(image_path: str, output_path: str) -> None:
    _open(image_path).transpose(Image.FLIP_LEFT_RIGHT).save(output_path, format="PNG")


def perspective(image_path: str, output_path: str, strength: float = 0.12,
                seed: int = 7) -> None:
    """Strong perspective warp (a photo of a screen or a print, taken at an angle)."""
    rng = np.random.default_rng(seed)
    img = np.array(_open(image_path))
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = rng.uniform(0, strength, (4, 2)) * [w, h]
    dst = src + jitter * [[1, 1], [-1, 1], [-1, -1], [1, -1]]
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    out = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderValue=(255, 255, 255))
    Image.fromarray(out).save(output_path, format="PNG")


def page_screenshot(image_path: str, output_path: str, scale: float = 0.55,
                    jpeg_quality: int = 80) -> None:
    """
    The image as it appears inside a screenshot of a shop page: scaled down,
    surrounded by page chrome, text and a price, then JPEG-compressed.
    """
    img = _open(image_path)
    w, h = img.size
    pw, ph = int(w * scale), int(h * scale)
    canvas = Image.new("RGB", (pw + 420, ph + 260), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 64), fill=(35, 47, 62))
    draw.text((24, 18), "MegaDeals  |  Search products...", fill=(255, 255, 255), font=_font(24))
    canvas.paste(img.resize((pw, ph), Image.LANCZOS), (40, 110))
    x = pw + 70
    draw.text((x, 120), "Brand New!", fill=(20, 20, 20), font=_font(30))
    draw.text((x, 170), "$ 199.99", fill=(177, 39, 4), font=_font(40))
    draw.rectangle((x, 240, x + 260, 296), fill=(255, 216, 20))
    draw.text((x + 50, 254), "Add to Cart", fill=(20, 20, 20), font=_font(24))
    for i in range(4):
        draw.rectangle((x, 330 + i * 28, x + 300 - 40 * i, 342 + i * 28), fill=(215, 215, 215))
    _jpeg(canvas, jpeg_quality).save(output_path, format="PNG")


def screen_photo(image_path: str, output_path: str, seed: int = 3) -> None:
    """Phone photo of a screen: perspective, moire pattern, blur, colour cast, JPEG."""
    perspective(image_path, output_path, strength=0.07, seed=seed)
    arr = np.array(_open(output_path), dtype=np.float32)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    moire = 6 * np.sin(2 * np.pi * (xx * 0.31 + yy * 0.07)) * np.sin(2 * np.pi * yy * 0.27)
    arr = arr + moire[:, :, None]
    arr *= np.array([1.03, 1.0, 0.94])          # warm cast
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    _jpeg(img.filter(ImageFilter.GaussianBlur(1.0)), 75).save(output_path, format="PNG")


def social_recompress(image_path: str, output_path: str, max_side: int = 1080) -> None:
    """Messenger / social upload: downscale to a size cap, JPEG twice."""
    img = _open(image_path)
    img.thumbnail((max_side * 0.75, max_side * 0.75), Image.LANCZOS)
    _jpeg(_jpeg(img, 72), 80).save(output_path, format="PNG")


# ─── Content edits ────────────────────────────────────────────────

def promo_overlay(image_path: str, output_path: str, text: str = "SALE -50%") -> None:
    """A reseller's banner, badge and their own visible logo over the image."""
    img = _open(image_path)
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle((0, int(h * 0.82), w, h), fill=(220, 30, 40, 235))
    draw.text((int(w * 0.05), int(h * 0.85)), text, fill=(255, 255, 255), font=_font(int(h * 0.1)))
    r = int(min(w, h) * 0.13)
    draw.ellipse((w - 2 * r - 20, 20, w - 20, 2 * r + 20), fill=(255, 210, 0, 240))
    draw.text((w - 2 * r, r), "HOT", fill=(0, 0, 0), font=_font(int(r * 0.6)))
    draw.text((20, 20), "shop-xyz.example", fill=(255, 255, 255, 200), font=_font(int(h * 0.04)))
    img.save(output_path, format="PNG")


def inpaint_region(image_path: str, output_path: str, x: float = 0.3, y: float = 0.3,
                   size: float = 0.3) -> None:
    """Object / logo removal: a region is erased and filled in (Telea inpainting)."""
    img = np.array(_open(image_path))
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    mask[int(h * y):int(h * (y + size)), int(w * x):int(w * (x + size))] = 255
    out = cv2.inpaint(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), mask, 5, cv2.INPAINT_TELEA)
    Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)).save(output_path, format="PNG")


def upscale_sharpen(image_path: str, output_path: str, factor: float = 2.0) -> None:
    """Enlarge and sharpen (what 'enhance' tools do; not a neural upscaler)."""
    img = _open(image_path)
    w, h = img.size
    img = img.resize((int(w * factor), int(h * factor)), Image.LANCZOS)
    img.filter(ImageFilter.UnsharpMask(radius=2, percent=120)).save(output_path, format="PNG")


# ─── Generative-AI edits (optional dependencies) ──────────────────

_vae = None
_rembg_session = None


def ai_available() -> dict:
    out = {}
    for name, module in (("regenerate_vae", "diffusers"), ("replace_background", "rembg")):
        try:
            __import__(module)
            out[name] = True
        except Exception:
            out[name] = False
    return out


def regenerate_vae(image_path: str, output_path: str, max_side: int = 768) -> None:
    """
    Regeneration attack: encode the image into the latent space of Stable
    Diffusion's VAE and decode it again. The picture looks the same, but every
    pixel is re-synthesised by a neural network.
    """
    global _vae
    import torch
    from diffusers import AutoencoderKL
    if _vae is None:
        _vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").eval()
    img = _open(image_path)
    orig_size = img.size
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    w, h = (img.width // 8) * 8, (img.height // 8) * 8
    x = torch.from_numpy(np.array(img.resize((w, h)), dtype=np.float32) / 127.5 - 1)
    x = x.permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        y = _vae.decode(_vae.encode(x).latent_dist.mean).sample
    out = ((y.squeeze(0).permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).numpy().astype(np.uint8)
    Image.fromarray(out).resize(orig_size, Image.LANCZOS).save(output_path, format="PNG")


def replace_background(image_path: str, output_path: str,
                       colour: tuple = (236, 229, 216)) -> None:
    """AI background removal (U2-Net) + a new studio-style gradient background."""
    global _rembg_session
    from rembg import remove, new_session
    if _rembg_session is None:
        _rembg_session = new_session("u2net")
    img = _open(image_path)
    cut = remove(img, session=_rembg_session)            # RGBA
    w, h = img.size
    grad = np.linspace(1.0, 0.82, h)[:, None, None] * np.array(colour)[None, None, :]
    bg = Image.fromarray(np.broadcast_to(grad, (h, w, 3)).astype(np.uint8))
    bg.paste(cut, (0, 0), cut)
    bg.save(output_path, format="PNG")
