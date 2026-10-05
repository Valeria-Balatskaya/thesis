# app/guard.py
# Service layer of the TraceMark app: protect images, analyse suspects,
# run attacks for the Attack Lab, record detections.

import hashlib
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from app import store
from src import attacks, ecommerce_attacks as ea, modern_attacks as ma
from src.defence import DefencePipeline
from src.hidden_v2_inference import HiddenV2Model
from src.metrics import compute as quality
from src.payload import (encode_product_id, blind_false_positive_rate, ID_BITS, TAG_BITS,
                         PAYLOAD_BITS, ECC_T)

try:                                  # iPhone photos
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# hidden_v3_best.pt (trained with a colour charge and flat-area masking, see
# notebooks/train_hidden_v2.ipynb) is preferred when present.
CHECKPOINT = next((c for c in ("checkpoints/hidden_v3_best.pt", "checkpoints/hidden_v2_best.pt")
                   if Path(c).exists()), "checkpoints/hidden_v2_best.pt")
DATA = store.DATA_DIR
MAX_SIDE = 1600                       # uploads are downscaled to this
# Watermark visibility settings. At full strength (1.0 = 40 dB, as trained) the
# current checkpoint shows a green/purple tint on white backgrounds, because it
# hides its signal in broad colour patches. Mitigation: 0.7 strength, and only
# 30% of that on flat areas (src/hidden_v2.texture_mask). Robustness with these
# values is measured by tests/test_defence_benchmark.py <folder> 35 0.7 0.3.
# A v3 checkpoint has the masking built in and is used as trained.
STRENGTH, MASK_FLOOR = (1.0, None) if "v3" in CHECKPOINT else (0.7, 0.3)

for sub in ("originals", "copies", "diff", "lab", "scans", "crawl", "aligned", "demo"):
    (DATA / sub).mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()              # the model and registry are shared across request threads
_pipeline: DefencePipeline | None = None


def pipeline() -> DefencePipeline:
    global _pipeline
    with _lock:
        if _pipeline is None:
            if not Path(CHECKPOINT).exists():
                raise FileNotFoundError(
                    f"{CHECKPOINT} is missing. Train it with notebooks/train_hidden_v2.ipynb "
                    "and copy it into checkpoints/.")
            p = DefencePipeline(HiddenV2Model(CHECKPOINT))
            for product in store.list_products():
                copy_ids = [c["id"] for c in store.list_copies(product["id"])]
                if Path(product["original_path"]).exists():
                    p.register(product["id"], product["original_path"], copy_ids)
            _pipeline = p
        return _pipeline


def media_url(path: str | None) -> str | None:
    """app/data/x/y.png -> /media/x/y.png"""
    if not path:
        return None
    return "/media/" + str(Path(path).relative_to(DATA))


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_info() -> dict:
    m = pipeline().model
    return {"checkpoint": CHECKPOINT, "payload_bits": PAYLOAD_BITS, "id_bits": ID_BITS,
            "tag_bits": TAG_BITS, "ecc_t": ECC_T, "target_psnr": m.target_psnr,
            "blind_fpr": blind_false_positive_rate(), "ai_attacks": ma.ai_available()}


# ─── protect ──────────────────────────────────────────────────────

def _save_normalised(src_file, dest: Path) -> tuple[int, int]:
    """Any upload (PNG/JPG/HEIC, any size) -> RGB PNG, longer side <= MAX_SIDE."""
    img = Image.open(src_file)
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    img.save(dest, format="PNG")
    return img.size


def _write_diff(original: str, watermarked: str, dest: Path, gain: int = 30) -> None:
    """Amplified difference image: what the watermark looks like if you could see it."""
    a = np.array(Image.open(original).convert("RGB"), dtype=np.int16)
    b = np.array(Image.open(watermarked).convert("RGB"), dtype=np.int16)
    Image.fromarray(np.clip(128 + (b - a) * gain, 0, 255).astype(np.uint8)).save(dest)


def create_copy(product_id: int, label: str) -> dict:
    product = store.get_product(product_id)
    copy_id = store.reserve_copy(product_id, label)
    path = DATA / "copies" / f"p{product_id}_c{copy_id}.png"
    p = pipeline()
    with _lock:
        p.model.embed_product(product["original_path"], copy_id, str(path),
                              strength=STRENGTH, mask_floor=MASK_FLOOR)
    q = quality(product["original_path"], str(path))
    store.finish_copy(copy_id, str(path), q["PSNR_dB"], q["SSIM"])
    _write_diff(product["original_path"], str(path), DATA / "diff" / f"c{copy_id}.png")
    with _lock:
        p.register(product_id, product["original_path"],
                   [c["id"] for c in store.list_copies(product_id)])
    return copy_json(store.get_copy(copy_id))


def protect(seller_id: int, title: str, upload_file, authorized_url: str | None) -> dict:
    tmp = DATA / "originals" / f"tmp_{uuid.uuid4().hex}.png"
    width, height = _save_normalised(upload_file, tmp)
    product_id = store.create_product(seller_id, title, str(tmp), width, height)
    final = DATA / "originals" / f"p{product_id}.png"
    tmp.rename(final)
    store.set_product_original(product_id, str(final))
    if authorized_url:
        store.add_authorized_url(product_id, authorized_url)
    create_copy(product_id, "Public listing")
    return product_json(product_id)


def copy_json(copy: dict) -> dict:
    bits = encode_product_id(copy["id"]).tolist()
    return {**copy, "url": media_url(copy["file_path"]),
            "diff_url": f"/media/diff/c{copy['id']}.png", "payload_bits": bits}


def product_json(product_id: int) -> dict | None:
    product = store.get_product(product_id)
    if not product:
        return None
    return {**product,
            "original_url": media_url(product["original_path"]),
            "copies": [copy_json(c) for c in store.list_copies(product_id)],
            "authorized_urls": store.list_authorized_urls(product_id),
            "detections": [detection_json(d) for d in store.list_detections(product_id)]}


def remove_product(product_id: int) -> None:
    store.delete_product(product_id)
    with _lock:
        pipeline_ = _pipeline
    if pipeline_:
        pipeline_.forget(product_id)


# ─── analyse ──────────────────────────────────────────────────────

def is_authorized(product_id: int, page_url: str | None) -> bool:
    """A sighting is authorised when its page is on a host the seller listed."""
    if not page_url:
        return False
    page = urlparse(page_url if "//" in page_url else "//" + page_url)
    for row in store.list_authorized_urls(product_id):
        allowed = urlparse(row["url"] if "//" in row["url"] else "//" + row["url"])
        if not allowed.netloc or allowed.netloc.lower() != page.netloc.lower():
            continue
        if page.path.startswith(allowed.path.rstrip("/")):
            return True
    return False


def analyse(suspect_path: str, page_url: str | None = None, image_url: str | None = None,
            record: bool = False) -> dict:
    """Run the staged pipeline on a suspect file and describe the result for the UI."""
    p = pipeline()
    t0 = time.time()
    with _lock:
        res = p.analyse(suspect_path)
    elapsed = round(time.time() - t0, 2)

    out = {"verdict": res["verdict"], "stages": res["stages"], "seconds": elapsed,
           "suspect_url": media_url(suspect_path), "product": None, "copy": None,
           "aligned_url": None, "bits": None, "expected_bits": None,
           "authorized": None, "detection_id": None}
    if res["verdict"] == "none":
        return out

    product = store.get_product(res["product_id"])
    out["product"] = {"id": product["id"], "title": product["title"],
                      "seller_name": product["seller_name"],
                      "original_url": media_url(product["original_path"])}
    if res["copy_id"] is not None:
        copy = store.get_copy(res["copy_id"])
        out["copy"] = {"id": copy["id"], "label": copy["label"], "created_at": copy["created_at"]}
        out["bits"] = res["bits"].tolist()
        out["expected_bits"] = encode_product_id(copy["id"]).tolist()

    aligned_path = None
    if res["aligned"] is not None:
        aligned_path = str(DATA / "aligned" / f"{uuid.uuid4().hex}.jpg")
        Image.fromarray(res["aligned"]).save(aligned_path, quality=90)
        out["aligned_url"] = media_url(aligned_path)

    out["authorized"] = is_authorized(product["id"], page_url)
    if record:
        out["detection_id"] = store.record_detection(
            product_id=product["id"], copy_id=res["copy_id"], verdict=res["verdict"],
            image_url=image_url, page_url=page_url, snapshot_path=suspect_path,
            aligned_path=aligned_path, authorized=out["authorized"],
            details={"stages": res["stages"], "sha256": sha256(suspect_path)})
    return out


def detection_json(d: dict) -> dict:
    return {**d, "snapshot_url": media_url(d.get("snapshot_path")),
            "aligned_url": media_url(d.get("aligned_path"))}


# ─── attack lab ───────────────────────────────────────────────────
# name -> (label, group, function, fixed kwargs, slider spec or None)
# slider spec: (param, min, max, step, default)

LAB_ATTACKS = {
    "jpeg":        ("JPEG compression", "Classic", attacks.jpeg_compress, {}, ("quality", 5, 95, 5, 30)),
    "resize":      ("Downscale and back", "Classic", attacks.resize_attack, {}, ("scale", 0.2, 0.9, 0.1, 0.5)),
    "noise":       ("Gaussian noise", "Classic", attacks.gaussian_noise, {}, ("sigma", 5, 40, 5, 15)),
    "brightness":  ("Brightness", "Classic", attacks.brightness_attack, {}, ("factor", 0.6, 1.6, 0.1, 1.3)),
    "instagram":   ("Instagram-style filter", "Classic", ea.instagram_filter, {}, None),
    "square":      ("Marketplace square crop", "Re-use", ea.marketplace_square, {"target_size": 800}, None),
    "crop":        ("Off-centre crop", "Re-use", ma.offcentre_crop, {"dx": 0.8, "dy": 0.2}, ("keep", 0.3, 0.9, 0.1, 0.6)),
    "rotate":      ("Rotate", "Re-use", ma.rotate, {}, ("degrees", -30, 30, 2, 8)),
    "mirror":      ("Mirror", "Re-use", ma.mirror, {}, None),
    "social":      ("Messenger re-upload", "Re-use", ma.social_recompress, {}, None),
    "marketplace": ("Marketplace CDN (WebP)", "Re-use", ma.marketplace_variant, {}, ("size", 64, 1024, 32, 360)),
    "overlay":     ("Sale banner + logo", "Re-use", ma.promo_overlay, {}, None),
    "screenshot":  ("Native screenshot", "Capture", ea.native_screenshot_simulation, {}, None),
    "page":        ("Screenshot of a shop page", "Capture", ma.page_screenshot, {}, ("scale", 0.3, 0.9, 0.05, 0.55)),
    "screen_photo": ("Phone photo of a screen", "Capture", ma.screen_photo, {}, None),
    "print":       ("Print and photograph", "Capture", ea.print_photograph_simulation, {}, None),
    "perspective": ("Perspective warp", "Capture", ma.perspective, {}, ("strength", 0.02, 0.2, 0.02, 0.12)),
    "inpaint":     ("Erase a region (inpaint)", "Edit", ma.inpaint_region, {}, ("size", 0.1, 0.5, 0.05, 0.3)),
    "upscale":     ("Enlarge + sharpen", "Edit", ma.upscale_sharpen, {}, ("factor", 1.2, 2.0, 0.2, 1.6)),
    "ai_background": ("AI background replacement", "Generative AI", ma.replace_background, {}, None),
    "ai_regenerate": ("AI regeneration (diffusion VAE)", "Generative AI", ma.regenerate_vae, {}, None),
}
_AI_REQUIRES = {"ai_background": "replace_background", "ai_regenerate": "regenerate_vae"}


def lab_catalogue() -> list[dict]:
    available = ma.ai_available()
    out = []
    for name, (label, group, _, _, slider) in LAB_ATTACKS.items():
        item = {"name": name, "label": label, "group": group, "slider": None,
                "available": available.get(_AI_REQUIRES.get(name, ""), True)}
        if slider:
            item["slider"] = dict(zip(("param", "min", "max", "step", "default"), slider))
        out.append(item)
    return out


def lab_apply(source_path: str, attack: str, value: float | None) -> str:
    label, _, fn, fixed, slider = LAB_ATTACKS[attack]
    kwargs = dict(fixed)
    if slider and value is not None:
        param, lo, hi, _, default = slider
        value = min(max(float(value), lo), hi)
        kwargs[param] = int(value) if param in ("quality", "size") else value
    out = DATA / "lab" / f"{uuid.uuid4().hex}.png"
    fn(source_path, str(out), **kwargs)
    return str(out)


def lab_source(copy_id: int, stacked_on: str | None) -> str:
    """The image an attack is applied to: a fresh copy, or the previous lab output."""
    if stacked_on:
        path = (DATA / Path(stacked_on).relative_to("/media")).resolve()
        if (DATA / "lab").resolve() in path.parents and path.exists():
            return str(path)
    return store.get_copy(copy_id)["file_path"]
