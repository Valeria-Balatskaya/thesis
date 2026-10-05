# scripts/prepare_training_images.py
# Shrink a folder of photos (e.g. COCO) to small JPEGs once, so the training
# data loader isn't the bottleneck. Shorter side -> 288 px.
#
# Run: python scripts/prepare_training_images.py <src_dir> <dst_dir> [max_images]

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

SHORT_SIDE = 288


def _convert(job):
    src, dst = job
    try:
        img = Image.open(src).convert("RGB")
        w, h = img.size
        if min(w, h) < 256:
            return 0
        scale = SHORT_SIDE / min(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
        img.save(dst, format="JPEG", quality=95)
        return 1
    except Exception:
        return 0


if __name__ == "__main__":
    src_dir, dst_dir = Path(sys.argv[1]), Path(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src_dir.rglob("*")
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:limit]
    jobs = [(str(p), str(dst_dir / f"{i:06d}.jpg")) for i, p in enumerate(files)]
    with ProcessPoolExecutor() as pool:
        done = sum(pool.map(_convert, jobs, chunksize=64))
    print(f"{done} images written to {dst_dir}")
