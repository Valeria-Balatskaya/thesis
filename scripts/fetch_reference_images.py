# scripts/fetch_reference_images.py
# Download two small, reproducible reference sets for the e-commerce image
# analysis (scripts/analyse_ecommerce_images.py):
#
#   data/abo_sample/   N product photos from Amazon Berkeley Objects (ABO),
#                      real Amazon listing images, full resolution, CC BY 4.0.
#                      Collins et al. (2022), ABO: Dataset and Benchmarks for
#                      Real-World 3D Object Understanding. CVPR.
#   data/coco_sample/  N "regular" photos from COCO test2017. Same source and
#                      style as the val2017 images HiDDeN v2 was trained on, but
#                      never seen in training. Lin et al. (2014), Microsoft COCO.
#
# Files are saved exactly as downloaded (no re-encoding), so resolution and
# JPEG quality can be measured. Both folders are git-ignored.
#
# Run: python scripts/fetch_reference_images.py [n_per_set]   (default 60, seed 0)

import io
import json
import random
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
SEED = 0
ABO = "https://amazon-berkeley-objects.s3.amazonaws.com"
COCO_INFO = "http://images.cocodataset.org/annotations/image_info_test2017.zip"
COCO_IMG = "http://images.cocodataset.org/test2017/{}"


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def fetch_abo(dest: Path, n: int, rng: random.Random) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    # Images are sharded into 256 folders 00..ff; take one image from each of n
    # randomly chosen folders, starting at a random key, so the sample spreads
    # over many products and categories.
    for shard in rng.sample([f"{i:02x}" for i in range(256)], n):
        start = f"images/original/{shard}/{rng.getrandbits(32):08x}"
        xml = get(f"{ABO}/?list-type=2&prefix=images/original/{shard}/"
                  f"&start-after={start}&max-keys=1").decode()
        keys = re.findall(r"<Key>([^<]+)</Key>", xml)
        if not keys:   # random start was past the last key: take the first one
            xml = get(f"{ABO}/?list-type=2&prefix=images/original/{shard}/&max-keys=1").decode()
            keys = re.findall(r"<Key>([^<]+)</Key>", xml)
        out = dest / Path(keys[0]).name
        if not out.exists():
            out.write_bytes(get(f"{ABO}/{keys[0]}"))
        print("abo ", out.name)


def fetch_coco(dest: Path, n: int, rng: random.Random) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(get(COCO_INFO))) as z:
        name = next(f for f in z.namelist() if f.endswith("image_info_test2017.json"))
        files = sorted(img["file_name"] for img in json.loads(z.read(name))["images"])
    for fname in rng.sample(files, n):
        out = dest / fname
        if not out.exists():
            out.write_bytes(get(COCO_IMG.format(fname)))
        print("coco", fname)


if __name__ == "__main__":
    rng = random.Random(SEED)
    fetch_abo(Path("data/abo_sample"), N, rng)
    fetch_coco(Path("data/coco_sample"), N, rng)
