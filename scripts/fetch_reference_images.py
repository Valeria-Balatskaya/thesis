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
# v3 training data (Part A1) adds two more steps:
#   manifest  split ABO at PRODUCT level: every product (item_id) that has an
#             image in the evaluation sample is excluded from training, with all
#             its other views, and N training images are drawn from the rest.
#             Writes data_manifests/abo_eval.csv, abo_train.csv, coco_eval.csv
#             (committed, so the split is fixed and auditable).
#   train     download the training manifest and shrink to 288 px (for Colab).
#
# Run:
#   python scripts/fetch_reference_images.py [n_per_set]          evaluation samples (default 60, seed 0)
#   python scripts/fetch_reference_images.py manifest --meta DIR [--n-train 2500]
#       DIR holds images/metadata/images.csv.gz and listings/metadata/listings_*.json.gz
#       (downloaded automatically into DIR if missing, ~95 MB)
#   python scripts/fetch_reference_images.py train [--manifest data_manifests/abo_train.csv]
#       [--dest data/abo_train] [--short-side 288]

import io
import json
import random
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 60
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


# ─── v3 training data ─────────────────────────────────────────────

MANIFESTS = Path("data_manifests")


def _abo_meta(meta: Path) -> tuple[dict, dict]:
    """path -> (image_id, w, h), image_id -> {item_id, ...} (from listings)."""
    import csv
    import gzip
    meta.mkdir(parents=True, exist_ok=True)
    if not (meta / "images.csv.gz").exists():
        (meta / "images.csv.gz").write_bytes(get(f"{ABO}/images/metadata/images.csv.gz"))
    images = {}
    with gzip.open(meta / "images.csv.gz", "rt") as f:
        for row in csv.DictReader(f):
            images[row["path"]] = (row["image_id"], int(row["width"]), int(row["height"]))
    items_of, product_type = {}, {}
    for i in range(16):
        name = f"listings_{i:x}.json.gz"
        if not (meta / name).exists():
            (meta / name).write_bytes(get(f"{ABO}/listings/metadata/{name}"))
        with gzip.open(meta / name, "rt") as f:
            for line in f:
                item = json.loads(line)
                ptype = (item.get("product_type") or [{}])[0].get("value", "")
                for img in [item.get("main_image_id")] + list(item.get("other_image_id") or []):
                    if img:
                        items_of.setdefault(img, set()).add(item["item_id"])
                        product_type.setdefault(img, ptype)
    return images, {"items_of": items_of, "product_type": product_type}


def make_manifest(meta: Path, n_train: int, min_side: int = 500) -> None:
    """Product-level train/eval split of ABO (see header)."""
    import csv
    images, info = _abo_meta(meta)
    items_of, ptype = info["items_of"], info["product_type"]
    MANIFESTS.mkdir(exist_ok=True)

    eval_paths = sorted(f"{p.name[:2]}/{p.name}" for p in Path("data/abo_sample").glob("*.jpg"))
    eval_items = set()
    with open(MANIFESTS / "abo_eval.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "image_id", "width", "height", "item_ids", "product_type"])
        for p in eval_paths:
            iid, wd, ht = images[p]
            items = items_of.get(iid, set())
            eval_items |= items
            w.writerow([p, iid, wd, ht, " ".join(sorted(items)), ptype.get(iid, "")])

    # every image of every product that appears in evaluation is excluded
    banned = {iid for iid, items in items_of.items() if items & eval_items}
    banned |= {images[p][0] for p in eval_paths}
    pool = sorted(p for p, (iid, wd, ht) in images.items()
                  if iid not in banned and iid in items_of and min(wd, ht) >= min_side)
    # Cap any single product category at 10% (ABO is ~30% phone cases)
    rng = random.Random(SEED)
    cap, per_type, train = max(1, n_train // 10), {}, []
    for p in rng.sample(pool, len(pool)):
        t = ptype.get(images[p][0], "")
        if per_type.get(t, 0) < cap:
            per_type[t] = per_type.get(t, 0) + 1
            train.append(p)
            if len(train) == n_train:
                break
    train.sort()
    with open(MANIFESTS / "abo_train.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "image_id", "width", "height", "item_ids", "product_type"])
        for p in train:
            iid, wd, ht = images[p]
            w.writerow([p, iid, wd, ht, " ".join(sorted(items_of[iid])), ptype.get(iid, "")])

    with open(MANIFESTS / "coco_eval.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "split"])
        for p in sorted(Path("data/coco_sample").glob("*.jpg")):
            w.writerow([p.name, "test2017"])

    train_items = set().union(*(items_of[images[p][0]] for p in train))
    print(f"eval: {len(eval_paths)} images, {len(eval_items)} products")
    print(f"train: {len(train)} images, {len(train_items)} products, {len(per_type)} categories "
          f"(cap {cap} each), drawn from {len(pool)} eligible")
    print(f"overlap: images {len(set(train) & set(eval_paths))}, products {len(train_items & eval_items)}")


def download_train(manifest: Path, dest: Path, short_side: int, workers: int = 16) -> None:
    import csv
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    dest.mkdir(parents=True, exist_ok=True)
    paths = [row["path"] for row in csv.DictReader(open(manifest))]

    def one(p: str) -> bool:
        out = dest / Path(p).name
        if out.exists():
            return True
        try:
            img = Image.open(io.BytesIO(get(f"{ABO}/images/original/{p}"))).convert("RGB")
        except Exception:
            return False
        scale = short_side / min(img.size)
        if scale < 1:
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        img.save(out, quality=95)
        return True

    with ThreadPoolExecutor(workers) as ex:
        ok = sum(ex.map(one, paths))
    print(f"{ok}/{len(paths)} training images in {dest}")


if __name__ == "__main__":
    import argparse
    if len(sys.argv) < 2 or sys.argv[1].isdigit():
        rng = random.Random(SEED)
        fetch_abo(Path("data/abo_sample"), N, rng)
        fetch_coco(Path("data/coco_sample"), N, rng)
        sys.exit()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--meta", default="data/abo_meta")
    m.add_argument("--n-train", type=int, default=2500)
    t = sub.add_parser("train")
    t.add_argument("--manifest", default="data_manifests/abo_train.csv")
    t.add_argument("--dest", default="data/abo_train")
    t.add_argument("--short-side", type=int, default=288)
    a = ap.parse_args()
    if a.cmd == "manifest":
        make_manifest(Path(a.meta), a.n_train)
    else:
        download_train(Path(a.manifest), Path(a.dest), a.short_side)
