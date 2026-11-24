# scripts/prepare_owner_forensics_data.py
# Photo sets for the owner-forensics experiments (docs/research_plan.md, section 4b).
#
#   pilot   40 photos = the A1/B1/B2 set (first 20 own + first 20 ABO sample); debugging only
#   e2pos   300 victim photos, none in the pilot: the remaining 27 own + 137 ABO + 136 COCO test2017
#   e2null  1,300 further photos (650 ABO + 650 COCO test2017) for the false-claim rate
#
# ABO / COCO photos are drawn with a fixed seed from the public buckets and exclude every image
# that is in the v3 training manifest, the evaluation manifests, data/abo_negatives (which are
# ABO *training* images) and the existing samples. Every photo is shrunk to <= 1024 px (long side)
# and saved as PNG, like the earlier tests. Output: data/owner_forensics/<set>/NNNN.png (git-ignored)
# and data_manifests/owner_forensics_<set>.csv (committed: the sets are fixed and auditable).
#
# Run: python scripts/prepare_owner_forensics_data.py        (resumable, ~15 min)

import csv
import io
import json
import os
import random
import re
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image

ABO = "https://amazon-berkeley-objects.s3.amazonaws.com"
COCO_INFO = "http://images.cocodataset.org/annotations/image_info_test2017.zip"
COCO_IMG = "http://images.cocodataset.org/test2017/{}"
SEED = 20261124
N_ABO_POS, N_COCO_POS, N_ABO_NULL, N_COCO_NULL = 137, 136, 650, 650
ROOT = Path("data/owner_forensics")
MAN = Path("data_manifests")
EXT = (".png", ".jpg", ".jpeg")


def get(url: str, tries: int = 4) -> bytes:
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
    raise RuntimeError


def used_names() -> set:
    used = set(os.listdir("data/abo_negatives")) | set(os.listdir("data/abo_sample")) | set(os.listdir("data/coco_sample"))
    for m in ("abo_train.csv", "abo_eval.csv"):
        used |= {os.path.basename(r["path"]) for r in csv.DictReader(open(MAN / m))}
    used |= {r["file"] for r in csv.DictReader(open(MAN / "coco_eval.csv"))}
    return used


def abo_candidates(rng: random.Random, need: int, used: set) -> list:
    keys = []
    for shard in rng.sample([f"{i:02x}" for i in range(256)], 256):
        start = f"images/original/{shard}/{rng.getrandbits(32):08x}"
        xml = get(f"{ABO}/?list-type=2&prefix=images/original/{shard}/&start-after={start}&max-keys=12").decode()
        ks = re.findall(r"<Key>([^<]+)</Key>", xml)
        if len(ks) < 12:                       # started near the end of the shard: take from the beginning
            xml = get(f"{ABO}/?list-type=2&prefix=images/original/{shard}/&max-keys=12").decode()
            ks += re.findall(r"<Key>([^<]+)</Key>", xml)
        keys += [k for k in dict.fromkeys(ks) if os.path.basename(k) not in used and k.lower().endswith(EXT)]
        if len(keys) >= 3 * need:
            break
    keys = sorted(set(keys))
    rng.shuffle(keys)
    return keys


def save(url: str, out: Path) -> tuple | None:
    if out.exists():
        im = Image.open(out)
        return im.size
    try:
        img = Image.open(io.BytesIO(get(url))).convert("RGB")
    except Exception:
        return None
    if min(img.size) < 300:
        return None
    img.thumbnail((1024, 1024), Image.LANCZOS)
    img.save(out, format="PNG")
    return img.size


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    used = used_names()
    # pilot: copies of the existing A1/B1/B2 photos
    pilot = []
    (ROOT / "pilot").mkdir(exist_ok=True)
    for tag, folder, n in (("own", "data/products_png", 20), ("abo", "data/abo_sample", 20)):
        for k, f in enumerate(sorted(x for x in os.listdir(folder) if x.lower().endswith(EXT))[:n]):
            out = ROOT / "pilot" / f"{len(pilot):04d}.png"
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            img.thumbnail((1024, 1024), Image.LANCZOS)
            img.save(out, format="PNG")
            pilot.append({"set": "pilot", "source": tag, "name": f, "file": out.name, "w": img.width, "h": img.height})
    # remaining own photos go to e2pos
    pos = []
    (ROOT / "e2pos").mkdir(exist_ok=True)
    own_rest = sorted(x for x in os.listdir("data/products_png") if x.lower().endswith(EXT))[20:]
    for f in own_rest:
        out = ROOT / "e2pos" / f"{len(pos):04d}.png"
        img = Image.open(os.path.join("data/products_png", f)).convert("RGB")
        img.thumbnail((1024, 1024), Image.LANCZOS)
        img.save(out, format="PNG")
        pos.append({"set": "e2pos", "source": "own", "name": f, "file": out.name, "w": img.width, "h": img.height})

    n_abo = N_ABO_POS + N_ABO_NULL
    abo_keys = abo_candidates(rng, n_abo, used)
    with zipfile.ZipFile(io.BytesIO(get(COCO_INFO))) as z:
        name = next(f for f in z.namelist() if f.endswith("image_info_test2017.json"))
        coco_files = sorted(i["file_name"] for i in json.loads(z.read(name))["images"])
    coco_files = [f for f in coco_files if f not in used]
    rng.shuffle(coco_files)
    print(f"candidates: ABO {len(abo_keys)}, COCO {len(coco_files)}", flush=True)

    def accept(src, names_urls, dest_dir, n, rows_out, prefix_start):
        dest_dir.mkdir(parents=True, exist_ok=True)
        got = 0
        with ThreadPoolExecutor(12) as ex:
            for start in range(0, len(names_urls), 48):
                chunk = names_urls[start:start + 48]
                tmp = [(nm, url, dest_dir / f"tmp_{start + k:05d}.png") for k, (nm, url) in enumerate(chunk)]
                sizes = list(ex.map(lambda t: save(t[1], t[2]), tmp))
                for (nm, url, path), size in zip(tmp, sizes):
                    if size is None or got >= n:
                        path.unlink(missing_ok=True)
                        continue
                    final = dest_dir / f"{prefix_start + len(rows_out):04d}.png"
                    path.rename(final)
                    rows_out.append({"set": dest_dir.name, "source": src, "name": nm, "file": final.name,
                                     "w": size[0], "h": size[1]})
                    got += 1
                print(f"  {src} -> {dest_dir.name}: {got}/{n}", flush=True)
                if got >= n:
                    break
        return got

    abo_items = [(os.path.basename(k), f"{ABO}/{k}") for k in abo_keys]
    coco_items = [(f, COCO_IMG.format(f)) for f in coco_files]
    null = []
    accept("abo", abo_items[:], ROOT / "e2pos", N_ABO_POS, pos, 0)
    used_abo = {r["name"] for r in pos if r["source"] == "abo"}
    accept("coco", coco_items[:], ROOT / "e2pos", N_COCO_POS, pos, 0)
    used_coco = {r["name"] for r in pos if r["source"] == "coco"}
    accept("abo", [t for t in abo_items if t[0] not in used_abo], ROOT / "e2null", N_ABO_NULL, null, 0)
    accept("coco", [t for t in coco_items if t[0] not in used_coco], ROOT / "e2null", N_COCO_NULL, null, 0)

    for name, rows in (("pilot", pilot), ("e2pos", pos), ("e2null", null)):
        with open(MAN / f"owner_forensics_{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["set", "source", "name", "file", "w", "h"])
            w.writeheader()
            w.writerows(rows)
        by = {}
        for r in rows:
            by[r["source"]] = by.get(r["source"], 0) + 1
        print(name, len(rows), by)


if __name__ == "__main__":
    main()
