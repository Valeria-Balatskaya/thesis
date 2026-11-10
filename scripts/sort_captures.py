# scripts/sort_captures.py
# File captures for the real-world dataset without renaming anything by hand.
#
#   python scripts/sort_captures.py <capture_type> <inbox_folder> [--device iPhone15] [--app WhatsApp] [--notes "..."]
#
# Every image in <inbox_folder> (PNG/JPEG/HEIC/WebP, any name) is identified by its
# picture: SIFT + RANSAC against the 50 registered originals in data/real_world/
# (also mirrored). This uses only the picture, never the watermark, so it does not
# bias the watermark results. Identified files are COPIED, byte for byte, to
#   data/real_world/<capture_type>/<copy_id>_<n>.<ext>
# and a row is appended to data/real_world/captures.csv. Files that can't be identified
# safely (too few matching points, or two originals match about equally) go to
# data/real_world/<capture_type>/_unidentified/. Rename those yourself to
# <copy_id>_<anything>.<ext>, move them up one folder, and run with --manual to log them.
# Unrelated photos you add as extra negatives: name them x_<anything>.<ext>.
#
# Capture types used in docs/real_world_protocol.md:
#   screenshot_100, screenshot_zoomed, page_screenshot, shop_grid_crop,
#   phone_photo_front, phone_photo_angle, whatsapp, telegram, instagram, print_photo

import argparse
import csv
import datetime
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

from src.defence import fingerprint, match

ROOT = "data/real_world"
EXT = (".png", ".jpg", ".jpeg", ".heic", ".heif", ".webp")
FIELDS = ["file", "capture_type", "copy_id", "item", "gt_source", "inliers", "runner_up",
          "width", "height", "device", "app", "notes", "original_name", "added"]
MIN_INLIERS, MARGIN = 15, 2.0


def load(path):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((1600, 1600), Image.LANCZOS)
    return np.asarray(img)


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_type")
    ap.add_argument("inbox")
    ap.add_argument("--device", default="")
    ap.add_argument("--app", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--root", default="data/real_world", help="dataset folder (default data/real_world)")
    ap.add_argument("--manual", action="store_true",
                    help="log files already named <copy_id>_... in the capture folder")
    a = ap.parse_args()
    ROOT = a.root
    items = list(csv.DictReader(open(f"{ROOT}/manifest.csv")))
    by_copy = {int(r["copy_id"]): r for r in items}
    dest = f"{ROOT}/{a.capture_type}"
    os.makedirs(f"{dest}/_unidentified", exist_ok=True)
    log = f"{ROOT}/captures.csv"
    logged = {r["file"] for r in csv.DictReader(open(log))} if os.path.exists(log) else set()
    new_rows = []
    now = datetime.datetime.now().isoformat(timespec="seconds")

    if a.manual:
        for name in sorted(os.listdir(dest)):
            rel = f"{a.capture_type}/{name}"
            if not name.lower().endswith(EXT) or rel in logged:
                continue
            head = name.split("_")[0]
            cid = int(head) if head.isdigit() else None
            img = Image.open(f"{dest}/{name}")
            new_rows.append({"file": rel, "capture_type": a.capture_type, "copy_id": cid or "",
                             "item": by_copy[cid]["item"] if cid in by_copy else "unrelated",
                             "gt_source": "manual", "inliers": "", "runner_up": "",
                             "width": img.width, "height": img.height, "device": a.device,
                             "app": a.app, "notes": a.notes, "original_name": name, "added": now})
    else:
        fps = {int(r["copy_id"]): fingerprint(load(f"{ROOT}/originals/{r['item']}.png")) for r in items}
        counts = {}
        for name in sorted(os.listdir(dest)):
            head = name.split("_")[0]
            if head.isdigit():
                counts[int(head)] = counts.get(int(head), 0) + 1
        for name in sorted(os.listdir(a.inbox)):
            src = os.path.join(a.inbox, name)
            if not name.lower().endswith(EXT) or not os.path.isfile(src):
                continue
            ext = os.path.splitext(name)[1].lower()
            if name.startswith("x_"):
                out = f"{dest}/{name}"
                shutil.copy2(src, out)
                cid, item, inl, second = "", "unrelated", "", ""
            else:
                img = load(src)
                scores = {}
                for flipped in (False, True):
                    fp = fingerprint(img[:, ::-1].copy() if flipped else img)
                    for cid_, ref in fps.items():
                        m = match(fp, ref)
                        if m:
                            scores[cid_] = max(scores.get(cid_, 0), m["inliers"])
                ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                best = ranked[0] if ranked else (None, 0)
                second = ranked[1][1] if len(ranked) > 1 else 0
                if best[1] < MIN_INLIERS or best[1] < MARGIN * second:
                    shutil.copy2(src, f"{dest}/_unidentified/{name}")
                    print(f"  ?  {name}: not identified (best {best[1]} points, runner-up {second})")
                    continue
                cid, inl = best
                item = by_copy[cid]["item"]
                counts[cid] = counts.get(cid, 0) + 1
                out = f"{dest}/{cid}_{counts[cid]}{ext}"
                shutil.copy2(src, out)
                print(f"  ok {name} -> {os.path.basename(out)} ({item}, {inl} points)")
            w, h = Image.open(out).size
            new_rows.append({"file": os.path.relpath(out, ROOT), "capture_type": a.capture_type,
                             "copy_id": cid, "item": item, "gt_source": "picture match" if cid else "unrelated",
                             "inliers": inl, "runner_up": second, "width": w, "height": h,
                             "device": a.device, "app": a.app, "notes": a.notes,
                             "original_name": name, "added": now})
    exists = os.path.exists(log)
    with open(log, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            wr.writeheader()
        wr.writerows(new_rows)
    print(f"{len(new_rows)} captures logged in {log}; unidentified ones in {dest}/_unidentified/")


if __name__ == "__main__":
    main()
