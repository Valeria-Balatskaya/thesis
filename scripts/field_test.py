# scripts/field_test.py
# Check a folder of images collected "in the wild" against the products
# protected in the TraceMark app (app/data/tracemark.db).
#
# Typical use: protect a few photos in the app, publish them somewhere real
# (a marketplace listing, a messenger chat, a social post), then download /
# screenshot / photograph them again and drop the results into sub-folders:
#
#   data/field/allegro_listing/      images saved from your own listing
#   data/field/mac_screenshots/      Cmd+Shift+4 captures
#   data/field/phone_photos/         phone photos of the screen
#   data/field/whatsapp/             images forwarded through a messenger
#   data/field/not_mine/             unrelated images (should all come back "none")
#
# If a file name starts with the copy ID it should trace to (e.g. 4097_shot1.png),
# the result is scored as correct / wrong. Otherwise it is only reported.
#
# Run: python scripts/field_test.py data/field

import csv
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PIL import Image

from app import guard, store

EXT = (".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif", ".bmp", ".gif")

if len(sys.argv) < 2:
    sys.exit("usage: python scripts/field_test.py <folder>")
root = sys.argv[1]
store.init_db()
if not store.list_products():
    sys.exit("No protected products in the app yet. Protect some images first.")
pipeline = guard.pipeline()

rows = []
tmp = guard.DATA / "scans" / "_field_tmp.png"
for folder, _, names in sorted(os.walk(root)):
    for name in sorted(names):
        if not name.lower().endswith(EXT):
            continue
        path = os.path.join(folder, name)
        try:
            guard._save_normalised(path, tmp)        # also converts HEIC / applies EXIF rotation
        except OSError:
            print(f"  unreadable: {path}")
            continue
        res = pipeline.analyse(str(tmp))
        expected = re.match(r"(\d{4,7})[_\-.]", name)
        expected = int(expected.group(1)) if expected else None
        if expected is None:
            score = ""
        elif res["copy_id"] == expected:
            score = "correct"
        elif res["copy_id"] is not None:
            score = "WRONG COPY"
        else:
            score = "not traced"
        retrieve = next((s for s in res["stages"] if s["stage"] == "retrieve"), {})
        rows.append({"source": os.path.relpath(folder, root), "file": name,
                     "size": "x".join(map(str, Image.open(tmp).size)),
                     "verdict": res["verdict"], "copy_id": res["copy_id"],
                     "product_id": res["product_id"], "expected_copy": expected, "score": score,
                     "inliers": retrieve.get("inliers"), "similarity": retrieve.get("similarity")})
tmp.unlink(missing_ok=True)

if not rows:
    sys.exit(f"No images found under {root}")

print(f"\n{'source':<24}{'file':<34}{'verdict':<20}{'copy':<8}{'score'}")
print("-" * 96)
for r in rows:
    print(f"{r['source'][:23]:<24}{r['file'][:33]:<34}{r['verdict']:<20}{str(r['copy_id'] or '-'):<8}{r['score']}")

print(f"\n{'source':<24}{'images':>7}{'traced':>8}{'picture only':>14}{'none':>6}{'correct':>9}{'wrong':>7}")
print("-" * 75)
for source in sorted({r["source"] for r in rows}):
    sub = [r for r in rows if r["source"] == source]
    print(f"{source[:23]:<24}{len(sub):>7}"
          f"{sum(r['copy_id'] is not None for r in sub):>8}"
          f"{sum(r['verdict'] == 'fingerprint_only' for r in sub):>14}"
          f"{sum(r['verdict'] == 'none' for r in sub):>6}"
          f"{sum(r['score'] == 'correct' for r in sub):>9}"
          f"{sum(r['score'] == 'WRONG COPY' for r in sub):>7}")

os.makedirs("output", exist_ok=True)
with open("output/field_test.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("\nCSV: output/field_test.csv")
