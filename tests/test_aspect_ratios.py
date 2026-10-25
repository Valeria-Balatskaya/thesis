# tests/test_aspect_ratios.py
# Part A3: is it worth adding Amazon's tall formats (6:7, 1:2) to the
# aspect-ratio retry of the blind read (src/hidden_v2_inference.ASPECT_RATIOS)?
#
#   gain   marketplace_square (centre crop to a square) on the 60 ABO evaluation
#          photos, watermarked at the app setting, read blind with the legacy
#          list vs the extended list
#   cost   every retry is one more decode that could land on a valid codeword by
#          chance. Measured: blind reads of clean, never-watermarked images
#          (COCO, ABO, own photos, 1,000 more ABO images, 500 procedural images).
#          Theoretical: attempts x per-read rate (src/payload.blind_false_positive_rate)
#
# Run: python tests/test_aspect_ratios.py
# Objectives 1.2.3 (robustness) and 1.2.5 (reliable detection).

import sys, os, csv, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image

from src.hidden_v2_inference import HiddenV2Model, ASPECT_RATIOS, LEGACY_ASPECT_RATIOS
from src.payload import blind_false_positive_rate
from src.distractor_gen import generate_distractor
from src import ecommerce_attacks as ea

OUT = "results/aspect_ratios"
os.makedirs(OUT, exist_ok=True)
os.makedirs("output", exist_ok=True)
model = HiddenV2Model("checkpoints/hidden_v2_best.pt")


def norm(src, dst, side=1024):
    img = Image.open(src).convert("RGB")
    img.thumbnail((side, side), Image.LANCZOS)
    img.save(dst, format="PNG")
    return img.size


# ─── gain ─────────────────────────────────────────────────────────
gain = []
for i, f in enumerate(sorted(os.listdir("data/abo_sample"))):
    orig = f"{OUT}/abo_{i:03d}.png"
    w, h = norm(f"data/abo_sample/{f}", orig)
    wm = f"{OUT}/abo_{i:03d}_wm.png"
    model.embed_product(orig, 2000 + i, wm, strength=0.7, mask_floor=0.3)
    sq = f"{OUT}/sq.png"
    ea.marketplace_square(wm, sq, target_size=512)
    legacy = model.read_product(sq, aspect_ratios=LEGACY_ASPECT_RATIOS)
    ext = model.read_product(sq, aspect_ratios=ASPECT_RATIOS)
    gain.append({"file": f, "aspect": round(w / h, 3),
                 "legacy_ok": int(legacy["product_id"] == 2000 + i),
                 "extended_ok": int(ext["product_id"] == 2000 + i),
                 "extended_via": ext["aspect"]})

# ─── cost ─────────────────────────────────────────────────────────
negatives = []
for folder in ["data/coco_sample", "data/abo_sample", "data/products_png", "data/abo_negatives"]:
    negatives += [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                  if f.lower().endswith((".png", ".jpg", ".jpeg"))]
for k in range(500):
    p = f"{OUT}/distractor_{k:03d}.png"
    if not os.path.exists(p):
        generate_distractor(f"aspect_neg_{k}", size=512, output_path=p)
    negatives.append(p)

t0 = time.time()
fp_legacy = fp_ext = 0
hits = []
for p in negatives:
    sus = f"{OUT}/neg.png"
    norm(p, sus)
    a = model.read_product(sus, aspect_ratios=LEGACY_ASPECT_RATIOS)["product_id"]
    b = model.read_product(sus, aspect_ratios=ASPECT_RATIOS)["product_id"]
    fp_legacy += a is not None
    fp_ext += b is not None
    if a is not None or b is not None:
        hits.append((p, a, b))
elapsed = time.time() - t0

per_read = blind_false_positive_rate()
n_leg, n_ext = len(LEGACY_ASPECT_RATIOS) + 1, len(ASPECT_RATIOS) + 1
n = len(negatives)
print(f"GAIN  marketplace_square on {len(gain)} ABO photos (app setting 0.7/0.3), blind:")
print(f"  legacy list   {sum(g['legacy_ok'] for g in gain)}/{len(gain)}")
print(f"  extended list {sum(g['extended_ok'] for g in gain)}/{len(gain)}")
for g in gain:
    if g["legacy_ok"] != g["extended_ok"] or not g["extended_ok"]:
        print(f"    {g['file']} aspect {g['aspect']}: legacy {g['legacy_ok']} extended "
              f"{g['extended_ok']} via {g['extended_via']}")
print(f"\nCOST  blind reads of {n} clean images ({elapsed:.0f}s):")
print(f"  false positives  legacy {fp_legacy}/{n}   extended {fp_ext}/{n}"
      f"   (95% upper bound for 0/{n}: {3 / n:.2%})")
print(f"  theoretical worst case per image: legacy {n_leg} x {per_read:.2e} = {n_leg * per_read:.2e}, "
      f"extended {n_ext} x {per_read:.2e} = {n_ext * per_read:.2e}  (+{(n_ext / n_leg - 1):.0%})")
for h in hits:
    print("  hit:", h)

with open("output/aspect_ratios_gain.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(gain[0].keys()))
    w.writeheader()
    w.writerows(gain)
with open("output/aspect_ratios_cost.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["list", "ratios", "attempts_per_image", "theoretical_fpr", "n_clean", "false_positives"])
    w.writerow(["legacy", len(LEGACY_ASPECT_RATIOS), n_leg, n_leg * per_read, n, fp_legacy])
    w.writerow(["extended", len(ASPECT_RATIOS), n_ext, n_ext * per_read, n, fp_ext])
