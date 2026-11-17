# tests/test_tiles.py
# Spatial redundancy experiment (Part B3, docs/spatial_redundancy.md), no training:
# v3 embedded per tile of a 2x2 grid (src/tiles.py: per-tile XOR mask, 10% overlap with a
# raised-cosine blend, soft combining of the tiles' logits) vs v3 on the whole image.
#
#   robustness   30 own + 30 ABO x the 23-attack suite: blind and full pipeline
#   visibility   PSNR, dE (whole image, flat background)
#   overwrite    a thief with our model file re-protects the copy at 1/1.5/2/3/4x, with
#                plain whole-image v3, and (tiled copies) with the tiled embedder as well:
#                our ID read blind / by the pipeline
#   false pos.   1,000 never-protected ABO photos: any ID from the blind reader
#                (tiled reader: 17 attempts per image, whole-image reader: 9)
# Decision rule of the design note (scaled from 90 to 60 photos): adopt only if
# offcentre_crop_50 >= 94% (85/90), jpeg_q30 and cdn_webp_360 drop <= 3 points and
# false positives stay 0/1,000.
#
# Run: python tests/test_tiles.py [n_own] [n_abo] [n_negatives]   (default 30 30 1000, ~2 h)

import csv
import os
import sys
import time
from shutil import copyfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image

from src.attack_suite import ATTACKS
from src.defence import DefencePipeline
from src.hidden_v2_inference import HiddenV2Model
from src.metrics import compute, visibility
from src.stats import pct_ci, frac_ci
from src.tiles import TiledModel

N_OWN = int(sys.argv[1]) if len(sys.argv) > 1 else 30
N_ABO = int(sys.argv[2]) if len(sys.argv) > 2 else 30
N_NEG = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
CKPT = "checkpoints/hidden_v3_best.pt"
OUT, WORK = os.environ.get("OUT_DIR", "output/tiles"), "results/tiles"
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

whole = HiddenV2Model(CKPT)
tiled = TiledModel(CKPT)
VARIANTS = {"whole": whole, "tiles_2x2": tiled}
pipes = {k: DefencePipeline(m) for k, m in VARIANTS.items()}

images = []
for tag, folder, n in [("own", "data/products_png", N_OWN), ("abo", "data/abo_sample", N_ABO)]:
    for k, f in enumerate(sorted(x for x in os.listdir(folder)
                                 if x.lower().endswith((".png", ".jpg", ".jpeg")))[:n]):
        p = f"{WORK}/{tag}_{k:03d}_orig.png"
        if not os.path.exists(p):
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            img.thumbnail((1024, 1024), Image.LANCZOS)
            img.save(p, format="PNG")
        images.append((tag, p))

pubs, quality = {}, []
for v, m in VARIANTS.items():
    for i, (tag, orig) in enumerate(images):
        pubs[(v, i)] = f"{WORK}/pub_{v}_{i:03d}.png"
        m.embed_product(orig, 1000 + i, pubs[(v, i)], strength=1.0)
        pipes[v].register(i, orig, [1000 + i])
        q, vis = compute(orig, pubs[(v, i)]), visibility(orig, pubs[(v, i)])
        quality.append({"variant": v, "set": tag, "image": i, "psnr": q["PSNR_dB"],
                        "dE": vis["dE_mean"], "dE_flat": vis["dE_flat"]})

rows, ow_rows = [], []
t0 = time.time()
for v, m in VARIANTS.items():
    for i, (tag, orig) in enumerate(images):
        cid = 1000 + i
        for atk, fn, akw in ATTACKS:
            sus = f"{WORK}/atk.png"
            np.random.seed(1000 * i + len(atk))
            copyfile(pubs[(v, i)], sus) if fn is None else fn(pubs[(v, i)], sus, **akw)
            b = m.read_product(sus)
            blind = b["product_id"] == cid
            res = None if blind else pipes[v].analyse(sus)
            rows.append({"variant": v, "set": tag, "image": i, "attack": atk, "blind": int(blind),
                         "full": int(blind or res["copy_id"] == cid),
                         "wrong": int(res is not None and res["copy_id"] not in (None, cid)),
                         "how": b.get("how") or ("direct" if blind and not b.get("aspect") else b.get("aspect") or "")})
        # thief A: our model file, plain whole-image embedding; thief B (tiled copies only):
        # also knows the tiling (the code is public) and re-protects tile by tile
        for thief, tm in (("whole", whole), ("tiled", tiled)):
            if thief == "tiled" and v == "whole":
                continue
            for lvl in (1.0, 1.5, 2.0, 3.0, 4.0):
                sus = f"{WORK}/thief.png"
                tm.embed_product(pubs[(v, i)], 500_000 + i, sus, strength=lvl)
                b = m.read_product(sus)
                blind = b["product_id"] == cid
                res = None if blind else pipes[v].analyse(sus)
                ow_rows.append({"variant": v, "thief": thief, "image": i, "level": lvl, "blind": int(blind),
                                "full": int(blind or res["copy_id"] == cid),
                                "thief_dE": round(visibility(orig, sus)["dE_mean"], 3)})
        if i % 10 == 9:
            print(f"{v}: {i + 1}/{len(images)} ({time.time() - t0:.0f}s)", flush=True)

negs = sorted(f for f in os.listdir("data/abo_negatives") if f.lower().endswith((".jpg", ".png")))[:N_NEG]
neg_rows = []
tmp = f"{WORK}/neg.png"
for k, f in enumerate(negs):
    img = Image.open(os.path.join("data/abo_negatives", f)).convert("RGB")
    img.thumbnail((1024, 1024), Image.LANCZOS)
    img.save(tmp, format="PNG")
    neg_rows.append({"file": f, **{f"id_{v}": (lambda r: "" if r["product_id"] is None else r["product_id"])(m.read_product(tmp))
                                   for v, m in VARIANTS.items()}})
    if k % 250 == 249:
        print(f"negatives {k + 1}/{len(negs)} ({time.time() - t0:.0f}s)", flush=True)

for name, data in (("rows", rows), ("overwrite", ow_rows), ("negatives", neg_rows), ("quality", quality)):
    with open(f"{OUT}/{name}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
        w.writeheader()
        w.writerows(data)

n_img = len(images)
print(f"\nTILES 2x2 vs WHOLE IMAGE — v3, {n_img} photos ({N_OWN} own + {N_ABO} ABO) x {len(ATTACKS)} attacks\n")
for v in VARIANTS:
    qs = [q for q in quality if q["variant"] == v]
    flat = [q["dE_flat"] for q in qs if q["dE_flat"] is not None]
    sub = [r for r in rows if r["variant"] == v]
    print(f"{v:<10} PSNR {np.mean([q['psnr'] for q in qs]):.1f}  dE {np.mean([q['dE'] for q in qs]):.2f}  "
          f"dE bg {np.mean(flat):.2f}   blind {pct_ci(sum(r['blind'] for r in sub), len(sub))}   "
          f"full {pct_ci(sum(r['full'] for r in sub), len(sub))}   wrong {sum(r['wrong'] for r in sub)}")
print(f"\n{'attack':<20}" + "".join(f"{v + ' full (blind)':<26}" for v in VARIANTS))
summary = []
for atk, _, _ in ATTACKS:
    line = f"{atk:<20}"
    for v in VARIANTS:
        sub = [r for r in rows if r["variant"] == v and r["attack"] == atk]
        kf, kb = sum(r["full"] for r in sub), sum(r["blind"] for r in sub)
        summary.append({"variant": v, "attack": atk, "n": len(sub), "full": kf, "blind": kb})
        line += f"{kf}/{len(sub)} ({kb})".ljust(26)
    print(line)
print("\nOverwrite by a thief with our model file (our ID read: full (blind), thief dE)")
for thief, v in (("whole", "whole"), ("whole", "tiles_2x2"), ("tiled", "tiles_2x2")):
    for lvl in (1.0, 1.5, 2.0, 3.0, 4.0):
        sub = [r for r in ow_rows if r["variant"] == v and r["thief"] == thief and r["level"] == lvl]
        kf, kb = sum(r["full"] for r in sub), sum(r["blind"] for r in sub)
        summary.append({"variant": v, "attack": f"overwrite_{thief}_{lvl}x", "n": len(sub), "full": kf, "blind": kb})
        print(f"  ours {v:<10} thief {thief:<6}{lvl:>4}x   {frac_ci(kf, len(sub)):<20}({kb})  "
              f"dE {np.mean([r['thief_dE'] for r in sub]):.2f}")
nn = len(neg_rows)
fps = {v: sum(r[f"id_{v}"] != "" for r in neg_rows) for v in VARIANTS}
for v in VARIANTS:
    summary.append({"variant": v, "attack": "false_positives", "n": nn, "full": "", "blind": fps[v]})
print(f"\nFalse positives on {nn} never-protected ABO photos (any blind ID): "
      + ", ".join(f"{v} {frac_ci(fps[v], nn)}" for v in VARIANTS))


def get(v, atk):
    return next(s for s in summary if s["variant"] == v and s["attack"] == atk)


crop = get("tiles_2x2", "offcentre_crop_50")
drops = {a: 100 * (get("whole", a)["full"] - get("tiles_2x2", a)["full"]) / get("whole", a)["n"]
         for a in ("jpeg_q30", "cdn_webp_360")}
ok = (crop["full"] / crop["n"] >= 85 / 90 and all(d <= 3 for d in drops.values()) and fps["tiles_2x2"] == 0)
print(f"\nDecision rule: crop 50 tiled {crop['full']}/{crop['n']} (need >= 94%), drops "
      + ", ".join(f"{a} {d:+.1f} pp" for a, d in drops.items()) + f" (need <= 3), FP {fps['tiles_2x2']}/{nn} (need 0)"
      + f"  ->  {'ADOPT' if ok else 'DO NOT ADOPT'}")
with open(f"{OUT}/summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)
print(f"\n{OUT}/ ({time.time() - t0:.0f}s)")
