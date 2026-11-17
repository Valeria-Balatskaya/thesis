# tests/test_order_proof.py
# Order proof (Part B2): the suspect carries a foreign message m. Did the thief put m on OUR
# issued copy (our copy came first), or on a leaked clean original (theirs came first)?
# src/defence.order_proof rebuilds both stories and calibrates against 25 never-issued
# decoy copies (z >= ORDER_Z = 4 -> "our copy came first").
#
#   positive  thief system S embeds m on our published v3 copy      -> must say "ours first"
#   null      thief system S embeds m on the leaked clean original  -> must NOT say "ours first"
# S = ours_v3 (our model file, unregistered ID), TrustMark Q, WAM, Video Seal, at 1x and 2x
# their default strength; each clean and after JPEG q70. 40 photos (20 own + 20 ABO).
# The message: what S's own decoder reads from the suspect; if it reads nothing, the
# thief's claimed message (in a dispute the thief presents their mark) — both counted.
#
# Run: python tests/test_order_proof.py [n_own] [n_abo]     (default 20 20, ~1.5 h)

import csv
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image

from src import attacks
from src.baselines import TrustMarkBaseline, WAMBaseline, VideoSealBaseline
from src.defence import OursPeeler, order_proof, ORDER_Z
from src.hidden_v2_inference import HiddenV2Model
from src.stats import frac_ci

N_OWN = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N_ABO = int(sys.argv[2]) if len(sys.argv) > 2 else 20
N_DECOYS = 25
OUT, WORK = os.environ.get("OUT_DIR", "output/order_proof"), "results/order_proof"
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

model = HiddenV2Model("checkpoints/hidden_v3_best.pt")
ext = {"trustmark": TrustMarkBaseline(), "wam": WAMBaseline(), "videoseal": VideoSealBaseline()}
systems = {"ours_v3": OursPeeler(model), **ext}


def rgb(p):
    return np.asarray(Image.open(p).convert("RGB"))


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


def claimed(name, i):
    """The thief's own message (what they would present in a dispute)."""
    if name == "ours_v3":
        return 500_000 + i
    e = ext[name]
    return e.message(500_000 + i)


def thief_embed(name, src, dst, i, lvl):
    if name == "ours_v3":
        model.embed_product(src, 500_000 + i, dst, strength=lvl)
    else:
        e = ext[name]
        e.embed(src, dst, 500_000 + i, strength=e.default_strength * lvl)


rows = []
t0 = time.time()
for i, (tag, orig) in enumerate(images):
    cid = 1000 + i
    pub = f"{WORK}/pub.png"
    model.embed_product(orig, cid, pub, strength=1.0)
    o = rgb(orig)
    for name, S in systems.items():
        for lvl in (1.0, 2.0):
            for direction, src in (("on_our_copy", pub), ("on_clean_original", orig)):
                base = f"{WORK}/thief.png"
                thief_embed(name, src, base, i, lvl)
                for post, q in (("clean", None), ("jpeg_q70", 70)):
                    sus = f"{WORK}/sus.png"
                    Image.open(base).save(sus) if q is None else attacks.jpeg_compress(base, sus, q)
                    s = rgb(sus)
                    m = S.foreign_read(s)
                    read = m is not None
                    if not read:
                        m = claimed(name, i)
                    r = order_proof(model, s, o, cid, S, m, n_decoys=N_DECOYS)
                    rows.append({"system": name, "level": lvl, "direction": direction, "post": post,
                                 "set": tag, "image": i, "message_read": int(read), "t": r["t"], "z": r["z"],
                                 "scale": r["scale"], "ours_first": int(r["our_copy_first"])})
    if i % 5 == 4:
        print(f"{i + 1}/{len(images)} ({time.time() - t0:.0f}s)", flush=True)

with open(f"{OUT}/rows.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

n = len(images)
print(f"\nORDER PROOF — {n} photos ({N_OWN} own + {N_ABO} ABO), {N_DECOYS} decoys, decision z >= {ORDER_Z}\n")
print(f"{'thief system':<12}{'x':>4} {'post':<10}{'message read':<16}"
      f"{'thief on OUR copy -> ours first':<36}{'thief on CLEAN original -> ours first (false)':<46}max null z")
summary = []
for name in systems:
    for lvl in (1.0, 2.0):
        for post in ("clean", "jpeg_q70"):
            sub = lambda d: [r for r in rows if r["system"] == name and r["level"] == lvl
                             and r["post"] == post and r["direction"] == d]
            pos, nul = sub("on_our_copy"), sub("on_clean_original")
            s = {"system": name, "level": lvl, "post": post, "n": len(pos),
                 "message_read_pos": sum(r["message_read"] for r in pos),
                 "message_read_null": sum(r["message_read"] for r in nul),
                 "ours_first_pos": sum(r["ours_first"] for r in pos),
                 "ours_first_null": sum(r["ours_first"] for r in nul),
                 "z_median_pos": round(float(np.median([r["z"] for r in pos])), 2),
                 "z_max_null": round(float(max(r["z"] for r in nul)), 2),
                 "t_median_pos": round(float(np.median([r["t"] for r in pos])), 3),
                 "t_median_null": round(float(np.median([r["t"] for r in nul])), 3)}
            summary.append(s)
            print(f"{name:<12}{lvl:>4} {post:<10}{s['message_read_pos']}/{len(pos)}, {s['message_read_null']}/{len(nul)}".ljust(42)
                  + f"{frac_ci(s['ours_first_pos'], len(pos)):<36}{frac_ci(s['ours_first_null'], len(nul)):<46}"
                  f"{s['z_max_null']}")
with open(f"{OUT}/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)
allnull = [r for r in rows if r["direction"] == "on_clean_original"]
allpos = [r for r in rows if r["direction"] == "on_our_copy"]
print(f"\nall null cases: 'ours first' {frac_ci(sum(r['ours_first'] for r in allnull), len(allnull))}, "
      f"max z {max(r['z'] for r in allnull):.2f}; all positives: {frac_ci(sum(r['ours_first'] for r in allpos), len(allpos))}")
print(f"{OUT}/ ({time.time() - t0:.0f}s)")
