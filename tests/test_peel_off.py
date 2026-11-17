# tests/test_peel_off.py
# Peel-off (Part B1): when someone's watermark sits on top of ours, regenerate it, subtract
# it, and read ours again (src/defence.DefencePipeline(peelers=...), verdict watermark_peeled).
#
# Victim: ours v3 as deployed (strength 1.0, mirror retry on). 40 photos (20 own + 20 ABO).
# Thieves on top of our published copy, then clean and JPEG q70:
#   ours_v3 at 1/1.5/2/3/4x (our model file, a random unregistered ID)
#   TrustMark Q, Watermark Anything, Video Seal at 1/2/3/4x their default strength
#   stacked: TrustMark then WAM (1x each); ours_v3 twice (1x each, two IDs)
# Per suspect: our ID read blind / by the full pipeline WITHOUT peel-off (before) and WITH
# it (after), and which foreign marks peel-off found.
# False positives: the pipeline with peel-off on
#   (a) every never-protected image of tests/test_mirror_fp.py (1,000 ABO negatives, the
#       clean originals of the own/ABO/COCO sets, the spare photos): 1,167 images
#   (b) the 40 registered products' clean originals carrying ONLY a foreign mark
#       (ours_v3 with an unregistered ID, TrustMark, WAM, Video Seal), clean and JPEG q70
# Any registered copy ID returned = false positive. Also: how often each foreign decoder
# says "present" on (a) (Video Seal's logit threshold is checked here).
#
# Run: python tests/test_peel_off.py [n_own] [n_abo] [n_negatives]   (default 20 20 1000, ~3 h)

import csv
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image

from src import attacks
from src.baselines import TrustMarkBaseline, WAMBaseline, VideoSealBaseline
from src.defence import DefencePipeline, OursPeeler
from src.hidden_v2_inference import HiddenV2Model
from src.metrics import visibility
from src.stats import frac_ci, pct_ci

N_OWN = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N_ABO = int(sys.argv[2]) if len(sys.argv) > 2 else 20
N_NEG = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
OUT, WORK = "output/peel_off", "results/peel_off"
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)
EXT = (".png", ".jpg", ".jpeg")

model = HiddenV2Model("checkpoints/hidden_v3_best.pt")
model.try_mirror = True                          # as in the app
ext = {"trustmark": TrustMarkBaseline(), "wam": WAMBaseline(), "videoseal": VideoSealBaseline()}
peelers = [OursPeeler(model)] + list(ext.values())
plain, peel = DefencePipeline(model), DefencePipeline(model, peelers)


def files(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(EXT))


def prep(src, dst):
    if not os.path.exists(dst):
        img = Image.open(src).convert("RGB")
        img.thumbnail((1024, 1024), Image.LANCZOS)
        img.save(dst, format="PNG")
    return dst


images = []
for tag, folder, n in [("own", "data/products_png", N_OWN), ("abo", "data/abo_sample", N_ABO)]:
    for k, f in enumerate(files(folder)[:n]):
        images.append((tag, prep(os.path.join(folder, f), f"{WORK}/{tag}_{k:03d}_orig.png")))
pubs = {}
for i, (tag, orig) in enumerate(images):
    pubs[i] = f"{WORK}/pub_{i:03d}.png"
    model.embed_product(orig, 1000 + i, pubs[i], strength=1.0)
    for p in (plain, peel):
        p.register(i, orig, [1000 + i])


def thief_ours(lvl, tid_offset=0):
    return lambda src, dst, i: model.embed_product(src, 500_000 + tid_offset + i, dst, strength=lvl)


def thief_ext(name, lvl):
    e = ext[name]
    return lambda src, dst, i: e.embed(src, dst, 500_000 + i, strength=e.default_strength * lvl)


def chain(*fns):
    def run(src, dst, i):
        cur = src
        for k, fn in enumerate(fns):
            nxt = f"{dst}.layer{k}.png"
            fn(cur, nxt, i)
            cur = nxt
        Image.open(cur).save(dst, format="PNG")
    return run


SCEN = [(f"ours_v3_{l}x", "ours_v3", l, thief_ours(l)) for l in (1.0, 1.5, 2.0, 3.0, 4.0)]
for name in ext:
    SCEN += [(f"{name}_{l}x", name, l, thief_ext(name, l)) for l in (1.0, 2.0, 3.0, 4.0)]
SCEN += [("stacked_trustmark_wam", "stacked", 1.0, chain(thief_ext("trustmark", 1.0), thief_ext("wam", 1.0))),
         ("stacked_ours_ours", "stacked", 1.0, chain(thief_ours(1.0), thief_ours(1.0, 1000)))]
POSTS = [("clean", None), ("jpeg_q70", 70)]
SMOKE = os.environ.get("SMOKE") == "1"           # tiny run to check the script
if SMOKE:
    SCEN = SCEN[1:2] + SCEN[6:7] + SCEN[-2:]
    OUT = WORK = "results/smoke"
    os.makedirs(WORK, exist_ok=True)

rows = []
t0 = time.time()
for scen, thief, lvl, fn in SCEN:
    for i, (tag, orig) in enumerate(images):
        cid = 1000 + i
        base = f"{WORK}/thief.png"
        fn(pubs[i], base, i)
        dE = visibility(orig, base)["dE_mean"]
        for post, q in POSTS:
            sus = f"{WORK}/sus.png"
            Image.open(base).save(sus) if q is None else attacks.jpeg_compress(base, sus, q)
            b = plain.analyse(sus)
            a = b if b["verdict"] == "watermark" else peel.analyse(sus)
            pe = a.get("peeled") or {}
            found = [st.get("found") for st in a["stages"] if st["stage"].startswith("peel")]
            rows.append({"scenario": scen, "thief": thief, "level": lvl, "set": tag, "image": i, "post": post,
                         "thief_dE": round(dE, 3),
                         "blind_before": int(b["verdict"] == "watermark" and b["copy_id"] == cid),
                         "pipeline_before": int(b["copy_id"] == cid),
                         "verdict_before": b["verdict"],
                         "blind_after": int(a["copy_id"] == cid and (a["verdict"] == "watermark"
                                                                    or pe.get("stage") == "blind")),
                         "pipeline_after": int(a["copy_id"] == cid),
                         "wrong_after": int(a["copy_id"] is not None and a["copy_id"] != cid),
                         "verdict_after": a["verdict"], "peel_stage": pe.get("stage", ""),
                         "peel_systems": "+".join(pe.get("systems", [])), "peel_scale": pe.get("scale", ""),
                         "found": "|".join("+".join(x["system"] for x in f) for f in found if f)})
    print(f"{scen:<24} done ({time.time() - t0:.0f}s)", flush=True)

# ─── false positives ──────────────────────────────────────────────
neg = [("abo_negative", os.path.join("data/abo_negatives", f)) for f in files("data/abo_negatives")[:N_NEG]]
for tag, folder in [("own", "data/products_png"), ("abo", "data/abo_sample"), ("coco", "data/coco_sample")]:
    neg += [(f"{tag}_photo", os.path.join(folder, f)) for f in files(folder)]
if SMOKE:
    neg = neg[:3] + neg[-3:]
neg_rows = []
tmp = f"{WORK}/neg.png"
for k, (kind, path) in enumerate(neg):
    img = Image.open(path).convert("RGB")
    img.thumbnail((1024, 1024), Image.LANCZOS)
    img.save(tmp, format="PNG")
    x = np.asarray(img)
    present = {p.name: int(p.foreign_read(x) is not None) for p in peelers}
    r = peel.analyse(tmp)
    neg_rows.append({"kind": kind, "file": os.path.basename(path), "verdict": r["verdict"],
                     "claim": int(r["copy_id"] is not None), "peel_claim": int(r["verdict"] == "watermark_peeled"),
                     **{f"present_{n}": v for n, v in present.items()}})
    if k % 200 == 199:
        print(f"negatives {k + 1}/{len(neg)} ({time.time() - t0:.0f}s)", flush=True)

foreign_rows = []
FOREIGN = [("ours_v3", thief_ours(1.0))] + [(n, thief_ext(n, 1.0)) for n in ext]
for name, fn in FOREIGN:
    for i, (tag, orig) in enumerate(images):
        base = f"{WORK}/foreign.png"
        fn(orig, base, i)
        for post, q in POSTS:
            sus = f"{WORK}/sus.png"
            Image.open(base).save(sus) if q is None else attacks.jpeg_compress(base, sus, q)
            r = peel.analyse(sus)
            foreign_rows.append({"foreign": name, "post": post, "image": i, "verdict": r["verdict"],
                                 "claim": int(r["copy_id"] is not None),
                                 "peel_claim": int(r["verdict"] == "watermark_peeled")})
    print(f"foreign mark on clean originals: {name} done ({time.time() - t0:.0f}s)", flush=True)

for name, data in (("rows", rows), ("negatives", neg_rows), ("foreign_on_original", foreign_rows)):
    with open(f"{OUT}/{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader()
        w.writerows(data)

# ─── summary ──────────────────────────────────────────────────────
n = len(images)
print(f"\nPEEL-OFF — victim ours v3, {n} photos ({N_OWN} own + {N_ABO} ABO)\n")
print(f"{'scenario':<24}{'post':<10}{'dE':>6}   {'blind before':<20}{'blind after':<20}"
      f"{'pipeline before':<20}{'pipeline after':<20}wrong")
summary = []
for scen, thief, lvl, _ in SCEN:
    for post, _ in POSTS:
        sub = [r for r in rows if r["scenario"] == scen and r["post"] == post]
        k = {c: sum(r[c] for r in sub) for c in ("blind_before", "blind_after", "pipeline_before",
                                                   "pipeline_after", "wrong_after")}
        s = {"scenario": scen, "thief": thief, "level": lvl, "post": post, "n": len(sub),
             "thief_dE": round(float(np.mean([r["thief_dE"] for r in sub])), 2), **k,
             "peeled": sum(r["verdict_after"] == "watermark_peeled" for r in sub),
             "found_any": sum(bool(r["found"]) for r in sub)}
        summary.append(s)
        print(f"{scen:<24}{post:<10}{s['thief_dE']:>6.2f}   {frac_ci(k['blind_before'], len(sub)):<20}"
              f"{frac_ci(k['blind_after'], len(sub)):<20}{frac_ci(k['pipeline_before'], len(sub)):<20}"
              f"{frac_ci(k['pipeline_after'], len(sub)):<20}{k['wrong_after']}")
with open(f"{OUT}/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)

nn = len(neg_rows)
print(f"\nFALSE POSITIVES, pipeline with peel-off ({N_OWN + N_ABO} products registered)")
print(f"  never-protected images: any registered ID {pct_ci(sum(r['claim'] for r in neg_rows), nn, 2)}, "
      f"via peel-off {sum(r['peel_claim'] for r in neg_rows)}/{nn}")
for p in peelers:
    c = sum(r[f"present_{p.name}"] for r in neg_rows)
    print(f"    {p.name:<12} decoder says 'present' on {pct_ci(c, nn, 2)}")
nf = len(foreign_rows)
print(f"  clean originals with only a foreign mark: any registered ID {pct_ci(sum(r['claim'] for r in foreign_rows), nf, 2)}, "
      f"via peel-off {sum(r['peel_claim'] for r in foreign_rows)}/{nf}")
for name, _ in FOREIGN:
    sub = [r for r in foreign_rows if r["foreign"] == name]
    print(f"    {name:<12} {sum(r['claim'] for r in sub)}/{len(sub)}")
print(f"\n{OUT}/ ({time.time() - t0:.0f}s)")
