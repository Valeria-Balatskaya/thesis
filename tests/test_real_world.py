# tests/test_real_world.py
# Real-world capture dataset (docs/real_world_protocol.md): genuine screenshots, phone
# photos and messenger round trips instead of simulations (objectives 1.2.3, 1.4).
#
# Reads data/real_world/manifest.csv (from scripts/prepare_real_world.py) and every
# capture in data/real_world/<capture_type>/<copy_id>_<n>.<ext> (filed by
# scripts/sort_captures.py). Each model has its own pipeline with its 20 products +
# the 10 controls registered (every product with 2 copies: the published one and an
# unpublished decoy, so a wrong-copy answer is possible).
#
# Per capture type and model:
#   blind      read_product returns the published copy's ID
#   full       Retrieve-Align-Verify returns it
#   wrong      any other copy ID (decoy, another product) = misattribution
#   verdicts   watermark / watermark_aligned / fingerprint_only / none
# Controls (clean originals of registered products) and x_* unrelated photos are run
# through every model's pipeline: any ID or watermark verdict is a FALSE POSITIVE.
# --informed also runs the calibrated informed check on fingerprint_only results
# (watermark_overwritten verdict; slow: ~15 s per product).
#
# Run: python tests/test_real_world.py [--root data/real_world] [--informed]
#      Writes output/real_world/{rows,summary}.csv and output/real_world/real_world.png

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

from src.hidden_v2_inference import HiddenV2Model
from src.defence import DefencePipeline, InformedChecker, ownership_check, mark_overwritten
from src.stats import pct_ci, wilson

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/real_world")
ap.add_argument("--informed", action="store_true")
args = ap.parse_args()
ROOT = args.root
OUT = "output/real_world"
os.makedirs(OUT, exist_ok=True)
CKPT = {"v3": "checkpoints/hidden_v3_best.pt", "v4": "checkpoints/hidden_v4_best.pt"}
ORDER = ["screenshot_100", "screenshot_zoomed", "page_screenshot", "shop_grid_crop",
         "phone_photo_front", "phone_photo_angle", "whatsapp", "telegram", "instagram", "print_photo"]
EXT = (".png", ".jpg", ".jpeg", ".heic", ".heif", ".webp")

items = list(csv.DictReader(open(f"{ROOT}/manifest.csv")))
by_copy = {int(r["copy_id"]): r for r in items}
models = sorted({r["model"] for r in items if r["kind"] == "protected"})
models = [m for m in models if os.path.exists(CKPT[m])]
pipes, nets = {}, {}
for m in models:
    nets[m] = HiddenV2Model(CKPT[m])
    pipes[m] = DefencePipeline(nets[m])
    for pid, r in enumerate(items):
        if r["model"] in (m, "none"):
            pipes[m].register(pid, f"{ROOT}/originals/{r['item']}.png",
                              [int(r["copy_id"]), int(r["decoy_copy_id"])])
print(f"models {models}; {len(items)} registered items")

tmp = f"{OUT}/_capture.png"
checkers = {}


def normalise(path):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((1600, 1600), Image.LANCZOS)       # same as the app's upload handling
    img.save(tmp, format="PNG")
    return img.size


def informed(m, res):
    if res["verdict"] != "fingerprint_only" or res["aligned"] is None:
        return res
    r = items[res["product_id"]]
    o = np.asarray(Image.open(f"{ROOT}/originals/{r['item']}.png").convert("RGB"))
    ev = []
    for cid in (int(r["copy_id"]), int(r["decoy_copy_id"])):
        key = (m, cid)
        if key not in checkers:
            checkers[key] = InformedChecker(nets[m], o, cid, embed_kwargs={"strength": 1.0, "mask_floor": None})
        c = ownership_check(nets[m], res["aligned"], o, cid, checker=checkers[key])
        ev.append({"copy_id": cid, "owned": c["owned"], "dct_sim": None,
                   "informed_z": c["informed"]["z"] if c["informed"] else None})
    return mark_overwritten(res, ev)


rows = []
types = [t for t in ORDER if os.path.isdir(f"{ROOT}/{t}")] + sorted(
    t for t in os.listdir(ROOT) if os.path.isdir(f"{ROOT}/{t}") and t not in ORDER
    and t not in ("originals", "to_capture") and not t.startswith("_"))
for t in types:
    for name in sorted(os.listdir(f"{ROOT}/{t}")):
        if not name.lower().endswith(EXT):
            continue
        head = name.split("_")[0]
        cid = int(head) if head.isdigit() else None
        r = by_copy.get(cid)
        kind = "unrelated" if r is None else r["kind"]
        size = normalise(f"{ROOT}/{t}/{name}")
        for m in ([r["model"]] if kind == "protected" and r["model"] in models else
                  [] if kind == "protected" else models):
            blind = nets[m].read_product(tmp)["product_id"]
            res = pipes[m].analyse(tmp)
            if args.informed:
                res = informed(m, res)
            got = res["copy_id"] if res["copy_id"] is not None else blind
            rows.append({"capture_type": t, "file": name, "kind": kind, "model": m,
                         "set": r["set"] if r else "", "w": size[0], "h": size[1],
                         "blind": int(kind == "protected" and blind == cid),
                         "full": int(kind == "protected" and res["copy_id"] == cid
                                     and res["verdict"] in ("watermark", "watermark_aligned")),
                         "wrong": int(kind == "protected" and got is not None and got != cid),
                         "false_positive": int(kind != "protected" and (
                             blind is not None or res["verdict"] in
                             ("watermark", "watermark_aligned", "watermark_overwritten"))),
                         "verdict": res["verdict"], "copy_read": got})
    print(f"{t}: {sum(1 for x in rows if x['capture_type'] == t)} reads", flush=True)

if not rows:
    sys.exit(f"no captures under {ROOT}/<capture_type>/ yet (see docs/real_world_protocol.md)")
with open(f"{OUT}/rows.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

summary = []
print(f"\n{'capture type':<20}{'model':<6}{'n':>4}  {'blind':<28}{'full pipeline':<28}{'wrong':>6}  verdicts"
      f"   | controls: false positives")
for t in types:
    for m in models:
        sub = [x for x in rows if x["capture_type"] == t and x["model"] == m and x["kind"] == "protected"]
        neg = [x for x in rows if x["capture_type"] == t and x["model"] == m and x["kind"] != "protected"]
        if not sub and not neg:
            continue
        v = defaultdict(int)
        for x in sub:
            v[x["verdict"]] += 1
        s = {"capture_type": t, "model": m, "n": len(sub),
             "blind": sum(x["blind"] for x in sub), "full": sum(x["full"] for x in sub),
             "wrong": sum(x["wrong"] for x in sub), "n_controls": len(neg),
             "false_positives": sum(x["false_positive"] for x in neg), **{f"v_{k}": n for k, n in v.items()}}
        summary.append(s)
        print(f"{t:<20}{m:<6}{len(sub):>4}  {pct_ci(s['blind'], len(sub)):<28}{pct_ci(s['full'], len(sub)):<28}"
              f"{s['wrong']:>6}  " + " ".join(f"{k[:11]} {n}" for k, n in sorted(v.items()))
              + f"   | {s['false_positives']}/{len(neg)}")
for m in models:
    neg = [x for x in rows if x["model"] == m and x["kind"] != "protected"]
    pos = [x for x in rows if x["model"] == m and x["kind"] == "protected"]
    print(f"\n{m}: all captures blind {pct_ci(sum(x['blind'] for x in pos), len(pos))}, "
          f"full {pct_ci(sum(x['full'] for x in pos), len(pos))}, wrong copy {sum(x['wrong'] for x in pos)}; "
          f"false positives {pct_ci(sum(x['false_positive'] for x in neg), len(neg), 2)}")
keys = sorted({k for s in summary for k in s})
with open(f"{OUT}/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(summary)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
shown = [t for t in types if any(s["capture_type"] == t and s["n"] for s in summary)]
fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(shown) + 2), 4.2))
width = 0.8 / (2 * len(models))
colours = {"v3": ("#9cc1ec", "#2a78d6"), "v4": ("#f3b79b", "#eb6834")}
for k, m in enumerate(models):
    for j, col in enumerate(("blind", "full")):
        xs, ys, lo, hi = [], [], [], []
        for n, t in enumerate(shown):
            s = next((s for s in summary if s["capture_type"] == t and s["model"] == m), None)
            if not s or not s["n"]:
                continue
            p = s[col] / s["n"]
            a, b = wilson(s[col], s["n"])
            xs.append(n + (2 * k + j - len(models) + 0.5) * width)
            ys.append(100 * p)
            lo.append(100 * (p - a))
            hi.append(100 * (b - p))
        ax.bar(xs, ys, width, yerr=[lo, hi], capsize=2, color=colours.get(m, ("#ccc", "#777"))[j],
               label=f"{m} {'blind read' if col == 'blind' else 'full pipeline'}", error_kw={"linewidth": 0.8})
ax.set_xticks(range(len(shown)))
ax.set_xticklabels([t.replace("_", "\n") for t in shown], fontsize=8)
ax.set_ylabel("% of captures traced to the right copy")
ax.set_ylim(0, 105)
ax.set_title("Real captures (not simulated), 95% Wilson intervals", loc="left", fontsize=11, fontweight="bold")
ax.legend(frameon=False, fontsize=8, ncol=2)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/real_world.png", dpi=200)
os.remove(tmp)
print(f"\n{OUT}/rows.csv, summary.csv, real_world.png")
