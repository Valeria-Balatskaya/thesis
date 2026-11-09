# tests/test_overwrite_cost.py
# How much does a thief have to damage the image to overwrite our watermark?
# (objectives 1.2.3 robustness to malicious attacks, 1.2.5 ownership verification)
#
# A thief re-protects our published copy with the SAME model and a random ID, at
# increasing strength (1x, 1.5x, 2x, 3x, 4x the system's own strength). For each
# level we measure:
#   damage      what the thief's version looks like next to the original:
#               CIEDE2000 dE (whole image, flat background) and PSNR
#   blind read  whose ID the blind decoder returns: ours / the thief's / nothing
#   ownership   our calibrated informed check (z >= 4), plus for "v3+dct" the keyed
#               luminance layer (src/dct_layer.py); owned = either fires
# Systems: v2 (app setting 0.7 / flat 0.3), v3 (as trained), v3+dct (v3 + DCT layer),
# plus key-conditioned models from the command line (v4=..., adds v4+dct too).
# Thief: for v2/v3 the same model file (there is no key). For a keyed model two thieves:
#   same_key   an insider with our model AND our key (what the upload pre-check stops)
#   other_key  someone with our model file but not our key (leaked / open-sourced model)
# "thief's ID read" = the thief's ID decoded with the THIEF's key (their own claim).
# Rule of thumb for dE: < 1 not noticeable, 1-2 on close inspection, > 2 at a glance.
# Percentages come with 95% Wilson intervals (src/stats.py).
#
# Run: python tests/test_overwrite_cost.py [n_own] [n_abo] [name=checkpoint ...]
#      (default 20 20), e.g. ... 20 20 v4=checkpoints/hidden_v4_best.pt

import sys, os, csv, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image

from src.hidden_v2_inference import HiddenV2Model
from src.defence import InformedChecker, Z_THRESHOLD
from src.metrics import compute, visibility
from src import dct_layer
from src.stats import frac_ci

N_OWN = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N_ABO = int(sys.argv[2]) if len(sys.argv) > 2 else 20
LEVELS = [1.0, 1.5, 2.0, 3.0, 4.0]
SYSTEMS = {
    "v2":     ("checkpoints/hidden_v2_best.pt", {"strength": 0.7, "mask_floor": 0.3}, False),
    "v3":     ("checkpoints/hidden_v3_best.pt", {"strength": 1.0, "mask_floor": None}, False),
    "v3+dct": ("checkpoints/hidden_v3_best.pt", {"strength": 1.0, "mask_floor": None}, True),
}
for arg in sys.argv[3:]:
    name, ckpt = arg.split("=", 1)
    SYSTEMS[name] = (ckpt, {"strength": 1.0, "mask_floor": None}, False)
    SYSTEMS[name + "+dct"] = (ckpt, {"strength": 1.0, "mask_floor": None}, True)
if os.environ.get("ONLY"):                      # e.g. ONLY=v3,v4 for a quick run
    SYSTEMS = {k: v for k, v in SYSTEMS.items() if k in os.environ["ONLY"].split(",")}
OUT = "results/overwrite_cost"
os.makedirs(OUT, exist_ok=True)
os.makedirs("output", exist_ok=True)


def rgb(p):
    return np.asarray(Image.open(p).convert("RGB"))


images = []
for tag, folder, n in [("own", "data/products_png", N_OWN), ("abo", "data/abo_sample", N_ABO)]:
    for k, f in enumerate(sorted(x for x in os.listdir(folder)
                                 if x.lower().endswith((".png", ".jpg", ".jpeg")))[:n]):
        p = f"{OUT}/{tag}_{k:03d}_orig.png"
        if not os.path.exists(p):
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            img.thumbnail((1024, 1024), Image.LANCZOS)
            img.save(p, format="PNG")
        images.append((tag, p))

rows = []
t0 = time.time()
models = {}
for sysname, (ckpt, kw, use_dct) in SYSTEMS.items():
    model = models.setdefault(ckpt, HiddenV2Model(ckpt))
    thieves = ["same_key", "other_key"] if model.key_dim else ["same_model"]
    for i, (tag, orig) in enumerate(images):
        cid = 1000 + i
        pub = f"{OUT}/pub.png"
        model.embed_product(orig, cid, pub, **kw)
        ref = rgb(pub)
        if use_dct:
            Image.fromarray(dct_layer.embed(ref, cid)).save(pub)
        own_q = visibility(orig, pub)
        checker = InformedChecker(model, rgb(orig), cid, embed_kwargs=kw)
        for thief_kind in thieves:
            tkey = model.other_key(10_000 + i) if thief_kind == "other_key" else None
            for lvl in LEVELS:
                thief = f"{OUT}/thief.png"
                model.embed_product(pub, 500_000 + i, thief, strength=kw["strength"] * lvl,
                                    mask_floor=kw["mask_floor"], key=tkey)
                q, v = compute(orig, thief), visibility(orig, thief)
                got = model.read_product(thief)["product_id"]
                theirs = model.read_product(thief, key=tkey)["product_id"] == 500_000 + i
                s_ = rgb(thief)
                z = checker.test(s_)["z"]
                dct_hit = bool(use_dct and dct_layer.detect(s_, ref, cid, n_null=200)["detected"])
                rows.append({"system": sysname, "thief": thief_kind, "set": tag, "image": i,
                             "level": lvl, "ours_dE": own_q["dE_mean"], "thief_dE": v["dE_mean"],
                             "thief_dE_flat": v["dE_flat"], "thief_psnr": q["PSNR_dB"],
                             "blind": "ours" if got == cid else "theirs" if got == 500_000 + i
                                      else "nothing" if got is None else "other",
                             "thief_id_read": int(theirs),
                             "informed_z": z, "dct_hit": int(dct_hit),
                             "owned": int(z >= Z_THRESHOLD or dct_hit)})
    print(f"{sysname} done ({time.time() - t0:.0f}s)", flush=True)

with open("output/overwrite_cost.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

n = len(images)
print(f"\n{n} photos ({N_OWN} own + {N_ABO} ABO); thief = same model file, random ID\n")
print(f"{'system':<9}{'thief':<11}{'x':>4}{'dE':>6}{'dE flat':>8}{'PSNR':>6}   blind ours/theirs/none"
      f"   {'ownership proven':<18}  thief's ID read")
summary = []
for sysname in SYSTEMS:
    for thief_kind in dict.fromkeys(r["thief"] for r in rows if r["system"] == sysname):
        for lvl in LEVELS:
            sub = [r for r in rows if r["system"] == sysname and r["thief"] == thief_kind
                   and r["level"] == lvl]
            flat = [r["thief_dE_flat"] for r in sub if r["thief_dE_flat"] is not None]
            c = lambda v: sum(r["blind"] == v for r in sub)
            s = {"system": sysname, "thief": thief_kind, "level": lvl,
                 "thief_dE": float(np.mean([r["thief_dE"] for r in sub])),
                 "thief_dE_flat": float(np.mean(flat)) if flat else None,
                 "thief_psnr": float(np.mean([r["thief_psnr"] for r in sub])),
                 "ours": c("ours"), "theirs": c("theirs"), "nothing": c("nothing"),
                 "owned": sum(r["owned"] for r in sub),
                 "thief_id_read": sum(r["thief_id_read"] for r in sub), "n": len(sub)}
            summary.append(s)
            print(f"{sysname:<9}{thief_kind:<11}{lvl:>4}{s['thief_dE']:>6.2f}{s['thief_dE_flat'] or 0:>8.2f}"
                  f"{s['thief_psnr']:>6.1f}   {s['ours']:>4}/{s['theirs']}/{s['nothing']:<14}"
                  f"{frac_ci(s['owned'], s['n']):<18}  {frac_ci(s['thief_id_read'], s['n'])}")
for sysname in SYSTEMS:
    sub = [r["ours_dE"] for r in rows if r["system"] == sysname and r["level"] == 1.0]
    print(f"  our own published copy, {sysname}: dE {np.mean(sub):.2f}")
with open("output/overwrite_cost_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
palette = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#c2417a", "#7a7a7a", "#d1a72a"]
fig, ax = plt.subplots(figsize=(8.5, 4.6))
curves = list(dict.fromkeys((s["system"], s["thief"]) for s in summary))
for k, (sysname, thief_kind) in enumerate(curves):
    pts = [s for s in summary if s["system"] == sysname and s["thief"] == thief_kind]
    label = sysname if thief_kind == "same_model" else f"{sysname} ({thief_kind.replace('_', ' ')})"
    col = palette[k % len(palette)]
    ax.plot([p["thief_dE"] for p in pts], [100 * p["owned"] / p["n"] for p in pts], marker="o",
            linewidth=2, color=col, label=f"{label}: ownership proven")
    ax.plot([p["thief_dE"] for p in pts], [100 * p["thief_id_read"] / p["n"] for p in pts],
            marker="o", linewidth=1.2, linestyle="--", color=col, label=f"{label}: thief's ID read")
ax.axvline(2.0, color="#52514e", linestyle=":", linewidth=1)
ax.text(2.03, 50, "visible at a glance (dE > 2)", fontsize=8, color="#52514e")
ax.set_xlabel("damage of the thief's version vs the original (CIEDE2000 dE)")
ax.set_ylabel("% of photos")
ax.set_title("Overwriting our watermark costs visible damage", loc="left", fontweight="bold", fontsize=11)
ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
ax.grid(color="#e4e3df")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig("output/overwrite_cost.png", dpi=200)
print(f"\nCSV: output/overwrite_cost.csv, chart: output/overwrite_cost.png ({time.time() - t0:.0f}s)")
