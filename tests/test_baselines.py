# tests/test_baselines.py
# Our system vs pretrained robust watermarks (Part 4; objectives 1.2.2, 1.2.3, 1.2.4, 1.4).
#
# Systems (src/baselines.py): ours (HiDDeN v3, and v4 if present), Adobe TrustMark (Q),
# Meta Watermark Anything (MIT weights). Each at
#   default   the system's own default strength
#   matched   strength tuned so the mean CIEDE2000 dE (whole image) on a calibration subset
#             (first 5 photos of each set) equals ours (v3) -> same visibility budget
# Data: own / ABO / COCO, n per set (default 30), longer side 1024 px. Same 23 attacks as
# the v3 results (src/attack_suite.py).
# Metrics: PSNR, SSIM, dE (whole, background), embed / extract time, false detections on
# the clean originals (each system's native decision), per attack: exact message read
#   blind     read the attacked file directly
#   +RAV      if blind fails: Retrieve (SIFT/RANSAC over all registered originals) -> Align
#             (warp into the original's frame, uncovered pixels from the clean original)
#             -> read again with the SAME system's native decoder. This is our pipeline
#             wrapped around another model, to show it is a contribution of its own.
#   ours also gets the deployed DefencePipeline (adds codeword matching after alignment).
# Rows are appended to output/baselines/rows.csv as they are computed; re-running resumes.
#
# Run: python tests/test_baselines.py [n_per_set]     (default 30; ~3-4 h on an M1 CPU)
#      SYSTEMS=v3,trustmark,wam  SETTINGS=default,matched  to restrict

import csv
import os
import sys
import time
from shutil import copyfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image

from src.attack_suite import ATTACKS, GEOMETRIC
from src.baselines import Ours, TrustMarkBaseline, WAMBaseline
from src.defence import DefencePipeline, fingerprint, match, align
from src.metrics import compute, visibility
from src.stats import pct_ci

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SETS = [("own", "data/products_png"), ("abo", "data/abo_sample"), ("coco", "data/coco_sample")]
OUT = "output/baselines"
WORK = "results/baselines"
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)
want = os.environ.get("SYSTEMS", "v3,v4,trustmark,wam").split(",")
settings_wanted = os.environ.get("SETTINGS", "default,matched").split(",")


def rgb(p):
    return np.asarray(Image.open(p).convert("RGB"))


images = []
for tag, folder in SETS:
    for k, f in enumerate(sorted(x for x in os.listdir(folder)
                                 if x.lower().endswith((".png", ".jpg", ".jpeg")))[:N]):
        p = f"{WORK}/{tag}_{k:03d}_orig.png"
        if not os.path.exists(p):
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            img.thumbnail((1024, 1024), Image.LANCZOS)
            img.save(p, format="PNG")
        images.append((tag, p))
calib = [i for i, (tag, _) in enumerate(images) if i % N < 5]

systems = {}
if "v3" in want:
    systems["v3"] = Ours("checkpoints/hidden_v3_best.pt", label="ours_v3")
if "v4" in want and os.path.exists("checkpoints/hidden_v4_best.pt"):
    systems["v4"] = Ours("checkpoints/hidden_v4_best.pt", label="ours_v4")
for key, cls in (("trustmark", TrustMarkBaseline), ("wam", WAMBaseline)):
    if key in want:
        try:
            systems[key] = cls()
        except ImportError as e:
            print(f"skipping {key}: {e}")
print("systems:", list(systems), flush=True)

# reference visibility: ours (v3) at its deployed setting on the calibration photos
ref = systems.get("v3") or Ours("checkpoints/hidden_v3_best.pt")
tmp = f"{WORK}/tmp.png"


def mean_de(sys_, strength):
    out = []
    for i in calib:
        sys_.embed(images[i][1], tmp, i, strength)
        out.append(visibility(images[i][1], tmp)["dE_mean"])
    return float(np.mean(out))


target_de = mean_de(ref, None)
print(f"target dE (ours v3, calibration n={len(calib)}): {target_de:.3f}", flush=True)
strengths = {}
for key, s in systems.items():
    strengths[(key, "default")] = s.default_strength
    if key in ("v3", "v4") or "matched" not in settings_wanted:
        continue
    lo, hi = s.default_strength / 16, s.default_strength * 4
    for _ in range(7):                                   # bisection in log-strength
        mid = float(np.sqrt(lo * hi))
        lo, hi = (mid, hi) if mean_de(s, mid) < target_de else (lo, mid)
    strengths[(key, "matched")] = float(np.sqrt(lo * hi))
    print(f"{key}: default strength {s.default_strength}, matched {strengths[(key, 'matched')]:.3f}", flush=True)

fps = {i: fingerprint(rgb(p)) for i, (_, p) in enumerate(images)}
originals = {i: rgb(p) for i, (_, p) in enumerate(images)}


def retrieve_align(sus_rgb):
    best = None
    for flipped in (False, True):
        img = sus_rgb[:, ::-1].copy() if flipped else sus_rgb
        f = fingerprint(img)
        for i, ref_fp in fps.items():
            m = match(f, ref_fp)
            if m and (best is None or m["inliers"] > best[1]["inliers"]):
                best = (i, m, img)
        if best:
            break
    if best is None:
        return None, None
    i, m, img = best
    return i, align(img, m["H"], originals[i])[0]


ROWS = f"{OUT}/rows.csv"
QUAL = f"{OUT}/quality.csv"
done = set()
if os.path.exists(QUAL):
    done = {(r["system"], r["setting"], int(r["image"])) for r in csv.DictReader(open(QUAL))}
rows_f = open(ROWS, "a", newline="")
qual_f = open(QUAL, "a", newline="")
rw = csv.DictWriter(rows_f, fieldnames=["system", "setting", "set", "image", "attack", "blind", "rav",
                                        "deployed", "detected", "bit_acc", "read_s"])
qw = csv.DictWriter(qual_f, fieldnames=["system", "setting", "set", "image", "strength", "psnr", "ssim",
                                        "dE_mean", "dE_flat", "embed_s", "fp_blind", "fp_rav"])
if os.path.getsize(ROWS) == 0:
    rw.writeheader()
if os.path.getsize(QUAL) == 0:
    qw.writeheader()

t0 = time.time()
for key, s in systems.items():
    for setting in settings_wanted:
        if (key, setting) not in strengths:
            continue
        st = strengths[(key, setting)]
        pipe = None
        if key in ("v3", "v4") and setting == "default":
            pipe = DefencePipeline(s.model)
            for i, (_, p) in enumerate(images):
                pipe.register(i, p, [s.copy_id(i)])
        for i, (tag, orig) in enumerate(images):
            if (s.name, setting, i) in done:
                continue
            pub = f"{WORK}/pub_{key}_{setting}.png"
            te = time.time()
            s.embed(orig, pub, i, st)
            embed_s = time.time() - te
            q, v = compute(orig, pub), visibility(orig, pub)
            # false detection on the clean original: blind, and after (self-)alignment
            fp_blind = s.read_path(orig, i)["detected"]
            _, al = retrieve_align(originals[i])
            fp_rav = fp_blind or (al is not None and s.read_rgb(al, i)["detected"])
            for atk, fn, akw in ATTACKS:
                sus = f"{WORK}/atk_{key}.png"
                np.random.seed(1000 * i + len(atk))
                copyfile(pub, sus) if fn is None else fn(pub, sus, **akw)
                tr = time.time()
                r = s.read_path(sus, i)
                read_s = time.time() - tr
                ok_rav = r["ok"]
                if not ok_rav:
                    j, al = retrieve_align(rgb(sus))
                    ok_rav = j == i and s.read_rgb(al, i)["ok"]
                deployed = ""
                if pipe is not None:
                    deployed = int(r["ok"] or pipe.analyse(sus)["copy_id"] == s.copy_id(i))
                rw.writerow({"system": s.name, "setting": setting, "set": tag, "image": i, "attack": atk,
                             "blind": int(r["ok"]), "rav": int(ok_rav), "deployed": deployed,
                             "detected": int(r["detected"]), "bit_acc": round(r["bit_acc"], 4),
                             "read_s": round(read_s, 4)})
            qw.writerow({"system": s.name, "setting": setting, "set": tag, "image": i, "strength": st,
                         "psnr": q["PSNR_dB"], "ssim": q["SSIM"], "dE_mean": v["dE_mean"],
                         "dE_flat": v["dE_flat"], "embed_s": round(embed_s, 3),
                         "fp_blind": int(fp_blind), "fp_rav": int(fp_rav)})
            rows_f.flush()
            qual_f.flush()
            if i % 10 == 9:
                print(f"{s.name} {setting}: {i + 1}/{len(images)} ({time.time() - t0:.0f}s)", flush=True)
rows_f.close()
qual_f.close()

# ─── summary ──────────────────────────────────────────────────────
rows = list(csv.DictReader(open(ROWS)))
qual = list(csv.DictReader(open(QUAL)))
f = lambda r, k: float(r[k]) if r[k] not in ("", "None") else None
combos = list(dict.fromkeys((r["system"], r["setting"]) for r in qual))
summary = []
print(f"\n{'system':<14}{'setting':<9}{'PSNR':>6}{'SSIM':>8}{'dE':>6}{'dE bg':>7}{'emb s':>7}{'read s':>7}"
      f"  {'FP clean':<12}{'blind (all 23)':<30}{'+RAV':<30}{'deployed RAV'}")
for sysname, setting in combos:
    qs = [r for r in qual if r["system"] == sysname and r["setting"] == setting]
    rs = [r for r in rows if r["system"] == sysname and r["setting"] == setting]
    flat = [f(r, "dE_flat") for r in qs if f(r, "dE_flat") is not None]
    dep = [r for r in rs if r["deployed"] != ""]
    s = {"system": sysname, "setting": setting, "n_images": len(qs), "n_reads": len(rs),
         "strength": float(qs[0]["strength"]),
         "psnr": np.mean([f(r, "psnr") for r in qs]), "ssim": np.mean([f(r, "ssim") for r in qs]),
         "dE": np.mean([f(r, "dE_mean") for r in qs]), "dE_bg": np.mean(flat) if flat else float("nan"),
         "embed_s": np.median([f(r, "embed_s") for r in qs]), "read_s": np.median([f(r, "read_s") for r in rs]),
         "fp": sum(int(r["fp_rav"]) for r in qs), "blind": sum(int(r["blind"]) for r in rs),
         "rav": sum(int(r["rav"]) for r in rs), "deployed": sum(int(r["deployed"]) for r in dep),
         "n_deployed": len(dep), "bit_acc": np.mean([f(r, "bit_acc") for r in rs])}
    for tag, _ in SETS:
        sub = [r for r in rs if r["set"] == tag]
        s[f"blind_{tag}"], s[f"rav_{tag}"], s[f"n_{tag}"] = (sum(int(r["blind"]) for r in sub),
                                                             sum(int(r["rav"]) for r in sub), len(sub))
    summary.append(s)
    print(f"{sysname:<14}{setting:<9}{s['psnr']:>6.1f}{s['ssim']:>8.4f}{s['dE']:>6.2f}{s['dE_bg']:>7.2f}"
          f"{s['embed_s']:>7.2f}{s['read_s']:>7.2f}  {s['fp']}/{len(qs):<9}"
          f"{pct_ci(s['blind'], len(rs)):<30}{pct_ci(s['rav'], len(rs)):<30}"
          + (pct_ci(s['deployed'], len(dep)) if dep else "-"))
with open(f"{OUT}/summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)

print("\nPer set, +RAV (blind):")
for s in summary:
    print(f"  {s['system']:<14}{s['setting']:<9}" + "  ".join(
        f"{tag} {100 * s[f'rav_{tag}'] / max(1, s[f'n_{tag}']):5.1f}% ({100 * s[f'blind_{tag}'] / max(1, s[f'n_{tag}']):.1f}%)"
        for tag, _ in SETS))
print("\nPer attack, all sets: +RAV / blind  (n = images per attack)")
print(f"  {'attack':<20}" + "".join(f"{(a + ' ' + b)[:20]:>22}" for a, b in combos))
for atk, _, _ in ATTACKS:
    line = f"  {atk + (' *' if atk in GEOMETRIC else ''):<20}"
    for a, b in combos:
        sub = [r for r in rows if r["system"] == a and r["setting"] == b and r["attack"] == atk]
        line += f"{sum(int(r['rav']) for r in sub):>12}/{len(sub)} ({sum(int(r['blind']) for r in sub)})".rjust(22)
    print(line)
print("  * geometric: position-dependent blind decoders need alignment")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2, 1]})
labels = [f"{s['system']}\n{s['setting']}" for s in summary]
x = np.arange(len(summary))
ax = axes[0]
for j, (col, colour, lab) in enumerate((("blind", "#9cc1ec", "blind read"), ("rav", "#2a78d6", "+ Retrieve-Align-Verify"))):
    ys = [100 * s[col] / max(1, s["n_reads"]) for s in summary]
    ax.bar(x + (j - 0.5) * 0.38, ys, 0.38, color=colour, label=lab)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel(f"% exact message read (23 attacks x {len(images)} photos)")
ax.set_ylim(0, 100)
ax.legend(frameon=False, fontsize=8)
ax.set_title("Robustness", loc="left", fontweight="bold", fontsize=11)
ax = axes[1]
for s in summary:
    ax.scatter(s["dE"], 100 * s["rav"] / max(1, s["n_reads"]), s=40)
    ax.annotate(f"{s['system']} ({s['setting'][0]})", (s["dE"], 100 * s["rav"] / max(1, s["n_reads"])),
                fontsize=7, xytext=(4, 2), textcoords="offset points")
ax.set_xlabel("visibility: mean CIEDE2000 dE")
ax.set_ylabel("% read with alignment")
ax.set_title("Robustness vs visibility", loc="left", fontweight="bold", fontsize=11)
for a in axes:
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/baselines.png", dpi=200)
print(f"\n{OUT}/summary.csv, rows.csv, quality.csv, baselines.png ({time.time() - t0:.0f}s)")
