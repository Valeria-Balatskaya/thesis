# tests/test_parameter_study.py
# Part C2 — impact of the watermarking parameters (registered objective 1.4)
# on quality, robustness and detection, on real marketplace photos (ABO).
#
# Grid: strength {0.3, 0.5, 0.7, 1.0} x mask {none, flat 0.3, flat 0.0}
#       (flat = local-std texture mask, src/hidden_v2.texture_mask)
#       optional second grid with the NVF mask (--nvf)
# Per setting, on the first N ABO evaluation photos:
#   quality     PSNR, SSIM, dE mean, dE on flat areas
#   robustness  blind and full Retrieve-Align-Verify read rate under the 12 attacks
#   detection   false positives on the clean originals (blind + pipeline)
# Outputs: output/parameter_study.csv, parameter_study_tradeoff.png (read rate vs
# dE, the invisibility/robustness trade-off), parameter_study_table.md.
#
# Run: python tests/test_parameter_study.py [n_abo] [--nvf]   (default 30)

import sys, os, csv, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shutil import copyfile

import numpy as np
from PIL import Image

from src.hidden_v2_inference import HiddenV2Model
from src.defence import DefencePipeline
from src.metrics import compute, visibility
from src import attacks, ecommerce_attacks as ea

args = [a for a in sys.argv[1:] if not a.startswith("--")]
N = int(args[0]) if args else 30
STRENGTHS = [0.3, 0.5, 0.7, 1.0]
MASKS = [("none", dict(mask_floor=1.0, mask_type="flat")),
         ("flat_0.3", dict(mask_floor=0.3, mask_type="flat")),
         ("flat_0.0", dict(mask_floor=0.0, mask_type="flat"))]
if "--nvf" in sys.argv:
    MASKS += [("nvf_0.3", dict(mask_floor=0.3, mask_type="nvf")),
              ("nvf_0.0", dict(mask_floor=0.0, mask_type="nvf"))]
ATTACK_SUITE = [
    ("clean",              None,                              {}),
    ("jpeg_q70",           attacks.jpeg_compress,             {"quality": 70}),
    ("jpeg_q30",           attacks.jpeg_compress,             {"quality": 30}),
    ("resize_0.5",         attacks.resize_attack,             {"scale": 0.5}),
    ("noise_s15",          attacks.gaussian_noise,            {"sigma": 15}),
    ("crop_0.1",           attacks.crop_attack,               {"crop_fraction": 0.1}),
    ("brightness_1.2",     attacks.brightness_attack,         {"factor": 1.2}),
    ("marketplace_square", ea.marketplace_square,             {"target_size": 512}),
    ("instagram_filter",   ea.instagram_filter,               {}),
    ("screenshot",         ea.screenshot_simulation,          {}),
    ("print_photograph",   ea.print_photograph_simulation,    {}),
    ("native_screenshot",  ea.native_screenshot_simulation,   {}),
]
OUT = "results/parameter_study"
os.makedirs(OUT, exist_ok=True)
os.makedirs("output", exist_ok=True)
model = HiddenV2Model("checkpoints/hidden_v2_best.pt")

originals = []
for k, f in enumerate(sorted(os.listdir("data/abo_sample"))[:N]):
    p = f"{OUT}/abo_{k:03d}_orig.png"
    if not os.path.exists(p):
        img = Image.open(f"data/abo_sample/{f}").convert("RGB")
        img.thumbnail((1024, 1024), Image.LANCZOS)
        img.save(p, format="PNG")
    originals.append(p)

pipe = DefencePipeline(model)
for i, p in enumerate(originals):
    pipe.register(i, p, [1000 + i])
fp_blind = sum(model.read_product(p)["product_id"] is not None for p in originals)
fp_pipe = sum(pipe.analyse(p)["copy_id"] is not None for p in originals)

rows = []
t0 = time.time()
for mask_name, opts in MASKS:
    for strength in STRENGTHS:
        np.random.seed(0)
        q_all, ok_b, ok_f = [], 0, 0
        per_attack = {a[0]: 0 for a in ATTACK_SUITE}
        for i, orig in enumerate(originals):
            wm = f"{OUT}/wm.png"
            model.embed_product(orig, 1000 + i, wm, strength=strength, **opts)
            q, v = compute(orig, wm), visibility(orig, wm)
            q_all.append((q["PSNR_dB"], q["SSIM"], v["dE_mean"], v["dE_flat"]))
            for atk, fn, kw in ATTACK_SUITE:
                sus = f"{OUT}/atk.png"
                copyfile(wm, sus) if fn is None else fn(wm, sus, **kw)
                b = model.read_product(sus)["product_id"] == 1000 + i
                full = b or pipe.analyse(sus)["copy_id"] == 1000 + i
                ok_b += b
                ok_f += full
                per_attack[atk] += full
        n_att = len(originals) * len(ATTACK_SUITE)
        flat = [x[3] for x in q_all if x[3] is not None]
        row = {"mask": mask_name, "strength": strength, "n_images": len(originals),
               "psnr": round(float(np.mean([x[0] for x in q_all])), 2),
               "ssim": round(float(np.mean([x[1] for x in q_all])), 4),
               "dE_mean": round(float(np.mean([x[2] for x in q_all])), 3),
               "dE_flat": round(float(np.mean(flat)), 3) if flat else None,
               "blind_pct": round(100 * ok_b / n_att, 1),
               "full_pct": round(100 * ok_f / n_att, 1),
               "false_pos_blind": f"{fp_blind}/{len(originals)}",
               "false_pos_pipeline": f"{fp_pipe}/{len(originals)}",
               **{f"full_{a}": per_attack[a] for a in per_attack}}
        rows.append(row)
        print(f"{mask_name:<9} strength {strength:.1f}: PSNR {row['psnr']:5.2f}  SSIM {row['ssim']:.4f}  "
              f"dE {row['dE_mean']:.2f}/{row['dE_flat']}  blind {row['blind_pct']:5.1f}%  "
              f"full {row['full_pct']:5.1f}%  ({time.time() - t0:.0f}s)", flush=True)

tag = "_nvf" if "--nvf" in sys.argv else ""
with open(f"output/parameter_study{tag}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

with open(f"output/parameter_study{tag}_table.md", "w") as f:
    f.write(f"Parameter study on {len(originals)} ABO product photos x 12 attacks "
            f"(HiDDeN v2). False positives on clean originals: blind {fp_blind}/{len(originals)}, "
            f"pipeline {fp_pipe}/{len(originals)}.\n\n")
    f.write("| Mask | Strength | PSNR (dB) | SSIM | dE mean | dE flat | Blind read | Full pipeline |\n")
    f.write("|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r['mask']} | {r['strength']} | {r['psnr']} | {r['ssim']} | {r['dE_mean']} | "
                f"{r['dE_flat']} | {r['blind_pct']}% | {r['full_pct']}% |\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
colours = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
for ax, key, label in [(axes[0], "dE_mean", "colour difference dE (whole image, ~1 = just noticeable)"),
                       (axes[1], "psnr", "PSNR (dB)")]:
    for (mask_name, _), c in zip(MASKS, colours):
        pts = [r for r in rows if r["mask"] == mask_name]
        ax.plot([r[key] for r in pts], [r["full_pct"] for r in pts], marker="o", color=c,
                linewidth=2, label=f"{mask_name} (full pipeline)")
        ax.plot([r[key] for r in pts], [r["blind_pct"] for r in pts], marker="o", color=c,
                linewidth=1.2, linestyle="--", alpha=0.8, label=f"{mask_name} (blind)")
        for r in pts:
            ax.annotate(f"{r['strength']}", (r[key], r["full_pct"]), textcoords="offset points",
                        xytext=(4, 4), fontsize=8, color="#52514e")
    ax.set_xlabel(label)
    ax.grid(color="#e4e3df")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("correct copy ID under 12 attacks (%)")
axes[1].legend(frameon=False, fontsize=8, loc="lower right")
fig.suptitle(f"Invisibility vs robustness on {len(originals)} ABO product photos (labels = strength)",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(f"output/parameter_study{tag}_tradeoff.png", dpi=200)
print(f"\nOutputs: output/parameter_study{tag}.csv, _table.md, _tradeoff.png  ({time.time() - t0:.0f}s)")
