# scripts/thesis_tables.py
# Build the thesis tables (Markdown, every percentage with n and a 95% Wilson interval)
# from the CSVs the tests write, and copy the figures to thesis/figures/.
# Inputs that don't exist yet are skipped with a note. Run: python scripts/thesis_tables.py

import csv
import os
import shutil
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.attack_suite import ATTACKS, GEOMETRIC
from src.stats import pct_ci, frac_ci

OUT = "output/thesis"
FIG = "thesis/figures"
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
written = []


def load(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else None


def num(x):
    return None if x in (None, "", "None", "nan") else float(x)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def write(name, title, source, lines):
    with open(f"{OUT}/{name}.md", "w") as f:
        f.write(f"# {title}\n\nSource: `{source}`. Percentages: 95% Wilson interval, (k/n).\n\n")
        f.write("\n".join(lines) + "\n")
    written.append(name)


# ─── visibility + robustness (test_v3_vs_v2) ─────────────────────
q, r = load("output/v3_vs_v2_quality.csv"), load("output/v3_vs_v2_robustness.csv")
if q and r:
    systems = list(dict.fromkeys(x["system"] for x in q))
    sets = list(dict.fromkeys(x["set"] for x in q))
    lines = ["| Set | Model | n | PSNR (dB) | SSIM | ΔE whole | ΔE background |", "|---|---|---|---|---|---|---|"]
    for s in sets:
        for m in systems:
            sub = [x for x in q if x["set"] == s and x["system"] == m]
            if sub:
                lines.append(f"| {s} | {m} | {len(sub)} | {mean(num(x['psnr']) for x in sub):.1f} | "
                             f"{mean(num(x['ssim']) for x in sub):.4f} | {mean(num(x['dE_mean']) for x in sub):.2f} | "
                             f"{mean(num(x['dE_flat']) for x in sub):.2f} |")
    write("visibility", "Visibility of the embedded watermark", "output/v3_vs_v2_quality.csv", lines)

    lines = ["| Set | Model | Blind read | Full pipeline (Retrieve–Align–Verify) |", "|---|---|---|---|"]
    for s in sets:
        for m in systems:
            sub = [x for x in r if x["set"] == s and x["system"] == m]
            if sub:
                lines.append(f"| {s} | {m} | {pct_ci(sum(int(x['blind']) for x in sub), len(sub))} | "
                             f"{pct_ci(sum(int(x['full']) for x in sub), len(sub))} |")
    lines += ["", f"Per attack, all sets (full pipeline; blind in brackets). * = geometric.", "",
              "| Attack | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
    for atk, _, _ in ATTACKS:
        cells = []
        for m in systems:
            sub = [x for x in r if x["system"] == m and x["attack"] == atk]
            cells.append(f"{frac_ci(sum(int(x['full']) for x in sub), len(sub))} ({sum(int(x['blind']) for x in sub)})"
                         if sub else "–")
        lines.append(f"| {atk}{' *' if atk in GEOMETRIC else ''} | " + " | ".join(cells) + " |")
    write("robustness", "Robustness under 23 attacks", "output/v3_vs_v2_robustness.csv", lines)
else:
    print("skip visibility/robustness: run tests/test_v3_vs_v2.py")

# ─── overwrite cost (Part 2.2) ────────────────────────────────────
ow = load("output/overwrite_cost_summary.csv")
if ow:
    lines = ["| Model | Thief | Thief strength | Thief's damage ΔE | ΔE background | PSNR (dB) | Ownership proven | Thief's ID read |",
             "|---|---|---|---|---|---|---|---|"]
    for x in ow:
        n = int(x["n"])
        lines.append(f"| {x['system']} | {x['thief'].replace('_', ' ')} | {float(x['level']):g}× | "
                     f"{num(x['thief_dE']):.2f} | {num(x['thief_dE_flat']) or float('nan'):.2f} | {num(x['thief_psnr']):.1f} | "
                     f"{frac_ci(int(x['owned']), n)} | {frac_ci(int(x['thief_id_read']), n)} |")
    lines += ["", "Ownership proven = calibrated informed check z ≥ 4 (500 never-issued copies) or the keyed DCT "
              "layer detected. Thief's ID read with the thief's own key. ΔE > 2 ≈ visible at a glance."]
    write("overwrite_cost", "Removing our mark by overwriting costs visible damage", "output/overwrite_cost_summary.csv", lines)
else:
    print("skip overwrite table: run tests/test_overwrite_cost.py (it now writes overwrite_cost_summary.csv)")

# ─── watermark_overwritten verdict ────────────────────────────────
for path in sorted(p for p in os.listdir("output") if p.startswith("overwritten_verdict_") and p.endswith(".csv")):
    rows = load(f"output/{path}")
    lines = ["| Scenario | Post-processing | Should fire | 'Overwritten' (correct copy) | Wrong copy |", "|---|---|---|---|---|"]
    for sc in dict.fromkeys(x["scenario"] for x in rows):
        for post in ("clean", "jpeg_q70"):
            sub = [x for x in rows if x["scenario"] == sc and x["post"] == post]
            if not sub:
                continue
            pos = sub[0]["group"] == "positive"
            fired = [x for x in sub if x["verdict"] == "watermark_overwritten"]
            k = sum(int(x["named_A"]) for x in fired) if pos else len(fired)
            lines.append(f"| {sc} | {post} | {'yes' if pos else 'no'} | {pct_ci(k, len(sub))} | "
                         f"{sum(int(x['named_B']) for x in fired)} |")
    neg = [x for x in rows if x["group"] == "negative"]
    fp = sum(x["verdict"] == "watermark_overwritten" for x in neg)
    zs = [num(x[c]) for x in neg for c in ("z_A", "z_B") if num(x[c]) is not None]
    lines += ["", f"False 'overwritten' verdicts on images without any issued copy: {pct_ci(fp, len(neg), 2)}; "
              f"{len(zs)} informed tests, highest z {max(zs) if zs else float('nan'):.2f} (threshold 4)."]
    write(path.replace(".csv", ""), "The watermark_overwritten verdict: hits and false positives", f"output/{path}", lines)

# ─── baselines ────────────────────────────────────────────────────
b = load("output/baselines/summary.csv")
if b:
    lines = ["| System | Setting | PSNR | SSIM | ΔE | ΔE bg | Embed s | Extract s | False detections (clean) | Blind | + Retrieve–Align–Verify | Deployed pipeline |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in b:
        n = int(x["n_reads"])
        dep = pct_ci(int(x["deployed"]), int(x["n_deployed"])) if int(x["n_deployed"]) else "–"
        lines.append(f"| {x['system']} | {x['setting']} (s={num(x['strength']):.2g}) | {num(x['psnr']):.1f} | {num(x['ssim']):.4f} | "
                     f"{num(x['dE']):.2f} | {num(x['dE_bg']):.2f} | {num(x['embed_s']):.2f} | {num(x['read_s']):.2f} | "
                     f"{frac_ci(int(x['fp']), int(x['n_images']))} | {pct_ci(int(x['blind']), n)} | {pct_ci(int(x['rav']), n)} | {dep} |")
    rows = load("output/baselines/rows.csv")
    combos = list(dict.fromkeys((x["system"], x["setting"]) for x in rows))
    lines += ["", "Per attack (+RAV; blind in brackets):", "",
              "| Attack | " + " | ".join(f"{a} {s}" for a, s in combos) + " |", "|---|" + "---|" * len(combos)]
    for atk, _, _ in ATTACKS:
        cells = []
        for a, s in combos:
            sub = [x for x in rows if x["system"] == a and x["setting"] == s and x["attack"] == atk]
            cells.append(f"{sum(int(x['rav']) for x in sub)}/{len(sub)} ({sum(int(x['blind']) for x in sub)})")
        lines.append(f"| {atk}{' *' if atk in GEOMETRIC else ''} | " + " | ".join(cells) + " |")
    write("baselines", "Comparison with pretrained watermarks", "output/baselines/", lines)
else:
    print("skip baselines: run tests/test_baselines.py")

# ─── ablation ─────────────────────────────────────────────────────
a = load("output/ablation/summary.csv")
if a:
    lines = ["| Configuration | Traced (23 attacks) | PSNR | ΔE | ΔE bg | False positives (never-protected photos) |",
             "|---|---|---|---|---|---|"]
    for x in a:
        tr = pct_ci(int(float(x["traced"])), int(x["n"])) if num(x["traced"]) is not None else "–"
        fp = pct_ci(int(float(x["false_positives"])), int(x["n_negatives"]), 2) if num(x["false_positives"]) is not None else "–"
        f = lambda k, d: f"{num(x[k]):.{d}f}" if num(x[k]) is not None else "–"
        lines.append(f"| {x['configuration']} | {tr} | {f('psnr', 1)} | {f('dE', 2)} | {f('dE_bg', 2)} | {fp} |")
    ow = load("output/overwrite_cost_summary.csv")
    if ow:
        lines += ["", "Components that only matter against overwriting (from test_overwrite_cost, ownership proven):", "",
                  "| Thief strength | v3 | v3 + DCT layer (− DCT: v3) | v4 other key (− key: v3) |", "|---|---|---|---|"]
        for lvl in sorted({float(x["level"]) for x in ow}):
            g = lambda s, t: next((frac_ci(int(x["owned"]), int(x["n"])) for x in ow
                                   if x["system"] == s and x["thief"] == t and float(x["level"]) == lvl), "–")
            lines.append(f"| {lvl:g}× | {g('v3', 'same_model')} | {g('v3+dct', 'same_model')} | {g('v4', 'other_key')} |")
    write("ablation", "Ablation: one component removed at a time", "output/ablation/summary.csv", lines)
else:
    print("skip ablation: run tests/test_ablation.py")

# ─── real world ───────────────────────────────────────────────────
rw = load("output/real_world/summary.csv")
if rw:
    lines = ["| Capture type | Model | n | Blind | Full pipeline | Wrong copy | False positives (controls) |", "|---|---|---|---|---|---|---|"]
    for x in rw:
        n = int(x["n"])
        lines.append(f"| {x['capture_type']} | {x['model']} | {n} | {pct_ci(int(x['blind']), n) if n else '–'} | "
                     f"{pct_ci(int(x['full']), n) if n else '–'} | {x['wrong']} | {frac_ci(int(x['false_positives']), int(x['n_controls']))} |")
    write("real_world", "Real captures (not simulated)", "output/real_world/summary.csv", lines)
else:
    print("skip real world: collect captures (docs/real_world_protocol.md), run tests/test_real_world.py")

# ─── figures ──────────────────────────────────────────────────────
FIGS = {"output/overwrite_cost.png": "overwrite_cost.png", "output/baselines/baselines.png": "baselines.png",
        "output/real_world/real_world.png": "real_world.png", "output/v2_vs_v3_signal.png": "v2_vs_v3_signal.png",
        "output/parameter_study_tradeoff.png": "parameter_study_tradeoff.png",
        "output/overhead_retrieval.png": "overhead_retrieval.png"}
if os.path.isdir("output/ecommerce_analysis"):
    for f in os.listdir("output/ecommerce_analysis"):
        if f.endswith(".png"):
            FIGS[f"output/ecommerce_analysis/{f}"] = f"ecommerce_{f}"
copied = [dst for src, dst in FIGS.items() if os.path.exists(src) and shutil.copy(src, f"{FIG}/{dst}")]
print(f"tables: {', '.join(written) or 'none'} -> {OUT}/;  figures: {len(copied)} -> {FIG}/")
