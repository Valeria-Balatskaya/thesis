# tests/test_ecommerce_vs_regular.py
# Does the HiDDeN v2 watermark behave differently on e-commerce product photos
# than on the regular photos it was trained on (COCO)?
#
# For each image set (default: own product photos, ABO product photos, COCO
# test2017 photos; see scripts/fetch_reference_images.py) and each embedding
# setting (the app default 0.7 strength + 0.3 flat mask, and full strength 1.0):
#   visibility   PSNR, CIEDE2000 dE mean and dE on flat bright areas
#   robustness   correct copy ID under the 12 attacks of test_hidden_v2_eval.py,
#                read blind, and (app setting only) with the full
#                Retrieve-Align-Verify pipeline (src/defence.py)
#   false pos.   blind reads of the clean, un-watermarked originals
#
# Uses the existing checkpoint only — no training.
# Run: python tests/test_ecommerce_vs_regular.py [n_per_set]   (default 30)

import sys, os, csv, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shutil import copyfile

import numpy as np
from PIL import Image

from src.hidden_v2_inference import HiddenV2Model
from src.defence import DefencePipeline
from src.metrics import compute, visibility
from src import attacks, ecommerce_attacks as ea

CHECKPOINT = "checkpoints/hidden_v2_best.pt"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
MAX_SIDE = 1024
SETS = [("own_products", "data/products_png"),
        ("abo_products", "data/abo_sample"),
        ("coco_regular", "data/coco_sample")]
SETTINGS = [("app_0.7_mask0.3", 0.7, 0.3), ("full_1.0", 1.0, None)]
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

OUT = "results/ecommerce_vs_regular"
os.makedirs(OUT, exist_ok=True)
os.makedirs("output", exist_ok=True)
np.random.seed(0)

model = HiddenV2Model(CHECKPOINT)


def normalise(src: str, dst: str) -> None:
    img = Image.open(src).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    img.save(dst, format="PNG")


rows, summary = [], []
t0 = time.time()
for set_name, folder in SETS:
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg")))[:N]
    originals = []
    for i, f in enumerate(files):
        p = f"{OUT}/{set_name}_{i:03d}_orig.png"
        normalise(os.path.join(folder, f), p)
        originals.append(p)

    fp = sum(model.read_product(p)["product_id"] is not None for p in originals)

    for setting, strength, mask in SETTINGS:
        pipe = DefencePipeline(model) if setting.startswith("app") else None
        vis, ok_blind, ok_full = [], {a[0]: 0 for a in ATTACK_SUITE}, {a[0]: 0 for a in ATTACK_SUITE}
        wms = []
        for i, orig in enumerate(originals):
            wm = f"{OUT}/{set_name}_{i:03d}_{setting}.png"
            model.embed_product(orig, 1000 + i, wm, strength=strength, mask_floor=mask)
            if pipe:
                pipe.register(i, orig, [1000 + i])   # whole set registered before any scan
            q, v = compute(orig, wm), visibility(orig, wm)
            vis.append((q["PSNR_dB"], v["dE_mean"], v["dE_flat"]))
            wms.append(wm)
        for i, wm in enumerate(wms):
            copy_id = 1000 + i
            for atk, fn, kw in ATTACK_SUITE:
                sus = f"{OUT}/atk.png"
                copyfile(wm, sus) if fn is None else fn(wm, sus, **kw)
                blind = model.read_product(sus)["product_id"] == copy_id
                full = blind or (pipe is not None and pipe.analyse(sus)["copy_id"] == copy_id)
                ok_blind[atk] += blind
                ok_full[atk] += full
                rows.append({"set": set_name, "setting": setting, "image": i, "attack": atk,
                             "blind": int(blind), "full_pipeline": int(full) if pipe else ""})
        psnr, de, de_flat = zip(*vis)
        flat_vals = [d for d in de_flat if d is not None]
        n_att = len(originals) * len(ATTACK_SUITE)
        s = {"set": set_name, "setting": setting, "n_images": len(originals),
             "psnr": round(float(np.mean(psnr)), 2), "dE_mean": round(float(np.mean(de)), 3),
             "dE_flat": round(float(np.mean(flat_vals)), 3) if flat_vals else None,
             "n_flat_images": len(flat_vals),
             "blind_total": f"{sum(ok_blind.values())}/{n_att}",
             "blind_pct": round(100 * sum(ok_blind.values()) / n_att, 1),
             "full_total": f"{sum(ok_full.values())}/{n_att}" if pipe else "",
             "full_pct": round(100 * sum(ok_full.values()) / n_att, 1) if pipe else "",
             "false_positives_clean": f"{fp}/{len(originals)}"}
        s.update({f"blind_{a}": ok_blind[a] for a in ok_blind})
        if pipe:
            s.update({f"full_{a}": ok_full[a] for a in ok_full})
        summary.append(s)
        print(f"{set_name:<14} {setting:<16} PSNR {s['psnr']:5.1f}  dE {s['dE_mean']:.2f}  "
              f"dE_flat {s['dE_flat']}  blind {s['blind_pct']}%  full {s['full_pct']}%  "
              f"FP {s['false_positives_clean']}  ({time.time() - t0:.0f}s)", flush=True)

with open("output/ecommerce_vs_regular_rows.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
keys = list(dict.fromkeys(k for s in summary for k in s))
with open("output/ecommerce_vs_regular_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(summary)

print("\nPer attack, blind read (correct / images):")
print(f"{'attack':<20}" + "".join(f"{s['set'][:6] + '|' + s['setting'][:4]:>13}" for s in summary))
for atk, _, _ in ATTACK_SUITE:
    print(f"{atk:<20}" + "".join(f"{s['blind_' + atk]:>13}" for s in summary))
print("\nCSVs: output/ecommerce_vs_regular_summary.csv, output/ecommerce_vs_regular_rows.csv")
