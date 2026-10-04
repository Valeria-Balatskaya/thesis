# tests/test_hidden_v2_eval.py
# Evaluation + regression test for a HiDDeN v2 checkpoint.
#
# Uses the REAL attack implementations (PIL / numpy, src/attacks.py and
# src/ecommerce_attacks.py) — not the differentiable approximations the model
# was trained with — so this measures generalisation, not memorisation.
#
# Reports, per attack:
#   BER       — raw bit error rate of the decoder
#   ID ok     — product ID recovered blindly (BCH corrects up to 6 bit errors)
#   wrong ID  — a valid but WRONG product ID was returned (must stay 0)
# and false positives: un-watermarked images, clean and attacked, must never
# yield a product ID.
#
# Run: python tests/test_hidden_v2_eval.py [checkpoint]

import sys, os, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shutil import copyfile

import numpy as np

from src.hidden_v2_inference import HiddenV2Model
from src.payload import encode_product_id, blind_false_positive_rate
from src.distractor_gen import generate_distractor
from src.metrics import compute
from src import attacks, ecommerce_attacks as ea

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/hidden_v2_best.pt"
SIPI = ["baboon", "airplane", "peppers", "splash", "house"]
N_SYNTHETIC = 25          # extra watermarked test images (procedural)
N_CLEAN_DISTRACTORS = 100  # un-watermarked images for the false-positive test

# Regression thresholds
MIN_PSNR = 38.0
MIN_CLEAN_ID_RATE = 1.0
MAX_FALSE_POSITIVES = 0

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

if not os.path.exists(CHECKPOINT):
    sys.exit(f"Checkpoint not found: {CHECKPOINT}\n"
             "Train it first (notebooks/train_hidden_v2.ipynb) and copy it into checkpoints/.")

OUT = "results/hidden_v2_eval"
os.makedirs(OUT, exist_ok=True)
os.makedirs("output", exist_ok=True)
np.random.seed(0)   # attacks.gaussian_noise / screenshot_simulation use the global RNG

model = HiddenV2Model(CHECKPOINT)
print(f"Checkpoint: {CHECKPOINT}  ({model.msg_len} bits, target {model.target_psnr} dB)\n")


def _attack(fn, kwargs, src, dst):
    if fn is None:
        copyfile(src, dst)
    else:
        fn(src, dst, **kwargs)


# ─── Test images: 5 SIPI + procedural ones ────────────────────────

originals = {name: f"data/sipi/{name}.png" for name in SIPI}
for i in range(N_SYNTHETIC):
    originals[f"synth{i}"] = generate_distractor(
        f"v2eval_wm_{i}", size=512, output_path=f"{OUT}/orig_synth{i}.png")
product_ids = {name: 1000 + 37 * i for i, name in enumerate(originals)}

# ─── Embed + quality ──────────────────────────────────────────────

quality_rows, published = [], {}
for name, orig in originals.items():
    published[name] = f"{OUT}/wm_{name}.png"
    model.embed_product(orig, product_ids[name], published[name])
    q = compute(orig, published[name])
    quality_rows.append({"image": name, "psnr": q["PSNR_dB"], "ssim": q["SSIM"]})

psnrs = [r["psnr"] for r in quality_rows]
ssims = [r["ssim"] for r in quality_rows]
print(f"QUALITY over {len(originals)} images:  PSNR mean {np.mean(psnrs):.2f} dB "
      f"(min {np.min(psnrs):.2f})   SSIM mean {np.mean(ssims):.4f} (min {np.min(ssims):.4f})\n")

# ─── Robustness ───────────────────────────────────────────────────

rows = []
print(f"{'Attack':<20}{'BER':>8}{'ID ok':>9}{'wrong ID':>10}")
print("-" * 47)
for atk_name, fn, kwargs in ATTACK_SUITE:
    bers, ok, wrong = [], 0, 0
    for name in originals:
        atk_path = f"{OUT}/atk_{name}_{atk_name}.png"
        _attack(fn, kwargs, published[name], atk_path)
        res = model.read_product(atk_path)
        ber = float((res["bits"] != encode_product_id(product_ids[name])).mean())
        bers.append(ber)
        ok += res["product_id"] == product_ids[name]
        wrong += res["product_id"] not in (None, product_ids[name])
        rows.append({"image": name, "attack": atk_name, "ber": round(ber, 4),
                     "decoded_id": res["product_id"], "true_id": product_ids[name]})
    print(f"{atk_name:<20}{np.mean(bers):>8.3f}{ok / len(originals):>9.0%}{wrong:>10}")

n_total = len(rows)
n_ok = sum(r["decoded_id"] == r["true_id"] for r in rows)
n_wrong = sum(r["decoded_id"] not in (None, r["true_id"]) for r in rows)
print("-" * 47)
print(f"{'OVERALL':<20}{np.mean([r['ber'] for r in rows]):>8.3f}{n_ok / n_total:>9.0%}{n_wrong:>10}")

# ─── False positives on un-watermarked images ─────────────────────

clean = dict(originals)
for i in range(N_CLEAN_DISTRACTORS):
    clean[f"clean{i}"] = generate_distractor(
        f"v2eval_clean_{i}", size=512, output_path=f"{OUT}/clean_{i}.png")

fp_rows = []
for name, path in clean.items():
    for atk_name, fn, kwargs in ATTACK_SUITE:
        atk_path = f"{OUT}/fp_tmp.png"
        _attack(fn, kwargs, path, atk_path)
        res = model.read_product(atk_path)
        fp_rows.append({"image": name, "attack": atk_name,
                        "decoded_id": res["product_id"], "reason": res["reason"]})
n_fp = sum(r["decoded_id"] is not None for r in fp_rows)
print(f"\nFALSE POSITIVES: {n_fp} / {len(fp_rows)} un-watermarked images returned a product ID "
      f"(theory: {blind_false_positive_rate():.1e} per image)")

# ─── CSVs + regression verdict ────────────────────────────────────

for fname, data in [("hidden_v2_quality.csv", quality_rows),
                    ("hidden_v2_robustness.csv", rows),
                    ("hidden_v2_false_positives.csv", fp_rows)]:
    with open(f"output/{fname}", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader(); w.writerows(data)

clean_rows = [r for r in rows if r["attack"] == "clean"]
clean_rate = sum(r["decoded_id"] == r["true_id"] for r in clean_rows) / len(clean_rows)
checks = [
    (f"mean PSNR >= {MIN_PSNR} dB", np.mean(psnrs) >= MIN_PSNR),
    (f"clean ID rate >= {MIN_CLEAN_ID_RATE:.0%}", clean_rate >= MIN_CLEAN_ID_RATE),
    (f"false positives <= {MAX_FALSE_POSITIVES}", n_fp <= MAX_FALSE_POSITIVES),
    ("no wrong product IDs", n_wrong == 0),
]
print("\nREGRESSION CHECKS")
for label, passed in checks:
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
print("\nCSVs: output/hidden_v2_quality.csv, output/hidden_v2_robustness.csv, "
      "output/hidden_v2_false_positives.csv")
sys.exit(0 if all(p for _, p in checks) else 1)
