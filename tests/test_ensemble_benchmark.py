# tests/test_ensemble_benchmark.py
# Head-to-head: DCT alone vs HiDDeN alone vs Ensemble.
# Runs against the full e-commerce attack suite.
# Produces the CSV that will become the main thesis result table.

import sys, os, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shutil import copyfile

from src.dct_watermark import embed as dct_embed, detect as dct_detect
from src.hidden_inference import HiddenModel
from src.ensemble import EnsembleWatermarker
from src.metrics import compute
from src import attacks, ecommerce_attacks as ea

CHECKPOINT = "checkpoints/hidden_final.pt"
IMAGES = ["baboon", "airplane", "peppers", "splash", "house"]
SELLERS = {
    "baboon":   "SELLER:acme_electronics",
    "airplane": "SELLER:aero_parts_co",
    "peppers":  "SELLER:fresh_market",
    "splash":   "SELLER:aqua_designs",
    "house":    "SELLER:home_goods_llc",
}
DCT_ALPHA = 0.02
DCT_N_COEFFS = 500
DCT_THRESHOLD = 6.0

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
]

os.makedirs("results/ensemble_benchmark", exist_ok=True)
os.makedirs("output", exist_ok=True)

print("Initializing ensemble...")
ensemble = EnsembleWatermarker(CHECKPOINT)
hidden_only = HiddenModel(CHECKPOINT)

# ─── Step 1: Embed with all three methods ─────────────────────────

dct_registry = []       # for DCT-only detection later
ensemble_registry = []  # for ensemble detection later

print("\nEmbedding watermarks...")
for name in IMAGES:
    orig = f"data/sipi/{name}.png"
    seller = SELLERS[name]

    # DCT-only
    dct_stego = f"results/ensemble_benchmark/dct_{name}.png"
    dct_meta = dct_embed(orig, seller, dct_stego,
                          alpha=DCT_ALPHA, n_coeffs=DCT_N_COEFFS)
    dct_registry.append({
        "seller_id": seller, "original_path": orig,
        "dct_meta": dct_meta, "stego_path": dct_stego,
    })

    # HiDDeN-only
    h_stego = f"results/ensemble_benchmark/hidden_{name}.png"
    hidden_only.embed(orig, seller, h_stego)

    # Ensemble
    ens_stego = f"results/ensemble_benchmark/ensemble_{name}.png"
    ens_result = ensemble.embed(orig, seller, ens_stego)
    ensemble_registry.append({
        "seller_id": seller, "original_path": orig,
        "dct_meta": ens_result["dct_meta"], "stego_path": ens_stego,
    })

    print(f"  {name}: DCT, HiDDeN, Ensemble")

# ─── Step 2: Report image quality for each method ─────────────────

print("\n" + "=" * 65)
print("IMAGE QUALITY (PSNR / SSIM on clean images)")
print("=" * 65)
print(f"{'Image':<10} {'DCT PSNR':<12} {'HiDDeN PSNR':<14} {'Ensemble PSNR':<14}")
print("-" * 55)

quality_rows = []
for name in IMAGES:
    orig = f"data/sipi/{name}.png"
    dct_q  = compute(orig, f"results/ensemble_benchmark/dct_{name}.png")
    h_q    = compute(orig, f"results/ensemble_benchmark/hidden_{name}.png")
    ens_q  = compute(orig, f"results/ensemble_benchmark/ensemble_{name}.png")
    quality_rows.append({
        "image": name,
        "dct_psnr": dct_q["PSNR_dB"], "dct_ssim": dct_q["SSIM"],
        "hidden_psnr": h_q["PSNR_dB"], "hidden_ssim": h_q["SSIM"],
        "ensemble_psnr": ens_q["PSNR_dB"], "ensemble_ssim": ens_q["SSIM"],
    })
    print(f"{name:<10} {dct_q['PSNR_dB']:<12} {h_q['PSNR_dB']:<14} {ens_q['PSNR_dB']:<14}")

# ─── Step 3: Robustness — run every attack against every method ───

seller_list = list(SELLERS.values())
robustness_rows = []

print("\n" + "=" * 90)
print("ROBUSTNESS (identification accuracy under attacks)")
print("=" * 90)
print(f"{'Image':<10} {'Attack':<22} {'DCT':<12} {'HiDDeN':<12} {'Ensemble':<14}")
print("-" * 75)

for name in IMAGES:
    true_seller = SELLERS[name]
    for atk_name, atk_fn, atk_kwargs in ATTACK_SUITE:

        def _attack(src, dst):
            if atk_fn is None: copyfile(src, dst)
            else: atk_fn(src, dst, **atk_kwargs)

        # DCT-only identification (find best match by DCT similarity)
        dct_best = None
        for r in dct_registry:
            atk_path = f"results/ensemble_benchmark/dct_{name}_{atk_name}.png"
            _attack(r["stego_path"] if r["seller_id"] == true_seller
                                    else r["stego_path"], atk_path) \
                if r["seller_id"] == true_seller else None
        # Simpler: attack the true seller's stego once, then check every registry entry
        atk_dct = f"results/ensemble_benchmark/dct_{name}_{atk_name}.png"
        true_stego = next(r["stego_path"] for r in dct_registry
                          if r["seller_id"] == true_seller)
        _attack(true_stego, atk_dct)
        dct_scores = []
        for r in dct_registry:
            res = dct_detect(atk_dct, r["original_path"], r["seller_id"],
                             r["dct_meta"], threshold=DCT_THRESHOLD)
            dct_scores.append((r["seller_id"], res["similarity"], res["detected"]))
        dct_scores.sort(key=lambda x: x[1], reverse=True)
        dct_top = dct_scores[0]
        dct_correct = (dct_top[0] == true_seller) and dct_top[2]

        # HiDDeN-only identification
        atk_h = f"results/ensemble_benchmark/hidden_{name}_{atk_name}.png"
        true_h_stego = f"results/ensemble_benchmark/hidden_{name}.png"
        _attack(true_h_stego, atk_h)
        h_res = hidden_only.identify(atk_h, seller_list)
        h_correct = (h_res["match"]["seller_id"] == true_seller
                     and h_res["match"]["ber"] < 0.40)

        # Ensemble identification
        atk_e = f"results/ensemble_benchmark/ensemble_{name}_{atk_name}.png"
        true_ens_stego = f"results/ensemble_benchmark/ensemble_{name}.png"
        _attack(true_ens_stego, atk_e)
        ens_res = ensemble.identify(atk_e, ensemble_registry)
        ens_correct = (ens_res["match"] is not None
                       and ens_res["match"]["seller_id"] == true_seller)
        ens_winner = ens_res["match"]["winning_detector"] if ens_res["match"] else "none"

        robustness_rows.append({
            "image": name, "attack": atk_name,
            "dct_correct": dct_correct,
            "hidden_correct": h_correct,
            "ensemble_correct": ens_correct,
            "ensemble_winner": ens_winner,
        })

        dct_str = "OK" if dct_correct else "FAIL"
        h_str   = "OK" if h_correct else "FAIL"
        ens_str = f"OK ({ens_winner})" if ens_correct else "FAIL"
        print(f"{name:<10} {atk_name:<22} {dct_str:<12} {h_str:<12} {ens_str:<14}")

# ─── Save CSVs ─────────────────────────────────────────────────────

with open("output/ensemble_quality.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(quality_rows[0].keys()))
    w.writeheader(); w.writerows(quality_rows)

with open("output/ensemble_robustness.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(robustness_rows[0].keys()))
    w.writeheader(); w.writerows(robustness_rows)

# ─── Summary table ─────────────────────────────────────────────────

print("\n" + "=" * 65)
print("SUMMARY: survival rate per attack (across 5 images)")
print("=" * 65)
print(f"{'Attack':<22} {'DCT':<10} {'HiDDeN':<10} {'Ensemble':<10}")
print("-" * 55)
attacks_seen = []
for r in robustness_rows:
    if r["attack"] not in attacks_seen: attacks_seen.append(r["attack"])
for a in attacks_seen:
    subset = [r for r in robustness_rows if r["attack"] == a]
    dct_rate = sum(1 for r in subset if r["dct_correct"]) / len(subset)
    h_rate   = sum(1 for r in subset if r["hidden_correct"]) / len(subset)
    ens_rate = sum(1 for r in subset if r["ensemble_correct"]) / len(subset)
    print(f"{a:<22} {dct_rate*100:>4.0f}%      {h_rate*100:>4.0f}%      {ens_rate*100:>4.0f}%")

overall = {
    "dct":      sum(1 for r in robustness_rows if r["dct_correct"]) / len(robustness_rows),
    "hidden":   sum(1 for r in robustness_rows if r["hidden_correct"]) / len(robustness_rows),
    "ensemble": sum(1 for r in robustness_rows if r["ensemble_correct"]) / len(robustness_rows),
}
print("-" * 55)
print(f"{'OVERALL':<22} {overall['dct']*100:>4.0f}%      {overall['hidden']*100:>4.0f}%      {overall['ensemble']*100:>4.0f}%")

print(f"\nCSVs: output/ensemble_quality.csv, output/ensemble_robustness.csv")