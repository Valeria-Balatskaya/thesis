# tests/test_quality_and_scale.py
# Two-part evaluation:
#   Part A — Imperceptibility: PSNR/SSIM of every method on clean SIPI images
#   Part B — Scale: HiDDeN identification with N competing distractor sellers
#
# Part A produces the "invisibility" table for Chapter 4.
# Part B answers "does 100% survival hold at realistic marketplace scale?"

import sys, os, csv, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shutil import copyfile

from src.lsb import embed as lsb_embed
from src.dct_watermark import embed as dct_embed
from src.hidden_inference import HiddenModel, _seller_to_bits, _ber
from src.metrics import compute
from src import attacks, ecommerce_attacks as ea

random.seed(42)

IMAGES = ["baboon", "airplane", "peppers", "splash", "house"]
SELLERS = {
    "baboon":   "SELLER:acme_electronics",
    "airplane": "SELLER:aero_parts_co",
    "peppers":  "SELLER:fresh_market",
    "splash":   "SELLER:aqua_designs",
    "house":    "SELLER:home_goods_llc",
}

os.makedirs("results/quality_scale", exist_ok=True)
os.makedirs("output", exist_ok=True)

# =====================================================================
# PART A — IMPERCEPTIBILITY
# =====================================================================
print("=" * 70)
print("PART A — IMPERCEPTIBILITY (PSNR / SSIM on clean SIPI images)")
print("=" * 70)

hidden = HiddenModel("checkpoints/hidden_final.pt")

quality_rows = []
print(f"\n{'Image':<10} {'Method':<10} {'PSNR (dB)':<12} {'SSIM':<8} {'Visually OK?'}")
print("-" * 55)

for name in IMAGES:
    orig = f"data/sipi/{name}.png"
    seller = SELLERS[name]

    # LSB
    lsb_out = f"results/quality_scale/lsb_{name}.png"
    lsb_embed(orig, seller, lsb_out, k=1)
    m = compute(orig, lsb_out)
    ok = m["PSNR_dB"] > 40 and m["SSIM"] > 0.98
    quality_rows.append({"image": name, "method": "LSB", **m, "visually_ok": ok})
    print(f"{name:<10} {'LSB':<10} {m['PSNR_dB']:<12} {m['SSIM']:<8} {'YES' if ok else 'no'}")

    # DCT
    dct_out = f"results/quality_scale/dct_{name}.png"
    dct_embed(orig, seller, dct_out, alpha=0.02, n_coeffs=500)
    m = compute(orig, dct_out)
    ok = m["PSNR_dB"] > 40 and m["SSIM"] > 0.98
    quality_rows.append({"image": name, "method": "DCT", **m, "visually_ok": ok})
    print(f"{name:<10} {'DCT':<10} {m['PSNR_dB']:<12} {m['SSIM']:<8} {'YES' if ok else 'no'}")

    # HiDDeN
    h_out = f"results/quality_scale/hidden_{name}.png"
    hidden.embed(orig, seller, h_out)
    m = compute(orig, h_out)
    ok = m["PSNR_dB"] > 40 and m["SSIM"] > 0.98
    quality_rows.append({"image": name, "method": "HiDDeN", **m, "visually_ok": ok})
    print(f"{name:<10} {'HiDDeN':<10} {m['PSNR_dB']:<12} {m['SSIM']:<8} {'YES' if ok else 'no'}")
    print()

with open("output/imperceptibility.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(quality_rows[0].keys()))
    w.writeheader()
    w.writerows(quality_rows)

# Per-method averages
print("SUMMARY (averaged over 5 images):")
for method in ["LSB", "DCT", "HiDDeN"]:
    subset = [r for r in quality_rows if r["method"] == method]
    avg_psnr = sum(r["PSNR_dB"] for r in subset) / len(subset)
    avg_ssim = sum(r["SSIM"] for r in subset) / len(subset)
    print(f"  {method:<8} avg PSNR = {avg_psnr:.2f} dB   avg SSIM = {avg_ssim:.4f}")

# =====================================================================
# PART B — HIDDEN SCALE TEST
# =====================================================================
print("\n" + "=" * 70)
print("PART B — HIDDEN IDENTIFICATION AT SCALE")
print("=" * 70)

REGISTRY_SIZES = [5, 20, 100, 500]
ATTACKS_TO_TEST = [
    ("clean",              None,                              {}),
    ("jpeg_q30",           attacks.jpeg_compress,             {"quality": 30}),
    ("brightness_1.2",     attacks.brightness_attack,         {"factor": 1.2}),
    ("crop_0.1",           attacks.crop_attack,               {"crop_fraction": 0.1}),
    ("instagram_filter",   ea.instagram_filter,               {}),
    ("print_photograph",   ea.print_photograph_simulation,    {}),
]

def random_seller_id() -> str:
    return f"SELLER:{random.randint(100000, 999999)}"

scale_rows = []

for size in REGISTRY_SIZES:
    print(f"\n--- Registry size: {size} sellers ---")

    # Build the candidate pool: 5 real + (size - 5) random distractors
    real_ids = list(SELLERS.values())
    distractors = [random_seller_id() for _ in range(size - 5)]
    candidate_pool = real_ids + distractors
    # Deduplicate in case of collision
    candidate_pool = list(dict.fromkeys(candidate_pool))

    correct = 0
    tested = 0

    for name in IMAGES:
        stego = f"results/quality_scale/hidden_{name}.png"
        true_seller = SELLERS[name]

        for atk_name, atk_fn, atk_kwargs in ATTACKS_TO_TEST:
            attacked = f"results/quality_scale/scale_{name}_{atk_name}.png"
            if atk_fn is None: copyfile(stego, attacked)
            else: atk_fn(stego, attacked, **atk_kwargs)

            result = hidden.identify(attacked, candidate_pool)
            predicted = result["match"]["seller_id"]
            correct_ber = result["match"]["ber"]

            # Also get the "margin" — how much better was the correct seller than the next?
            correct_score = next((r["ber"] for r in result["ranking"]
                                  if r["seller_id"] == true_seller), 1.0)
            top_wrong_score = next((r["ber"] for r in result["ranking"]
                                    if r["seller_id"] != true_seller), 1.0)
            margin = top_wrong_score - correct_score  # positive if correct wins

            is_right = (predicted == true_seller)
            correct += int(is_right)
            tested += 1

            scale_rows.append({
                "registry_size": size,
                "image": name,
                "attack": atk_name,
                "correct": is_right,
                "correct_ber": round(correct_score, 3),
                "top_wrong_ber": round(top_wrong_score, 3),
                "margin": round(margin, 3),
            })

    acc = 100 * correct / tested
    subset = [r for r in scale_rows if r["registry_size"] == size]
    margins = [r["margin"] for r in subset]
    print(f"Accuracy: {correct}/{tested} = {acc:.1f}%")
    print(f"Mean margin: {sum(margins)/len(margins):.3f}   Min margin: {min(margins):.3f}")

with open("output/hidden_scale.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(scale_rows[0].keys()))
    w.writeheader()
    w.writerows(scale_rows)

print("\n" + "=" * 70)
print("HIDDEN SCALE SUMMARY")
print("=" * 70)
print(f"{'Size':<8} {'Accuracy':<12} {'Mean margin':<15} {'Min margin':<12}")
print("-" * 50)
for size in REGISTRY_SIZES:
    subset = [r for r in scale_rows if r["registry_size"] == size]
    acc = 100 * sum(r["correct"] for r in subset) / len(subset)
    margins = [r["margin"] for r in subset]
    print(f"{size:<8} {acc:<12.1f}% {sum(margins)/len(margins):<15.3f} {min(margins):<12.3f}")

print("\nCSVs saved: output/imperceptibility.csv, output/hidden_scale.csv")