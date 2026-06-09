# tests/test_attacks.py
# Robustness benchmark: LSB watermark under all attack conditions
# Reference: An et al. (2024) WAVES benchmark, arXiv:2401.08573
#            Chan & Cheng (2004) Pattern Recognition 37(3):469-474

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.lsb import embed, extract
from src.metrics import compute
from src import attacks
import numpy as np

IMAGES  = ["baboon", "airplane", "peppers", "splash", "house"]
MESSAGE = "SELLER_ID:0042"
BITS    = len(MESSAGE.encode("utf-8")) * 8
os.makedirs("results/attacks", exist_ok=True)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def bit_error_rate(original_msg: str, recovered_msg: str) -> float:
    """
    Compute Bit Error Rate (BER) between original and recovered message.
    BER = 0.0 means perfect recovery. BER = 0.5 means random noise (total failure).
    """
    orig_bits = []
    for byte in original_msg.encode("utf-8"):
        for i in range(7, -1, -1):
            orig_bits.append((byte >> i) & 1)

    try:
        rec_bytes = recovered_msg.encode("utf-8")
    except Exception:
        return 1.0

    rec_bits = []
    for byte in rec_bytes[:len(orig_bits)//8]:
        for i in range(7, -1, -1):
            rec_bits.append((byte >> i) & 1)

    # Pad if shorter
    while len(rec_bits) < len(orig_bits):
        rec_bits.append(0)

    errors = sum(a != b for a, b in zip(orig_bits, rec_bits[:len(orig_bits)]))
    return round(errors / len(orig_bits), 4)


def banner(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def run_attack_test(attack_name, attack_fn, attack_params):
    """Run one attack across all 5 images and print results."""
    print(f"\n{'─'*70}")
    print(f"  Attack: {attack_name}")
    print(f"{'─'*70}")
    print(f"{'Image':<12} {'BER':<8} {'Match':<8} {'PSNR (dB)':<12} {'SSIM':<10} Result")
    print("-" * 60)

    results = []
    for name in IMAGES:
        orig  = f"data/sipi/{name}.png"
        stego = f"results/attacks/stego_{name}.png"
        attacked = f"results/attacks/attacked_{attack_name.replace(' ','_')}_{name}.png"

        # embed clean watermark
        embed(orig, MESSAGE, stego, k=1)

        # apply attack
        attack_fn(stego, attacked, **attack_params)

        # try to extract
        try:
            recovered = extract(attacked, k=1)
        except Exception:
            recovered = ""

        match = recovered == MESSAGE
        ber   = bit_error_rate(MESSAGE, recovered)
        m     = compute(orig, attacked)
        status = PASS if match else FAIL

        print(f"{name:<12} {ber:<8} {str(match):<8} {m['PSNR_dB']:<12} {m['SSIM']:<10} {status}")
        results.append({"name": name, "ber": ber, "match": match, "psnr": m["PSNR_dB"], "ssim": m["SSIM"]})

    avg_ber = round(np.mean([r["ber"] for r in results]), 4)
    accuracy = round(sum(r["match"] for r in results) / len(results) * 100, 1)
    print(f"\n  → Average BER: {avg_ber}  |  Recovery accuracy: {accuracy}%")
    return results


# ══════════════════════════════════════════════════════════════════════════════
banner("LSB ROBUSTNESS BENCHMARK — All Attacks")
print("Reference: An et al. (2024) WAVES benchmark | Chan & Cheng (2004)")
# ══════════════════════════════════════════════════════════════════════════════

all_results = {}

# JPEG attacks at different quality levels
for q in [90, 70, 50]:
    r = run_attack_test(
        f"JPEG q={q}",
        attacks.jpeg_compress,
        {"quality": q}
    )
    all_results[f"JPEG q={q}"] = r

# Resize attacks
for scale in [0.75, 0.5]:
    r = run_attack_test(
        f"Resize {int(scale*100)}%",
        attacks.resize_attack,
        {"scale": scale}
    )
    all_results[f"Resize {int(scale*100)}%"] = r

# Gaussian noise
for sigma in [5, 15, 30]:
    r = run_attack_test(
        f"Noise sigma={sigma}",
        attacks.gaussian_noise,
        {"sigma": sigma}
    )
    all_results[f"Noise sigma={sigma}"] = r

# Crop attack
for frac in [0.05, 0.10]:
    r = run_attack_test(
        f"Crop {int(frac*100)}%",
        attacks.crop_attack,
        {"crop_fraction": frac}
    )
    all_results[f"Crop {int(frac*100)}%"] = r

# Brightness
for factor in [1.2, 0.8]:
    r = run_attack_test(
        f"Brightness x{factor}",
        attacks.brightness_attack,
        {"factor": factor}
    )
    all_results[f"Brightness x{factor}"] = r

# ══════════════════════════════════════════════════════════════════════════════
banner("SUMMARY TABLE — LSB survival rate per attack")
print(f"{'Attack':<22} {'Avg BER':<10} {'Accuracy':<12} {'Verdict'}")
print("-" * 55)
for attack_name, results in all_results.items():
    avg_ber  = round(np.mean([r["ber"] for r in results]), 4)
    accuracy = round(sum(r["match"] for r in results) / len(results) * 100, 1)
    verdict  = "SURVIVES" if accuracy == 100 else ("PARTIAL" if accuracy > 0 else "DESTROYED")
    color    = "\033[92m" if verdict == "SURVIVES" else ("\033[93m" if verdict == "PARTIAL" else "\033[91m")
    print(f"{attack_name:<22} {avg_ber:<10} {accuracy}%{'':<8} {color}{verdict}\033[0m")

print("\nDone. Results saved in results/attacks/")
print("Next step: implement DCT watermarking (Section 4.3)\n")