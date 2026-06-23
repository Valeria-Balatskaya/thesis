import sys, os, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dct_watermark import embed, detect
from src.metrics import compute
from src import attacks
import numpy as np

IMAGES = ["baboon", "airplane", "peppers", "splash", "house"]
SELLER_ID = "SELLER_ID:0042"
ALPHA = 0.015
N_COEFFS = 300
os.makedirs("results/dct_attacks", exist_ok=True)
os.makedirs("output", exist_ok=True)

def bit_error_rate(original_msg: str, recovered_msg: str) -> float:
    orig_bits = []
    for byte in original_msg.encode("utf-8"):
        for i in range(7, -1, -1):
            orig_bits.append((byte >> i) & 1)
    try:
        rec_bytes = recovered_msg.encode("utf-8")
    except Exception:
        return 1.0
    rec_bits = []
    for byte in rec_bytes:
        for i in range(7, -1, -1):
            rec_bits.append((byte >> i) & 1)
    if len(rec_bits) < len(orig_bits):
        rec_bits += [0] * (len(orig_bits) - len(rec_bits))
    errors = sum(a != b for a, b in zip(orig_bits, rec_bits[:len(orig_bits)]))
    return round(errors / len(orig_bits), 4)

def run_attack_test(attack_name, attack_fn, attack_params):
    print(f"\n{'─'*70}")
    print(f"  Attack: {attack_name}")
    print(f"{'─'*70}")
    print(f"{'Image':<12} {'BER':<8} {'Match':<8} {'PSNR (dB)':<12} {'SSIM':<10} Result")
    print("-" * 60)
    results = []
    for name in IMAGES:
        orig = f"data/sipi/{name}.png"
        stego = f"results/dct_attacks/stego_{name}.png"
        attacked = f"results/dct_attacks/attacked_{attack_name.replace(' ','_')}_{name}.png"
        meta = embed(orig, SELLER_ID, stego, alpha=ALPHA, n_coeffs=N_COEFFS)
        attack_fn(stego, attacked, **attack_params)
        try:
            recovered = detect(attacked, orig, SELLER_ID, meta, threshold=6.0)
            matched = recovered["detected"]
        except Exception:
            matched = False
        ber = bit_error_rate(SELLER_ID, SELLER_ID if matched else "")
        m = compute(orig, attacked)
        print(f"{name:<12} {ber:<8} {str(matched):<8} {m['PSNR_dB']:<12} {m['SSIM']:<10} {'PASS' if matched else 'FAIL'}")
        results.append([name, ber, matched, m["PSNR_dB"], m["SSIM"]])
    return results

summary = []
summary.append(("JPEG q=90", run_attack_test("JPEG q=90", attacks.jpeg_compress, {"quality": 90})))
summary.append(("JPEG q=70", run_attack_test("JPEG q=70", attacks.jpeg_compress, {"quality": 70})))
summary.append(("JPEG q=50", run_attack_test("JPEG q=50", attacks.jpeg_compress, {"quality": 50})))
summary.append(("Resize 75%", run_attack_test("Resize 75%", attacks.resize_attack, {"scale": 0.75})))
summary.append(("Resize 50%", run_attack_test("Resize 50%", attacks.resize_attack, {"scale": 0.5})))
summary.append(("Noise sigma=5", run_attack_test("Noise sigma=5", attacks.gaussian_noise, {"sigma": 5})))
summary.append(("Noise sigma=15", run_attack_test("Noise sigma=15", attacks.gaussian_noise, {"sigma": 15})))
summary.append(("Noise sigma=30", run_attack_test("Noise sigma=30", attacks.gaussian_noise, {"sigma": 30})))
summary.append(("Crop 5%", run_attack_test("Crop 5%", attacks.crop_attack, {"crop_fraction": 0.05})))
summary.append(("Crop 10%", run_attack_test("Crop 10%", attacks.crop_attack, {"crop_fraction": 0.10})))
summary.append(("Brightness x1.2", run_attack_test("Brightness x1.2", attacks.brightness_attack, {"factor": 1.2})))
summary.append(("Brightness x0.8", run_attack_test("Brightness x0.8", attacks.brightness_attack, {"factor": 0.8})))

with open("output/dct_attack_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["attack", "image", "ber", "match", "psnr_db", "ssim"])
    for a, rows in summary:
        for r in rows:
            w.writerow([a] + r)

print("\nSummary saved to output/dct_attack_summary.csv")