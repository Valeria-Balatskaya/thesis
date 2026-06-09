# main.py
from src.lsb import embed, extract
from src.metrics import compute
import os

IMAGES  = ["baboon", "airplane", "peppers", "splash", "house"]
MESSAGE = "SELLER_ID:0042"

print(f"\n{'Image':<12} {'Recovered':<20} {'Match':<6} {'PSNR (dB)':<12} {'SSIM':<10} {'Status'}")
print("-" * 70)

for name in IMAGES:
    original = f"data/sipi/{name}.png"
    stego    = f"results/stego_lsb_{name}.png"

    # embed
    embed(original, MESSAGE, stego, k=1)

    # extract (blind)
    recovered = extract(stego, k=1)
    match = recovered == MESSAGE

    # measure quality
    m = compute(original, stego)
    psnr_ok = m["PSNR_dB"] > 38
    ssim_ok = m["SSIM"] > 0.98
    status  = "PASS" if (match and psnr_ok and ssim_ok) else "FAIL"

    print(f"{name:<12} {recovered:<20} {str(match):<6} {m['PSNR_dB']:<12} {m['SSIM']:<10} {status}")

print("-" * 70)
print("\nDone. Stego images saved in results/")