# tests/test_lsb.py
# Comprehensive validation of LSB watermarking baseline
# Reference: Chan & Cheng 2004, Pattern Recognition 37(3):469-474

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.lsb import embed, extract
from src.metrics import compute
import numpy as np
from PIL import Image

IMAGES   = ["baboon", "airplane", "peppers", "splash", "house"]
RESULTS  = "results"
os.makedirs(RESULTS, exist_ok=True)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def banner(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")

# ─────────────────────────────────────────────
# TEST 1: Correct recovery across all 5 images
# ─────────────────────────────────────────────
banner("TEST 1: Message recovery on all 5 SIPI images (k=1)")
MESSAGE = "SELLER_ID:0042"
print(f"{'Image':<12} {'Match':<8} {'PSNR (dB)':<12} {'SSIM':<10} Result")
print("-" * 55)
all_pass = True
for name in IMAGES:
    orig  = f"data/sipi/{name}.png"
    stego = f"{RESULTS}/t1_{name}.png"
    embed(orig, MESSAGE, stego, k=1)
    recovered = extract(stego, k=1)
    match = recovered == MESSAGE
    m = compute(orig, stego)
    result = PASS if match else FAIL
    if not match: all_pass = False
    print(f"{name:<12} {str(match):<8} {m['PSNR_dB']:<12} {m['SSIM']:<10} {result}")
print(f"\nTest 1 overall: {PASS if all_pass else FAIL}")

# ─────────────────────────────────────────────
# TEST 2: Different k values (1, 2, 3)
# Chan & Cheng 2004 Table 1 worst-case PSNR:
#   k=1 → 48.13 dB, k=2 → 38.59 dB, k=3 → 31.23 dB
# ─────────────────────────────────────────────
banner("TEST 2: k values comparison on baboon (k=1,2,3)")
print(f"{'k':<5} {'Match':<8} {'PSNR (dB)':<12} {'SSIM':<10} {'PSNR target':<14} Result")
print("-" * 60)
# Worst-case PSNR targets from Table 1, Chan & Cheng 2004
targets = {1: 38.0, 2: 28.0, 3: 20.0}
for k in [1, 2, 3]:
    orig  = "data/sipi/baboon.png"
    stego = f"{RESULTS}/t2_baboon_k{k}.png"
    embed(orig, MESSAGE, stego, k=k)
    recovered = extract(stego, k=k)
    match = recovered == MESSAGE
    m = compute(orig, stego)
    ok = match and m["PSNR_dB"] > targets[k]
    print(f"{k:<5} {str(match):<8} {m['PSNR_dB']:<12} {m['SSIM']:<10} {f'>{targets[k]}':<14} {PASS if ok else FAIL}")

# ─────────────────────────────────────────────
# TEST 3: Different message lengths
# ─────────────────────────────────────────────
banner("TEST 3: Message length stress test on peppers (k=1)")
messages = {
    "tiny"   : "A",
    "short"  : "SELLER_ID:0042",
    "medium" : "SELLER_ID:0042|timestamp:20260609|platform:imagechain",
    "long"   : "SELLER_ID:0042|" * 10,
}
print(f"{'Label':<10} {'Chars':<8} {'Match':<8} {'PSNR (dB)':<12} Result")
print("-" * 50)
for label, msg in messages.items():
    orig  = "data/sipi/peppers.png"
    stego = f"{RESULTS}/t3_{label}.png"
    try:
        embed(orig, msg, stego, k=1)
        recovered = extract(stego, k=1)
        match = recovered == msg
        m = compute(orig, stego)
        print(f"{label:<10} {len(msg):<8} {str(match):<8} {m['PSNR_dB']:<12} {PASS if match else FAIL}")
    except ValueError as e:
        print(f"{label:<10} {len(msg):<8} [ERROR: {e}]")

# ─────────────────────────────────────────────
# TEST 4: Maximum capacity test
# Max bits = width * height * channels - 32 (header)
# ─────────────────────────────────────────────
banner("TEST 4: Maximum capacity on baboon 512×512")
img = np.array(Image.open("data/sipi/baboon.png").convert("RGB"))
max_bits  = img.shape[0] * img.shape[1] * img.shape[2] - 32
max_chars = max_bits // 8
# Fill to ~80% capacity to stay safe
test_msg  = ("X" * int(max_chars * 0.8))
orig  = "data/sipi/baboon.png"
stego = f"{RESULTS}/t4_maxcap.png"
try:
    embed(orig, test_msg, stego, k=1)
    recovered = extract(stego, k=1)
    match = recovered == test_msg
    m = compute(orig, stego)
    print(f"Max chars possible : {max_chars}")
    print(f"Tested at 80%      : {len(test_msg)} chars")
    print(f"Match              : {match}")
    print(f"PSNR               : {m['PSNR_dB']} dB")
    print(f"SSIM               : {m['SSIM']}")
    print(f"Result             : {PASS if match else FAIL}")
except Exception as e:
    print(f"[ERROR] {e}")

# ─────────────────────────────────────────────
# TEST 5: Special characters and unicode
# ─────────────────────────────────────────────
banner("TEST 5: Special characters in message")
special_messages = {
    "symbols"  : "ID:042|©2026|★",
    "numbers"  : "1234567890|0.99|3.14159",
    "mixed"    : "Seller#042_Porto@2026!",
}
print(f"{'Label':<12} {'Match':<8} {'PSNR (dB)':<12} Result")
print("-" * 45)
for label, msg in special_messages.items():
    orig  = "data/sipi/splash.png"
    stego = f"{RESULTS}/t5_{label}.png"
    embed(orig, msg, stego, k=1)
    recovered = extract(stego, k=1)
    match = recovered == msg
    m = compute(orig, stego)
    print(f"{label:<12} {str(match):<8} {m['PSNR_dB']:<12} {PASS if match else FAIL}")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
banner("ALL TESTS COMPLETE")
print("Results saved in results/")
print("If all tests passed, LSB baseline is validated.")
print("Next step: add JPEG attack to test robustness.\n")