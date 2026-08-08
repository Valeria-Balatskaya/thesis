# tests/test_robustness_compare.py
# Phase 1: Head-to-head robustness of LSB vs DCT under a fixed attack suite.
# Produces a single CSV that becomes Chapter 4, Table 1 of the thesis.

import sys, os, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.lsb import embed as lsb_embed, extract as lsb_extract
from src.dct_watermark import embed as dct_embed, detect as dct_detect
from src.metrics import compute
from src import attacks

IMAGES    = ["baboon", "airplane", "peppers", "splash", "house"]
SELLER_ID = "SELLER_ID:0042"
DCT_ALPHA = 0.02
DCT_N     = 500  # sweet spot from your tuning run

# Attack suite: (name, function, kwarg_name, value)
ATTACK_SUITE = [
    ("clean",         None,                        None,      None),
    ("jpeg_q90",      attacks.jpeg_compress,       "quality", 90),
    ("jpeg_q70",      attacks.jpeg_compress,       "quality", 70),
    ("jpeg_q50",      attacks.jpeg_compress,       "quality", 50),
    ("jpeg_q30",      attacks.jpeg_compress,       "quality", 30),
    ("resize_0.5",    attacks.resize_attack,       "scale",   0.5),
    ("noise_s5",      attacks.gaussian_noise,      "sigma",   5),
    ("noise_s15",     attacks.gaussian_noise,      "sigma",   15),
    ("crop_0.1",      attacks.crop_attack,         "crop_fraction", 0.1),
    ("brightness_1.2", attacks.brightness_attack,  "factor",  1.2),
]

os.makedirs("results/robustness", exist_ok=True)
os.makedirs("output", exist_ok=True)

def bits_of(msg: str) -> list[int]:
    out = []
    for byte in msg.encode("utf-8"):
        for i in range(7, -1, -1):
            out.append((byte >> i) & 1)
    return out

def ber(orig_msg: str, recovered_msg: str) -> float:
    o = bits_of(orig_msg)
    try:
        r = bits_of(recovered_msg)
    except Exception:
        return 1.0
    if len(r) < len(o):
        r += [0] * (len(o) - len(r))
    r = r[:len(o)]
    return sum(a != b for a, b in zip(o, r)) / len(o)

rows = []
print(f"\n{'Method':<6} {'Image':<10} {'Attack':<16} {'BER/Sim':<12} {'Survived':<10}")
print("-" * 60)

for name in IMAGES:
    orig = f"data/sipi/{name}.png"
    lsb_stego = f"results/robustness/lsb_{name}.png"
    dct_stego = f"results/robustness/dct_{name}.png"

    # Embed once per method per image
    lsb_embed(orig, SELLER_ID, lsb_stego, k=1)
    dct_meta = dct_embed(orig, SELLER_ID, dct_stego, alpha=DCT_ALPHA, n_coeffs=DCT_N)

    for atk_name, atk_fn, kwarg_name, kwarg_val in ATTACK_SUITE:
        # Attack each stego image
        lsb_attacked = f"results/robustness/lsb_{name}_{atk_name}.png"
        dct_attacked = f"results/robustness/dct_{name}_{atk_name}.png"

        if atk_fn is None:
            # Clean case: copy through
            from shutil import copyfile
            copyfile(lsb_stego, lsb_attacked)
            copyfile(dct_stego, dct_attacked)
        else:
            atk_fn(lsb_stego, lsb_attacked, **{kwarg_name: kwarg_val})
            atk_fn(dct_stego, dct_attacked, **{kwarg_name: kwarg_val})

        # LSB: try to recover, measure BER. Survived = BER == 0 (perfect match).
        try:
            lsb_rec = lsb_extract(lsb_attacked, k=1)
            lsb_ber = ber(SELLER_ID, lsb_rec)
        except Exception:
            lsb_ber = 1.0
        lsb_survived = lsb_ber == 0.0

        # DCT: run detector, survived = sim > threshold
        dct_result = dct_detect(dct_attacked, orig, SELLER_ID, dct_meta, threshold=6.0)
        dct_survived = dct_result["detected"]

        rows.append({
            "image": name, "attack": atk_name,
            "lsb_ber": round(lsb_ber, 4), "lsb_survived": lsb_survived,
            "dct_similarity": dct_result["similarity"], "dct_survived": dct_survived,
        })

        print(f"{'LSB':<6} {name:<10} {atk_name:<16} {lsb_ber:<12.4f} {str(lsb_survived):<10}")
        print(f"{'DCT':<6} {name:<10} {atk_name:<16} {dct_result['similarity']:<12} {str(dct_survived):<10}")

# Save CSV
csv_path = "output/robustness_compare.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# Summary: survival rate per attack, per method
print("\n" + "=" * 60)
print("SUMMARY: survival rate per attack (across 5 images)")
print("=" * 60)
print(f"{'Attack':<16} {'LSB survival':<15} {'DCT survival':<15}")
print("-" * 60)
attacks_seen = []
for r in rows:
    if r["attack"] not in attacks_seen:
        attacks_seen.append(r["attack"])
for a in attacks_seen:
    lsb_rate = sum(1 for r in rows if r["attack"] == a and r["lsb_survived"]) / len(IMAGES)
    dct_rate = sum(1 for r in rows if r["attack"] == a and r["dct_survived"]) / len(IMAGES)
    print(f"{a:<16} {lsb_rate*100:<14.0f}% {dct_rate*100:<14.0f}%")

print(f"\nCSV saved to {csv_path}")