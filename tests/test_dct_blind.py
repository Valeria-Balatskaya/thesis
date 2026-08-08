# tests/test_dct_blind.py
# Compare informed detector (Cox, needs original) vs blind detector (Barni).
# Also re-runs the attack suite so we can see how much the blind detector
# gains us on brightness / resize / crop.

import sys, os, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dct_watermark import embed, detect, detect_blind
from src import attacks
from shutil import copyfile

IMAGES    = ["baboon", "airplane", "peppers", "splash", "house"]
SELLER_ID = "SELLER_ID:0042"
WRONG_ID  = "SELLER_ID:9999"  # sanity check: should NOT detect
ALPHA, N_COEFFS = 0.02, 500

ATTACK_SUITE = [
    ("clean",          None,                        None,      None),
    ("jpeg_q50",       attacks.jpeg_compress,       "quality", 50),
    ("jpeg_q30",       attacks.jpeg_compress,       "quality", 30),
    ("resize_0.5",     attacks.resize_attack,       "scale",   0.5),
    ("noise_s15",      attacks.gaussian_noise,      "sigma",   15),
    ("crop_0.1",       attacks.crop_attack,         "crop_fraction", 0.1),
    ("brightness_1.2", attacks.brightness_attack,   "factor",  1.2),
]

os.makedirs("results/dct_blind", exist_ok=True)
os.makedirs("output", exist_ok=True)

rows = []
print(f"\n{'Image':<10} {'Attack':<16} {'Informed sim':<14} {'Blind sim':<12} {'Blind (wrong ID)':<18}")
print("-" * 75)

for name in IMAGES:
    orig  = f"data/sipi/{name}.png"
    stego = f"results/dct_blind/dct_{name}.png"
    meta  = embed(orig, SELLER_ID, stego, alpha=ALPHA, n_coeffs=N_COEFFS)

    for atk_name, atk_fn, kw, val in ATTACK_SUITE:
        attacked = f"results/dct_blind/dct_{name}_{atk_name}.png"
        if atk_fn is None:
            copyfile(stego, attacked)
        else:
            atk_fn(stego, attacked, **{kw: val})

        informed = detect(attacked, orig, SELLER_ID, meta, threshold=6.0)
        blind    = detect_blind(attacked, SELLER_ID, n_coeffs=N_COEFFS)
        blind_wrong = detect_blind(attacked, WRONG_ID, n_coeffs=N_COEFFS)

        rows.append({
            "image": name, "attack": atk_name,
            "informed_sim": informed["similarity"],
            "blind_sim": blind["similarity"],
            "blind_wrong_id_sim": blind_wrong["similarity"],
            "informed_detected": informed["detected"],
            "blind_detected": blind["detected"],
        })
        print(f"{name:<10} {atk_name:<16} {informed['similarity']:<14} {blind['similarity']:<12} {blind_wrong['similarity']:<18}")

with open("output/dct_blind_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("\n" + "=" * 60)
print("SUMMARY: blind detector survival per attack")
print("=" * 60)
attacks_seen = []
for r in rows:
    if r["attack"] not in attacks_seen:
        attacks_seen.append(r["attack"])
for a in attacks_seen:
    blind_rate = sum(1 for r in rows if r["attack"] == a and r["blind_detected"]) / len(IMAGES)
    print(f"{a:<16} blind survival: {blind_rate*100:.0f}%")

print("\nCSV: output/dct_blind_results.csv")