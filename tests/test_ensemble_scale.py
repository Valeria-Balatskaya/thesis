# tests/test_ensemble_scale.py
# Realistic scale test: each distractor seller gets a UNIQUE image (not the
# same 5 SIPI images reused). This models actual marketplace conditions where
# every seller has different products.
#
# We report both accuracy AND the number of times each detector "won", so we
# can see whether DCT or HiDDeN carries the ensemble at scale.

import sys, os, csv, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shutil import copyfile

from src.ensemble import EnsembleWatermarker
from src.distractor_gen import generate_distractor
from src import attacks, ecommerce_attacks as ea

random.seed(42)

CHECKPOINT = "checkpoints/hidden_final.pt"
IMAGES = ["baboon", "airplane", "peppers", "splash", "house"]
SELLERS = {
    "baboon":   "SELLER:acme_electronics",
    "airplane": "SELLER:aero_parts_co",
    "peppers":  "SELLER:fresh_market",
    "splash":   "SELLER:aqua_designs",
    "house":    "SELLER:home_goods_llc",
}
REGISTRY_SIZES = [5, 20, 100, 500]

ATTACKS_TO_TEST = [
    ("clean",              None,                              {}),
    ("jpeg_q30",           attacks.jpeg_compress,             {"quality": 30}),
    ("brightness_1.2",     attacks.brightness_attack,         {"factor": 1.2}),
    ("crop_0.1",           attacks.crop_attack,               {"crop_fraction": 0.1}),
    ("instagram_filter",   ea.instagram_filter,               {}),
    ("print_photograph",   ea.print_photograph_simulation,    {}),
]

os.makedirs("results/ensemble_scale", exist_ok=True)
os.makedirs("results/distractors", exist_ok=True)
os.makedirs("output", exist_ok=True)


def random_seller_id() -> str:
    return f"SELLER:{random.randint(100000, 999999)}"


print("Initializing ensemble...")
ensemble = EnsembleWatermarker(CHECKPOINT)

rows = []

# Pre-generate distractor content pool (we reuse across sizes to save time
# — a size-500 test uses the same distractors as size-100, plus more)
print("\nGenerating distractor content pool (up to 500 unique images)...")
max_distractors = max(REGISTRY_SIZES) - 5
distractor_originals = []
for i in range(max_distractors):
    path = f"results/distractors/distractor_{i:04d}.png"
    if not os.path.exists(path):
        generate_distractor(f"distractor_seed_{i}", size=256, output_path=path)
    distractor_originals.append(path)
print(f"  {len(distractor_originals)} distractor images ready.")

for size in REGISTRY_SIZES:
    print(f"\n=== Registry size: {size} sellers ===")

    real_ids = list(SELLERS.values())
    distractor_ids = [random_seller_id() for _ in range(size - 5)]
    # Deduplicate any accidental collisions
    all_ids = list(dict.fromkeys(real_ids + distractor_ids))

    print(f"  Registering {len(all_ids)} sellers (this takes ~{len(all_ids)*0.6:.0f}s)...")
    registry = []
    for i, sid in enumerate(all_ids):
        if sid in real_ids:
            img_name = next(k for k, v in SELLERS.items() if v == sid)
            orig = f"data/sipi/{img_name}.png"
        else:
            # Each distractor gets a UNIQUE image — this is the key change
            distractor_idx = i - len(real_ids)
            orig = distractor_originals[distractor_idx]

        dct_out    = f"results/ensemble_scale/n{size}_{i}_dct.png"
        hidden_out = f"results/ensemble_scale/n{size}_{i}_hidden.png"
        r = ensemble.embed(orig, sid, dct_out, hidden_out)
        registry.append({
            "seller_id": sid, "original_path": orig,
            "dct_meta": r["dct_meta"],
            "dct_stego": dct_out, "hidden_stego": hidden_out,
        })

    # Test: for each real seller, attack their watermarked images and try to identify
    correct = 0
    tested = 0
    for img_name, true_seller in SELLERS.items():
        real_entry = next(r for r in registry if r["seller_id"] == true_seller)
        for atk_name, atk_fn, atk_kwargs in ATTACKS_TO_TEST:
            atk_dct    = f"results/ensemble_scale/n{size}_{img_name}_{atk_name}_dct.png"
            atk_hidden = f"results/ensemble_scale/n{size}_{img_name}_{atk_name}_hidden.png"
            if atk_fn is None:
                copyfile(real_entry["dct_stego"],    atk_dct)
                copyfile(real_entry["hidden_stego"], atk_hidden)
            else:
                atk_fn(real_entry["dct_stego"],    atk_dct,    **atk_kwargs)
                atk_fn(real_entry["hidden_stego"], atk_hidden, **atk_kwargs)

            res_d = ensemble.identify(atk_dct,    registry)
            res_h = ensemble.identify(atk_hidden, registry)
            m_d = res_d["match"]
            m_h = res_h["match"]
            best = None
            if m_d and m_h:
                best = m_d if m_d["confidence"] >= m_h["confidence"] else m_h
            elif m_d: best = m_d
            elif m_h: best = m_h

            is_right = best is not None and best["seller_id"] == true_seller
            correct += int(is_right)
            tested += 1

            rows.append({
                "registry_size": size,
                "image": img_name,
                "attack": atk_name,
                "correct": is_right,
                "winner": best["winning_detector"] if best else "none",
                "confidence": round(best["confidence"], 3) if best else 0.0,
            })

    acc = 100 * correct / tested
    print(f"  Accuracy: {correct}/{tested} = {acc:.1f}%")

with open("output/ensemble_scale.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

print("\n" + "=" * 60)
print("ENSEMBLE SCALE SUMMARY (unique-content distractors)")
print("=" * 60)
print(f"{'Size':<8} {'Accuracy':<12} {'DCT wins':<12} {'HiDDeN wins':<12}")
print("-" * 45)
for size in REGISTRY_SIZES:
    subset = [r for r in rows if r["registry_size"] == size]
    acc = 100 * sum(r["correct"] for r in subset) / len(subset)
    dct_wins = sum(1 for r in subset if r["correct"] and r["winner"] == "dct")
    h_wins   = sum(1 for r in subset if r["correct"] and r["winner"] == "hidden")
    print(f"{size:<8} {acc:<12.1f}% {dct_wins:<12} {h_wins:<12}")

print("\nCSV: output/ensemble_scale.csv")