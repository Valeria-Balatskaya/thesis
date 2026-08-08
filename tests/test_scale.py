# tests/test_scale.py
# How does traceability accuracy scale with seller registry size?
# We reuse the same 5 SIPI images but register N *variants* of each seller_id.
# Then we test: given a watermarked+attacked image, does the correct seller
# still rank first when the registry has 10, 50, 100, 500 competing sellers?

import sys, os, csv, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.traceability import SellerRegistry
from src import attacks
from shutil import copyfile

random.seed(42)

IMAGES = ["baboon", "airplane", "peppers", "splash", "house"]
REGISTRY_SIZES = [10, 50, 100, 500]
ATTACKS = [
    ("clean",     None,                   None,      None),
    ("jpeg_q50",  attacks.jpeg_compress,  "quality", 50),
    ("noise_s15", attacks.gaussian_noise, "sigma",   15),
]

os.makedirs("results/scale", exist_ok=True)
os.makedirs("output", exist_ok=True)

def random_seller_id() -> str:
    return f"SELLER:{random.randint(10000, 99999)}"

# For each registry size:
#   - Register N-5 random "distractor" sellers on random SIPI images
#   - Register the 5 "real" sellers we'll test on
#   - Test identification for each real seller under each attack
rows = []

for size in REGISTRY_SIZES:
    print(f"\n=== Registry size: {size} sellers ===")
    registry = SellerRegistry()

    # Distractors: (size - 5) fake sellers, each on a random SIPI image
    real_ids = [f"REAL_SELLER:{img}" for img in IMAGES]
    for i in range(size - 5):
        distractor_id = random_seller_id()
        img = random.choice(IMAGES)
        orig = f"data/sipi/{img}.png"
        wmk  = f"results/scale/n{size}_distractor_{i}.png"
        registry.register(distractor_id, orig, wmk)

    # Real sellers: one per SIPI image
    for img, seller_id in zip(IMAGES, real_ids):
        orig = f"data/sipi/{img}.png"
        wmk  = f"results/scale/n{size}_{img}_real.png"
        registry.register(seller_id, orig, wmk)

    print(f"Registered {len(registry)} sellers ({size - 5} distractors + 5 real)")

    # Test each real seller under each attack
    correct_total = 0
    tested_total  = 0
    for img, true_seller in zip(IMAGES, real_ids):
        stego = f"results/scale/n{size}_{img}_real.png"
        for atk_name, atk_fn, kw, val in ATTACKS:
            attacked = f"results/scale/n{size}_{img}_{atk_name}.png"
            if atk_fn is None:
                copyfile(stego, attacked)
            else:
                atk_fn(stego, attacked, **{kw: val})

            result = registry.identify(attacked, threshold=6.0)
            predicted = result["match"]["seller_id"] if result["match"] else "NO_MATCH"
            is_right  = (predicted == true_seller)
            correct_total += int(is_right)
            tested_total  += 1

            # margin = correct seller sim - top wrong seller sim
            ranking = result["ranking"]
            correct_sim = next((r["similarity"] for r in ranking
                                if r["seller_id"] == true_seller), 0.0)
            top_wrong_sim = next((r["similarity"] for r in ranking
                                  if r["seller_id"] != true_seller), 0.0)
            margin = correct_sim - top_wrong_sim

            rows.append({
                "registry_size": size,
                "image": img,
                "attack": atk_name,
                "correct": is_right,
                "correct_sim": round(correct_sim, 3),
                "top_wrong_sim": round(top_wrong_sim, 3),
                "margin": round(margin, 3),
            })

    acc = 100 * correct_total / tested_total
    print(f"Accuracy at size {size}: {correct_total}/{tested_total} = {acc:.1f}%")

# Save CSV
with open("output/scale_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# Summary
print("\n" + "=" * 60)
print("SUMMARY: accuracy and mean margin per registry size")
print("=" * 60)
print(f"{'Size':<8} {'Accuracy':<12} {'Mean margin':<15} {'Min margin':<12}")
for size in REGISTRY_SIZES:
    subset = [r for r in rows if r["registry_size"] == size]
    acc = 100 * sum(r["correct"] for r in subset) / len(subset)
    margins = [r["margin"] for r in subset]
    print(f"{size:<8} {acc:<12.1f}% {sum(margins)/len(margins):<15.3f} {min(margins):<12.3f}")

print("\nCSV: output/scale_results.csv")