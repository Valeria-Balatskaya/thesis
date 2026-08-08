# tests/test_traceability.py
# Simulate a small marketplace: 5 sellers, each with their own product image.
# Attacker steals one seller's image, applies an attack, uploads it elsewhere.
# Can we identify the original seller?

import sys, os, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.traceability import SellerRegistry
from src import attacks
from shutil import copyfile

IMAGES = ["baboon", "airplane", "peppers", "splash", "house"]

# Each SIPI image belongs to a different fictional seller.
SELLERS = {
    "baboon":   "SELLER:acme_electronics",
    "airplane": "SELLER:aero_parts_co",
    "peppers":  "SELLER:fresh_market",
    "splash":   "SELLER:aqua_designs",
    "house":    "SELLER:home_goods_llc",
}

ATTACKS = [
    ("clean",     None,                   None,      None),
    ("jpeg_q70",  attacks.jpeg_compress,  "quality", 70),
    ("jpeg_q30",  attacks.jpeg_compress,  "quality", 30),
    ("noise_s15", attacks.gaussian_noise, "sigma",   15),
    ("resize",    attacks.resize_attack,  "scale",   0.7),
]

os.makedirs("results/traceability", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Step 1: register every seller with their watermarked image
registry = SellerRegistry()
for image_name, seller_id in SELLERS.items():
    orig = f"data/sipi/{image_name}.png"
    wmk  = f"results/traceability/wmk_{image_name}.png"
    registry.register(seller_id, orig, wmk)

print(f"Registered {len(registry)} sellers.\n")

# Step 2: for every image, apply each attack, try to identify seller
rows = []
correct = 0
total   = 0

print(f"{'True seller':<30} {'Attack':<12} {'Predicted':<30} {'Sim':<8} {'Correct?'}")
print("-" * 90)

for image_name, true_seller in SELLERS.items():
    stego = f"results/traceability/wmk_{image_name}.png"

    for atk_name, atk_fn, kw, val in ATTACKS:
        attacked = f"results/traceability/{image_name}_{atk_name}.png"
        if atk_fn is None:
            copyfile(stego, attacked)
        else:
            atk_fn(stego, attacked, **{kw: val})

        result = registry.identify(attacked, threshold=6.0)
        predicted = result["match"]["seller_id"] if result["match"] else "NO MATCH"
        sim       = result["match"]["similarity"] if result["match"] else 0.0
        is_right  = (predicted == true_seller)
        correct  += int(is_right)
        total    += 1

        rows.append({
            "true_seller": true_seller,
            "attack": atk_name,
            "predicted": predicted,
            "similarity": sim,
            "correct": is_right,
            "top_wrong_sim": result["ranking"][1]["similarity"]
                             if len(result["ranking"]) > 1 else None,
        })

        mark = "OK" if is_right else "WRONG"
        print(f"{true_seller:<30} {atk_name:<12} {predicted:<30} {sim:<8} {mark}")

print("-" * 90)
print(f"\nIdentification accuracy: {correct}/{total} = {100*correct/total:.1f}%")

with open("output/traceability_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("CSV: output/traceability_results.csv")