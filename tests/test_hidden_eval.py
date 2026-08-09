# tests/test_hidden_eval.py
# Local evaluation of trained HiDDeN checkpoint on SIPI images
# under the same attack suite used for DCT. Direct head-to-head.

import sys, os, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.hidden_inference import HiddenModel
from src import attacks
from shutil import copyfile

CHECKPOINT = "checkpoints/hidden_final.pt"

IMAGES = ["baboon", "airplane", "peppers", "splash", "house"]
SELLERS = {
    "baboon":   "SELLER:acme_electronics",
    "airplane": "SELLER:aero_parts_co",
    "peppers":  "SELLER:fresh_market",
    "splash":   "SELLER:aqua_designs",
    "house":    "SELLER:home_goods_llc",
}

ATTACKS = [
    ("clean",          None,                       None,      None),
    ("jpeg_q70",       attacks.jpeg_compress,      "quality", 70),
    ("jpeg_q50",       attacks.jpeg_compress,      "quality", 50),
    ("jpeg_q30",       attacks.jpeg_compress,      "quality", 30),
    ("resize_0.5",     attacks.resize_attack,      "scale",   0.5),
    ("noise_s15",      attacks.gaussian_noise,     "sigma",   15),
    ("crop_0.1",       attacks.crop_attack,        "crop_fraction", 0.1),
    ("brightness_1.2", attacks.brightness_attack,  "factor",  1.2),
]

os.makedirs("results/hidden_eval", exist_ok=True)
os.makedirs("output", exist_ok=True)

print(f"Loading checkpoint: {CHECKPOINT}")
model = HiddenModel(CHECKPOINT)
print(f"msg_len={model.msg_len}, image_size={model.image_size}, device={model.device}\n")

# Embed each seller's image once
for img, seller in SELLERS.items():
    orig = f"data/sipi/{img}.png"
    wmk = f"results/hidden_eval/wmk_{img}.png"
    model.embed(orig, seller, wmk)
    print(f"  embedded {seller} into {img}")

rows = []
correct = 0
total = 0

print(f"\n{'True seller':<30} {'Attack':<16} {'Predicted':<30} {'BER':<8} {'Correct?'}")
print("-" * 100)

for img, true_seller in SELLERS.items():
    stego = f"results/hidden_eval/wmk_{img}.png"
    for atk_name, atk_fn, kw, val in ATTACKS:
        attacked = f"results/hidden_eval/{img}_{atk_name}.png"
        if atk_fn is None:
            copyfile(stego, attacked)
        else:
            atk_fn(stego, attacked, **{kw: val})

        result = model.identify(attacked, list(SELLERS.values()))
        pred = result["match"]["seller_id"]
        ber = result["match"]["ber"]
        is_right = (pred == true_seller)
        correct += int(is_right)
        total += 1

        rows.append({
            "true": true_seller, "attack": atk_name,
            "predicted": pred, "ber": round(ber, 3),
            "correct": is_right,
        })
        mark = "OK" if is_right else "WRONG"
        print(f"{true_seller:<30} {atk_name:<16} {pred:<30} {ber:<8.3f} {mark}")

print(f"\nHiDDeN identification accuracy: {correct}/{total} = {100*correct/total:.1f}%")

with open("output/hidden_eval.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("CSV: output/hidden_eval.csv")