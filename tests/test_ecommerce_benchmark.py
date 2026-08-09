# tests/test_ecommerce_benchmark.py
# Master benchmark: LSB, DCT, HiDDeN against the classical + e-commerce
# attack suites. Produces a single CSV that becomes the main thesis result table.

import sys, os, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shutil import copyfile

from src.lsb import embed as lsb_embed, extract as lsb_extract
from src.dct_watermark import embed as dct_embed, detect as dct_detect
from src.hidden_inference import HiddenModel
from src import attacks
from src import ecommerce_attacks as ea

CHECKPOINT = "checkpoints/hidden_final.pt"
IMAGES = ["baboon", "airplane", "peppers", "splash", "house"]
SELLERS = {
    "baboon":   "SELLER:acme_electronics",
    "airplane": "SELLER:aero_parts_co",
    "peppers":  "SELLER:fresh_market",
    "splash":   "SELLER:aqua_designs",
    "house":    "SELLER:home_goods_llc",
}
DCT_ALPHA = 0.02
DCT_N_COEFFS = 500
DCT_THRESHOLD = 6.0

# (name, function, kwargs) — merged classical + e-commerce
ATTACK_SUITE = [
    ("clean",              None,                              {}),
    # Classical
    ("jpeg_q70",           attacks.jpeg_compress,             {"quality": 70}),
    ("jpeg_q30",           attacks.jpeg_compress,             {"quality": 30}),
    ("resize_0.5",         attacks.resize_attack,             {"scale": 0.5}),
    ("noise_s15",          attacks.gaussian_noise,            {"sigma": 15}),
    ("crop_0.1",           attacks.crop_attack,               {"crop_fraction": 0.1}),
    ("brightness_1.2",     attacks.brightness_attack,         {"factor": 1.2}),
    # E-commerce specific
    ("marketplace_square", ea.marketplace_square,             {"target_size": 512}),
    ("instagram_filter",   ea.instagram_filter,               {}),
    ("screenshot",         ea.screenshot_simulation,          {}),
    ("print_photograph",   ea.print_photograph_simulation,    {}),
]

os.makedirs("results/ecommerce_benchmark", exist_ok=True)
os.makedirs("output", exist_ok=True)


def bits_of(msg: str) -> list[int]:
    out = []
    for byte in msg.encode("utf-8"):
        for i in range(7, -1, -1):
            out.append((byte >> i) & 1)
    return out


def ber(orig: str, recovered: str) -> float:
    o = bits_of(orig)
    try:
        r = bits_of(recovered)
    except Exception:
        return 1.0
    if len(r) < len(o):
        r += [0] * (len(o) - len(r))
    r = r[:len(o)]
    return sum(a != b for a, b in zip(o, r)) / len(o)


print("Loading HiDDeN checkpoint...")
hidden = HiddenModel(CHECKPOINT)
print(f"HiDDeN device={hidden.device}, msg_len={hidden.msg_len}\n")

# Embed once per method per image
print("Embedding watermarks...")
for name in IMAGES:
    orig = f"data/sipi/{name}.png"
    lsb_embed(orig, SELLERS[name], f"results/ecommerce_benchmark/lsb_{name}.png", k=1)
    dct_embed(orig, SELLERS[name], f"results/ecommerce_benchmark/dct_{name}.png",
              alpha=DCT_ALPHA, n_coeffs=DCT_N_COEFFS)
    hidden.embed(orig, SELLERS[name], f"results/ecommerce_benchmark/hidden_{name}.png")
    print(f"  {name} — LSB, DCT, HiDDeN")

# Pre-compute DCT metadata for identification (need to re-embed once per image)
dct_meta = {}
for name in IMAGES:
    orig = f"data/sipi/{name}.png"
    meta = dct_embed(orig, SELLERS[name],
                     f"results/ecommerce_benchmark/dct_{name}.png",
                     alpha=DCT_ALPHA, n_coeffs=DCT_N_COEFFS)
    dct_meta[name] = (orig, meta)

seller_list = list(SELLERS.values())
rows = []

print(f"\n{'Image':<10} {'Attack':<22} {'LSB':<8} {'DCT':<10} {'HiDDeN':<10}")
print("-" * 65)

for name in IMAGES:
    true_seller = SELLERS[name]
    for atk_name, atk_fn, atk_kwargs in ATTACK_SUITE:
        # LSB
        lsb_stego = f"results/ecommerce_benchmark/lsb_{name}.png"
        lsb_att   = f"results/ecommerce_benchmark/lsb_{name}_{atk_name}.png"
        if atk_fn is None: copyfile(lsb_stego, lsb_att)
        else: atk_fn(lsb_stego, lsb_att, **atk_kwargs)
        try:
            lsb_rec = lsb_extract(lsb_att, k=1)
            lsb_survived = (ber(true_seller, lsb_rec) == 0)
        except Exception:
            lsb_survived = False

        # DCT (informed detector against the true seller's carrier)
        dct_stego = f"results/ecommerce_benchmark/dct_{name}.png"
        dct_att   = f"results/ecommerce_benchmark/dct_{name}_{atk_name}.png"
        if atk_fn is None: copyfile(dct_stego, dct_att)
        else: atk_fn(dct_stego, dct_att, **atk_kwargs)
        orig_path, meta = dct_meta[name]
        dct_result = dct_detect(dct_att, orig_path, true_seller, meta, threshold=DCT_THRESHOLD)
        dct_survived = dct_result["detected"]

        # HiDDeN (identify among the 5 candidate sellers)
        h_stego = f"results/ecommerce_benchmark/hidden_{name}.png"
        h_att   = f"results/ecommerce_benchmark/hidden_{name}_{atk_name}.png"
        if atk_fn is None: copyfile(h_stego, h_att)
        else: atk_fn(h_stego, h_att, **atk_kwargs)
        h_result = hidden.identify(h_att, seller_list)
        h_survived = (h_result["match"]["seller_id"] == true_seller)

        rows.append({
            "image": name, "attack": atk_name,
            "lsb_survived": lsb_survived,
            "dct_similarity": dct_result["similarity"],
            "dct_survived": dct_survived,
            "hidden_ber": round(h_result["match"]["ber"], 3),
            "hidden_survived": h_survived,
        })

        lsb_str = "OK" if lsb_survived else "FAIL"
        dct_str = f"{dct_result['similarity']:.1f} {'OK' if dct_survived else 'FAIL'}"
        h_str   = f"{h_result['match']['ber']:.2f} {'OK' if h_survived else 'FAIL'}"
        print(f"{name:<10} {atk_name:<22} {lsb_str:<8} {dct_str:<10} {h_str:<10}")

# Save CSV
csv_path = "output/ecommerce_benchmark.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# Summary: per-attack survival rate for each method
print("\n" + "=" * 70)
print("SUMMARY: survival rate per attack (across 5 images)")
print("=" * 70)
print(f"{'Attack':<22} {'LSB':<10} {'DCT':<10} {'HiDDeN':<10}")
print("-" * 55)
seen = []
for r in rows:
    if r["attack"] not in seen: seen.append(r["attack"])
for a in seen:
    subset = [r for r in rows if r["attack"] == a]
    lsb_rate = sum(1 for r in subset if r["lsb_survived"]) / len(subset)
    dct_rate = sum(1 for r in subset if r["dct_survived"]) / len(subset)
    h_rate   = sum(1 for r in subset if r["hidden_survived"]) / len(subset)
    print(f"{a:<22} {lsb_rate*100:>4.0f}%      {dct_rate*100:>4.0f}%      {h_rate*100:>4.0f}%")

print(f"\nCSV: {csv_path}")