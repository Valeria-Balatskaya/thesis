# tests/test_deployed_benchmark.py
# Honest benchmark of what a thief actually gets: ONE published image.
#
# Every system below publishes exactly one file per product. That file — and
# only that file — is attacked, and the attacked copy is what the detector sees.
#
#   v3_dual_storage : previous design. Public file = DCT-only copy; the HiDDeN
#                     copy never leaves the server, so it cannot help detection.
#   hidden_residual : HiDDeN alone, new residual embedding (component analysis).
#   v4_layered      : new design. HiDDeN residual + DCT layered in one file.
#
# Also measures false positives: attacked UN-watermarked originals should never
# be attributed to any seller.
#
# Run: python tests/test_deployed_benchmark.py

import sys, os, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shutil import copyfile

from src.ensemble import EnsembleWatermarker
from src.metrics import compute
from src import attacks, ecommerce_attacks as ea

CHECKPOINT = "checkpoints/hidden_final.pt"
IMAGES = ["baboon", "airplane", "peppers", "splash", "house"]
SELLERS = {
    "baboon":   "SELLER:acme_electronics",
    "airplane": "SELLER:aero_parts_co",
    "peppers":  "SELLER:fresh_market",
    "splash":   "SELLER:aqua_designs",
    "house":    "SELLER:home_goods_llc",
}

ATTACK_SUITE = [
    ("clean",              None,                              {}),
    ("jpeg_q70",           attacks.jpeg_compress,             {"quality": 70}),
    ("jpeg_q30",           attacks.jpeg_compress,             {"quality": 30}),
    ("resize_0.5",         attacks.resize_attack,             {"scale": 0.5}),
    ("noise_s15",          attacks.gaussian_noise,            {"sigma": 15}),
    ("crop_0.1",           attacks.crop_attack,               {"crop_fraction": 0.1}),
    ("brightness_1.2",     attacks.brightness_attack,         {"factor": 1.2}),
    ("marketplace_square", ea.marketplace_square,             {"target_size": 512}),
    ("instagram_filter",   ea.instagram_filter,               {}),
    ("screenshot",         ea.screenshot_simulation,          {}),
    ("print_photograph",   ea.print_photograph_simulation,    {}),
    ("native_screenshot",  ea.native_screenshot_simulation,   {}),
]

OUT = "results/deployed_benchmark"
os.makedirs(OUT, exist_ok=True)
os.makedirs("output", exist_ok=True)

ensemble = EnsembleWatermarker(CHECKPOINT)
hidden = ensemble.hidden
seller_list = list(SELLERS.values())


def _attack(fn, kwargs, src, dst):
    if fn is None:
        copyfile(src, dst)
    else:
        fn(src, dst, **kwargs)


# ─── Embed: one published file per system ─────────────────────────

v3_registry, v4_registry = [], []
published = {}   # (system, image) -> published path
quality_rows = []

print("Embedding...")
for name in IMAGES:
    orig = f"data/sipi/{name}.png"
    sid = SELLERS[name]

    v3_pub = f"{OUT}/v3_{name}.png"
    v3_meta = ensemble.embed_dual_storage(orig, sid, v3_pub, f"{OUT}/v3_{name}_serverside.png")
    v3_registry.append({"seller_id": sid, "original_path": orig,
                        "dct_meta": v3_meta["dct_meta"]})

    h_pub = f"{OUT}/hidden_{name}.png"
    hidden.embed(orig, sid, h_pub, mode="residual")

    v4_pub = f"{OUT}/v4_{name}.png"
    v4_ref = f"{OUT}/v4_{name}_ref.png"
    v4_meta = ensemble.embed(orig, sid, v4_pub, v4_ref)
    v4_registry.append({"seller_id": sid, "original_path": orig,
                        "reference_path": v4_ref, "dct_meta": v4_meta["dct_meta"]})

    published.update({("v3_dual_storage", name): v3_pub,
                      ("hidden_residual", name): h_pub,
                      ("v4_layered", name): v4_pub})

    row = {"image": name}
    for system in ("v3_dual_storage", "hidden_residual", "v4_layered"):
        q = compute(orig, published[(system, name)])
        row[f"{system}_psnr"] = q["PSNR_dB"]
        row[f"{system}_ssim"] = q["SSIM"]
    quality_rows.append(row)

SYSTEMS = ["v3_dual_storage", "hidden_residual", "v4_layered"]

print("\n" + "=" * 70)
print("IMAGE QUALITY of the PUBLISHED file (vs. original)")
print("=" * 70)
print(f"{'Image':<10}" + "".join(f"{s:>20}" for s in SYSTEMS))
for r in quality_rows:
    print(f"{r['image']:<10}" + "".join(
        f"{r[s + '_psnr']:>11.1f} dB/{r[s + '_ssim']:.3f}" for s in SYSTEMS))


# ─── Robustness: attack the published file only ───────────────────

def _identify(system, suspect):
    if system == "v3_dual_storage":
        m = ensemble.identify(suspect, v3_registry)["match"]
        return m["seller_id"] if m else None
    if system == "v4_layered":
        m = ensemble.identify(suspect, v4_registry)["match"]
        return m["seller_id"] if m else None
    m = hidden.identify(suspect, seller_list)["match"]
    return m["seller_id"] if m["ber"] < 0.40 else None


rows = []
print("\n" + "=" * 70)
print("ROBUSTNESS (correct seller identified from the attacked published file)")
print("=" * 70)
for name in IMAGES:
    for atk_name, fn, kw in ATTACK_SUITE:
        row = {"image": name, "attack": atk_name}
        for system in SYSTEMS:
            suspect = f"{OUT}/atk_{system}_{name}_{atk_name}.png"
            _attack(fn, kw, published[(system, name)], suspect)
            row[system] = _identify(system, suspect) == SELLERS[name]
        rows.append(row)

# ─── False positives: attacked originals that were never watermarked ──

fp_rows = []
for name in IMAGES:
    for atk_name, fn, kw in ATTACK_SUITE:
        suspect = f"{OUT}/fp_{name}_{atk_name}.png"
        _attack(fn, kw, f"data/sipi/{name}.png", suspect)
        row = {"image": name, "attack": atk_name}
        for system in SYSTEMS:
            row[system] = _identify(system, suspect) is not None
        fp_rows.append(row)

# ─── Summary ──────────────────────────────────────────────────────

print(f"{'Attack':<22}" + "".join(f"{s:>18}" for s in SYSTEMS))
print("-" * 76)
for atk_name, _, _ in ATTACK_SUITE:
    sub = [r for r in rows if r["attack"] == atk_name]
    print(f"{atk_name:<22}" + "".join(
        f"{100 * sum(r[s] for r in sub) / len(sub):>17.0f}%" for s in SYSTEMS))
print("-" * 76)
print(f"{'OVERALL':<22}" + "".join(
    f"{100 * sum(r[s] for r in rows) / len(rows):>17.0f}%" for s in SYSTEMS))
print(f"{'FALSE POSITIVES':<22}" + "".join(
    f"{sum(r[s] for r in fp_rows):>11d}/{len(fp_rows):<6d}" for s in SYSTEMS))

for path, data in [("output/deployed_quality.csv", quality_rows),
                   ("output/deployed_robustness.csv", rows),
                   ("output/deployed_false_positives.csv", fp_rows)]:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader()
        w.writerows(data)
print("\nCSVs: output/deployed_quality.csv, output/deployed_robustness.csv, "
      "output/deployed_false_positives.csv")
