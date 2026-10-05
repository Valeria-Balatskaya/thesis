# tests/test_defence_benchmark.py
# Benchmark of the Retrieve-Align-Verify pipeline (src/defence.py) against
# modern re-use and AI-edit attacks (src/modern_attacks.py).
#
# For every attacked, watermarked image it compares
#   blind only      — HiDDeN v2 read directly
#   full pipeline   — blind, else fingerprint retrieval + alignment + verify
# and counts three outcomes: copy traced (watermark), picture recognised but
# watermark gone (fingerprint only), missed.
#
# Negative controls (the part that keeps the result honest):
#   A. un-watermarked ORIGINALS of registered products, clean and attacked:
#      may be recognised as the picture, must NEVER be reported as watermarked
#   B. photos that are NOT registered (held-out product photos — often the same
#      object photographed again — plus unrelated images): must return nothing
#   C. leak tracing: a second copy with a different ID must resolve to itself
#
# Run: python tests/test_defence_benchmark.py [image_folder] [n_registered] [strength]
#   strength scales the watermark (1.0 = the 40 dB it was trained at, 0.5 ~ 46 dB)

import sys, os, csv, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shutil import copyfile

import numpy as np

from src.hidden_v2_inference import HiddenV2Model
from src.defence import DefencePipeline
from src.distractor_gen import generate_distractor
from src import attacks, ecommerce_attacks as ea, modern_attacks as ma

CHECKPOINT = "checkpoints/hidden_v2_best.pt"
IMAGE_DIR = sys.argv[1] if len(sys.argv) > 1 else "data/products_png"
N_REGISTERED = int(sys.argv[2]) if len(sys.argv) > 2 else 35
STRENGTH = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
N_AI = 12            # AI attacks are slow on CPU: run them on the first N_AI products

ATTACKS = [
    ("clean",              None,                            {}),
    ("jpeg_q30",           attacks.jpeg_compress,           {"quality": 30}),
    ("marketplace_square", ea.marketplace_square,           {"target_size": 512}),
    ("native_screenshot",  ea.native_screenshot_simulation, {}),
    ("social_recompress",  ma.social_recompress,            {}),
    ("offcentre_crop_70",  ma.offcentre_crop,               {"keep": 0.7, "dx": 1.0, "dy": 0.0}),
    ("offcentre_crop_50",  ma.offcentre_crop,               {"keep": 0.5, "dx": 0.2, "dy": 0.9}),
    ("rotate_8",           ma.rotate,                       {"degrees": 8}),
    ("mirror",             ma.mirror,                       {}),
    ("perspective",        ma.perspective,                  {"strength": 0.12}),
    ("page_screenshot",    ma.page_screenshot,              {}),
    ("screen_photo",       ma.screen_photo,                 {}),
    ("promo_overlay",      ma.promo_overlay,                {}),
    ("inpaint_region",     ma.inpaint_region,               {}),
    ("upscale_sharpen",    ma.upscale_sharpen,              {"factor": 1.5}),
]
AI_ATTACKS = [
    ("ai_regenerate_vae",     ma.regenerate_vae,     {}),
    ("ai_replace_background", ma.replace_background, {}),
]
available = ma.ai_available()
AI_ATTACKS = [a for a in AI_ATTACKS if available[a[1].__name__]]
CONTROL_ATTACKS = ["clean", "jpeg_q30", "offcentre_crop_70", "page_screenshot", "promo_overlay"]

OUT = "results/defence_benchmark"
os.makedirs(OUT, exist_ok=True)
os.makedirs("output", exist_ok=True)
np.random.seed(0)

if not os.path.exists(CHECKPOINT):
    sys.exit(f"Checkpoint not found: {CHECKPOINT}")
files = sorted(f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg")))
if len(files) < 2:
    sys.exit(f"Need images in {IMAGE_DIR}")
N_REGISTERED = min(N_REGISTERED, max(1, len(files) * 3 // 4))
registered, held_out = files[:N_REGISTERED], files[N_REGISTERED:]

model = HiddenV2Model(CHECKPOINT)
pipeline = DefencePipeline(model)


def _attack(fn, kwargs, src, dst):
    if fn is None:
        copyfile(src, dst)
    else:
        fn(src, dst, **kwargs)


# ─── Register: every product gets two copies with different IDs ───

print(f"Watermark strength {STRENGTH}")
print(f"Registering {len(registered)} products from {IMAGE_DIR} "
      f"({len(held_out)} photos held out as unregistered)...")
products = {}
for i, fname in enumerate(registered):
    pid = i + 1
    orig = os.path.join(IMAGE_DIR, fname)
    listing_id, partner_id = 5000 + 2 * i, 5001 + 2 * i
    pipeline.register(pid, orig, [listing_id, partner_id])
    listing = f"{OUT}/pub_{pid}_listing.png"
    partner = f"{OUT}/pub_{pid}_partner.png"
    model.embed_product(orig, listing_id, listing, strength=STRENGTH)
    model.embed_product(orig, partner_id, partner, strength=STRENGTH)
    products[pid] = {"orig": orig, "listing": listing, "partner": partner,
                     "listing_id": listing_id, "partner_id": partner_id}

from src.metrics import compute as _quality
_q = [_quality(p["orig"], p["listing"]) for p in list(products.values())[:10]]
print(f"Published copies: PSNR {np.mean([q['PSNR_dB'] for q in _q]):.1f} dB, "
      f"SSIM {np.mean([q['SSIM'] for q in _q]):.4f} (first 10)")

# ─── Robustness of the published (listing) copy ───────────────────

rows = []
t0 = time.time()
# "picture only" = recognised as the right picture, watermark unreadable.
# "wrong pic"    = recognised as a DIFFERENT registered picture (no watermark claim).
# "wrong copy"   = a watermark ID was reported and it was the wrong one (must be 0).
print(f"\n{'Attack':<24}{'n':>4}{'blind':>8}{'full':>8}{'picture only':>14}{'missed':>8}"
      f"{'wrong pic':>11}{'wrong copy':>12}")
print("-" * 89)
for atk_name, fn, kwargs in ATTACKS + AI_ATTACKS:
    is_ai = atk_name.startswith("ai_")
    pids = list(products)[:N_AI] if is_ai else list(products)
    tally = {"blind": 0, "full": 0, "fp_only": 0, "missed": 0, "wrong_pic": 0, "wrong_copy": 0}
    for pid in pids:
        p = products[pid]
        atk_path = f"{OUT}/atk_{pid}_{atk_name}.png"
        _attack(fn, kwargs, p["listing"], atk_path)
        blind = pipeline.analyse(atk_path, use_alignment=False)
        full = blind if blind["verdict"] == "watermark" else pipeline.analyse(atk_path)
        traced = full["copy_id"] == p["listing_id"]
        wrong_copy = full["copy_id"] not in (None, p["listing_id"])
        wrong_pic = full["verdict"] == "fingerprint_only" and full["product_id"] != pid
        tally["blind"] += blind["copy_id"] == p["listing_id"]
        tally["full"] += traced
        tally["fp_only"] += full["verdict"] == "fingerprint_only" and full["product_id"] == pid
        tally["missed"] += full["verdict"] == "none"
        tally["wrong_pic"] += wrong_pic
        tally["wrong_copy"] += wrong_copy
        retrieve = next((s for s in full["stages"] if s["stage"] == "retrieve"), {})
        verify = next((s for s in full["stages"] if s["stage"] == "align_verify"), {})
        rows.append({"product": pid, "attack": atk_name, "blind_ok": blind["copy_id"] == p["listing_id"],
                     "verdict": full["verdict"], "copy_id": full["copy_id"], "true_copy": p["listing_id"],
                     "matched_product": full["product_id"],
                     "inliers": retrieve.get("inliers"), "similarity": retrieve.get("similarity"),
                     "bit_errors": verify.get("bit_errors"), "method": verify.get("method")})
    n = len(pids)
    print(f"{atk_name:<24}{n:>4}{tally['blind'] / n:>8.0%}{tally['full'] / n:>8.0%}"
          f"{tally['fp_only'] / n:>14.0%}{tally['missed'] / n:>8.0%}"
          f"{tally['wrong_pic']:>11}{tally['wrong_copy']:>12}")

n = len(rows)
blind_all = sum(r["blind_ok"] for r in rows) / n
full_all = sum(r["copy_id"] == r["true_copy"] for r in rows) / n
fp_all = sum(r["verdict"] == "fingerprint_only" and r["matched_product"] == r["product"] for r in rows) / n
print("-" * 89)
print(f"{'OVERALL':<24}{n:>4}{blind_all:>8.0%}{full_all:>8.0%}{fp_all:>14.0%}"
      f"{sum(r['verdict'] == 'none' for r in rows) / n:>8.0%}"
      f"{sum(r['verdict'] == 'fingerprint_only' and r['matched_product'] != r['product'] for r in rows):>11}"
      f"{sum(r['copy_id'] not in (None, r['true_copy']) for r in rows):>12}")
print(f"({time.time() - t0:.0f}s)")

# ─── Control A: un-watermarked originals of registered products ───

suite = {a[0]: a for a in ATTACKS}
ctrl_a = {"watermark_claimed": 0, "fingerprint_only": 0, "none": 0, "n": 0}
for pid, p in products.items():
    for atk_name in CONTROL_ATTACKS:
        _, fn, kwargs = suite[atk_name]
        path = f"{OUT}/ctrl_tmp.png"
        _attack(fn, kwargs, p["orig"], path)
        res = pipeline.analyse(path)
        ctrl_a["n"] += 1
        if res["verdict"] in ("watermark", "watermark_aligned"):
            ctrl_a["watermark_claimed"] += 1
        else:
            ctrl_a[res["verdict"]] += 1

# ─── Control B: unregistered images ───────────────────────────────

unregistered = [os.path.join(IMAGE_DIR, f) for f in held_out]
n_similar = len(unregistered)
unregistered += [generate_distractor(f"defence_neg_{i}", size=512,
                                     output_path=f"{OUT}/neg_{i}.png") for i in range(30)]
unregistered += [f"data/sipi/{f}" for f in sorted(os.listdir("data/sipi")) if f.endswith(".png")]
ctrl_b = {"similar": {"watermark_claimed": 0, "fingerprint_only": 0, "none": 0, "n": 0},
          "unrelated": {"watermark_claimed": 0, "fingerprint_only": 0, "none": 0, "n": 0}}
for i, src in enumerate(unregistered):
    group = ctrl_b["similar" if i < n_similar else "unrelated"]
    for atk_name in CONTROL_ATTACKS:
        _, fn, kwargs = suite[atk_name]
        path = f"{OUT}/ctrl_tmp.png"
        _attack(fn, kwargs, src, path)
        res = pipeline.analyse(path)
        group["n"] += 1
        key = "watermark_claimed" if res["verdict"] in ("watermark", "watermark_aligned") else res["verdict"]
        group[key] += 1

# ─── Control C: leak tracing between two copies of the same product ──

trace = {"n": 0, "correct": 0, "swapped": 0}
for pid in list(products)[:15]:
    p = products[pid]
    for atk_name in ["clean", "jpeg_q30", "offcentre_crop_70", "page_screenshot", "perspective"]:
        _, fn, kwargs = suite[atk_name]
        path = f"{OUT}/trace_tmp.png"
        _attack(fn, kwargs, p["partner"], path)
        res = pipeline.analyse(path)
        trace["n"] += 1
        trace["correct"] += res["copy_id"] == p["partner_id"]
        trace["swapped"] += res["copy_id"] == p["listing_id"]

print("\nNEGATIVE CONTROLS")
print(f"  A. un-watermarked originals of registered products ({ctrl_a['n']} tests):")
print(f"       falsely reported as watermarked: {ctrl_a['watermark_claimed']}   "
      f"recognised as the picture: {ctrl_a['fingerprint_only']}   nothing: {ctrl_a['none']}")
for label, key in (("held-out product photos (similar objects)", "similar"),
                   ("unrelated images", "unrelated")):
    g = ctrl_b[key]
    print(f"  B. {label} ({g['n']} tests):")
    print(f"       falsely reported as watermarked: {g['watermark_claimed']}   "
          f"falsely recognised as a registered picture: {g['fingerprint_only']}   nothing: {g['none']}")
print(f"  C. leak tracing, partner copy ({trace['n']} tests): correct copy {trace['correct']}, "
      f"confused with the listing copy {trace['swapped']}")

SUFFIX = "" if STRENGTH == 1.0 else f"_s{STRENGTH}"
with open(f"output/defence_benchmark{SUFFIX}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
with open(f"output/defence_controls{SUFFIX}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["control", "n", "watermark_claimed", "fingerprint_only", "none"])
    w.writerow(["A_unwatermarked_originals", ctrl_a["n"], ctrl_a["watermark_claimed"],
                ctrl_a["fingerprint_only"], ctrl_a["none"]])
    for key in ("similar", "unrelated"):
        g = ctrl_b[key]
        w.writerow([f"B_{key}", g["n"], g["watermark_claimed"], g["fingerprint_only"], g["none"]])
    w.writerow(["C_leak_tracing", trace["n"], trace["correct"], trace["swapped"], ""])
print(f"\nCSVs: output/defence_benchmark{SUFFIX}.csv, output/defence_controls{SUFFIX}.csv")

false_claims = ctrl_a["watermark_claimed"] + sum(g["watermark_claimed"] for g in ctrl_b.values())
wrong = sum(r["copy_id"] not in (None, r["true_copy"]) for r in rows)
sys.exit(0 if false_claims == 0 and wrong == 0 and trace["swapped"] == 0 else 1)
