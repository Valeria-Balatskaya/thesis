# tests/test_texture_masks.py
# Supervisor request (meeting 2): embed the watermark only where there is
# texture / noise, not on flat backgrounds.  Objectives 1.2.2 (invisibility)
# and 1.2.3 (robustness), parameter study 1.4.
#
# With the EXISTING v2 checkpoint (no retraining) the masks are applied at
# embedding time only, so this measures what masking alone does to a model
# that was not trained with it. v3 is trained with the mask (Part A5).
#
# Settings (all at strength 0.7, the app default):
#   none            no mask
#   flat_0.3        local-std texture mask, floor 0.3 (current app default)
#   flat_0.1, flat_0.0
#   nvf_0.3, nvf_0.1, nvf_0.0   noise visibility function (Voloshynovskiy 1999)
#   nvf_0.0_white0  NVF floor 0 + nothing at all on pure white
#   nvf_0.1_renorm  NVF floor 0.1, energy moved from flat areas into texture
#   flat_0.3_head2  current default + 2-level headroom at pure white/black
#
# Per setting: PSNR, SSIM, dE (whole image, flat areas), blind and full
# Retrieve-Align-Verify read rate under the 12 attacks of test_hidden_v2_eval.
# Per image: which images fail, with how much of the image is product.
#
# Run: python tests/test_texture_masks.py [n_own] [n_abo]   (default 30 30)

import sys, os, csv, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shutil import copyfile

import numpy as np
from PIL import Image

from src.hidden_v2_inference import HiddenV2Model
from src.defence import DefencePipeline
from src.metrics import compute, visibility
from src import attacks, ecommerce_attacks as ea

CHECKPOINT = "checkpoints/hidden_v2_best.pt"
N_OWN = int(sys.argv[1]) if len(sys.argv) > 1 else 30
N_ABO = int(sys.argv[2]) if len(sys.argv) > 2 else 30
STRENGTH = 0.7
SETTINGS = [
    ("none",           dict(mask_floor=1.0, mask_type="flat")),
    ("flat_0.3",       dict(mask_floor=0.3, mask_type="flat")),
    ("flat_0.1",       dict(mask_floor=0.1, mask_type="flat")),
    ("flat_0.0",       dict(mask_floor=0.0, mask_type="flat")),
    ("nvf_0.3",        dict(mask_floor=0.3, mask_type="nvf")),
    ("nvf_0.1",        dict(mask_floor=0.1, mask_type="nvf")),
    ("nvf_0.0",        dict(mask_floor=0.0, mask_type="nvf")),
    ("nvf_0.0_white0", dict(mask_floor=0.0, mask_type="nvf", zero_white=True)),
    ("nvf_0.1_renorm", dict(mask_floor=0.1, mask_type="nvf", mask_renorm=True)),
    ("flat_0.3_head2", dict(mask_floor=0.3, mask_type="flat", headroom=2)),
]
if len(sys.argv) > 3:      # optional: run a subset, e.g. "none,flat_0.3"
    keep = sys.argv[3].split(",")
    SETTINGS = [s for s in SETTINGS if s[0] in keep]
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
OUT = "results/texture_masks"
os.makedirs(OUT, exist_ok=True)
os.makedirs("output", exist_ok=True)

model = HiddenV2Model(CHECKPOINT)


def load_set(folder, n, tag):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg")))[:n]
    out = []
    for k, f in enumerate(files):
        p = f"{OUT}/{tag}_{k:03d}_orig.png"
        if not os.path.exists(p):
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            img.thumbnail((1024, 1024), Image.LANCZOS)
            img.save(p, format="PNG")
        out.append((tag, f, p))
    return out


def image_stats(path):
    """Share of the image that is product (not plain background) and pure white."""
    rgb = np.asarray(Image.open(path).convert("RGB"), np.float32)
    border = np.concatenate([rgb[:8].reshape(-1, 3), rgb[-8:].reshape(-1, 3),
                             rgb[:, :8].reshape(-1, 3), rgb[:, -8:].reshape(-1, 3)])
    bg = np.linalg.norm(rgb - np.median(border, axis=0), axis=2) < 18
    return {"product_share": round(1 - float(bg.mean()), 3),
            "pure_white_share": round(float((rgb.min(axis=2) >= 254).mean()), 3)}


images = load_set("data/products_png", N_OWN, "own") + load_set("data/abo_sample", N_ABO, "abo")
stats = {p: image_stats(p) for _, _, p in images}

summary, per_image = [], []
t0 = time.time()
for name, opts in SETTINGS:
    np.random.seed(0)          # same noise for every setting
    pipe = DefencePipeline(model)
    wms = []
    for i, (tag, f, orig) in enumerate(images):
        wm = f"{OUT}/{name}_{i:03d}.png"
        model.embed_product(orig, 1000 + i, wm, strength=STRENGTH, **opts)
        pipe.register(i, orig, [1000 + i])
        wms.append(wm)
    for i, (tag, f, orig) in enumerate(images):
        wm = wms[i]
        q, v = compute(orig, wm), visibility(orig, wm)
        row = {"setting": name, "set": tag, "file": f, **stats[orig],
               "psnr": q["PSNR_dB"], "ssim": q["SSIM"], "dE_mean": v["dE_mean"],
               "dE_flat": v["dE_flat"], "blind_ok": 0, "full_ok": 0, "failed_attacks": ""}
        failed = []
        for atk, fn, kw in ATTACK_SUITE:
            sus = f"{OUT}/atk.png"
            copyfile(wm, sus) if fn is None else fn(wm, sus, **kw)
            blind = model.read_product(sus)["product_id"] == 1000 + i
            full = blind or pipe.analyse(sus)["copy_id"] == 1000 + i
            row["blind_ok"] += blind
            row["full_ok"] += full
            if not full:
                failed.append(atk)
        row["failed_attacks"] = " ".join(failed)
        per_image.append(row)

    rows = [r for r in per_image if r["setting"] == name]
    n_att = len(rows) * len(ATTACK_SUITE)
    s = {"setting": name, "n_images": len(rows)}
    for tag in ("own", "abo", "all"):
        sub = rows if tag == "all" else [r for r in rows if r["set"] == tag]
        if not sub:
            continue
        na = len(sub) * len(ATTACK_SUITE)
        flat = [r["dE_flat"] for r in sub if r["dE_flat"] is not None]
        s.update({
            f"{tag}_psnr": round(float(np.mean([r["psnr"] for r in sub])), 2),
            f"{tag}_ssim": round(float(np.mean([r["ssim"] for r in sub])), 4),
            f"{tag}_dE": round(float(np.mean([r["dE_mean"] for r in sub])), 3),
            f"{tag}_dE_flat": round(float(np.mean(flat)), 3) if flat else None,
            f"{tag}_blind_pct": round(100 * sum(r["blind_ok"] for r in sub) / na, 1),
            f"{tag}_full_pct": round(100 * sum(r["full_ok"] for r in sub) / na, 1),
            f"{tag}_clean_pct": round(100 * sum("clean" not in r["failed_attacks"].split() for r in sub) / len(sub), 1),
        })
    summary.append(s)
    print(f"{name:<16} own PSNR {s['own_psnr']:5.1f} dE {s['own_dE']:.2f}/{s['own_dE_flat']}  "
          f"blind {s['own_blind_pct']:5.1f}% full {s['own_full_pct']:5.1f}% | "
          f"ABO PSNR {s['abo_psnr']:5.1f} dE {s['abo_dE']:.2f}/{s['abo_dE_flat']}  "
          f"blind {s['abo_blind_pct']:5.1f}% full {s['abo_full_pct']:5.1f}%  ({time.time() - t0:.0f}s)",
          flush=True)

tag = "_" + sys.argv[3].replace(",", "+") if len(sys.argv) > 3 else ""
with open(f"output/texture_masks_summary{tag}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for s in summary for k in s)))
    w.writeheader()
    w.writerows(summary)
with open(f"output/texture_masks_per_image{tag}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
    w.writeheader()
    w.writerows(per_image)

print("\nImages failing >= 3 of 12 attacks with the full pipeline (setting: file product% white%: attacks)")
for r in per_image:
    if len(ATTACK_SUITE) - r["full_ok"] >= 3:
        print(f"  {r['setting']:<16} {r['set']} {r['file']:<16} product {r['product_share']:.0%} "
              f"white {r['pure_white_share']:.0%}: {r['failed_attacks']}")
print(f"\nCSVs: output/texture_masks_summary{tag}.csv, output/texture_masks_per_image{tag}.csv")
