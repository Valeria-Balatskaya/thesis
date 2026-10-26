# tests/test_v3_vs_v2.py
# Head-to-head of the retrained v3 model against v2 (objectives 1.2.2, 1.2.3, 1.2.5).
#
#   v2  checkpoints/hidden_v2_best.pt at the app setting (strength 0.7, flat mask 0.3)
#   v3  checkpoints/hidden_v3_best.pt as trained (strength 1.0, NVF mask 0.1, nothing on
#       pure white, energy moved into texture, colour charge 8, trained on COCO + ABO,
#       levels distortion, overwrite augmentation)
#
# On own product photos, ABO evaluation photos (never used in training, product-level split)
# and COCO test2017 photos:
#   visibility   PSNR, SSIM, dE whole image, dE on flat bright areas (backgrounds)
#   robustness   correct copy ID, blind and full Retrieve-Align-Verify, under 12 classic
#                + 11 modern attacks
#   detection    false positives on the clean originals
#   overwrite    a thief re-protects the published copy with the SAME model (native
#                strength and 1.5x): whose ID does the blind decoder return, and does the
#                calibrated informed check (z >= 4) still find ours?
#
# Run: python tests/test_v3_vs_v2.py [n_per_set] [name=checkpoint ...]   (default 30)
#      e.g. python tests/test_v3_vs_v2.py 30 v4=checkpoints/hidden_v4_best.pt

import sys, os, csv, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shutil import copyfile

import numpy as np
from PIL import Image

from src.hidden_v2_inference import HiddenV2Model
from src.defence import DefencePipeline, InformedChecker, Z_THRESHOLD
from src.metrics import compute, visibility
from src import attacks, ecommerce_attacks as ea, modern_attacks as ma

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SYSTEMS = {
    "v2": ("checkpoints/hidden_v2_best.pt", {"strength": 0.7, "mask_floor": 0.3}),
    "v3": ("checkpoints/hidden_v3_best.pt", {"strength": 1.0, "mask_floor": None}),
}
# extra systems from the command line, e.g.  v4=checkpoints/hidden_v4_best.pt
for arg in sys.argv[2:]:
    name, ckpt = arg.split("=", 1)
    SYSTEMS[name] = (ckpt, {"strength": 1.0, "mask_floor": None})
SETS = [("own", "data/products_png"), ("abo", "data/abo_sample"), ("coco", "data/coco_sample")]
ATTACKS = [
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
    ("social_recompress",  ma.social_recompress,              {}),
    ("cdn_webp_360",       ma.marketplace_variant,            {"size": 360}),
    ("offcentre_crop_70",  ma.offcentre_crop,                 {"keep": 0.7, "dx": 1.0, "dy": 0.0}),
    ("offcentre_crop_50",  ma.offcentre_crop,                 {"keep": 0.5, "dx": 0.2, "dy": 0.9}),
    ("rotate_8",           ma.rotate,                         {"degrees": 8}),
    ("mirror",             ma.mirror,                         {}),
    ("perspective",        ma.perspective,                    {"strength": 0.12}),
    ("page_screenshot",    ma.page_screenshot,                {}),
    ("screen_photo",       ma.screen_photo,                   {}),
    ("promo_overlay",      ma.promo_overlay,                  {}),
    ("inpaint_region",     ma.inpaint_region,                 {}),
]
OUT = "results/v3_vs_v2"
os.makedirs(OUT, exist_ok=True)
os.makedirs("output", exist_ok=True)


def rgb(p):
    return np.asarray(Image.open(p).convert("RGB"))


images = []
for tag, folder in SETS:
    for k, f in enumerate(sorted(x for x in os.listdir(folder)
                                 if x.lower().endswith((".png", ".jpg", ".jpeg")))[:N]):
        p = f"{OUT}/{tag}_{k:03d}_orig.png"
        if not os.path.exists(p):
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            img.thumbnail((1024, 1024), Image.LANCZOS)
            img.save(p, format="PNG")
        images.append((tag, f, p))

quality, robust, overwrite, fps = [], [], [], {}
t0 = time.time()
for sysname, (ckpt, kw) in SYSTEMS.items():
    model = HiddenV2Model(ckpt)
    pipe = DefencePipeline(model)
    pub = {}
    for i, (tag, f, orig) in enumerate(images):
        pub[i] = f"{OUT}/{sysname}_{i:03d}.png"
        model.embed_product(orig, 1000 + i, pub[i], **kw)
        pipe.register(i, orig, [1000 + i])
        q, v = compute(orig, pub[i]), visibility(orig, pub[i])
        quality.append({"system": sysname, "set": tag, "image": i, "psnr": q["PSNR_dB"],
                        "ssim": q["SSIM"], "dE_mean": v["dE_mean"], "dE_flat": v["dE_flat"]})
    fps[sysname] = sum(model.read_product(p)["product_id"] is not None
                       or pipe.analyse(p)["copy_id"] is not None for _, _, p in images)

    np.random.seed(0)
    for i, (tag, f, orig) in enumerate(images):
        for atk, fn, akw in ATTACKS:
            sus = f"{OUT}/atk.png"
            copyfile(pub[i], sus) if fn is None else fn(pub[i], sus, **akw)
            blind = model.read_product(sus)["product_id"] == 1000 + i
            full = blind or pipe.analyse(sus)["copy_id"] == 1000 + i
            robust.append({"system": sysname, "set": tag, "image": i, "attack": atk,
                           "blind": int(blind), "full": int(full)})
    print(f"{sysname}: robustness done ({time.time() - t0:.0f}s)", flush=True)

    # overwrite with the same model, on the own + ABO photos
    native = kw["strength"]
    for i, (tag, f, orig) in enumerate(images):
        if tag == "coco":
            continue
        checker = InformedChecker(model, rgb(orig), 1000 + i, embed_kwargs=kw)
        for name, st in (("same_strength", native), ("x1.5", native * 1.5)):
            thief = f"{OUT}/thief.png"
            model.embed_product(pub[i], 500_000 + i, thief, strength=st,
                                mask_floor=kw["mask_floor"])
            for post in ("clean", "jpeg_q70"):
                sus = f"{OUT}/ow.png"
                copyfile(thief, sus) if post == "clean" else attacks.jpeg_compress(thief, sus, 70)
                cid = model.read_product(sus)["product_id"]
                z = checker.test(rgb(sus))["z"]
                overwrite.append({"system": sysname, "set": tag, "image": i, "thief": name,
                                  "post": post,
                                  "blind": "ours" if cid == 1000 + i else
                                           "theirs" if cid is not None and cid >= 500_000 else
                                           "nothing" if cid is None else "other",
                                  "informed_z": z, "informed_hit": int(z >= Z_THRESHOLD)})
    print(f"{sysname}: overwrite done ({time.time() - t0:.0f}s)", flush=True)

for name, rows in [("quality", quality), ("robustness", robust), ("overwrite", overwrite)]:
    with open(f"output/v3_vs_v2_{name}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

print(f"\n{N} photos per set; v2 at 0.7/flat 0.3, v3 as trained\n")
print("VISIBILITY (mean)            PSNR    SSIM    dE   dE flat")
for tag, _ in SETS:
    for s in SYSTEMS:
        sub = [r for r in quality if r["system"] == s and r["set"] == tag]
        flat = [r["dE_flat"] for r in sub if r["dE_flat"] is not None]
        print(f"  {tag:<5} {s}                {np.mean([r['psnr'] for r in sub]):6.2f}  "
              f"{np.mean([r['ssim'] for r in sub]):.4f}  {np.mean([r['dE_mean'] for r in sub]):.2f}  "
              f"{np.mean(flat) if flat else float('nan'):.2f} (n={len(flat)})")

print("\nROBUSTNESS (blind / full pipeline), all 23 attacks")
for tag, _ in SETS:
    line = f"  {tag:<5}"
    for s in SYSTEMS:
        sub = [r for r in robust if r["system"] == s and r["set"] == tag]
        line += (f"  {s}: {100 * np.mean([r['blind'] for r in sub]):5.1f}% / "
                 f"{100 * np.mean([r['full'] for r in sub]):5.1f}%")
    print(line)
print(f"\n{'attack (all sets): full pipeline (blind)':<42}" + "".join(f"{s:>14}" for s in SYSTEMS))
for atk, _, _ in ATTACKS:
    line = f"  {atk:<40}"
    for s in SYSTEMS:
        sub = [r for r in robust if r["system"] == s and r["attack"] == atk]
        line += f"{sum(r['full'] for r in sub):>7}/{len(sub)} ({sum(r['blind'] for r in sub)})"
    print(line)
print(f"\nFALSE POSITIVES on {len(images)} clean originals: " + ", ".join(f"{s} {fps[s]}" for s in SYSTEMS))

print("\nOVERWRITE with the same model (own + ABO): blind ours/theirs/nothing, informed z>=4")
for s in SYSTEMS:
    for name in ("same_strength", "x1.5"):
        for post in ("clean", "jpeg_q70"):
            sub = [r for r in overwrite if r["system"] == s and r["thief"] == name and r["post"] == post]
            c = lambda v: sum(r["blind"] == v for r in sub)
            print(f"  {s} {name:<14} {post:<9} {c('ours'):>3}/{c('theirs')}/{c('nothing')} of {len(sub)}   "
                  f"informed {sum(r['informed_hit'] for r in sub)}/{len(sub)} "
                  f"(median z {np.median([r['informed_z'] for r in sub]):.2f})")
print(f"\nCSVs: output/v3_vs_v2_*.csv ({time.time() - t0:.0f}s)")
