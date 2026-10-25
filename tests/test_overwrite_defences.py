# tests/test_overwrite_defences.py
# Part B1 + B2: defences against watermark overwriting beyond the upload
# pre-check. Objectives 1.2.3 (robustness to malicious attacks) and 1.2.5
# (ownership verification).
#
# B1  second, secret-keyed layer: DCT spread spectrum in luminance
#     (src/dct_layer.py). Three published variants per photo:
#       v2_only     our current watermark (app setting 0.7 / 0.3)
#       v2_then_dct DCT layer embedded on the v2 copy  (DCT reference = v2 copy)
#       dct_then_v2 v2 embedded on the DCT-marked image (DCT reference = original)
#     - quality vs the original (PSNR, SSIM, dE whole / flat)
#     - v2 robustness: blind and full-pipeline read under the 12 attacks
#     - DCT layer under the same attacks (suspect aligned to the reference frame
#       with SIFT/RANSAC first, src/defence.align)
#     - DCT layer after a thief re-protects the image with v2 (same strength,
#       full strength, two layers), clean and after JPEG q70
# B2  calibrated informed v2 check (src/defence.InformedChecker): z-score
#     against 500 never-issued copies per photo, on the same overwritten images.
#
# False-positive calibration of both ownership checks, on images that do NOT
# contain the tested copy:
#     N1  the clean original (JPEG q70)
#     N2  a sibling copy of the same photo with another copy ID (+ its own DCT layer)
#     N3  the original protected by a thief with v2 only (thief's ID)
#
# Run: python tests/test_overwrite_defences.py [n_own] [n_abo]   (default 20 20)

import sys, os, csv, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shutil import copyfile

import numpy as np
from PIL import Image

from src.hidden_v2_inference import HiddenV2Model
from src.defence import DefencePipeline, InformedChecker, Z_THRESHOLD, fingerprint, match, align
from src.metrics import compute, visibility
from src import attacks, ecommerce_attacks as ea, dct_layer

N_OWN = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N_ABO = int(sys.argv[2]) if len(sys.argv) > 2 else 20
STRENGTH, MASK = 0.7, 0.3
DCT_THRESHOLD = 6.0
OUT = "results/overwrite_defences"
os.makedirs(OUT, exist_ok=True)
os.makedirs("output", exist_ok=True)

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

model = HiddenV2Model("checkpoints/hidden_v2_best.pt")


def rgb(p):
    return np.asarray(Image.open(p).convert("RGB"))


def save(arr, p):
    Image.fromarray(arr).save(p, format="PNG")
    return p


def to_frame(suspect, original, ref_fp):
    """Suspect warped into the original's frame (SIFT/RANSAC), else resized."""
    if suspect.shape == original.shape:
        return suspect
    m = match(fingerprint(suspect), ref_fp)
    if m is None:
        m = match(fingerprint(suspect[:, ::-1].copy()), ref_fp)
        if m is not None:
            suspect = suspect[:, ::-1].copy()
    if m is not None:
        return align(suspect, m["H"], original)[0]
    h, w = original.shape[:2]
    return np.asarray(Image.fromarray(suspect).resize((w, h), Image.BICUBIC))


def v2_embed(src, cid, dst, strength=STRENGTH, mask=MASK):
    model.embed_product(src, cid, dst, strength=strength, mask_floor=mask)
    return dst


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


images = load_set("data/products_png", N_OWN, "own") + load_set("data/abo_sample", N_ABO, "abo")
VARIANTS = ["v2_only", "v2_then_dct", "dct_then_v2"]

quality, robust, dct_rows, overwrite, negatives = [], [], [], [], []
pipe = DefencePipeline(model)
pub, dct_ref = {}, {}
t0 = time.time()

# ─── build the published variants ─────────────────────────────────
for i, (tag, f, orig) in enumerate(images):
    cid = 1000 + i
    pipe.register(i, orig, [cid])
    o = rgb(orig)
    v2c = v2_embed(orig, cid, f"{OUT}/{i:03d}_v2_only.png")
    pub[(i, "v2_only")] = v2c
    pub[(i, "v2_then_dct")] = save(dct_layer.embed(rgb(v2c), cid), f"{OUT}/{i:03d}_v2_then_dct.png")
    dct_ref[(i, "v2_then_dct")] = rgb(v2c)
    d = save(dct_layer.embed(o, cid), f"{OUT}/{i:03d}_dct_only.png")
    pub[(i, "dct_then_v2")] = v2_embed(d, cid, f"{OUT}/{i:03d}_dct_then_v2.png")
    dct_ref[(i, "dct_then_v2")] = o
    for v in VARIANTS:
        q, vis = compute(orig, pub[(i, v)]), visibility(orig, pub[(i, v)])
        quality.append({"image": i, "set": tag, "variant": v, "psnr": q["PSNR_dB"], "ssim": q["SSIM"],
                        "dE_mean": vis["dE_mean"], "dE_flat": vis["dE_flat"]})
print(f"embedded ({time.time() - t0:.0f}s)", flush=True)

# ─── robustness of v2 and of the DCT layer under the 12 attacks ───
np.random.seed(0)
for i, (tag, f, orig) in enumerate(images):
    cid, o = 1000 + i, rgb(orig)
    ofp = pipe.products[i]["fp"]
    for v in VARIANTS:
        for atk, fn, kw in ATTACK_SUITE:
            sus = f"{OUT}/atk.png"
            copyfile(pub[(i, v)], sus) if fn is None else fn(pub[(i, v)], sus, **kw)
            blind = model.read_product(sus)["product_id"] == cid
            full = blind or pipe.analyse(sus)["copy_id"] == cid
            row = {"image": i, "set": tag, "variant": v, "attack": atk,
                   "v2_blind": int(blind), "v2_full": int(full)}
            if v != "v2_only":
                s = to_frame(rgb(sus), o, ofp)
                sim = float(dct_layer.similarity(s, dct_ref[(i, v)], [cid])[0])
                row.update({"dct_sim": round(sim, 2), "dct_detected": int(sim > DCT_THRESHOLD)})
            robust.append(row)
print(f"robustness done ({time.time() - t0:.0f}s)", flush=True)

# ─── overwrite: thief re-protects with v2; ownership checks ───────
THIEF = [("thief_v2_same", STRENGTH, MASK, 1), ("thief_v2_full", 1.0, None, 1),
         ("thief_v2_x2", STRENGTH, MASK, 2)]
for i, (tag, f, orig) in enumerate(images):
    cid, o = 1000 + i, rgb(orig)
    checker = InformedChecker(model, o, cid)
    for v in VARIANTS:
        for name, st, mk, layers in THIEF:
            cur = pub[(i, v)]
            for k in range(layers):
                cur = v2_embed(cur, 500_000 + i + 1000 * k, f"{OUT}/thief_{k}.png", st, mk)
            for post in ("clean", "jpeg_q70"):
                sus = f"{OUT}/ow.png"
                copyfile(cur, sus) if post == "clean" else attacks.jpeg_compress(cur, sus, 70)
                s = rgb(sus)
                inf = checker.test(s)
                row = {"image": i, "set": tag, "variant": v, "thief": name, "post": post,
                       "v2_blind_ours": int(model.read_product(sus)["product_id"] == cid),
                       "informed_z": inf["z"], "informed_hit": int(inf["z"] >= Z_THRESHOLD)}
                if v != "v2_only":
                    det = dct_layer.detect(s, dct_ref[(i, v)], cid, n_null=500)
                    row.update({"dct_sim": det["sim"], "dct_detected": int(det["detected"]),
                                "dct_null_std": det["null_std"]})
                row["owned"] = int(row["informed_hit"] or row.get("dct_detected", 0))
                overwrite.append(row)

    # negatives for calibration: the tested copy (cid) is NOT in these images
    sib = 900_000 + i
    sib_v2 = v2_embed(orig, sib, f"{OUT}/sib.png")
    cases = {
        "N1_clean_original": orig,
        "N2_sibling_copy": save(dct_layer.embed(rgb(sib_v2), sib), f"{OUT}/sib_dct.png"),
        "N3_thief_only": v2_embed(orig, 600_000 + i, f"{OUT}/thief_only.png"),
    }
    for name, path in cases.items():
        for post in ("clean", "jpeg_q70"):
            sus = f"{OUT}/neg.png"
            copyfile(path, sus) if post == "clean" else attacks.jpeg_compress(path, sus, 70)
            s = rgb(sus)
            inf = checker.test(s)
            row = {"image": i, "set": tag, "case": name, "post": post,
                   "informed_z": inf["z"], "informed_hit": int(inf["z"] >= Z_THRESHOLD)}
            for v in ("v2_then_dct", "dct_then_v2"):
                sim = float(dct_layer.similarity(s, dct_ref[(i, v)], [cid])[0])
                row[f"dct_sim_{v}"] = round(sim, 2)
                row[f"dct_hit_{v}"] = int(sim > DCT_THRESHOLD)
            negatives.append(row)
    if i % 5 == 4:
        print(f"overwrite + negatives: {i + 1}/{len(images)} ({time.time() - t0:.0f}s)", flush=True)

# ─── save and summarise ───────────────────────────────────────────
for name, rows in [("quality", quality), ("robustness", robust), ("overwrite", overwrite),
                   ("negatives", negatives)]:
    with open(f"output/overwrite_defences_{name}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)

n = len(images)
print(f"\n{n} photos ({N_OWN} own + {N_ABO} ABO); v2 at {STRENGTH}/{MASK}; DCT alpha "
      f"{dct_layer.ALPHA}, {dct_layer.N_COEFFS} coefficients, threshold {DCT_THRESHOLD}; "
      f"informed z >= {Z_THRESHOLD}\n")
print("QUALITY vs original (mean)")
for v in VARIANTS:
    sub = [r for r in quality if r["variant"] == v]
    flat = [r["dE_flat"] for r in sub if r["dE_flat"] is not None]
    print(f"  {v:<12} PSNR {np.mean([r['psnr'] for r in sub]):5.2f}  SSIM {np.mean([r['ssim'] for r in sub]):.4f}  "
          f"dE {np.mean([r['dE_mean'] for r in sub]):.2f}  dE flat {np.mean(flat):.2f}")

print("\nROBUSTNESS under 12 attacks: v2 blind / v2 full pipeline / DCT layer detected")
for v in VARIANTS:
    for tag in ("own", "abo"):
        sub = [r for r in robust if r["variant"] == v and r["set"] == tag]
        line = f"  {v:<12} {tag}: v2 blind {sum(r['v2_blind'] for r in sub)}/{len(sub)}  full {sum(r['v2_full'] for r in sub)}/{len(sub)}"
        if v != "v2_only":
            line += f"  DCT {sum(r['dct_detected'] for r in sub)}/{len(sub)}"
        print(line)
print("  DCT layer per attack (v2_then_dct / dct_then_v2):")
for atk, _, _ in ATTACK_SUITE:
    a = [r for r in robust if r["attack"] == atk and r["variant"] == "v2_then_dct"]
    b = [r for r in robust if r["attack"] == atk and r["variant"] == "dct_then_v2"]
    print(f"    {atk:<20} {sum(r['dct_detected'] for r in a):>3}/{len(a)}  {sum(r['dct_detected'] for r in b):>3}/{len(b)}"
          f"   median sim {np.median([r['dct_sim'] for r in a]):6.1f} / {np.median([r['dct_sim'] for r in b]):6.1f}")

print("\nOVERWRITE (thief re-protects with v2): our v2 blind / informed z>=thr / DCT detected / owned (either)")
for v in VARIANTS:
    for name, *_ in THIEF:
        for post in ("clean", "jpeg_q70"):
            sub = [r for r in overwrite if r["variant"] == v and r["thief"] == name and r["post"] == post]
            dct = f"{sum(r['dct_detected'] for r in sub)}/{n}" if v != "v2_only" else "  -  "
            print(f"  {v:<12} {name:<14} {post:<9} blind {sum(r['v2_blind_ours'] for r in sub):>2}/{n}  "
                  f"informed {sum(r['informed_hit'] for r in sub):>2}/{n} (median z {np.median([r['informed_z'] for r in sub]):5.2f})  "
                  f"DCT {dct}  owned {sum(r['owned'] for r in sub):>2}/{n}")

print("\nFALSE POSITIVES on images without the tested copy (informed z>=thr / DCT v2_then_dct / DCT dct_then_v2)")
for case in ("N1_clean_original", "N2_sibling_copy", "N3_thief_only"):
    sub = [r for r in negatives if r["case"] == case]
    zs = [r["informed_z"] for r in sub]
    print(f"  {case:<18} informed {sum(r['informed_hit'] for r in sub)}/{len(sub)} (max z {max(zs):.2f})  "
          f"DCT {sum(r['dct_hit_v2_then_dct'] for r in sub)}/{len(sub)} (max sim {max(r['dct_sim_v2_then_dct'] for r in sub):.2f})  "
          f"{sum(r['dct_hit_dct_then_v2'] for r in sub)}/{len(sub)} (max sim {max(r['dct_sim_dct_then_v2'] for r in sub):.2f})")
all_neg = len(negatives)
print(f"  total: informed {sum(r['informed_hit'] for r in negatives)}/{all_neg}, "
      f"DCT {sum(r['dct_hit_v2_then_dct'] + r['dct_hit_dct_then_v2'] for r in negatives)}/{2 * all_neg}")
print(f"\nCSVs: output/overwrite_defences_*.csv  ({time.time() - t0:.0f}s)")
