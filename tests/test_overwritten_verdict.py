# tests/test_overwritten_verdict.py
# The "watermark_overwritten" verdict (ownership dispute): how often does it fire
# when it should, and how often when it must not? (objectives 1.2.3, 1.2.5)
#
# Setup as in the app: every product has two issued copies (A and B, as for two
# channels). Detection = DefencePipeline.analyse, then for "fingerprint_only" the
# calibrated informed check (z >= 4, 500 never-issued copies) for BOTH copies, and
# mark_overwritten() names the copy with the strongest evidence.
#
#   should fire (copy A was published, a thief wrote over it)
#     thief re-protects copy A with the same model at 1x and 1.5x, a random ID
#     (keyed model: also with another key), each clean and after JPEG q70
#     -> overwritten naming A = correct; naming B = WRONG COPY
#   must not fire (no issued copy of ours is in the image)
#     clean original; original + the thief's own mark (thief watermarked a leaked
#     clean original); original + another system's mark (Cox DCT, LSB, visible logo);
#     original after rotate 8 / off-centre crop / page screenshot (aligned path)
#     -> any "watermark_overwritten" here is a FALSE POSITIVE of the verdict
#
# Run: python tests/test_overwritten_verdict.py [n_own] [n_abo] [checkpoint]
#      (default 20 20 checkpoints/hidden_v3_best.pt; ~35 min on an M1 CPU)

import sys, os, csv, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shutil import copyfile

import numpy as np
from PIL import Image, ImageDraw

from src.hidden_v2_inference import HiddenV2Model
from src.defence import DefencePipeline, InformedChecker, ownership_check, mark_overwritten
from src import attacks, lsb, modern_attacks as ma
from src.dct_watermark import embed as dct_embed
from src.modern_attacks import _font
from src.stats import pct_ci

N_OWN = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N_ABO = int(sys.argv[2]) if len(sys.argv) > 2 else 20
CKPT = sys.argv[3] if len(sys.argv) > 3 else "checkpoints/hidden_v3_best.pt"
KW = ({"strength": 0.7, "mask_floor": 0.3} if "v2" in os.path.basename(CKPT)
      else {"strength": 1.0, "mask_floor": None})
OUT = "results/overwritten_verdict"
os.makedirs(OUT, exist_ok=True)
os.makedirs("output", exist_ok=True)
TAG = os.path.basename(CKPT).replace("hidden_", "").replace("_best", "").replace(".pt", "")

model = HiddenV2Model(CKPT)


def rgb(p):
    return np.asarray(Image.open(p).convert("RGB"))


def visible_logo(src, dst):
    img = Image.open(src).convert("RGB")
    w, h = img.size
    d = ImageDraw.Draw(img, "RGBA")
    d.text((int(w * 0.6), int(h * 0.9)), "© thief-shop", fill=(255, 255, 255, 150),
           font=_font(int(h * 0.05)), stroke_width=1, stroke_fill=(0, 0, 0, 90))
    img.save(dst, format="PNG")


images = []
for tag, folder, n in [("own", "data/products_png", N_OWN), ("abo", "data/abo_sample", N_ABO)]:
    for k, f in enumerate(sorted(x for x in os.listdir(folder)
                                 if x.lower().endswith((".png", ".jpg", ".jpeg")))[:n]):
        p = f"{OUT}/{tag}_{k:03d}_orig.png"
        if not os.path.exists(p):
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            img.thumbnail((1024, 1024), Image.LANCZOS)
            img.save(p, format="PNG")
        images.append((tag, p))

pipe = DefencePipeline(model)
copies = {}
for i, (tag, orig) in enumerate(images):
    a, b = 1000 + 2 * i, 1001 + 2 * i
    for cid in (a, b):
        model.embed_product(orig, cid, f"{OUT}/c{cid}.png", **KW)
    copies[i] = (a, b)
    pipe.register(i, orig, [a, b])

rows = []
t0 = time.time()
for i, (tag, orig) in enumerate(images):
    a, b = copies[i]
    o = rgb(orig)
    checkers = {cid: InformedChecker(model, o, cid, embed_kwargs=KW) for cid in (a, b)}
    pub = f"{OUT}/c{a}.png"

    def evidence(product_id, aligned):
        out = []
        for cid in (a, b):
            r = ownership_check(model, aligned, o, cid, checker=checkers[cid])
            out.append({"copy_id": cid, "owned": r["owned"],
                        "informed_z": r["informed"]["z"] if r["informed"] else None,
                        "p_empirical": r["informed"]["p_empirical"] if r["informed"] else None,
                        "dct_sim": None})
        return out

    # (scenario, group, function writing the suspect)
    tkey = model.other_key(30_000 + i)
    scen = [
        ("thief_1x", "positive", lambda d: model.embed_product(pub, 500_000 + i, d, **KW)),
        ("thief_1.5x", "positive", lambda d: model.embed_product(
            pub, 500_000 + i, d, strength=1.5 * KW["strength"], mask_floor=KW["mask_floor"])),
    ]
    if model.key_dim:
        scen.append(("thief_other_key_1x", "positive",
                     lambda d: model.embed_product(pub, 500_000 + i, d, key=tkey, **KW)))
    scen += [
        ("clean_original", "negative", lambda d: copyfile(orig, d)),
        ("original+thief_mark", "negative", lambda d: model.embed_product(orig, 500_000 + i, d, **KW)),
        ("original+dct", "negative", lambda d: dct_embed(orig, f"THIEF:{i}", d, alpha=0.02, n_coeffs=500)),
        ("original+lsb", "negative", lambda d: lsb.embed(orig, f"(c) thief {i}", d, k=1)),
        ("original+logo", "negative", lambda d: visible_logo(orig, d)),
        ("original_rotate8", "negative", lambda d: ma.rotate(orig, d, degrees=8)),
        ("original_crop70", "negative", lambda d: ma.offcentre_crop(orig, d, keep=0.7, dx=1.0, dy=0.0)),
        ("original_page", "negative", lambda d: ma.page_screenshot(orig, d)),
    ]
    for name, group, make in scen:
        base = f"{OUT}/sus_base.png"
        make(base)
        posts = [("clean", None), ("jpeg_q70", 70)] if not name.startswith("original_") else [("clean", None)]
        for post, q in posts:
            sus = f"{OUT}/sus.png"
            copyfile(base, sus) if q is None else attacks.jpeg_compress(base, sus, q)
            res = pipe.analyse(sus)
            ev = evidence(i, res["aligned"]) if (res["verdict"] == "fingerprint_only"
                                                 and res["aligned"] is not None) else []
            mark_overwritten(res, ev)
            zs = {e["copy_id"]: e["informed_z"] for e in ev}
            rows.append({"set": tag, "image": i, "scenario": name, "group": group, "post": post,
                         "verdict": res["verdict"], "copy_named": res["copy_id"],
                         "named_A": int(res["copy_id"] == a), "named_B": int(res["copy_id"] == b),
                         "their_id": (res.get("overwrite") or {}).get("their_id"),
                         "z_A": zs.get(a), "z_B": zs.get(b)})
    if i % 5 == 4:
        print(f"{i + 1}/{len(images)} ({time.time() - t0:.0f}s)", flush=True)

with open(f"output/overwritten_verdict_{TAG}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

n = len(images)
print(f"\n{CKPT} at {KW}: {n} photos ({N_OWN} own + {N_ABO} ABO), 2 copies each\n")
print(f"{'scenario':<22}{'post':<10}{'overwritten (right copy)':<34}{'wrong copy':<12}other verdicts")
for name in dict.fromkeys(r["scenario"] for r in rows):
    for post in ("clean", "jpeg_q70"):
        sub = [r for r in rows if r["scenario"] == name and r["post"] == post]
        if not sub:
            continue
        ow = [r for r in sub if r["verdict"] == "watermark_overwritten"]
        right = sum(r["named_A"] for r in ow)
        wrong = sum(r["named_B"] for r in ow)
        others = {}
        for r in sub:
            if r["verdict"] != "watermark_overwritten":
                others[r["verdict"]] = others.get(r["verdict"], 0) + 1
        label = pct_ci(right, len(sub)) if sub[0]["group"] == "positive" else pct_ci(len(ow), len(sub))
        print(f"{name:<22}{post:<10}{label:<34}{wrong:<12}"
              + ", ".join(f"{k} {v}" for k, v in sorted(others.items())))

neg = [r for r in rows if r["group"] == "negative"]
pos = [r for r in rows if r["group"] == "positive"]
fp = sum(r["verdict"] == "watermark_overwritten" for r in neg)
tests = sum((r["z_A"] is not None) + (r["z_B"] is not None) for r in neg)
null_z = [z for r in neg for z in (r["z_A"], r["z_B"]) if z is not None]
print(f"\nFALSE 'overwritten' on images without any issued copy: {pct_ci(fp, len(neg), 2)}"
      f"  ({tests} informed tests, highest z {max(null_z) if null_z else float('nan'):.2f})")
print(f"Correct copy named among positives: {pct_ci(sum(r['named_A'] for r in pos), len(pos))}; "
      f"wrong copy {sum(r['named_B'] for r in pos)}")
print(f"\nCSV: output/overwritten_verdict_{TAG}.csv ({time.time() - t0:.0f}s)")
