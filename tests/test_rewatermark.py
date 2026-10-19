# tests/test_rewatermark.py
# What happens when someone puts their own watermark on top of ours?
#
# Each product photo is protected with OUR HiDDeN v2 watermark at the app
# setting (0.7 strength, 0.3 flat mask), then a second watermark is added on top:
#
#   other system   DCT spread spectrum (src/dct_watermark.py, Cox et al.),
#                  LSB (src/lsb.py), HiDDeN v1 (src/hidden_inference.py)
#   our system     a second HiDDeN v2 watermark with a different copy ID —
#                  a thief re-protecting the stolen image in TraceMark
#   visible        semi-transparent text/logo: small corner mark, large tiled
#                  diagonal "stock photo" watermark
#   layers         2 and 3 watermarks stacked (other systems, and v2 on v2)
#
# Every case is read clean and after a JPEG q70 re-save (what any platform does
# next). For each read we report whether OUR copy ID, the THIEF's ID, or nothing
# comes back, blind and with the full Retrieve-Align-Verify pipeline.
#
# In the v2-on-v2 cases the thief's copy is registered in the same pipeline, as
# it would be if the thief protected the stolen image in the same TraceMark
# server (their "original" is our published copy). That is the worst case the
# pre-check in app/guard.py protect() now refuses.
#
# Negative control: the second watermarks applied to the UN-watermarked
# originals must never read as one of our IDs.
#
# Uses the existing checkpoints only — no training.
# Run: python tests/test_rewatermark.py [n_own] [n_abo]   (default 20 20)

import sys, os, csv, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image, ImageDraw

from src.hidden_v2_inference import HiddenV2Model
from src.hidden_inference import HiddenModel
from src.defence import DefencePipeline, informed_score
from src.metrics import compute
from src import attacks, lsb
from src.dct_watermark import embed as dct_embed
from src.modern_attacks import _font

V2 = "checkpoints/hidden_v2_best.pt"
V1 = "checkpoints/hidden_final.pt"
N_OWN = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N_ABO = int(sys.argv[2]) if len(sys.argv) > 2 else 20
STRENGTH, MASK = 0.7, 0.3
OUT = "results/rewatermark"
os.makedirs(OUT, exist_ok=True)
os.makedirs("output", exist_ok=True)
np.random.seed(0)

v2 = HiddenV2Model(V2)
v1 = HiddenModel(V1, device="cpu")


# ─── second watermarks ────────────────────────────────────────────

def w_dct(src, dst, i):
    dct_embed(src, f"THIEF:{i}", dst, alpha=0.02, n_coeffs=500)


def w_lsb(src, dst, i):
    lsb.embed(src, f"(c) thief-shop {i}", dst, k=1)


def w_hidden_v1(src, dst, i):
    v1.embed(src, f"THIEF:{i}", dst, mode="residual")


def w_v2_theirs(src, dst, i):
    v2.embed_product(src, thief_id(i), dst, strength=STRENGTH, mask_floor=MASK)


def w_v2_theirs_full(src, dst, i):
    # thief uses full strength to try to overpower ours
    v2.embed_product(src, thief_id(i), dst, strength=1.0)


def w_visible_small(src, dst, i):
    img = Image.open(src).convert("RGB")
    w, h = img.size
    d = ImageDraw.Draw(img, "RGBA")
    d.text((int(w * 0.62), int(h * 0.92)), "© thief-shop", fill=(255, 255, 255, 150),
           font=_font(int(h * 0.04)), stroke_width=1, stroke_fill=(0, 0, 0, 90))
    img.save(dst, format="PNG")


def w_visible_large(src, dst, i):
    img = Image.open(src).convert("RGBA")
    w, h = img.size
    layer = Image.new("RGBA", (w * 2, h * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    font = _font(int(h * 0.07))
    for yy in range(0, h * 2, int(h * 0.18)):
        for xx in range(0, w * 2, int(w * 0.45)):
            d.text((xx, yy), "THIEF-SHOP ©", fill=(255, 255, 255, 90), font=font,
                   stroke_width=2, stroke_fill=(0, 0, 0, 60))
    layer = layer.rotate(30, resample=Image.BICUBIC).crop((w // 2, h // 2, w // 2 + w, h // 2 + h))
    Image.alpha_composite(img, layer).convert("RGB").save(dst, format="PNG")


def chain(*fns):
    def run(src, dst, i):
        cur = src
        for k, fn in enumerate(fns):
            nxt = f"{dst}.layer{k}.png"
            fn(cur, nxt, i if k == 0 else i + 1000 * k)
            cur = nxt
        Image.open(cur).save(dst, format="PNG")
    return run


def thief_id(i: int) -> int:
    return 500_000 + i


CASES = [
    ("other_dct",            "other system", w_dct),
    ("other_lsb",            "other system", w_lsb),
    ("other_hidden_v1",      "other system", w_hidden_v1),
    ("ours_v2_again",        "our system",   w_v2_theirs),
    ("ours_v2_again_full",   "our system",   w_v2_theirs_full),
    ("visible_small",        "visible",      w_visible_small),
    ("visible_large_tiled",  "visible",      w_visible_large),
    ("layers2_dct_v1",       "layers",       chain(w_dct, w_hidden_v1)),
    ("layers3_dct_v1_lsb",   "layers",       chain(w_dct, w_hidden_v1, w_lsb)),
    ("layers2_v2_v2",        "layers",       chain(w_v2_theirs, w_v2_theirs)),
    ("layers3_v2_v2_v2",     "layers",       chain(w_v2_theirs, w_v2_theirs, w_v2_theirs)),
]
V2_CASES = {"ours_v2_again", "ours_v2_again_full", "layers2_v2_v2", "layers3_v2_v2_v2"}
POST = [("clean", None, {}), ("jpeg_q70", attacks.jpeg_compress, {"quality": 70})]


# ─── data ─────────────────────────────────────────────────────────

def load_set(folder, n, tag):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg")))[:n]
    out = []
    for k, f in enumerate(files):
        p = f"{OUT}/{tag}_{k:03d}_orig.png"
        img = Image.open(os.path.join(folder, f)).convert("RGB")
        img.thumbnail((1024, 1024), Image.LANCZOS)
        img.save(p, format="PNG")
        out.append(p)
    return out


originals = load_set("data/products_png", N_OWN, "own") + load_set("data/abo_sample", N_ABO, "abo")
N_DECOYS = 8   # copies never issued: the null distribution of the informed check


def rgb(p):
    return np.asarray(Image.open(p).convert("RGB"))


ours, decoys = {}, {}
pipe = DefencePipeline(v2)
for i, orig in enumerate(originals):
    wm = f"{OUT}/{i:03d}_ours.png"
    v2.embed_product(orig, 1000 + i, wm, strength=STRENGTH, mask_floor=MASK)
    ours[i] = wm
    pipe.register(i, orig, [1000 + i])
    decoys[i] = []
    for d in range(N_DECOYS):
        v2.embed_product(orig, 900_000 + d, f"{OUT}/decoy.png", strength=STRENGTH, mask_floor=MASK)
        decoys[i].append(rgb(f"{OUT}/decoy.png"))


def outcome(copy_id, i):
    if copy_id is None:
        return "nothing"
    if copy_id == 1000 + i:
        return "ours"
    if copy_id >= 500_000:
        return "theirs"
    return "other_id"


rows = []
t0 = time.time()
for case, group, fn in CASES:
    for i, orig in enumerate(originals):
        stacked = f"{OUT}/{case}_{i:03d}.png"
        fn(ours[i], stacked, i)
        q = compute(ours[i], stacked)
        # A thief who re-protects in TraceMark registers our published copy as
        # their "original", with their own copy ID(s)
        thief_pid = 10_000 + i
        if case in V2_CASES:
            ids = [thief_id(i)] + ([thief_id(i + 1000)] if "layers" in case else []) \
                  + ([thief_id(i + 2000)] if "layers3" in case else [])
            pipe.register(thief_pid, ours[i], ids)
        for post, pfn, kw in POST:
            sus = f"{OUT}/sus.png"
            if pfn is None:
                Image.open(stacked).save(sus)
            else:
                pfn(stacked, sus, **kw)
            blind = v2.read_product(sus)["product_id"]
            res = pipe.analyse(sus)
            s_rgb, o_rgb = rgb(sus), rgb(orig)
            inf_ours = informed_score(s_rgb, o_rgb, rgb(ours[i]))
            inf_decoy = max(informed_score(s_rgb, o_rgb, d) for d in decoys[i])
            rows.append({"case": case, "group": group, "image": i, "post": post,
                         "psnr_vs_ours": q["PSNR_dB"],
                         "blind": outcome(blind, i),
                         "pipeline": outcome(res["copy_id"], i),
                         "pipeline_verdict": res["verdict"],
                         "informed_ours": round(inf_ours, 3),
                         "informed_decoy_max": round(inf_decoy, 3)})
        if case in V2_CASES:
            pipe.forget(thief_pid)
    print(f"{case:<22} done ({time.time() - t0:.0f}s)", flush=True)

# negative control: second watermark on the un-watermarked original
control = []
for case, group, fn in CASES:
    if case in V2_CASES:
        continue   # a v2 mark on a clean image is simply a valid (thief) watermark
    false = 0
    for i, orig in enumerate(originals):
        sus = f"{OUT}/ctrl.png"
        fn(orig, sus, i)
        cid = v2.read_product(sus)["product_id"]
        res = pipe.analyse(sus)
        false += (cid is not None) or (res["copy_id"] is not None)
    control.append({"case": case, "false_positives": f"{false}/{len(originals)}"})

with open("output/rewatermark_rows.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

summary = []
n = len(originals)
print(f"\n{n} product photos ({N_OWN} own + {N_ABO} ABO), ours = v2 at {STRENGTH}/{MASK}\n")
print(f"{'case':<22}{'post':<10}{'PSNR':>6}  {'blind: ours/theirs/none':>24}  {'pipeline: ours/theirs/none':>27}  informed: ours>decoys")
for case, group, _ in CASES:
    for post, _, _ in POST:
        sub = [r for r in rows if r["case"] == case and r["post"] == post]
        def cnt(col, val):
            return sum(r[col] == val for r in sub)
        s = {"case": case, "group": group, "post": post, "n": len(sub),
             "psnr_vs_ours": round(float(np.mean([r["psnr_vs_ours"] for r in sub])), 1),
             "blind_ours": cnt("blind", "ours"), "blind_theirs": cnt("blind", "theirs"),
             "blind_nothing": cnt("blind", "nothing"), "blind_other": cnt("blind", "other_id"),
             "pipe_ours": cnt("pipeline", "ours"), "pipe_theirs": cnt("pipeline", "theirs"),
             "pipe_nothing": cnt("pipeline", "nothing"), "pipe_other": cnt("pipeline", "other_id"),
             "pipe_fingerprint_only": sum(r["pipeline_verdict"] == "fingerprint_only" for r in sub),
             "informed_wins": sum(r["informed_ours"] > r["informed_decoy_max"] for r in sub),
             "informed_ours_median": round(float(np.median([r["informed_ours"] for r in sub])), 3),
             "informed_decoy_max_median": round(float(np.median([r["informed_decoy_max"] for r in sub])), 3),
             "informed_min_margin": round(float(min(r["informed_ours"] - r["informed_decoy_max"] for r in sub)), 3)}
        summary.append(s)
        print(f"{case:<22}{post:<10}{s['psnr_vs_ours']:>6}  "
              f"{s['blind_ours']:>8}/{s['blind_theirs']}/{s['blind_nothing']:<10}"
              f"{s['pipe_ours']:>12}/{s['pipe_theirs']}/{s['pipe_nothing']:<6}"
              f"{s['informed_wins']:>10}/{s['n']} (med {s['informed_ours_median']} vs {s['informed_decoy_max_median']})"
              + (f"  (other id {s['blind_other']}/{s['pipe_other']})" if s['blind_other'] or s['pipe_other'] else ""))

with open("output/rewatermark_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)
with open("output/rewatermark_control.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["case", "false_positives"])
    w.writeheader()
    w.writerows(control)
print("\nNegative control (second watermark on un-watermarked originals -> must read nothing):")
for c in control:
    print(f"  {c['case']:<22} {c['false_positives']}")
print("\nCSVs: output/rewatermark_summary.csv, rewatermark_rows.csv, rewatermark_control.csv")
