# tests/test_copy_attack.py
# Part B4: the copy attack (Kutter, Voloshynovskiy & Herrigel 2000, "The
# watermark copy attack", Proc. SPIE 3971): estimate the watermark from a
# protected image and add it to a DIFFERENT image, so that image appears to
# carry the victim's ID (framing an innocent seller, or faking provenance).
# Objective 1.2.5 (reliable detection / ownership verification).
#
#   attack     residual estimates from the protected copy W:
#                oracle    W - O (the exact residual; upper bound, needs the original)
#                median    W - median_5x5(W)
#                lowpass   W - gaussian(W, sigma=4)  (v2 lives in low frequencies)
#              scaled x1 and x2, added to an unrelated, unregistered product photo
#   detection  does the blind decoder read the victim's ID on the target?
#   defence    content binding (DefencePipeline.analyse(verify_content=True)):
#              the suspect must show the registered picture of the decoded ID
#              (SIFT/RANSAC + gradient structure check) -> "watermark_unbound"
#   cost       how many GENUINE attacked copies (12 attacks) would be wrongly
#              flagged as unbound
#   tag design why not bind the HMAC tag to a perceptual hash instead: pHash
#              Hamming distance between original and attacked genuine copies
#
# Run: python tests/test_copy_attack.py [n_own] [n_abo]   (default 20 20)

import sys, os, csv, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shutil import copyfile

import cv2
import numpy as np
from PIL import Image
from scipy.fft import dctn

from src.hidden_v2_inference import HiddenV2Model
from src.defence import DefencePipeline
from src import attacks, ecommerce_attacks as ea

N_OWN = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N_ABO = int(sys.argv[2]) if len(sys.argv) > 2 else 20
OUT = "results/copy_attack"
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


def norm(src, dst, size=None):
    img = Image.open(src).convert("RGB")
    if size:
        img = img.resize(size, Image.LANCZOS)
    else:
        img.thumbnail((1024, 1024), Image.LANCZOS)
    img.save(dst, format="PNG")
    return dst


def phash(arr: np.ndarray) -> np.ndarray:
    """64-bit perceptual hash (DCT of a 32x32 grey image, sign of the 8x8 low band)."""
    g = cv2.resize(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY), (32, 32), interpolation=cv2.INTER_AREA)
    low = dctn(g.astype(np.float64), norm="ortho")[:8, :8].ravel()
    return (low > np.median(low[1:])).astype(np.uint8)


def files(folder, n):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg")))[:n]


sources = [("own", f"data/products_png/{f}") for f in files("data/products_png", N_OWN)] + \
          [("abo", f"data/abo_sample/{f}") for f in files("data/abo_sample", N_ABO)]
targets = [f"data/abo_negatives/{f}" for f in files("data/abo_negatives", len(sources) + 20)][20:]

pipe = DefencePipeline(model)
transplants, genuine, hashes = [], [], []
np.random.seed(0)
t0 = time.time()
for i, (tag, src) in enumerate(sources):
    cid = 1000 + i
    orig = norm(src, f"{OUT}/{i:03d}_orig.png")
    wm = f"{OUT}/{i:03d}_wm.png"
    model.embed_product(orig, cid, wm, strength=0.7, mask_floor=0.3)
    pipe.register(i, orig, [cid])

for i, (tag, src) in enumerate(sources):
    cid = 1000 + i
    o, w = rgb(f"{OUT}/{i:03d}_orig.png").astype(np.float32), rgb(f"{OUT}/{i:03d}_wm.png")
    wf = w.astype(np.float32)
    estimates = {
        "oracle": wf - o,
        "median": wf - cv2.medianBlur(w, 5).astype(np.float32),
        "lowpass": wf - cv2.GaussianBlur(wf, (0, 0), 4),
    }
    h, wd = w.shape[:2]
    target = rgb(norm(targets[i], f"{OUT}/target.png", size=(wd, h))).astype(np.float32)
    for est, r in estimates.items():
        for scale in (1, 2):
            fake = np.clip(np.round(target + scale * r), 0, 255).astype(np.uint8)
            p = f"{OUT}/fake.png"
            Image.fromarray(fake).save(p)
            blind = model.read_product(p)["product_id"]
            res = pipe.analyse(p, verify_content=True)
            transplants.append({"image": i, "set": tag, "estimate": est, "scale": scale,
                                "blind_reads_victim": int(blind == cid),
                                "verdict": res["verdict"],
                                "framed": int(res["verdict"] in ("watermark", "watermark_aligned")
                                              and res["copy_id"] == cid)})

    # cost of the content check on genuine attacked copies + pHash fragility
    ph_o = phash(o.astype(np.uint8))
    for atk, fn, kw in ATTACK_SUITE:
        sus = f"{OUT}/sus.png"
        copyfile(f"{OUT}/{i:03d}_wm.png", sus) if fn is None else fn(f"{OUT}/{i:03d}_wm.png", sus, **kw)
        plain = pipe.analyse(sus)
        bound = pipe.analyse(sus, verify_content=True)
        genuine.append({"image": i, "set": tag, "attack": atk, "verdict_plain": plain["verdict"],
                        "verdict_bound": bound["verdict"],
                        "traced_plain": int(plain["copy_id"] == cid),
                        "traced_bound": int(bound["copy_id"] == cid and bound["verdict"] != "watermark_unbound")})
        hashes.append({"image": i, "attack": atk,
                       "phash_distance": int((phash(rgb(sus)) != ph_o).sum())})
    if i % 10 == 9:
        print(f"{i + 1}/{len(sources)} ({time.time() - t0:.0f}s)", flush=True)

for name, rows in [("transplants", transplants), ("genuine", genuine), ("phash", hashes)]:
    with open(f"output/copy_attack_{name}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

n = len(sources)
print(f"\nCOPY ATTACK: victim's residual estimated and added to {n} unrelated product photos")
print(f"{'estimate':<9}{'scale':>6}  {'blind decoder reads victim ID':>30}  {'framed (verdict watermark)':>28}  unbound")
for est in ("oracle", "median", "lowpass"):
    for scale in (1, 2):
        sub = [r for r in transplants if r["estimate"] == est and r["scale"] == scale]
        print(f"{est:<9}{scale:>6}  {sum(r['blind_reads_victim'] for r in sub):>27}/{n}  "
              f"{sum(r['framed'] for r in sub):>25}/{n}  {sum(r['verdict'] == 'watermark_unbound' for r in sub):>5}/{n}")
print("\nCOST on genuine attacked copies (traced without -> with content binding)")
tp = sum(r["traced_plain"] for r in genuine)
tb = sum(r["traced_bound"] for r in genuine)
print(f"  all: {tp}/{len(genuine)} -> {tb}/{len(genuine)}  (flagged unbound: "
      f"{sum(r['verdict_bound'] == 'watermark_unbound' for r in genuine)})")
for atk, _, _ in ATTACK_SUITE:
    sub = [r for r in genuine if r["attack"] == atk]
    a, b = sum(r["traced_plain"] for r in sub), sum(r["traced_bound"] for r in sub)
    if a != b:
        print(f"    {atk:<20} {a}/{len(sub)} -> {b}/{len(sub)}")
print("\nTAG DESIGN: pHash Hamming distance (of 64) between original and attacked genuine copy")
for atk, _, _ in ATTACK_SUITE:
    d = [r["phash_distance"] for r in hashes if r["attack"] == atk]
    print(f"  {atk:<20} median {np.median(d):4.0f}  max {max(d):3d}  exact match {sum(x == 0 for x in d)}/{len(d)}")
d = [r["phash_distance"] for r in hashes]
print(f"  any attack: exact pHash match in {sum(x == 0 for x in d)}/{len(d)} genuine copies "
      f"-> a tag bound to the suspect's pHash would fail {sum(x > 0 for x in d)}/{len(d)} genuine reads")
print(f"\nCSVs: output/copy_attack_*.csv ({time.time() - t0:.0f}s)")
