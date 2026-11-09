# tests/test_ablation.py
# Ablation: what does each component contribute? Remove one at a time from the deployed
# system and measure the impact (Part 5.2; objectives 1.2.2, 1.2.3, 1.2.5, 1.4).
#
# Deployed system = HiDDeN v3 as trained (NVF texture mask) + payload with BCH(63,30,6) and
# a 10-bit HMAC tag + aspect-ratio retry + Retrieve-Align-Verify. Measured here, on
# n_own + n_abo photos x the 23 attacks of src/attack_suite.py, and false positives on
# N_NEG never-protected ABO photos (data/abo_negatives):
#   full                 everything
#   - alignment          blind read only (with aspect retry)
#   - aspect retry       blind read without padding to other aspect ratios (+ alignment)
#   - BCH                no error correction: the 30 message bits (ID + tag) must be read
#                        exactly (BCH is systematic, so they are the first 30 bits)
#   - HMAC tag           any BCH-decodable word counts as an ID -> false positives only
#   - texture mask       same model, embedded with mask_type="none" (signal also on
#                        flat / white background): visibility and robustness
#   - residual embedding HiDDeN v1: residual mode vs the legacy full-image output (PSNR only,
#                        10 photos; v1 = the first model of the project)
# Components measured in other tests (rows added by scripts/thesis_tables.py):
#   - DCT layer          tests/test_overwrite_cost.py (v3 vs v3+dct)
#   - v4 secret key      tests/test_overwrite_cost.py (v4 other_key) + tests/test_v4_keys.py
#   - content binding    tests/test_copy_attack.py
#
# Run: python tests/test_ablation.py [n_own] [n_abo] [n_negatives]   (default 30 30 500, ~60 min)

import csv
import os
import sys
import time
from shutil import copyfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image

from src.attack_suite import ATTACKS
from src.defence import DefencePipeline
from src.hidden_v2_inference import HiddenV2Model
from src.metrics import compute, visibility
from src.payload import decode_payload, encode_product_id, _mask, _CODE
from src.stats import pct_ci

N_OWN = int(sys.argv[1]) if len(sys.argv) > 1 else 30
N_ABO = int(sys.argv[2]) if len(sys.argv) > 2 else 30
N_NEG = int(sys.argv[3]) if len(sys.argv) > 3 else 500
CKPT = "checkpoints/hidden_v3_best.pt"
OUT, WORK = "output/ablation", "results/ablation"
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

model = HiddenV2Model(CKPT)
MASK = _mask()
K = _CODE.k
# sanity: systematic code, message in the first k bits
_c = encode_product_id(1234) ^ MASK
assert _CODE.decode(_c)[0] is not None and (_CODE.decode(_c)[0] == _c[:K]).all()


def info_exact(bits, cid):
    return bool(((np.asarray(bits) ^ MASK)[:K] == (encode_product_id(cid) ^ MASK)[:K]).all())


class NoAspect:
    """The same model, but read_product never pads to other aspect ratios."""
    def __init__(self, m):
        self.m = m

    def __getattr__(self, name):
        return getattr(self.m, name)

    def read_product(self, path, **kw):
        return self.m.read_product(path, try_aspects=False)


images = []
for tag, folder, n in [("own", "data/products_png", N_OWN), ("abo", "data/abo_sample", N_ABO)]:
    for k, f in enumerate(sorted(x for x in os.listdir(folder)
                                 if x.lower().endswith((".png", ".jpg", ".jpeg")))[:n]):
        p = f"{WORK}/{tag}_{k:03d}_orig.png"
        if not os.path.exists(p):
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            img.thumbnail((1024, 1024), Image.LANCZOS)
            img.save(p, format="PNG")
        images.append((tag, p))

VARIANTS = {"deployed": {}, "no_texture_mask": {"mask_type": "none"}}
rows, quality = [], []
t0 = time.time()
for vname, mask_kw in VARIANTS.items():
    pipe, pipe_na = DefencePipeline(model), DefencePipeline(NoAspect(model))
    pubs = {}
    for i, (tag, orig) in enumerate(images):
        pubs[i] = f"{WORK}/{vname}_{i:03d}.png"
        model.embed_product(orig, 1000 + i, pubs[i], strength=1.0, **mask_kw)
        for p in (pipe, pipe_na):
            p.register(i, orig, [1000 + i])
        q, v = compute(orig, pubs[i]), visibility(orig, pubs[i])
        quality.append({"variant": vname, "set": tag, "image": i, "psnr": q["PSNR_dB"], "ssim": q["SSIM"],
                        "dE": v["dE_mean"], "dE_flat": v["dE_flat"]})
    for i, (tag, orig) in enumerate(images):
        cid = 1000 + i
        for atk, fn, akw in ATTACKS:
            sus = f"{WORK}/atk.png"
            np.random.seed(1000 * i + len(atk))
            copyfile(pubs[i], sus) if fn is None else fn(pubs[i], sus, **akw)
            direct = model.read_product(sus, try_aspects=False)
            blind_aspect = direct if direct["product_id"] == cid else model.read_product(sus)
            b_direct, b_aspect = direct["product_id"] == cid, blind_aspect["product_id"] == cid
            res = pipe.analyse(sus) if not b_aspect else None
            full = b_aspect or res["copy_id"] == cid
            if b_direct:
                full_na = True
            elif b_aspect:                         # only the aspect retry found it: re-run without
                full_na = pipe_na.analyse(sus)["copy_id"] == cid
            else:
                full_na = res["copy_id"] == cid
            aligned_bits = None
            if res is not None and res.get("aligned") is not None:
                aligned_bits = pipe._read(res["aligned"])
            no_ecc_blind = info_exact(direct["bits"], cid)
            no_ecc_full = no_ecc_blind or (aligned_bits is not None and info_exact(aligned_bits, cid))
            rows.append({"variant": vname, "set": tag, "image": i, "attack": atk,
                         "full": int(full), "no_alignment": int(b_aspect), "no_aspect_retry": int(full_na),
                         "no_bch_blind": int(no_ecc_blind), "no_bch_full": int(no_ecc_full),
                         "blind_direct": int(b_direct)})
        if i % 10 == 9:
            print(f"{vname}: {i + 1}/{len(images)} ({time.time() - t0:.0f}s)", flush=True)

# false positives on never-protected photos: with and without the HMAC tag / aspect retry / BCH
negs = sorted(f for f in os.listdir("data/abo_negatives") if f.lower().endswith((".jpg", ".png")))[:N_NEG]
fp = {"deployed": 0, "no_hmac_tag": 0, "no_aspect_retry": 0, "no_bch_tag_only": 0}
tmp = f"{WORK}/neg.png"
for f in negs:
    img = Image.open(os.path.join("data/abo_negatives", f)).convert("RGB")
    img.thumbnail((1024, 1024), Image.LANCZOS)
    img.save(tmp, format="PNG")
    direct = model.read_product(tmp, try_aspects=False)
    full = direct if direct["product_id"] is not None else model.read_product(tmp)
    fp["deployed"] += full["product_id"] is not None
    fp["no_aspect_retry"] += direct["product_id"] is not None
    fp["no_hmac_tag"] += decode_payload(direct["bits"])["reason"] in ("ok", "tag_mismatch")
    # no BCH: take the 30 message bits as read; the HMAC tag is the only check
    info = (direct["bits"] ^ MASK)[:K]
    pid = int("".join(map(str, info[:20])), 2)
    from src.payload import _tag
    fp["no_bch_tag_only"] += int("".join(map(str, info[20:])), 2) == _tag(pid)

# residual vs legacy embedding (HiDDeN v1)
res_rows = []
if os.path.exists("checkpoints/hidden_final.pt"):
    from src.hidden_inference import HiddenModel
    v1 = HiddenModel("checkpoints/hidden_final.pt", device="cpu")
    for i, (tag, orig) in enumerate(images[:5] + images[N_OWN:N_OWN + 5]):
        for mode in ("residual", "legacy"):
            out = f"{WORK}/v1_{mode}.png"
            v1.embed(orig, f"SELLER:{i}", out, mode=mode)
            res_rows.append({"mode": mode, "psnr": compute(orig, out)["PSNR_dB"]})

for name, data in (("rows", rows), ("quality", quality)):
    with open(f"{OUT}/{name}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
        w.writeheader()
        w.writerows(data)

n_img = len(images)
summary = []
print(f"\nABLATION — {CKPT}, {n_img} photos ({N_OWN} own + {N_ABO} ABO) x {len(ATTACKS)} attacks\n")
print(f"{'configuration':<34}{'traced (all attacks)':<32}{'PSNR':>6}{'dE':>6}{'dE bg':>7}  false positives (n={len(negs)})")


def qline(v):
    qs = [q for q in quality if q["variant"] == v]
    flat = [q["dE_flat"] for q in qs if q["dE_flat"] is not None]
    return np.mean([q["psnr"] for q in qs]), np.mean([q["dE"] for q in qs]), np.mean(flat) if flat else float("nan")


def add(label, k, n, v="deployed", fp_k=None):
    p, d, df = qline(v)
    summary.append({"configuration": label, "traced": k, "n": n, "psnr": round(p, 2), "dE": round(d, 3),
                    "dE_bg": round(df, 3), "false_positives": fp_k, "n_negatives": len(negs)})
    print(f"{label:<34}{pct_ci(k, n):<32}{p:>6.1f}{d:>6.2f}{df:>7.2f}  "
          + ("-" if fp_k is None else pct_ci(fp_k, len(negs), 2)))


dep = [r for r in rows if r["variant"] == "deployed"]
nm = [r for r in rows if r["variant"] == "no_texture_mask"]
N = len(dep)
add("full system", sum(r["full"] for r in dep), N, fp_k=fp["deployed"])
add("- alignment (blind only)", sum(r["no_alignment"] for r in dep), N, fp_k=fp["deployed"])
add("- aspect-ratio retry", sum(r["no_aspect_retry"] for r in dep), N, fp_k=fp["no_aspect_retry"])
add("- BCH (exact 30 bits), full", sum(r["no_bch_full"] for r in dep), N, fp_k=fp["no_bch_tag_only"])
add("- BCH, blind only", sum(r["no_bch_blind"] for r in dep), N)
add("- HMAC tag", sum(r["full"] for r in dep), N, fp_k=fp["no_hmac_tag"])
add("- texture mask (mask none)", sum(r["full"] for r in nm), len(nm), v="no_texture_mask")
for tag in ("own", "abo"):
    sub = [r for r in dep if r["set"] == tag]
    sub2 = [r for r in nm if r["set"] == tag]
    print(f"  {tag}: full {pct_ci(sum(r['full'] for r in sub), len(sub))}, blind {pct_ci(sum(r['no_alignment'] for r in sub), len(sub))}, "
          f"no mask {pct_ci(sum(r['full'] for r in sub2), len(sub2))}")
if res_rows:
    for mode in ("residual", "legacy"):
        ps = [r["psnr"] for r in res_rows if r["mode"] == mode]
        print(f"HiDDeN v1 {mode:<9} embedding: PSNR {np.mean(ps):.1f} dB (n={len(ps)})")
        summary.append({"configuration": f"v1 {mode} embedding", "traced": None, "n": len(ps),
                        "psnr": round(float(np.mean(ps)), 2), "dE": None, "dE_bg": None,
                        "false_positives": None, "n_negatives": None})
with open(f"{OUT}/summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)
print(f"\n{OUT}/summary.csv ({time.time() - t0:.0f}s)")
