# tests/test_self_overwrite.py
# Who survives being overwritten by their OWN system? (Meeting 3 follow-up, Part A1;
# objectives 1.2.3, 1.2.5)
#
# For every system (ours v3, ours v4_final, TrustMark Q, Watermark Anything, Video Seal):
# protect a photo with message A, then a thief protects that copy again with the SAME
# system and message B at 1x / 1.5x / 2x / 3x of the system's default strength.
# v4 is keyed, so it gets two thieves: same_key (insider) and other_key (has the model
# file, not our key; thief's ID read with the thief's key).
# Measured per level, clean and after JPEG q70:
#   original read   message A still read blind with the system's own native decision
#   thief read      message B read (the thief's claim)
#   thief dE        CIEDE2000 of the thief's version vs the clean original
#   ownership       ours only: calibrated informed check z >= 4 (src/defence.InformedChecker)
# Baselines have no informed check of their own (their residuals are not kept), so only
# the blind columns apply to them.
#
# Run: python tests/test_self_overwrite.py [n_own] [n_abo]      (default 20 20, ~40 min)
#      SYSTEMS=v3,v4,trustmark,wam,videoseal to restrict

import csv
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image

from src import attacks
from src.defence import InformedChecker, Z_THRESHOLD
from src.hidden_v2_inference import HiddenV2Model
from src.metrics import visibility
from src.stats import frac_ci

N_OWN = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N_ABO = int(sys.argv[2]) if len(sys.argv) > 2 else 20
LEVELS = [1.0, 1.5, 2.0, 3.0]
WANT = os.environ.get("SYSTEMS", "v3,v4,trustmark,wam,videoseal").split(",")
OUT, WORK = "output/self_overwrite", "results/self_overwrite"
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)


def rgb(p):
    return np.asarray(Image.open(p).convert("RGB"))


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


class OursSys:
    """Our model behind the same calls as the baselines (message i -> copy ID)."""

    def __init__(self, name, ckpt):
        self.name, self.model = name, HiddenV2Model(ckpt)
        self.default_strength = 1.0

    def embed(self, src, dst, cid, strength=None, key=None):
        self.model.embed_product(src, cid, dst, strength=strength or 1.0, mask_floor=None, key=key)

    def read(self, path, cid, key=None):
        return self.model.read_product(path, key=key)["product_id"] == cid


systems = {}
if "v3" in WANT:
    systems["ours_v3"] = OursSys("ours_v3", "checkpoints/hidden_v3_best.pt")
if "v4" in WANT and os.path.exists("checkpoints/hidden_v4_final.pt"):
    systems["ours_v4_final"] = OursSys("ours_v4_final", "checkpoints/hidden_v4_final.pt")
from src.baselines import TrustMarkBaseline, WAMBaseline, VideoSealBaseline
for key, cls in (("trustmark", TrustMarkBaseline), ("wam", WAMBaseline), ("videoseal", VideoSealBaseline)):
    if key in WANT:
        systems[key] = cls()
print("systems:", list(systems), flush=True)

POSTS = [("clean", None), ("jpeg_q70", 70)]
rows = []
t0 = time.time()
for sname, s in systems.items():
    ours = isinstance(s, OursSys)
    thieves = (["same_key", "other_key"] if ours and s.model.key_dim else ["same_system"])
    for i, (tag, orig) in enumerate(images):
        pub = f"{WORK}/pub.png"
        a_id, b_id = (1000 + i, 500_000 + i) if ours else (i, 500_000 + i)
        s.embed(orig, pub, a_id, s.default_strength)
        checker = (InformedChecker(s.model, rgb(orig), a_id, embed_kwargs={"strength": 1.0, "mask_floor": None})
                   if ours else None)
        for thief in thieves:
            tkey = s.model.other_key(10_000 + i) if thief == "other_key" else None
            for lvl in LEVELS:
                th = f"{WORK}/thief.png"
                if ours:
                    s.embed(pub, th, b_id, s.default_strength * lvl, key=tkey)
                else:
                    s.embed(pub, th, b_id, s.default_strength * lvl)
                dE = visibility(orig, th)["dE_mean"]
                for post, q in POSTS:
                    sus = f"{WORK}/sus.png"
                    if q is None:
                        Image.open(th).save(sus)
                    else:
                        attacks.jpeg_compress(th, sus, q)
                    if ours:
                        orig_read = s.read(sus, a_id)
                        thief_read = s.read(sus, b_id, key=tkey)
                        z = checker.test(rgb(sus))["z"]
                    else:
                        orig_read = s.read_path(sus, a_id)["ok"]
                        thief_read = s.read_path(sus, b_id)["ok"]
                        z = None
                    rows.append({"system": sname, "thief": thief, "set": tag, "image": i, "level": lvl,
                                 "post": post, "thief_dE": round(dE, 3), "original_read": int(orig_read),
                                 "thief_read": int(thief_read), "informed_z": z,
                                 "owned": "" if z is None else int(orig_read or z >= Z_THRESHOLD)})
        if i % 10 == 9:
            print(f"{sname}: {i + 1}/{len(images)} ({time.time() - t0:.0f}s)", flush=True)

with open(f"{OUT}/rows.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

n = len(images)
print(f"\nWHO SURVIVES BEING OVERWRITTEN BY THEIR OWN SYSTEM — {n} photos ({N_OWN} own + {N_ABO} ABO)\n")
print(f"{'system':<15}{'thief':<12}{'post':<10}{'x':>4}{'thief dE':>9}   {'original read blind':<22}"
      f"{'thief read':<22}ownership (ours: blind or z>=4)")
summary = []
for sname in systems:
    for thief in dict.fromkeys(r["thief"] for r in rows if r["system"] == sname):
        for post, _ in POSTS:
            for lvl in LEVELS:
                sub = [r for r in rows if r["system"] == sname and r["thief"] == thief
                       and r["post"] == post and r["level"] == lvl]
                k_o, k_t = sum(r["original_read"] for r in sub), sum(r["thief_read"] for r in sub)
                own = [r for r in sub if r["owned"] != ""]
                k_w = sum(r["owned"] for r in own) if own else None
                summary.append({"system": sname, "thief": thief, "post": post, "level": lvl, "n": len(sub),
                                "thief_dE": round(float(np.mean([r["thief_dE"] for r in sub])), 2),
                                "original_read": k_o, "thief_read": k_t, "owned": k_w,
                                "informed_z_median": (round(float(np.median([r["informed_z"] for r in own])), 2)
                                                      if own else None)})
                print(f"{sname:<15}{thief:<12}{post:<10}{lvl:>4}{summary[-1]['thief_dE']:>9.2f}   "
                      f"{frac_ci(k_o, len(sub)):<22}{frac_ci(k_t, len(sub)):<22}"
                      + ("-" if k_w is None else frac_ci(k_w, len(own))))
with open(f"{OUT}/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)
print(f"\n{OUT}/summary.csv, rows.csv ({time.time() - t0:.0f}s)")
