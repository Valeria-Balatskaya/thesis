# tests/test_ai_attacks.py
# Generative-AI attacks and the TrustMark remover on every system (Parts A2 + A3;
# objectives 1.2.3, 1.4). Same photos, scoring and alignment as tests/test_baselines.py.
#
# Systems at their default strength: ours v3 (deployed), TrustMark Q, Watermark Anything,
# Video Seal 1.0. Attacks on each protected copy:
#   regenerate_vae      SD VAE encode/decode at 768 px (src/modern_attacks)
#   replace_background  rembg (U2-Net) cut-out on a new gradient background
#   img2img_0.1..0.5    Stable Diffusion 1.5 img2img, empty prompt, no guidance, DDIM 50 steps,
#                       512 px, seed 0 (same noise for every system)
#   trustmark_rm        TrustMark-RM, the remover network shipped with the trustmark package
# Per attacked image: exact message read blind; + Retrieve-Align-Verify (that system's own
# decoder on the aligned image); ours also: the deployed DefencePipeline and the calibrated
# informed check (z >= 4 -> ownership proven); dE of the attacked image vs the clean
# original (how much the attack itself changed the picture).
# Rows are appended to output/ai_attacks/rows.csv; re-running resumes.
#
# Run: python tests/test_ai_attacks.py [n_per_set]   (default 30 -> 90 photos, ~3-4 h, MPS)
#      SYSTEMS=v3,trustmark,wam,videoseal  ATTACKS=img2img_0.1,...  to restrict
#      PER_SET=10  only the first 10 photos of each set (same image numbering as the full run,
#                  so rows stay resumable); summary.csv then compares all systems on the
#                  photos every system has, summary_all.csv uses every row

import csv
import os
import sys
import time
from functools import partial

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image

from src import modern_attacks as ma
from src.baselines import Ours, TrustMarkBaseline, WAMBaseline, VideoSealBaseline
from src.defence import DefencePipeline, InformedChecker, Z_THRESHOLD, fingerprint, match, align
from src.metrics import visibility
from src.stats import pct_ci

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SETS = [("own", "data/products_png"), ("abo", "data/abo_sample"), ("coco", "data/coco_sample")]
OUT, WORK = os.environ.get("OUT_DIR", "output/ai_attacks"), "results/ai_attacks"
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)
want = os.environ.get("SYSTEMS", "v3,trustmark,wam,videoseal").split(",")


def rgb(p):
    return np.asarray(Image.open(p).convert("RGB"))


images = []
for tag, folder in SETS:
    for k, f in enumerate(sorted(x for x in os.listdir(folder)
                                 if x.lower().endswith((".png", ".jpg", ".jpeg")))[:N]):
        p = f"results/baselines/{tag}_{k:03d}_orig.png"          # same files as test_baselines
        if not os.path.exists(p):
            os.makedirs("results/baselines", exist_ok=True)
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            img.thumbnail((1024, 1024), Image.LANCZOS)
            img.save(p, format="PNG")
        images.append((tag, p))
PER_SET = int(os.environ.get("PER_SET", N))
keep = [i for i, (_, p) in enumerate(images) if int(p.rsplit("_", 2)[-2]) < PER_SET]

tm = TrustMarkBaseline()
systems = {}
if "v3" in want:
    systems["v3"] = Ours("checkpoints/hidden_v3_best.pt", label="ours_v3")
if "trustmark" in want:
    systems["trustmark"] = tm
if "wam" in want:
    systems["wam"] = WAMBaseline()
if "videoseal" in want:
    systems["videoseal"] = VideoSealBaseline()

ATTACKS = [("regenerate_vae", ma.regenerate_vae), ("replace_background", ma.replace_background)]
ATTACKS += [(f"img2img_{s}", partial(ma.img2img, strength=s)) for s in (0.1, 0.2, 0.3, 0.5)]
ATTACKS += [("trustmark_rm", tm.remove)]
if os.environ.get("ATTACKS"):
    ATTACKS = [a for a in ATTACKS if a[0] in os.environ["ATTACKS"].split(",")]

fps = {i: fingerprint(rgb(p)) for i, (_, p) in enumerate(images)}
originals = {i: rgb(p) for i, (_, p) in enumerate(images)}


def retrieve_align(sus_rgb):
    best = None
    for flipped in (False, True):
        img = sus_rgb[:, ::-1].copy() if flipped else sus_rgb
        f = fingerprint(img)
        for i, ref_fp in fps.items():
            m = match(f, ref_fp)
            if m and (best is None or m["inliers"] > best[1]["inliers"]):
                best = (i, m, img)
        if best:
            break
    if best is None:
        return None, None
    i, m, img = best
    return i, align(img, m["H"], originals[i])[0]


ROWS = f"{OUT}/rows.csv"
FIELDS = ["system", "set", "image", "attack", "blind", "rav", "deployed", "informed_z", "owned",
          "detected", "bit_acc", "dE_attack", "retrieved"]
done = set()
if os.path.exists(ROWS):
    done = {(r["system"], int(r["image"]), r["attack"]) for r in csv.DictReader(open(ROWS))}
fh = open(ROWS, "a", newline="")
w = csv.DictWriter(fh, fieldnames=FIELDS)
if os.path.getsize(ROWS) == 0:
    w.writeheader()

t0 = time.time()
for key, s in systems.items():
    ours = isinstance(s, Ours)
    pipe = None
    if ours:
        pipe = DefencePipeline(s.model)
        for i, (_, p) in enumerate(images):
            pipe.register(i, p, [s.copy_id(i)])
    for i in keep:
        tag, orig = images[i]
        todo = [(a, fn) for a, fn in ATTACKS if (s.name, i, a) not in done]
        if not todo:
            continue
        pub = f"{WORK}/pub_{key}.png"
        s.embed(orig, pub, i)
        checker = InformedChecker(s.model, originals[i], s.copy_id(i),
                                  embed_kwargs={"strength": 1.0, "mask_floor": None}) if ours else None
        for atk, fn in todo:
            sus = f"{WORK}/atk_{key}.png"
            fn(pub, sus)
            r = s.read_path(sus, i)
            j, al = (None, None)
            ok_rav = r["ok"]
            if not ok_rav or ours:
                j, al = retrieve_align(rgb(sus))
                ok_rav = ok_rav or (j == i and s.read_rgb(al, i)["ok"])
            dep, z, owned = "", "", ""
            if ours:
                dep = int(r["ok"] or pipe.analyse(sus)["copy_id"] == s.copy_id(i))
                if j == i:
                    z = checker.test(al)["z"]
                else:
                    sr = rgb(sus)
                    z = checker.test(sr)["z"] if sr.shape == originals[i].shape else None
                owned = int(bool(dep) or (z is not None and z >= Z_THRESHOLD))
            w.writerow({"system": s.name, "set": tag, "image": i, "attack": atk, "blind": int(r["ok"]),
                        "rav": int(ok_rav), "deployed": dep, "informed_z": z, "owned": owned,
                        "detected": int(r["detected"]), "bit_acc": round(r["bit_acc"], 4),
                        "dE_attack": round(visibility(orig, sus)["dE_mean"], 3), "retrieved": int(j == i)})
        fh.flush()
        if i % 10 == 9:
            print(f"{s.name}: {i + 1}/{len(images)} ({time.time() - t0:.0f}s)", flush=True)
fh.close()

all_rows = list(csv.DictReader(open(ROWS)))
names = list(dict.fromkeys(r["system"] for r in all_rows))
per_sys = {n: {(r["image"], r["attack"]) for r in all_rows if r["system"] == n} for n in names}
shared = set.intersection(*per_sys.values())


def report(rows, out_name, title):
  print(f"\nAI ATTACKS + TrustMark-RM — {title}, default strengths")
  print("cells: +RAV (blind) of n; ours: deployed pipeline / ownership (pipeline or informed z>=4)\n")
  print(f"{'attack':<20}{'dE':>6}  " + "".join(f"{n:<26}" for n in names))
  summary = []
  for atk, _ in ATTACKS:
      line = ""
      sub_all = [r for r in rows if r["attack"] == atk]
      if not sub_all:
          continue
      for n in names:
          sub = [r for r in sub_all if r["system"] == n]
          if not sub:
              line += f"{'-':<26}"
              continue
          k = {c: sum(int(r[c]) for r in sub if r[c] != "") for c in ("blind", "rav", "deployed", "owned")}
          summary.append({"attack": atk, "system": n, "n": len(sub), **k,
                          "dE_attack": round(float(np.mean([float(r["dE_attack"]) for r in sub])), 2),
                          "bit_acc": round(float(np.mean([float(r["bit_acc"]) for r in sub])), 3)})
          cell = f"{k['rav']}/{len(sub)} ({k['blind']})"
          if sub[0]["deployed"] != "":
              cell += f" {k['deployed']}/{k['owned']}"
          line += f"{cell:<26}"
      print(f"{atk:<20}{np.mean([float(r['dE_attack']) for r in sub_all]):>6.2f}  {line}")
  with open(f"{OUT}/{out_name}", "w", newline="") as f:
      wr = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
      wr.writeheader()
      wr.writerows(summary)
  print("\nWith 95% Wilson intervals (+RAV):")
  for s in summary:
      print(f"  {s['attack']:<20}{s['system']:<14}{pct_ci(s['rav'], s['n'])}"
            + (f"   ownership {pct_ci(s['owned'], s['n'])}" if s["system"].startswith("ours") else ""))


shared_rows = [r for r in all_rows if (r["image"], r["attack"]) in shared]
n_sh = len({r["image"] for r in shared_rows})
report(shared_rows, "summary.csv", f"{n_sh} photos every system has")
report(all_rows, "summary_all.csv", "every row (ours may have more photos)")
print(f"\n{OUT}/ ({time.time() - t0:.0f}s)")
