# tests/test_retry_rule.py
# Retry false-positive mitigation (Part B4). The blind reader retries an unreadable image
# padded to 8 other aspect ratios and mirrored (18 reads in all); each read is another chance
# of a random valid ID (the single blind false read in tests/test_mirror_fp.py came from the
# 16:9 retry with 6 corrected errors). Rule: a read produced by a RETRY is accepted only with
# <= 4 corrected errors; otherwise it is "tentative" and the pipeline accepts it only if
# alignment shows the picture of the product the ID belongs to
# (HiddenV2Model.retry_max_errors = 4, DefencePipeline "retry_confirmed").
#
# Cost: the 23-attack suite on 90 photos (30 own + 30 ABO + 30 COCO, as test_baselines),
# v3 at strength 1.0, mirror retry on (as in the app): blind and full pipeline, rule off/on.
# Benefit: the 1,167 never-protected images of test_mirror_fp: blind IDs and pipeline claims.
# Keep the rule only if the full pipeline loses <= 0.5 percentage points.
#
# Run: python tests/test_retry_rule.py [n_per_set]   (default 30, ~1.5 h)

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
from src.stats import pct_ci

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
LIMIT = 4
OUT, WORK = os.environ.get("OUT_DIR", "output/retry_rule"), "results/retry_rule"
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)
EXT = (".png", ".jpg", ".jpeg")
SETS = [("own", "data/products_png"), ("abo", "data/abo_sample"), ("coco", "data/coco_sample")]


def files(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(EXT))


images = []
for tag, folder in SETS:
    for k, f in enumerate(files(folder)[:N]):
        p = f"results/baselines/{tag}_{k:03d}_orig.png"
        if not os.path.exists(p):
            os.makedirs("results/baselines", exist_ok=True)
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            img.thumbnail((1024, 1024), Image.LANCZOS)
            img.save(p, format="PNG")
        images.append((tag, p))

model = HiddenV2Model("checkpoints/hidden_v3_best.pt")
model.try_mirror = True
pipe = DefencePipeline(model)
pubs = {}
for i, (tag, orig) in enumerate(images):
    pubs[i] = f"{WORK}/pub_{i:03d}.png"
    if not os.path.exists(pubs[i]):
        model.embed_product(orig, 1000 + i, pubs[i], strength=1.0)
    pipe.register(i, orig, [1000 + i])


def run(path, limit):
    model.retry_max_errors = limit
    b = model.read_product(path)
    return b


rows = []
t0 = time.time()
for i, (tag, orig) in enumerate(images):
    cid = 1000 + i
    for atk, fn, akw in ATTACKS:
        sus = f"{WORK}/atk.png"
        np.random.seed(1000 * i + len(atk))
        copyfile(pubs[i], sus) if fn is None else fn(pubs[i], sus, **akw)
        b0 = run(sus, None)
        retry = b0["product_id"] is not None and (b0["aspect"] is not None or b0["mirrored"])
        if b0["product_id"] == cid and not retry:
            r = dict(blind0=1, blind1=1, full0=1, full1=1)
        else:
            full0 = b0["product_id"] == cid or pipe.analyse(sus)["copy_id"] == cid
            b1 = run(sus, LIMIT)
            if b1["product_id"] == b0["product_id"] and b1["aspect"] == b0["aspect"]:
                full1 = full0
            else:
                full1 = b1["product_id"] == cid or pipe.analyse(sus)["copy_id"] == cid
            r = dict(blind0=int(b0["product_id"] == cid), blind1=int(b1["product_id"] == cid),
                     full0=int(full0), full1=int(full1))
        rows.append({"set": tag, "image": i, "attack": atk, **r, "retry_read": int(retry),
                     "retry_errors": b0["errors_corrected"] if retry else "",
                     "retry_aspect": b0["aspect"] or "", "retry_mirrored": int(b0["mirrored"])})
        model.retry_max_errors = None
    if i % 10 == 9:
        print(f"positives {i + 1}/{len(images)} ({time.time() - t0:.0f}s)", flush=True)

# never-protected images: same 1,167 as tests/test_mirror_fp.py
negs = [("abo_negative", os.path.join("data/abo_negatives", f)) for f in files("data/abo_negatives")[:1000]]
for tag, folder in SETS:
    negs += [(f"{tag}_photo", os.path.join(folder, f)) for f in files(folder)]
if os.environ.get("SMOKE") == "1":
    negs = negs[:5]
neg_rows = []
tmp = f"{WORK}/neg.png"
for k, (kind, path) in enumerate(negs):
    img = Image.open(path).convert("RGB")
    img.thumbnail((1024, 1024), Image.LANCZOS)
    img.save(tmp, format="PNG")
    r = {"kind": kind, "file": os.path.basename(path)}
    for lim, sfx in ((None, "0"), (LIMIT, "1")):
        b = run(tmp, lim)
        r["blind_id" + sfx] = "" if b["product_id"] is None else b["product_id"]
        r["errors" + sfx] = "" if b["product_id"] is None else b["errors_corrected"]
        r["tentative" + sfx] = "" if not b.get("tentative") else b["tentative"]["product_id"]
        # the pipeline only needs to run if the blind stage produced something
        claim = ""
        if b["product_id"] is not None or b.get("tentative"):
            claim = int(pipe.analyse(tmp)["copy_id"] is not None)
        r["claim" + sfx] = claim
    model.retry_max_errors = None
    neg_rows.append(r)
    if k % 300 == 299:
        print(f"negatives {k + 1}/{len(negs)} ({time.time() - t0:.0f}s)", flush=True)

for name, data in (("rows", rows), ("negatives", neg_rows)):
    with open(f"{OUT}/{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader()
        w.writerows(data)

n = len(rows)
k = {c: sum(r[c] for r in rows) for c in ("blind0", "blind1", "full0", "full1", "retry_read")}
print(f"\nRETRY RULE (retry reads need <= {LIMIT} corrected errors or alignment confirmation)")
print(f"{len(images)} photos x {len(ATTACKS)} attacks = {n} reads; mirror retry on\n")
print(f"  blind:          rule off {pct_ci(k['blind0'], n)}   rule on {pct_ci(k['blind1'], n)}")
print(f"  full pipeline:  rule off {pct_ci(k['full0'], n)}   rule on {pct_ci(k['full1'], n)}"
      f"   difference {100 * (k['full0'] - k['full1']) / n:.2f} pp")
ret = [r for r in rows if r["retry_read"]]
print(f"  correct reads that came from a retry: {len(ret)}; corrected errors: "
      + ", ".join(f"{e}: {sum(r['retry_errors'] == e for r in ret)}" for e in range(7)))
print("  per attack where the rule changes anything (blind off/on, full off/on):")
for atk, _, _ in ATTACKS:
    sub = [r for r in rows if r["attack"] == atk]
    a = [sum(r[c] for r in sub) for c in ("blind0", "blind1", "full0", "full1")]
    if a[0] != a[1] or a[2] != a[3]:
        print(f"    {atk:<20}{a[0]}/{a[1]}  {a[2]}/{a[3]}  of {len(sub)}")
nn = len(neg_rows)
for sfx, lab in (("0", "rule off"), ("1", "rule on ")):
    ids = sum(r["blind_id" + sfx] != "" for r in neg_rows)
    cl = sum(r["claim" + sfx] == 1 for r in neg_rows)
    print(f"  never-protected ({nn}), {lab}: blind ID {pct_ci(ids, nn, 2)}, pipeline claim {pct_ci(cl, nn, 2)}"
          + (f", tentative {sum(r['tentative1'] != '' for r in neg_rows)}" if sfx == "1" else ""))
hits = [r for r in neg_rows if r["blind_id0"] != ""]
for r in hits:
    print(f"    {r['kind']} {r['file']}: ID {r['blind_id0']} with {r['errors0']} errors -> rule on: "
          f"{'ID ' + str(r['blind_id1']) if r['blind_id1'] != '' else 'tentative only' if r['tentative1'] != '' else 'nothing'}")
with open(f"{OUT}/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["n", "blind_off", "blind_on", "full_off", "full_on",
                                      "neg_n", "neg_blind_off", "neg_blind_on", "neg_claim_off", "neg_claim_on"])
    w.writeheader()
    w.writerow({"n": n, "blind_off": k["blind0"], "blind_on": k["blind1"], "full_off": k["full0"],
                "full_on": k["full1"], "neg_n": nn,
                "neg_blind_off": sum(r["blind_id0"] != "" for r in neg_rows),
                "neg_blind_on": sum(r["blind_id1"] != "" for r in neg_rows),
                "neg_claim_off": sum(r["claim0"] == 1 for r in neg_rows),
                "neg_claim_on": sum(r["claim1"] == 1 for r in neg_rows)})
print(f"\n{OUT}/ ({time.time() - t0:.0f}s)")
