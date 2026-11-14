# tests/test_mirror_fp.py
# Mirror fix + 1,000-image false-positive check (Meeting 3 follow-up; objectives 1.2.3, 1.4)
#
# 1. Mirror: v3 reads mirrored copies 0/90 blind and 83/90 with alignment (test_baselines).
#    read_product(try_mirror=True) also reads the horizontally flipped image when every other
#    attempt fails. Positives: 90 photos (30 own + 30 ABO + 30 COCO), protected, then
#    mirrored / mirrored + JPEG q70 / mirrored + resize 0.5. Blind and full pipeline,
#    without and with the retry.
# 2. False positives: every never-protected image we have (1,000 ABO negatives + the 90
#    clean originals of the positives + remaining own/COCO/ABO photos), with the 90 products
#    registered in the pipeline. Counts: blind ID read (any ID, registered or not), and a
#    watermark verdict from the full pipeline; each without and with the mirror retry.
#    Mirror retry doubles the attempts: theoretical rate per image 8.6e-6 x 9 -> x 18.
#
# Run: python tests/test_mirror_fp.py [n_per_set] [n_negatives] [checkpoint]
#      (default 30 1000 checkpoints/hidden_v3_best.pt; ~40-60 min on an M1 CPU)

import sys, os, csv, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PIL import Image

from src.hidden_v2_inference import HiddenV2Model
from src.defence import DefencePipeline
from src import attacks, modern_attacks as ma
from src.stats import pct_ci

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
N_NEG = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
CKPT = sys.argv[3] if len(sys.argv) > 3 else "checkpoints/hidden_v3_best.pt"
OUT = "results/mirror_fp"
os.makedirs(OUT, exist_ok=True)
os.makedirs("output/mirror_fp", exist_ok=True)
EXT = (".png", ".jpg", ".jpeg")

model = HiddenV2Model(CKPT)
pipe = DefencePipeline(model)


def files(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(EXT))


def prep(src, dst):
    if not os.path.exists(dst):
        img = Image.open(src).convert("RGB")
        img.thumbnail((1024, 1024), Image.LANCZOS)
        img.save(dst, format="PNG")
    return dst


# ─── positives ────────────────────────────────────────────────────
pos = []
spare = []     # photos of the sets not used as positives -> extra negatives
for tag, folder in [("own", "data/products_png"), ("abo", "data/abo_sample"), ("coco", "data/coco_sample")]:
    fs = files(folder)
    for k, f in enumerate(fs[:N]):
        pos.append((tag, prep(os.path.join(folder, f), f"{OUT}/{tag}_{k:03d}_orig.png")))
    spare += [(tag, os.path.join(folder, f)) for f in fs[N:]]

POST = {
    "mirror": lambda s, d: ma.mirror(s, d),
    "mirror+jpeg70": lambda s, d: (ma.mirror(s, d), attacks.jpeg_compress(d, d, 70)),
    "mirror+resize0.5": lambda s, d: (ma.mirror(s, d), attacks.resize_attack(d, d, 0.5)),
}
rows = []
t0 = time.time()
for i, (tag, orig) in enumerate(pos):
    cid = 1000 + i
    prot = f"{OUT}/p{i:03d}.png"
    model.embed_product(orig, cid, prot, strength=1.0)
    pipe.register(i, orig, [cid])
for i, (tag, orig) in enumerate(pos):
    cid, prot = 1000 + i, f"{OUT}/p{i:03d}.png"
    for name, fn in POST.items():
        sus = f"{OUT}/sus.png"
        fn(prot, sus)
        r = {"set": tag, "image": i, "attack": name}
        for mir in (False, True):
            model.try_mirror = mir
            b = model.read_product(sus)["product_id"]
            res = pipe.analyse(sus)
            sfx = "_mirror" if mir else ""
            r["blind" + sfx] = int(b == cid)
            r["full" + sfx] = int(res["copy_id"] == cid)
            r["wrong" + sfx] = int(res["copy_id"] is not None and res["copy_id"] != cid)
        rows.append(r)
    if i % 30 == 29:
        print(f"positives {i + 1}/{len(pos)} ({time.time() - t0:.0f}s)", flush=True)
model.try_mirror = False

# ─── negatives ────────────────────────────────────────────────────
negs = [("abo_negative", os.path.join("data/abo_negatives", f)) for f in files("data/abo_negatives")[:N_NEG]]
negs += [(f"{tag}_clean_original", p) for tag, p in pos] + [(f"{tag}_spare", p) for tag, p in spare]
neg_rows = []
for k, (kind, path) in enumerate(negs):
    p = path                                   # clean originals of positives are already 1024 px PNGs
    if not path.startswith(OUT):
        img = Image.open(path).convert("RGB")
        img.thumbnail((1024, 1024), Image.LANCZOS)
        img.save(f"{OUT}/neg.png", format="PNG")
        p = f"{OUT}/neg.png"
    r = {"kind": kind, "file": os.path.basename(path)}
    for mir in (False, True):
        model.try_mirror = mir
        sfx = "_mirror" if mir else ""
        b = model.read_product(p)
        res = pipe.analyse(p)
        r["blind_id" + sfx] = "" if b["product_id"] is None else b["product_id"]
        r["verdict" + sfx] = res["verdict"]
        r["claim" + sfx] = int(res["verdict"].startswith("watermark"))
    neg_rows.append(r)
    if k % 200 == 199:
        print(f"negatives {k + 1}/{len(negs)} ({time.time() - t0:.0f}s)", flush=True)
model.try_mirror = False

for name, data in [("positives", rows), ("negatives", neg_rows)]:
    with open(f"output/mirror_fp/{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader()
        w.writerows(data)

print(f"\n{CKPT}: {len(pos)} positives ({N} own + {N} ABO + {N} COCO), {len(negs)} never-protected images\n")
print(f"{'attack':<18}{'blind':<30}{'blind + mirror retry':<30}{'full':<30}full + mirror retry")
summary = []
for name in POST:
    sub = [r for r in rows if r["attack"] == name]
    n = len(sub)
    k = {c: sum(r[c] for r in sub) for c in ("blind", "blind_mirror", "full", "full_mirror", "wrong", "wrong_mirror")}
    summary.append({"attack": name, "n": n, **k})
    print(f"{name:<18}{pct_ci(k['blind'], n):<30}{pct_ci(k['blind_mirror'], n):<30}"
          f"{pct_ci(k['full'], n):<30}{pct_ci(k['full_mirror'], n)}  wrong {k['wrong']}/{k['wrong_mirror']}")
n = len(neg_rows)
fb = sum(r["blind_id"] != "" for r in neg_rows)
fbm = sum(r["blind_id_mirror"] != "" for r in neg_rows)
fc = sum(r["claim"] for r in neg_rows)
fcm = sum(r["claim_mirror"] for r in neg_rows)
print(f"\nFALSE POSITIVES on {n} never-protected images ({sum(r['kind'] == 'abo_negative' for r in neg_rows)} ABO negatives "
      f"+ {len(pos)} clean originals of registered products + {len(spare)} other photos)")
print(f"  blind decoder returns any ID:      {pct_ci(fb, n, 2)}   with mirror retry {pct_ci(fbm, n, 2)}")
print(f"  pipeline claims a watermark:       {pct_ci(fc, n, 2)}   with mirror retry {pct_ci(fcm, n, 2)}")
summary.append({"attack": "FALSE_POSITIVES", "n": n, "blind": fb, "blind_mirror": fbm, "full": fc,
                "full_mirror": fcm, "wrong": "", "wrong_mirror": ""})
with open("output/mirror_fp/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)
print(f"\noutput/mirror_fp/ ({time.time() - t0:.0f}s)")
