# tests/test_v4_keys.py
# Key-conditioned watermark (v4): is our watermark immune to a second watermark
# written with the SAME model but ANOTHER key? (objectives 1.2.3, 1.2.5)
#
#   wrong keys   our protected copies (clean and JPEG q70) decoded with N_WRONG random
#                wrong keys: a valid ID must never come out (false reads), bit error
#                rate should be ~0.5
#   other key    a thief with the same model file but without our key re-protects our
#                published copy (1x, 1.5x, 2x, 3x strength): our key must still read
#                OUR ID; the thief's key reads the thief's ID; damage (dE) recorded
#   same key     for comparison, an insider with our key (what the upload pre-check
#                in TraceMark prevents)
#
# Run: python tests/test_v4_keys.py [checkpoint] [n_own] [n_abo]
#      (default checkpoints/hidden_v4_best.pt 20 20)

import sys, os, csv, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from PIL import Image

from src.hidden_v2_inference import HiddenV2Model
from src.hidden_v2 import random_keys
from src.payload import encode_product_id, decode_payload
from src.metrics import visibility
from src import attacks

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/hidden_v4_best.pt"
N_OWN = int(sys.argv[2]) if len(sys.argv) > 2 else 20
N_ABO = int(sys.argv[3]) if len(sys.argv) > 3 else 20
N_WRONG = 200
LEVELS = [1.0, 1.5, 2.0, 3.0]
OUT = "results/v4_keys"
os.makedirs(OUT, exist_ok=True)
os.makedirs("output", exist_ok=True)

model = HiddenV2Model(CKPT)
if not model.key_dim:
    sys.exit(f"{CKPT} is not key-conditioned (key_dim = 0)")
net, dev = model.model, model.device
torch.manual_seed(0)


def load(p):
    return model._load(p)


def save(t, p):
    arr = (t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(p, format="PNG")
    return p


@torch.no_grad()
def embed(x, cid, key, strength=1.0):
    msg = torch.tensor(encode_product_id(cid), dtype=torch.float32, device=dev).view(1, -1)
    return net.watermark(x, msg, strength, key=key)


@torch.no_grad()
def read(x, key):
    bits = (net.decode(x, key).squeeze(0) > 0).cpu().numpy().astype(np.uint8)
    return decode_payload(bits)["product_id"], bits


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

ours_key = model.key()
wrong_rows, over_rows = [], []
t0 = time.time()
for i, (tag, orig) in enumerate(images):
    cid = 1000 + i
    x = load(orig)
    pub = save(embed(x, cid, ours_key), f"{OUT}/pub.png")
    jpg = f"{OUT}/pub_jpg.png"
    attacks.jpeg_compress(pub, jpg, 70)
    for post, path in (("clean", pub), ("jpeg_q70", jpg)):
        y = load(path)
        got, _ = read(y, ours_key)
        false_reads, bers = 0, []
        wrong = random_keys(N_WRONG, model.key_dim, dev, avoid=ours_key.expand(N_WRONG, -1))
        for k in range(N_WRONG):
            g, bits = read(y, wrong[k:k + 1])
            false_reads += g is not None
            bers.append(float((bits != encode_product_id(cid)).mean()))
        wrong_rows.append({"image": i, "set": tag, "post": post, "our_key_reads": int(got == cid),
                           "wrong_key_false_reads": false_reads, "wrong_key_mean_ber": round(np.mean(bers), 3)})

    pub_t = load(pub)
    for who in ("other_key", "same_key"):
        thief_key = (random_keys(1, model.key_dim, dev, avoid=ours_key) if who == "other_key" else ours_key)
        for lvl in LEVELS:
            t = save(embed(pub_t, 500_000 + i, thief_key, strength=lvl), f"{OUT}/thief.png")
            for post in ("clean", "jpeg_q70"):
                sus = f"{OUT}/sus.png"
                (Image.open(t).save(sus) if post == "clean" else attacks.jpeg_compress(t, sus, 70))
                y = load(sus)
                ours_read, _ = read(y, ours_key)
                theirs_read, _ = read(y, thief_key)
                over_rows.append({"image": i, "set": tag, "thief": who, "level": lvl, "post": post,
                                  "our_key": "ours" if ours_read == cid else
                                             "theirs" if ours_read == 500_000 + i else
                                             "nothing" if ours_read is None else "other",
                                  "thief_key_reads_theirs": int(theirs_read == 500_000 + i),
                                  "thief_dE": visibility(orig, t)["dE_mean"]})
    if i % 10 == 9:
        print(f"{i + 1}/{len(images)} ({time.time() - t0:.0f}s)", flush=True)

for name, rows in (("wrong_keys", wrong_rows), ("overwrite", over_rows)):
    with open(f"output/v4_keys_{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

n = len(images)
print(f"\n{CKPT}: {n} photos ({N_OWN} own + {N_ABO} ABO)\n")
print(f"WRONG KEYS ({N_WRONG} random keys per image)")
for post in ("clean", "jpeg_q70"):
    sub = [r for r in wrong_rows if r["post"] == post]
    print(f"  {post:<9} our key reads ours {sum(r['our_key_reads'] for r in sub)}/{n}   "
          f"false reads with wrong keys {sum(r['wrong_key_false_reads'] for r in sub)}/{n * N_WRONG}   "
          f"mean BER {np.mean([r['wrong_key_mean_ber'] for r in sub]):.3f}")
print("\nOVERWRITE (thief re-protects our copy with the same model)")
print(f"  {'thief':<10}{'x':>5}{'post':>10}   our key: ours/theirs/nothing   thief key reads theirs   thief dE")
for who in ("other_key", "same_key"):
    for lvl in LEVELS:
        for post in ("clean", "jpeg_q70"):
            sub = [r for r in over_rows if r["thief"] == who and r["level"] == lvl and r["post"] == post]
            c = lambda v: sum(r["our_key"] == v for r in sub)
            print(f"  {who:<10}{lvl:>5}{post:>10}   {c('ours'):>8}/{c('theirs')}/{c('nothing'):<14}"
                  f"{sum(r['thief_key_reads_theirs'] for r in sub):>10}/{n}   "
                  f"{np.mean([r['thief_dE'] for r in sub]):.2f}")
print(f"\nCSVs: output/v4_keys_*.csv ({time.time() - t0:.0f}s)")
