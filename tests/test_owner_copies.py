# tests/test_owner_copies.py
# E3: leak attribution survives re-marking (docs/research_plan.md sections 3 and 4b, criterion C3).
# The owner issues 5 copies of every product (5 sales channels, 5 messages of the victim system V).
# One copy leaks and a thief re-marks it (T in all four systems, 1x and 2x, clean and JPEG q70).
# Which channel leaked?
#   informed naming (pre-registered, C3): z_k of every issued copy's residual against 50 never-issued
#       copies; the named copy = argmax z_k if max z_k >= 4, otherwise "none"
#   order naming (secondary): the same with the order-proof t of every issued copy's reconstruction
#       A_k = T(C_k, m_T) against the decoys' t; thief message = what the public decoders read
# Wrong copy = a named copy that is not the leaked one. Pass: 0 wrong copies.
#
# Run: VICTIM=ours_v3 python tests/test_owner_copies.py [n_photos]    (default 100, resumable)
#      photos = a seeded sample of the E2 positive set (disjoint from the pilot)

import csv
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image

from src import owner_forensics as of
from src.stats import frac_ci

VICTIM = os.environ.get("VICTIM", "ours_v3")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
D = int(os.environ.get("N_DECOYS", of.N_DECOYS))
K = 5
OUT = os.environ.get("OUT_DIR", "output/owner_forensics/copies")
os.makedirs(OUT, exist_ok=True)
ROWS = f"{OUT}/{VICTIM}.csv"

manifest = list(csv.DictReader(open("data_manifests/owner_forensics_e2pos.csv")))
sample = sorted(np.random.default_rng(5151).choice(len(manifest), size=min(N, len(manifest)), replace=False))
systems = [of.load_system(n) for n in of.SYSTEMS]
victim = of.load_system(VICTIM)

FIELDS = ["victim", "image", "source", "thief", "level", "post", "leaked", "named_informed", "max_z_informed",
          "z_true_informed", "named_order", "max_z_order", "z_true_order", "thief_read", "secs"]
done = set()
if os.path.exists(ROWS):
    done = {(int(r["image"]), r["thief"], r["level"], r["post"]) for r in csv.DictReader(open(ROWS))}
fh = open(ROWS, "a", newline="")
w = csv.DictWriter(fh, fieldnames=FIELDS)
if os.path.getsize(ROWS) == 0:
    w.writeheader()


def z_of(vals, k_true):
    """z of every issued copy against the decoys (vals[K:]); returns (named or None, max z, z of the true copy)."""
    null = np.array(vals[K:])
    z = (np.array(vals[:K]) - null.mean()) / (null.std(ddof=1) + 1e-12)
    top = int(z.argmax())
    return (top if z[top] >= of.Z_THRESHOLD else None), float(z[top]), float(z[k_true])


t0 = time.time()
for n_done, idx in enumerate(sample, 1):
    meta = manifest[idx]
    O = np.asarray(Image.open(f"data/owner_forensics/e2pos/{meta['file']}").convert("RGB"))
    rng = np.random.default_rng(88_000 + idx)
    plan = [(T, lv, p, int(rng.integers(K))) for T in of.SYSTEMS for lv in (1.0, 2.0) for p in ("clean", "jpeg70")]
    plan = [x for x in plan if (idx, of.load_system(x[0]).name, str(x[1]), x[2]) not in done]
    if not plan:
        continue
    issued = [victim.mark(O, victim.message(100_000 + 10 * idx + k)) for k in range(K)]
    decoys = [victim.mark(O, victim.decoy_message(k)) for k in range(D)]
    copies = issued + decoys
    bank = of.InformedBank(O, copies)
    own_msgs = [victim.message(100_000 + 10 * idx + k) for k in range(K)]
    thief_banks: dict = {}
    for tname, lv, post, leaked in plan:
        T = of.load_system(tname)
        tt = time.time()
        m_t = T.thief_message(idx)
        S = T.mark(issued[leaked], m_t, lv)
        if post == "jpeg70":
            S = of.jpeg(S)
        sc = bank.scores(S)
        zi = list((sc[:K] - sc[K:].mean()) / (sc[K:].std(ddof=1) + 1e-12))
        named_i = int(np.argmax(zi)) if max(zi) >= of.Z_THRESHOLD else None
        marks = [(sy, m) for sy, m in of.find_marks(S, systems, victim, 10**9)
                 if not (sy.name == victim.name and any(victim.same(m, om) for om in own_msgs))]
        named_o, zmax_o, z_true_o, thief_read = None, None, None, 0
        for s, msg in marks:
            key = (s.name, s.key(msg))
            if key not in thief_banks:
                if len(thief_banks) >= 1:
                    thief_banks.clear()
                thief_banks[key] = of.ThiefBank(s, O, copies, msg)
            ts, _ = thief_banks[key].ts(S)
            nm, zm, zt = z_of(ts, leaked)
            if zmax_o is None or zm > zmax_o:
                named_o, zmax_o, z_true_o = nm, zm, zt
            thief_read |= int(s.name == T.name and T.same(msg, m_t))
        w.writerow({"victim": VICTIM, "image": idx, "source": meta["source"], "thief": T.name, "level": lv, "post": post,
                    "leaked": leaked, "named_informed": "" if named_i is None else named_i,
                    "max_z_informed": round(max(zi), 2), "z_true_informed": round(zi[leaked], 2),
                    "named_order": "" if named_o is None else named_o,
                    "max_z_order": "" if zmax_o is None else round(zmax_o, 2),
                    "z_true_order": "" if z_true_o is None else round(z_true_o, 2),
                    "thief_read": thief_read, "secs": round(time.time() - tt, 1)})
    fh.flush()
    if n_done % 5 == 0 or n_done == len(sample):
        print(f"{VICTIM}: {n_done}/{len(sample)} photos ({time.time() - t0:.0f}s)", flush=True)
fh.close()

rows = list(csv.DictReader(open(ROWS)))
n = len(rows)
print(f"\nE3 leak attribution, victim {VICTIM}: {n} cases, {len({r['image'] for r in rows})} photos, "
      f"{K} issued copies + {D} decoys")
for name, col in (("informed (pre-registered)", "named_informed"), ("order proof", "named_order")):
    right = sum(r[col] != "" and int(r[col]) == int(r["leaked"]) for r in rows)
    wrong = sum(r[col] != "" and int(r[col]) != int(r["leaked"]) for r in rows)
    print(f"  {name:<26} correct {frac_ci(right, n)}   wrong copy {wrong}/{n} {frac_ci(wrong, n)}   none {n - right - wrong}")
print(ROWS)
