# tests/test_owner_forensics.py
# E1 (pilot, 40 photos) and E2 (300 victim photos + >= 1,300 null photos): owner-side forensics
# against re-watermarking for ANY victim system. Protocol and pass/fail criteria are frozen in
# docs/research_plan.md section 4b; the defences are src/owner_forensics.py.
#
# Positives (SET=pilot|e2pos): the owner publishes C = V(O, m_V); a thief T in {ours v3, TrustMark,
#   WAM, Video Seal} re-marks C at 1 / 1.5 / 2 / 3x its default strength; clean and JPEG q70.
#   Per case: blind read, informed check, peel-off, order proof (message read by the public decoders
#   and message disclosed by the thief), thief's message read, thief dE vs the clean original.
# Null (SET=e2null): the owner "registered" (O, C) but never published C. Cases per photo: the thief
#   marks the leaked clean original (random T, strength, post) and the clean O / JPEG q70 of O.
#   A photo is a false claim if ANY of its cases fires informed, peel-off or order proof.
#
# Run (one victim at a time, resumable):
#   VICTIM=ours_v3 SET=pilot python tests/test_owner_forensics.py [start end]
#   VICTIM=wam SET=e2null python tests/test_owner_forensics.py 0 650
# Env: THIEVES=ours_v3,trustmark,wam,videoseal  LEVELS=1,1.5,2,3  N_DECOYS=50  OUT_DIR=...

import csv
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image

from src import owner_forensics as of
from src.stats import frac_ci

SET = os.environ.get("SET", "pilot")
VICTIM = os.environ.get("VICTIM", "ours_v3")
THIEVES = os.environ.get("THIEVES", ",".join(of.SYSTEMS)).split(",")
LEVELS = [float(x) for x in os.environ.get("LEVELS", "1,1.5,2,3").split(",")]
D = int(os.environ.get("N_DECOYS", of.N_DECOYS))
OUT = os.environ.get("OUT_DIR", f"output/owner_forensics/{SET}")
os.makedirs(OUT, exist_ok=True)
ROWS = f"{OUT}/{VICTIM}.csv"

manifest = list(csv.DictReader(open(f"data_manifests/owner_forensics_{SET}.csv")))
start = int(sys.argv[1]) if len(sys.argv) > 2 else 0
end = int(sys.argv[2]) if len(sys.argv) > 2 else len(manifest)
end = min(end, len(manifest))

systems = [of.load_system(n) for n in of.SYSTEMS]
victim = of.load_system(VICTIM)
thieves = [of.load_system(n) for n in THIEVES]

FIELDS = ["victim", "set", "source", "image", "kind", "thief", "level", "post", "dE_O", "blind", "informed_z",
          "informed", "peel_ok", "peel_only", "peel_scale", "marks", "thief_read", "order_read_z", "order_read_t",
          "order_read", "order_claim_z", "order_claim_t", "order_claim", "order_scale", "restored",
          "restored_claim", "secs"]
done = set()
if os.path.exists(ROWS):
    done = {(int(r["image"]), r["kind"], r["thief"], r["level"], r["post"]) for r in csv.DictReader(open(ROWS))}
fh = open(ROWS, "a", newline="")
w = csv.DictWriter(fh, fieldnames=FIELDS)
if os.path.getsize(ROWS) == 0:
    w.writeheader()


def row(idx, meta, kind, T, level, post, S, ev, dE, m_t, secs):
    thief_read = ""
    if T is not None:
        thief_read = int(any(s.name == T.name and T.same(m, m_t) for s, m in ev.raw_marks))
    w.writerow({
        "victim": VICTIM, "set": SET, "source": meta["source"], "image": idx, "kind": kind,
        "thief": T.name if T else "none", "level": level if T else "", "post": post,
        "dE_O": "" if dE is None else round(dE, 3), "blind": int(victim.read_ok(S, idx)),
        "informed_z": round(ev.informed_z, 2), "informed": int(ev.informed), "peel_ok": int(ev.peel_ok),
        "peel_only": int(ev.peel_ok and not ev.peel_blind), "peel_scale": ev.peel_scale or "",
        "marks": ";".join(f"{a}:{b[:12]}" for a, b in ev.marks), "thief_read": thief_read,
        "order_read_z": "" if ev.order_read_z is None else round(ev.order_read_z, 2),
        "order_read_t": "" if ev.order_read_t is None else round(ev.order_read_t, 3),
        "order_read": int(ev.order_read),
        "order_claim_z": "" if ev.order_claim_z is None else round(ev.order_claim_z, 2),
        "order_claim_t": "" if ev.order_claim_t is None else round(ev.order_claim_t, 3),
        "order_claim": int(ev.order_claim),
        "order_scale": "" if ev.order_scale is None else round(ev.order_scale, 2),
        "restored": int(ev.restored), "restored_claim": int(ev.restored_claim), "secs": round(secs, 1)})


t0 = time.time()
for idx in range(start, end):
    meta = manifest[idx]
    O = np.asarray(Image.open(f"data/owner_forensics/{SET}/{meta['file']}").convert("RGB"))
    if SET == "e2null":
        rng = np.random.default_rng(777_000 + idx)
        T = thieves[int(rng.integers(len(thieves)))]
        level = float(rng.choice(LEVELS))
        post = str(rng.choice(["clean", "jpeg70"]))
        plan = [("null_thief", T, level, post), ("null_clean", None, 0, "clean"), ("null_clean", None, 0, "jpeg70")]
        if all((idx, k, t.name if t else "none", str(lv if t else ""), p) in done for k, t, lv, p in plan):
            continue
        case = of.Case(victim, idx, O, systems, D)
        for kind, T_, lv, p in plan:
            tc = time.time()
            if T_ is not None:
                m_t = T_.thief_message(idx)
                S0 = T_.mark(O, m_t, lv)
                dE = of.delta_e(O, S0)
                S = S0 if p == "clean" else of.jpeg(S0)
                ev = case.evidence(S, thief_claim=(T_, m_t))
            else:
                m_t, dE = None, None
                S = O if p == "clean" else of.jpeg(O)
                ev = case.evidence(S)
            row(idx, meta, kind, T_, lv, p, S, ev, dE, m_t, time.time() - tc)
        fh.flush()
    else:
        todo = [(T, lv, p) for T in thieves for lv in LEVELS for p in ("clean", "jpeg70")
                if (idx, "pos", T.name, str(lv), p) not in done]
        if not todo:
            continue
        case = of.Case(victim, idx, O, systems, D)
        for T in thieves:
            m_t = T.thief_message(idx)
            for lv in LEVELS:
                S0 = None
                for p in ("clean", "jpeg70"):
                    if (idx, "pos", T.name, str(lv), p) in done:
                        continue
                    tc = time.time()
                    if S0 is None:
                        S0 = T.mark(case.copy, m_t, lv)
                        dE = of.delta_e(O, S0)
                    S = S0 if p == "clean" else of.jpeg(S0)
                    ev = case.evidence(S, thief_claim=(T, m_t))
                    row(idx, meta, "pos", T, lv, p, S, ev, dE, m_t, time.time() - tc)
            fh.flush()
    n_done = idx - start + 1
    if n_done % 5 == 0 or idx == end - 1:
        print(f"{VICTIM} {SET}: {n_done}/{end - start} photos ({time.time() - t0:.0f}s)", flush=True)
fh.close()

rows = list(csv.DictReader(open(ROWS)))
print(f"\n{VICTIM} {SET}: {len(rows)} cases, {len({r['image'] for r in rows})} photos, "
      f"decoys {D}, {time.time() - t0:.0f}s this run")
if SET == "e2null":
    photos = {}
    for r in rows:
        any_ = int(r["informed"]) or int(r["peel_ok"]) or int(r["order_read"]) or int(r["order_claim"])
        photos[r["image"]] = photos.get(r["image"], 0) or any_
    print(f"false claims: {frac_ci(sum(photos.values()), len(photos))} photos (informed | peel-off | order proof)")
    for c in ("informed", "peel_ok", "order_read", "order_claim"):
        print(f"  {c}: {frac_ci(sum(int(r[c]) for r in rows), len(rows))} cases")
else:
    print("kind thief level post: n | blind | informed | peel | order(read) | order(claim) | restored | thief dE")
    for T in [t.name for t in thieves]:
        for lv in LEVELS:
            sub = [r for r in rows if r["thief"] == T and r["level"] and float(r["level"]) == lv]
            if not sub:
                continue
            k = lambda c: sum(int(r[c]) for r in sub)
            print(f"  {T:<10} {lv:>3} n={len(sub):<3} blind {k('blind'):<3} inf {k('informed'):<3} peel {k('peel_ok'):<3} "
                  f"ord {k('order_read'):<3} ordc {k('order_claim'):<3} restored {k('restored'):<3} "
                  f"dE {np.mean([float(r['dE_O']) for r in sub]):.2f}")
print(f"\n{ROWS}")
