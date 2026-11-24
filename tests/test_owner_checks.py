# tests/test_owner_checks.py
# Checks that run BEFORE E2 (docs/research_plan.md section 4b, deviation 3).
#
# MODE=sanity  10 pilot photos per victim. Two real suspects per photo (S = T(C) with T = the victim's own
#              system at 1x = the hardest cell, and T = the next system at 2x), then one broken input at a
#              time. Each control lists the defences that CONSUME the broken input; none of them may fire.
#                wrong_original   the owner's case is built on another photo's original (resized)
#                                   consumes: informed, order_read, order_claim   (peel-off never uses O)
#                wrong_copy       the claim is for another issued copy of the same photo (message i + 1000)
#                                   consumes: blind, informed, peel_ok, order_read, order_claim
#                random_message   thief_claim carries a random message of the thief's system
#                                   consumes: order_claim
#                foreign_system   thief_claim carries a message of a DIFFERENT system's embedder
#                                   consumes: order_claim
#                no_thief         S = C (nothing re-marked); the claim names a thief that never acted
#                                   consumes: order_read (no decoder-read mark may exist); the rest is
#                                   informational (the owner's mark IS in C, so informed / blind are true)
#              Any fired "consumed" defence = FAIL (stop, fix). Every other firing is printed too.
# MODE=null    null pilot: the first 100 photos of the E2 null set per victim. Per photo:
#                thief_<system> x4  the thief marks the leaked clean original with every system (random
#                                   strength 1/1.5/2/3 and post clean/JPEG q70): own system, other systems
#                unrelated_marked   another photo (resized) marked by a random thief
#                clean              the clean original
#              A photo is a false claim if ANY case fires informed, peel-off or order proof.
#
# Run: MODE=sanity VICTIM=ours_v3 python tests/test_owner_checks.py
#      MODE=null   VICTIM=wam     python tests/test_owner_checks.py [n_photos]      (resumable)

import csv
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from PIL import Image

from src import owner_forensics as of
from src.stats import frac_ci

MODE = os.environ.get("MODE", "sanity")
VICTIM = os.environ.get("VICTIM", "ours_v3")
OUT = os.environ.get("OUT_DIR", f"output/owner_forensics/{'sanity' if MODE == 'sanity' else 'null_pilot'}")
os.makedirs(OUT, exist_ok=True)
ROWS = f"{OUT}/{VICTIM}.csv"

systems = [of.load_system(n) for n in of.SYSTEMS]
victim = of.load_system(VICTIM)
SET = "pilot" if MODE == "sanity" else "e2null"
manifest = list(csv.DictReader(open(f"data_manifests/owner_forensics_{SET}.csv")))
N = int(sys.argv[1]) if len(sys.argv) > 1 else (10 if MODE == "sanity" else 100)
if MODE == "sanity":
    sample = sorted(np.random.default_rng(909).choice(len(manifest), size=N, replace=False))
else:
    sample = list(range(N))

FIELDS = ["mode", "victim", "image", "case", "thief", "level", "post", "blind", "informed_z", "informed", "peel_ok",
          "peel_only", "marks", "thief_read", "order_read_z", "order_read", "order_claim_z", "order_claim", "secs"]
done = set()
if os.path.exists(ROWS):
    done = {(int(r["image"]), r["case"], r["thief"], r["level"], r["post"]) for r in csv.DictReader(open(ROWS))}
fh = open(ROWS, "a", newline="")
w = csv.DictWriter(fh, fieldnames=FIELDS)
if os.path.getsize(ROWS) == 0:
    w.writeheader()


def load(idx):
    return np.asarray(Image.open(f"data/owner_forensics/{SET}/{manifest[idx]['file']}").convert("RGB"))


def fit(img, shape):
    return cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


REG = {of.load_system(n).name: n for n in of.SYSTEMS}      # system name ("trustmark_Q") -> registry key


def other(T):
    """The next system in of.SYSTEMS (a different embedder)."""
    return of.load_system(of.SYSTEMS[(of.SYSTEMS.index(REG[T.name]) + 1) % len(of.SYSTEMS)])


def log(case_name, idx, case, S, T, level, post, claim, secs=0.0):
    ev = case.evidence(S, thief_claim=claim)
    tr = ""
    if T is not None:
        m_t = T.thief_message(idx)
        tr = int(any(s.name == T.name and T.same(m, m_t) for s, m in ev.raw_marks))
    f = lambda v: "" if v is None else round(v, 2)
    w.writerow({"mode": MODE, "victim": VICTIM, "image": idx, "case": case_name, "thief": T.name if T else "none",
                "level": level if T else "", "post": post, "blind": int(victim.read_ok(S, case.i)),
                "informed_z": f(ev.informed_z), "informed": int(ev.informed), "peel_ok": int(ev.peel_ok),
                "peel_only": int(ev.peel_ok and not ev.peel_blind), "marks": ";".join(f"{a}" for a, _ in ev.marks),
                "thief_read": tr, "order_read_z": f(ev.order_read_z), "order_read": int(ev.order_read),
                "order_claim_z": f(ev.order_claim_z), "order_claim": int(ev.order_claim), "secs": round(secs, 1)})


t0 = time.time()
for n_done, idx in enumerate(sample, 1):
    O = load(idx)
    rng = np.random.default_rng(606_000 + idx)
    if MODE == "sanity":
        true_case = of.Case(victim, idx, O, systems)
        wrong_O = fit(load((int(idx) + 7) % len(manifest)), O.shape)
        wrong_O_case = of.Case(victim, idx, wrong_O, systems)
        wrong_copy_case = of.Case(victim, idx + 1000, O, systems)
        for T, lv in ((victim, 1.0), (other(victim), 2.0)):
            m_t = T.thief_message(idx)
            S = T.mark(true_case.copy, m_t, lv)
            T2 = other(T)
            rand_msg = T.thief_message(idx + 777_000)
            foreign = (T2, T2.thief_message(idx))
            controls = [("wrong_original", wrong_O_case, S, (T, m_t)),
                        ("wrong_copy", wrong_copy_case, S, (T, m_t)),
                        ("random_message", true_case, S, (T, rand_msg)),
                        ("foreign_system", true_case, S, foreign),
                        ("no_thief", true_case, true_case.copy, (T, m_t))]
            for name, case, susp, claim in controls:
                if (idx, name, T.name, str(lv), "clean") in done:
                    continue
                tt = time.time()
                log(name, idx, case, susp, T, lv, "clean", claim, secs=time.time() - tt)
            # the real thing, for reference: same suspect with the true inputs
            if (idx, "true_inputs", T.name, str(lv), "clean") not in done:
                log("true_inputs", idx, true_case, S, T, lv, "clean", (T, m_t))
        fh.flush()
    else:
        plan = []
        for tn in of.SYSTEMS:
            plan.append(("thief_" + tn, of.load_system(tn), float(rng.choice([1, 1.5, 2, 3])), str(rng.choice(["clean", "jpeg70"]))))
        plan.append(("unrelated_marked", of.load_system(of.SYSTEMS[int(rng.integers(4))]), float(rng.choice([1, 1.5, 2, 3])),
                     str(rng.choice(["clean", "jpeg70"]))))
        plan.append(("clean", None, 0, "clean"))
        if all((idx, c, t.name if t else "none", str(lv if t else ""), p) in done for c, t, lv, p in plan):
            continue
        case = of.Case(victim, idx, O, systems)
        for name, T, lv, post in plan:
            if (idx, name, T.name if T else "none", str(lv if T else ""), post) in done:
                continue
            tt = time.time()
            if name == "clean":
                S, claim = O, None
            else:
                base = O if name.startswith("thief_") else fit(load((idx + 1) % len(manifest)), O.shape)
                m_t = T.thief_message(idx)
                S = T.mark(base, m_t, lv)
                S = S if post == "clean" else of.jpeg(S)
                claim = (T, m_t)
            log(name, idx, case, S, T, lv, post, claim, secs=time.time() - tt)
        fh.flush()
    if n_done % 5 == 0 or n_done == len(sample):
        print(f"{MODE} {VICTIM}: {n_done}/{len(sample)} photos ({time.time() - t0:.0f}s)", flush=True)
fh.close()

rows = list(csv.DictReader(open(ROWS)))
print(f"\n{MODE} {VICTIM}: {len(rows)} cases, {len({r['image'] for r in rows})} photos, {time.time() - t0:.0f}s this run")
DEF = ("blind", "informed", "peel_ok", "peel_only", "order_read", "order_claim", "thief_read")
if MODE == "sanity":
    CONSUMED = {"wrong_original": ("informed", "order_read", "order_claim"),
                "wrong_copy": ("blind", "informed", "peel_ok", "order_read", "order_claim"),
                "random_message": ("order_claim",), "foreign_system": ("order_claim",), "no_thief": ("order_read",)}
    bad = 0
    print(f"{'control':<16}{'n':<4}" + "".join(f"{d:<12}" for d in DEF) + "  consumed-defence firings")
    for c in ("true_inputs", *CONSUMED):
        sub = [r for r in rows if r["case"] == c]
        if not sub:
            continue
        fired = {d: sum(int(r[d] or 0) for r in sub) for d in DEF}
        n_bad = sum(fired[d] for d in CONSUMED.get(c, ()))
        bad += n_bad
        print(f"{c:<16}{len(sub):<4}" + "".join(f"{fired[d]:<12}" for d in DEF) +
              (f"  {n_bad}  {'FAIL' if n_bad else 'ok'}" if c in CONSUMED else "  (reference)"))
    print("SANITY FAIL: a consumed defence fired" if bad else "SANITY OK: no consumed defence fired")
else:
    photos = {}
    for r in rows:
        a = int(r["informed"]) or int(r["peel_ok"]) or int(r["order_read"]) or int(r["order_claim"])
        photos[r["image"]] = photos.get(r["image"], 0) or a
    print(f"false claims: {frac_ci(sum(photos.values()), len(photos))} photos (any defence, any case)")
    for c in ("informed", "peel_ok", "order_read", "order_claim"):
        print(f"  {c}: {frac_ci(sum(int(r[c]) for r in rows), len(rows))} cases")
    for name in sorted({r["case"] for r in rows}):
        sub = [r for r in rows if r["case"] == name]
        print(f"  {name:<20} n={len(sub):<4} informed {sum(int(r['informed']) for r in sub)}  peel {sum(int(r['peel_ok']) for r in sub)}  "
              f"order_read {sum(int(r['order_read']) for r in sub)}  order_claim {sum(int(r['order_claim']) for r in sub)}  "
              f"max informed z {max(float(r['informed_z']) for r in sub):.2f}  max order z "
              f"{max([float(r['order_read_z']) for r in sub if r['order_read_z']] + [-9]):.2f}/"
              f"{max([float(r['order_claim_z']) for r in sub if r['order_claim_z']] + [-9]):.2f}")
print(ROWS)
