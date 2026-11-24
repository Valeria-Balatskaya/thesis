# scripts/owner_bundle.py
# E4: third-party verification of the owner-side forensics (docs/research_plan.md, criterion C5).
#
#   make    python scripts/owner_bundle.py make OUT_DIR --set e2pos --victim ours_v3 --n 5 [--null]
#           Saves self-contained bundles: original O, suspect S (PNG), and bundle.json with the
#           victim system and product index, the thief system and message, the protocol constants
#           and the decisions found when the bundle was made.
#   verify  python scripts/owner_bundle.py verify BUNDLE_DIR [BUNDLE_DIR ...]
#           A neutral party needs only the bundle, this repository's code and the public models:
#           it rebuilds the issued copy C and the 50 never-issued copies from O, re-runs informed
#           check, peel-off and order proof and compares with the saved decisions
#           (pass: identical decisions and |dz| <= 0.05).

import argparse
import csv
import json
import os
import platform
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image


def summary(ev) -> dict:
    return {"informed_z": ev.informed_z, "informed": bool(ev.informed), "peel_ok": bool(ev.peel_ok),
            "order_read_z": ev.order_read_z, "order_read": bool(ev.order_read),
            "order_claim_z": ev.order_claim_z, "order_claim": bool(ev.order_claim),
            "restored": bool(ev.restored)}


def msg_to_json(sysm, msg):
    return msg if isinstance(msg, (int, str)) else [int(b) for b in np.asarray(msg)]


def msg_from_json(m):
    return m if isinstance(m, (int, str)) else np.asarray(m, dtype=np.uint8)


def make(a):
    from src import owner_forensics as of
    systems = [of.load_system(n) for n in of.SYSTEMS]
    victim = of.load_system(a.victim)
    manifest = list(csv.DictReader(open(f"data_manifests/owner_forensics_{a.set}.csv")))
    rng = np.random.default_rng(2024)
    idxs = sorted(rng.choice(len(manifest), size=a.n, replace=False))
    os.makedirs(a.out, exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    for k, idx in enumerate(idxs):
        meta = manifest[idx]
        O = np.asarray(Image.open(f"data/owner_forensics/{a.set}/{meta['file']}").convert("RGB"))
        T = of.load_system(of.SYSTEMS[int(rng.integers(4))])
        level = float(rng.choice([1.0, 1.5, 2.0, 3.0]))
        case = of.Case(victim, int(idx), O, systems)
        m_t = T.thief_message(int(idx))
        S = T.mark(O if a.null else case.copy, m_t, level)
        ev = case.evidence(S, thief_claim=(T, m_t))
        d = f"{a.out}/{a.victim}_{'null' if a.null else 'pos'}_{k:02d}"
        os.makedirs(d, exist_ok=True)
        Image.fromarray(O).save(f"{d}/original.png")
        Image.fromarray(S).save(f"{d}/suspect.png")
        json.dump({"victim": a.victim, "image_index": int(idx), "thief": T.name, "thief_level": level,
                   "thief_message": msg_to_json(T, m_t), "null_case": bool(a.null),
                   "n_decoys": of.N_DECOYS, "z_threshold": of.Z_THRESHOLD, "git": git,
                   "python": platform.python_version(), "platform": platform.platform(),
                   "decisions": summary(ev)}, open(f"{d}/bundle.json", "w"), indent=1)
        print(d, "restored" if ev.restored else "not restored", flush=True)


def verify(a):
    from src import owner_forensics as of
    systems = [of.load_system(n) for n in of.SYSTEMS]
    ok_all = True
    for d in a.bundles:
        b = json.load(open(f"{d}/bundle.json"))
        victim = of.load_system(b["victim"])
        thief = of.load_system(next(s.name for s in systems if s.name.startswith(b["thief"].split("_")[0])))
        O = np.asarray(Image.open(f"{d}/original.png").convert("RGB"))
        S = np.asarray(Image.open(f"{d}/suspect.png").convert("RGB"))
        case = of.Case(victim, b["image_index"], O, systems, b["n_decoys"])
        ev = case.evidence(S, thief_claim=(thief, msg_from_json(b["thief_message"])))
        now, then = summary(ev), b["decisions"]
        same = all(now[k] == then[k] for k in ("informed", "peel_ok", "order_read", "order_claim", "restored"))
        dz = max(abs(now[k] - then[k]) for k in ("informed_z", "order_claim_z") if now[k] is not None and then[k] is not None)
        ok = same and dz <= 0.05
        ok_all &= ok
        print(f"{'PASS' if ok else 'FAIL'}  {d}  restored={now['restored']}  max|dz|={dz:.4f}")
    print("ALL PASS" if ok_all else "SOME FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("out")
    m.add_argument("--set", default="e2pos")
    m.add_argument("--victim", default="ours_v3")
    m.add_argument("--n", type=int, default=5)
    m.add_argument("--null", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("bundles", nargs="+")
    a = ap.parse_args()
    make(a) if a.cmd == "make" else verify(a)
