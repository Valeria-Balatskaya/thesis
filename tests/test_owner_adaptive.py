# tests/test_owner_adaptive.py
# Adaptive attacks A1-A6 on the owner-side forensics (docs/research_plan.md sections 4 and 4b).
# One victim per run (VICTIM=...), photos = a seeded sample of the E2 positive set (not the pilot).
#
#   A1  post-processing after re-marking (thief 1x / 2x): JPEG 90 / 70 / 50, resize 0.75, 3% crop on every
#       side, screenshot simulation; geometry is undone by retrieve-align (SIFT homography)
#   A2  unknown thief strength, log-uniform 0.5-3x (clean and JPEG q70), for the four systems and for
#       TrustMark variants B / C / P (the defender regenerates with Q). Two defenders on the variants:
#       "default" (the four systems, claim = Q with the disclosed message) and "sweep" (+ TrustMark B/C/P
#       as decoders and regenerators). WAM "other scaling" = the random strength (scaling_w is a linear
#       gain); Video Seal has no second checkpoint in this repository (videoseal_0.0 needs configs that
#       are not shipped), so it is not tested.
#   A3  closed watermark: the thief uses a model the defender does not have (our v2 checkpoint,
#       our v4 checkpoint with another key); only the informed check can work
#   A4  estimated-original attack: the thief denoises / regenerates C (non-local means, SD VAE,
#       SD img2img 0.1 / 0.2 / 0.3) and re-marks that (1x / 2x). Same attack on the clean original
#       gives the false-claim rate. Win for the thief = owner claim fails while their message reads
#   A5  stacked re-marking (2 and 3 layers, same or mixed systems), peel-off with 2 and 3 rounds
#   A6  fake original (Craver): the thief subtracts their own residual from the suspect and calls
#       the result "the original"; does it carry OUR residual? (positives) vs a thief who really
#       had the original (null)
# A7 (thief marks a leaked clean original) is the null of tests/test_owner_forensics.py.
#
# Order proof is reported twice: as pre-registered ("order") and post-processing tolerant ("rob", A1/A2 only;
# src.owner_forensics.test_robust, NOT pre-registered; computed only when informed and peel-off both failed,
# blank otherwise). Rows carry every defence separately.
#
# Run: ATTACK=a1 VICTIM=ours_v3 OFFSET=0 python tests/test_owner_adaptive.py [n_photos]   (default 20)
#      pilot = OFFSET 0, 20 photos; E2 = OFFSET 20, 60 photos (disjoint). LEVELS=1,2  THIEVES=... restrict.
#      Rows are resumable in OUT_DIR (default output/owner_forensics/adaptive/)

import csv
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from PIL import Image

from src import owner_forensics as of
from src.defence import align, fingerprint, match
from src.stats import frac_ci

ATTACK = os.environ.get("ATTACK", "a1")
VICTIM = os.environ.get("VICTIM", "ours_v3")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
OFFSET = int(os.environ.get("OFFSET", 0))
LEVELS = [float(x) for x in os.environ.get("LEVELS", "1,2").split(",")]
THIEVES = os.environ.get("THIEVES", ",".join(of.SYSTEMS)).split(",")
OUT = os.environ.get("OUT_DIR", "output/owner_forensics/adaptive")
os.makedirs(OUT, exist_ok=True)
ROWS = f"{OUT}/{ATTACK}_{VICTIM}.csv"
TMP = tempfile.mkdtemp(prefix="of_adaptive_")

manifest = list(csv.DictReader(open("data_manifests/owner_forensics_e2pos.csv")))
sample = sorted(int(i) for i in np.random.default_rng(4242).permutation(len(manifest))[OFFSET:OFFSET + N])
systems = [of.load_system(n) for n in of.SYSTEMS]
victim = of.load_system(VICTIM)
thieves = [of.load_system(n) for n in THIEVES]

FIELDS = ["attack", "victim", "image", "source", "thief", "variant", "level", "post", "dE_O", "retrieved", "blind",
          "informed_z", "informed", "peel_ok", "peel_rounds3", "marks", "thief_read", "order_read_z", "order_read",
          "order_claim_z", "order_claim", "order_scale", "order_rob_read_z", "order_rob_claim_z", "restored",
          "restored_rob", "defender", "extra", "secs"]
done = set()
if os.path.exists(ROWS):
    done = {(int(r["image"]), r["thief"], r["variant"], r["level"], r["post"]) for r in csv.DictReader(open(ROWS))}
fh = open(ROWS, "a", newline="")
w = csv.DictWriter(fh, fieldnames=FIELDS)
if os.path.getsize(ROWS) == 0:
    w.writeheader()


def rgb(p):
    return np.asarray(Image.open(p).convert("RGB"))


def via_file(fn, arr, **kw):
    """Apply a file-to-file attack (src/attacks, ecommerce_attacks, modern_attacks) to an array."""
    a, b = f"{TMP}/in.png", f"{TMP}/out.png"
    Image.fromarray(arr).save(a, format="PNG")
    fn(a, b, **kw)
    return rgb(b)


def to_frame(S, O):
    """Retrieve-align a (possibly resized / cropped) suspect into the original's frame."""
    if S.shape == O.shape:
        return S, True
    m = match(fingerprint(S), fingerprint(O))
    if m is None:
        return None, False
    return align(S, m["H"], O)[0], True


def log(case, idx, meta, T, variant, level, post, S, ev, dE, m_t, retrieved, secs, p3="", extra="", rob=(None, None),
        defender="default"):
    tr = ""
    if T is not None and m_t is not None and ev is not None:
        tr = int(any(s.name == T.name and T.same(m, m_t) for s, m in ev.raw_marks))
    f = lambda v, d=2: "" if v is None else round(v, d)
    rrz, rcz = rob
    informed_or_peel = ev is not None and (ev.informed or ev.peel_ok)
    w.writerow({"attack": ATTACK, "victim": VICTIM, "image": idx, "source": meta["source"],
                "thief": T.name if T else "none", "variant": variant, "level": level, "post": post,
                "dE_O": f(dE, 3), "retrieved": int(retrieved), "blind": "" if ev is None else int(victim.read_ok(S, idx)),
                "informed_z": "" if ev is None else f(ev.informed_z), "informed": "" if ev is None else int(ev.informed),
                "peel_ok": "" if ev is None else int(ev.peel_ok), "peel_rounds3": p3,
                "marks": "" if ev is None else ";".join(f"{a}:{b[:10]}" for a, b in ev.marks), "thief_read": tr,
                "order_read_z": "" if ev is None else f(ev.order_read_z), "order_read": "" if ev is None else int(ev.order_read),
                "order_claim_z": "" if ev is None else f(ev.order_claim_z), "order_claim": "" if ev is None else int(ev.order_claim),
                "order_scale": "" if ev is None else f(ev.order_scale),
                "order_rob_read_z": f(rrz), "order_rob_claim_z": f(rcz),
                "restored": 0 if ev is None else int(ev.restored),
                "restored_rob": 0 if ev is None else int(bool(informed_or_peel) or (rrz is not None and rrz >= of.ORDER_Z)),
                "defender": defender, "extra": extra, "secs": round(secs, 1)})


REGISTRY = {s.name for s in systems}


def run(case, idx, meta, T, variant, level, post, S0, m_t, claim=True, run_order=True, claim_sys=None,
        defender="default", case_systems=None, robust=False):
    """
    S0 = what the thief publishes (before any post-processing). One row per call. claim_sys = the system
    the DEFENDER regenerates with for the disclosed message (default: the thief's own, if it is one of
    the four); case_systems = the decoders / peelers the defender owns (default: the four systems).
    """
    key = (idx, T.name if T else "none", variant, str(level), post)
    if key in done:
        return
    t = time.time()
    dE = of.delta_e(case.o, S0) if S0.shape == case.o.shape else None
    S, ok = to_frame(S0, case.o)
    if not ok:
        log(case, idx, meta, T, variant, level, post, S0, None, dE, m_t, False, time.time() - t, defender=defender)
        return
    claim_obj = None
    if claim and T is not None:
        cs = claim_sys or T
        claim_obj = (cs, m_t) if cs.name in REGISTRY or claim_sys is not None or case_systems is not None else None
    if case_systems is not None:
        case.systems = case_systems
    try:
        ev = case.evidence(S, thief_claim=claim_obj, run_order=run_order)
    finally:
        case.systems = systems
    # tolerant order proof only where the order proof decides (informed and peel-off both failed); blank elsewhere
    rob = case.robust_order(S, ev, claim_obj) if (robust and run_order and not (ev.informed or ev.peel_ok)) else (None, None)
    log(case, idx, meta, T, variant, level, post, S, ev, dE, m_t, True, time.time() - t, rob=rob, defender=defender)


def attack_variants():
    """A1 transformations of the thief's output S0 (name -> function)."""
    from src import ecommerce_attacks as ea

    def crop3(a):
        h, wd = a.shape[:2]
        return a[round(.03 * h):h - round(.03 * h), round(.03 * wd):wd - round(.03 * wd)]

    def resz(f):
        return lambda a: cv2.resize(a, (round(a.shape[1] * f), round(a.shape[0] * f)), interpolation=cv2.INTER_AREA)

    return {"jpeg90": lambda a: of.jpeg(a, 90), "jpeg70": lambda a: of.jpeg(a, 70), "jpeg50": lambda a: of.jpeg(a, 50),
            "resize0.75": resz(.75), "crop3": crop3,
            "screenshot": lambda a: via_file(ea.screenshot_simulation, a)}


def denoisers():
    from src import modern_attacks as ma
    d = {"nlmeans": lambda a: cv2.fastNlMeansDenoisingColored(a, None, 3, 3, 5, 15),
         "vae": lambda a: via_file(ma.regenerate_vae, a)}
    for sgt in (0.1, 0.2, 0.3):
        d[f"img2img{sgt}"] = (lambda s: lambda a: via_file(ma.img2img, a, strength=s))(sgt)
    return d


_TM_VARIANTS: dict = {}


def tm_variant(t):
    """TrustMark model variant B / C / P as a System (weights are fetched on first use by the trustmark package)."""
    if t not in _TM_VARIANTS:
        from src.baselines import TrustMarkBaseline
        _TM_VARIANTS[t] = of.TrustMarkSystem(TrustMarkBaseline(model_type=t))
    return _TM_VARIANTS[t]


closed = []
if ATTACK == "a3":
    closed = [of.ClosedSystem("checkpoints/hidden_v2_best.pt", "closed_v2"),
              of.ClosedSystem("checkpoints/hidden_v4_final.pt", "closed_v4_otherkey", key_seed=99)]

t0 = time.time()
for n_done, idx in enumerate(sample, 1):
    meta = manifest[idx]
    O = rgb(f"data/owner_forensics/e2pos/{meta['file']}")
    case = of.Case(victim, idx, O, systems)
    rng = np.random.default_rng(31_000 + idx)

    if ATTACK == "a1":
        variants = attack_variants()
        for T in thieves:
            m_t = T.thief_message(idx)
            for lv in LEVELS:
                S0 = T.mark(case.copy, m_t, lv)
                for name, fn in variants.items():
                    run(case, idx, meta, T, name, lv, "x", fn(S0), m_t, robust=True)

    elif ATTACK == "a2":
        for T in thieves:                                       # unknown strength, the four systems
            m_t = T.thief_message(idx)
            lv = round(float(np.exp(rng.uniform(np.log(0.5), np.log(3.0)))), 2)
            S0 = T.mark(case.copy, m_t, lv)
            run(case, idx, meta, T, "random_strength", lv, "clean", S0, m_t, robust=True)
            run(case, idx, meta, T, "random_strength", lv, "jpeg70", of.jpeg(S0), m_t, robust=True)
        if "trustmark" in THIEVES:                              # unknown strength AND another model variant
            default_tm = of.load_system("trustmark")
            ext = systems + [tm_variant(v) for v in "BCP"]
            for v in "BCP":
                T = tm_variant(v)
                m_t = T.thief_message(idx)
                lv = round(float(np.exp(rng.uniform(np.log(0.5), np.log(3.0)))), 2)
                S0 = T.mark(case.copy, m_t, lv)
                for post, S1 in (("clean", S0), ("jpeg70", of.jpeg(S0))):
                    run(case, idx, meta, T, f"variant|default", lv, post, S1, m_t, claim_sys=default_tm,
                        defender="default", robust=True)
                    run(case, idx, meta, T, f"variant|sweep", lv, post, S1, m_t, claim_sys=T, defender="sweep",
                        case_systems=ext, robust=True)

    elif ATTACK == "a3":
        for T in closed:
            m_t = T.thief_message(idx)
            for lv in (1.0, 2.0):
                S0 = T.mark(case.copy, m_t, lv)
                run(case, idx, meta, T, "closed", lv, "clean", S0, m_t, claim=False)
                run(case, idx, meta, T, "closed", lv, "jpeg70", of.jpeg(S0), m_t, claim=False)

    elif ATTACK == "a4":
        for name, fn in denoisers().items():
            todo = [(T, lv, k) for T in thieves for lv in LEVELS for k in ("on_copy", "on_original")
                    if (idx, T.name, f"{name}|{k}", str(lv), "clean") not in done]
            if not todo:
                continue
            ests = {}
            for src_kind, src in (("on_copy", case.copy), ("on_original", O)):      # on_original = the null
                est = fn(src)
                if est.shape != O.shape:
                    est = cv2.resize(est, (O.shape[1], O.shape[0]), interpolation=cv2.INTER_AREA)
                ests[src_kind] = (est, of.delta_e(O, est))
            for T in thieves:
                m_t = T.thief_message(idx)
                for lv in LEVELS:
                    for src_kind, (est, att_dE) in ests.items():
                        key = (idx, T.name, f"{name}|{src_kind}", str(lv), "clean")
                        if key in done:
                            continue
                        tt = time.time()
                        S0 = T.mark(est, m_t, lv)
                        ev = case.evidence(S0, thief_claim=(T, m_t))
                        log(case, idx, meta, T, f"{name}|{src_kind}", lv, "clean", S0, ev, of.delta_e(O, S0), m_t, True,
                            time.time() - tt, extra=f"attack_dE={att_dE:.3f}")

    elif ATTACK == "a5":
        pool = of.SYSTEMS
        stacks = [(a, b) for a in pool for b in pool] + [tuple(pool[j] for j in rng.integers(0, 4, 3)) for _ in range(8)]
        for st in stacks:
            tt = time.time()
            key = (idx, "+".join(st), "stack", "1.5", "clean")
            if key in done:
                continue
            S = case.copy
            for layer, nme in enumerate(st):         # every layer carries its own message
                sysm = of.load_system(nme)
                m_last = sysm.thief_message(idx + 1000 * (layer + 1))
                S = sysm.mark(S, m_last, 1.5)
            last = of.load_system(st[-1])
            ev = case.evidence(S, thief_claim=(last, m_last))
            p3 = int(of.peel(S, victim, idx, systems, rounds=3)["ok"])
            class _T:                         # row() wants a thief-like object for the name
                name = "+".join(st)
                same = staticmethod(lambda a, b: False)
            log(case, idx, meta, _T, "stack", 1.5, "clean", S, ev, of.delta_e(O, S), None, True, time.time() - tt, p3=p3)

    elif ATTACK == "a6":
        for T in thieves:
            m_t = T.thief_message(idx)
            for kind, src in (("derived", case.copy), ("real_original", O)):     # real_original = thief really had O
                key = (idx, T.name, f"fake_original|{kind}", "2.0", "clean")
                if key in done:
                    continue
                tt = time.time()
                S0 = T.mark(src, m_t, 2.0)
                fake = of.to_u8(S0.astype(np.float32) / 255 - T.residual(S0, m_t))      # the thief's "original"
                z = case.informed_bank.test(fake)["z"]
                w.writerow({"attack": ATTACK, "victim": VICTIM, "image": idx, "source": meta["source"], "thief": T.name,
                            "variant": f"fake_original|{kind}", "level": 2.0, "post": "clean",
                            "dE_O": round(of.delta_e(O, S0), 3), "retrieved": 1,
                            "informed_z": round(z, 2), "informed": int(z >= of.Z_THRESHOLD),
                            "restored": int(z >= of.Z_THRESHOLD), "secs": round(time.time() - tt, 1)})
    fh.flush()
    if n_done % 5 == 0 or n_done == len(sample):
        print(f"{ATTACK} {VICTIM}: {n_done}/{len(sample)} photos ({time.time() - t0:.0f}s)", flush=True)
fh.close()

rows = list(csv.DictReader(open(ROWS)))
print(f"\n{ATTACK} {VICTIM}: {len(rows)} cases, {len({r['image'] for r in rows})} photos, {time.time() - t0:.0f}s this run")
print("per defence (count of cases where it fires):  thief | variant | defender   n  informed peel order(read) order(claim)"
      "  tolerant order(read/claim)  restored restored_tolerant  thief_read")
c = lambda sub, k: sum(int(r[k] or 0) for r in sub)
rz = lambda sub, k: sum(1 for r in sub if r[k] != "" and float(r[k]) >= of.ORDER_Z)
for key in sorted({(r["thief"], r["variant"], r["defender"]) for r in rows}):
    sub = [r for r in rows if (r["thief"], r["variant"], r["defender"]) == key]
    print(f"  {key[0]:<20}{key[1]:<22}{key[2]:<8}n={len(sub):<4} inf {c(sub,'informed'):<3} peel {c(sub,'peel_ok'):<3} "
          f"ord {c(sub,'order_read'):<3}/{c(sub,'order_claim'):<3} rob {rz(sub,'order_rob_read_z'):<3}/{rz(sub,'order_rob_claim_z'):<3} "
          f"restored {c(sub,'restored'):<3} rob {c(sub,'restored_rob'):<3} thief_read {c(sub,'thief_read')}")
print(ROWS)
