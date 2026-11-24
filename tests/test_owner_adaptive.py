# tests/test_owner_adaptive.py
# Adaptive attacks A1-A6 on the owner-side forensics (docs/research_plan.md sections 4 and 4b).
# One victim per run (VICTIM=...), photos = a seeded sample of the E2 positive set (not the pilot).
#
#   A1  post-processing after re-marking: JPEG 50 / 90, resize 0.5 / 0.7 / 0.9, 5% crop on every
#       side, screenshot simulations; geometry is undone by retrieve-align (SIFT homography)
#   A2  unknown thief strength, log-uniform 0.5-4x (clean and JPEG q70): accuracy vs thief dE
#   A3  closed watermark: the thief uses a model the defender does not have (our v2 checkpoint,
#       our v4 checkpoint with another key); only the informed check can work
#   A4  estimated-original attack: the thief denoises / regenerates C (median, non-local means,
#       SD VAE, SD img2img 0.1 / 0.2 / 0.3) and re-marks that. Same attack on the clean original
#       gives the false-claim rate. Win for the thief = owner claim fails while their message reads
#   A5  stacked re-marking (2 and 3 layers, same or mixed systems), peel-off with 2 and 3 rounds
#   A6  fake original (Craver): the thief subtracts their own residual from the suspect and calls
#       the result "the original"; does it carry OUR residual? (positives) vs a thief who really
#       had the original (null)
# A7 (thief marks a leaked clean original) is the null of tests/test_owner_forensics.py.
#
# Run: ATTACK=a1 VICTIM=ours_v3 python tests/test_owner_adaptive.py [n_photos]   (default 40)
#      THIEVES=... restricts the thief systems; rows are resumable in output/owner_forensics/adaptive/

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
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
THIEVES = os.environ.get("THIEVES", ",".join(of.SYSTEMS)).split(",")
OUT = os.environ.get("OUT_DIR", "output/owner_forensics/adaptive")
os.makedirs(OUT, exist_ok=True)
ROWS = f"{OUT}/{ATTACK}_{VICTIM}.csv"
TMP = tempfile.mkdtemp(prefix="of_adaptive_")

manifest = list(csv.DictReader(open("data_manifests/owner_forensics_e2pos.csv")))
sample = sorted(np.random.default_rng(4242).choice(len(manifest), size=min(N, len(manifest)), replace=False))
systems = [of.load_system(n) for n in of.SYSTEMS]
victim = of.load_system(VICTIM)
thieves = [of.load_system(n) for n in THIEVES]

FIELDS = ["attack", "victim", "image", "source", "thief", "variant", "level", "post", "dE_O", "retrieved", "blind",
          "informed_z", "informed", "peel_ok", "peel_rounds3", "marks", "thief_read", "order_read_z", "order_read",
          "order_claim_z", "order_claim", "order_scale", "restored", "extra", "secs"]
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


def log(case, idx, meta, T, variant, level, post, S, ev, dE, m_t, retrieved, secs, p3="", extra=""):
    tr = ""
    if T is not None and m_t is not None and ev is not None:
        tr = int(any(s.name == T.name and T.same(m, m_t) for s, m in ev.raw_marks))
    f = lambda v, d=2: "" if v is None else round(v, d)
    w.writerow({"attack": ATTACK, "victim": VICTIM, "image": idx, "source": meta["source"],
                "thief": T.name if T else "none", "variant": variant, "level": level, "post": post,
                "dE_O": f(dE, 3), "retrieved": int(retrieved), "blind": "" if ev is None else int(victim.read_ok(S, idx)),
                "informed_z": "" if ev is None else f(ev.informed_z), "informed": "" if ev is None else int(ev.informed),
                "peel_ok": "" if ev is None else int(ev.peel_ok), "peel_rounds3": p3,
                "marks": "" if ev is None else ";".join(f"{a}:{b[:10]}" for a, b in ev.marks), "thief_read": tr,
                "order_read_z": "" if ev is None else f(ev.order_read_z), "order_read": "" if ev is None else int(ev.order_read),
                "order_claim_z": "" if ev is None else f(ev.order_claim_z), "order_claim": "" if ev is None else int(ev.order_claim),
                "order_scale": "" if ev is None else f(ev.order_scale),
                "restored": 0 if ev is None else int(ev.restored), "extra": extra, "secs": round(secs, 1)})


def run(case, idx, meta, T, variant, level, post, S0, m_t, claim=True, run_order=True):
    """S0 = what the thief publishes (before any post-processing). Returns after logging one row."""
    key = (idx, T.name if T else "none", variant, str(level), post)
    if key in done:
        return
    t = time.time()
    dE = of.delta_e(case.o, S0) if S0.shape == case.o.shape else None
    S = S0
    S, ok = to_frame(S, case.o)
    if not ok:
        log(case, idx, meta, T, variant, level, post, S0, None, dE, m_t, False, time.time() - t)
        return
    ev = case.evidence(S, thief_claim=(T, m_t) if (claim and T is not None and T.name in {s.name for s in systems}) else None,
                       run_order=run_order)
    log(case, idx, meta, T, variant, level, post, S, ev, dE, m_t, True, time.time() - t)


def attack_variants(A):
    """A1 transformations of the thief's output S0 (name -> function)."""
    from src import attacks, ecommerce_attacks as ea
    def crop5(a):
        h, wd = a.shape[:2]
        return a[round(.05 * h):h - round(.05 * h), round(.05 * wd):wd - round(.05 * wd)]
    def resz(f):
        return lambda a: cv2.resize(a, (round(a.shape[1] * f), round(a.shape[0] * f)), interpolation=cv2.INTER_AREA)
    return {"jpeg50": lambda a: of.jpeg(a, 50), "jpeg90": lambda a: of.jpeg(a, 90),
            "resize0.5": resz(.5), "resize0.7": resz(.7), "resize0.9": resz(.9), "crop5": crop5,
            "screenshot": lambda a: via_file(ea.screenshot_simulation, a),
            "native_screenshot": lambda a: via_file(ea.native_screenshot_simulation, a)}


def denoisers():
    from src import modern_attacks as ma
    d = {"median3": lambda a: cv2.medianBlur(a, 3),
         "nlmeans": lambda a: cv2.fastNlMeansDenoisingColored(a, None, 3, 3, 5, 15),
         "gauss1": lambda a: cv2.GaussianBlur(a, (0, 0), 1.0),
         "vae": lambda a: via_file(ma.regenerate_vae, a)}
    for sgt in (0.1, 0.2, 0.3):
        d[f"img2img{sgt}"] = (lambda s: lambda a: via_file(ma.img2img, a, strength=s))(sgt)
    return d


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
        variants = attack_variants(None)
        for T in thieves:
            m_t = T.thief_message(idx)
            S0 = T.mark(case.copy, m_t, 2.0)
            for name, fn in variants.items():
                run(case, idx, meta, T, name, 2.0, "x", fn(S0), m_t)

    elif ATTACK == "a2":
        for T in thieves:
            m_t = T.thief_message(idx)
            lv = round(float(np.exp(rng.uniform(np.log(0.5), np.log(4.0)))), 2)
            S0 = T.mark(case.copy, m_t, lv)
            run(case, idx, meta, T, "random_strength", lv, "clean", S0, m_t)
            run(case, idx, meta, T, "random_strength", lv, "jpeg70", of.jpeg(S0), m_t)

    elif ATTACK == "a3":
        for T in closed:
            m_t = T.thief_message(idx)
            for lv in (1.0, 2.0):
                S0 = T.mark(case.copy, m_t, lv)
                run(case, idx, meta, T, "closed", lv, "clean", S0, m_t, claim=False)
                run(case, idx, meta, T, "closed", lv, "jpeg70", of.jpeg(S0), m_t, claim=False)

    elif ATTACK == "a4":
        den = denoisers()
        for T in thieves:
            m_t = T.thief_message(idx)
            for name, fn in den.items():
                for src_kind, src in (("on_copy", case.copy), ("on_original", O)):    # on_original = the null
                    est = fn(src)
                    if est.shape != O.shape:
                        est = cv2.resize(est, (O.shape[1], O.shape[0]), interpolation=cv2.INTER_AREA)
                    S0 = T.mark(est, m_t, 1.5)
                    key = (idx, T.name, f"{name}|{src_kind}", "1.5", "clean")
                    if key in done:
                        continue
                    tt = time.time()
                    ev = case.evidence(S0, thief_claim=(T, m_t))
                    log(case, idx, meta, T, f"{name}|{src_kind}", 1.5, "clean", S0, ev, of.delta_e(O, S0), m_t, True,
                        time.time() - tt, extra=f"attack_dE={of.delta_e(O, est):.3f}")

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
for key in sorted({(r["thief"], r["variant"]) for r in rows}):
    sub = [r for r in rows if (r["thief"], r["variant"]) == key]
    k = sum(int(r["restored"]) for r in sub)
    print(f"  {key[0]:<22}{key[1]:<28} restored {frac_ci(k, len(sub))}")
print(ROWS)
