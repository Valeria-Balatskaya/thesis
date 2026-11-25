# src/owner_forensics.py
# Owner-side forensics against re-watermarking for ANY watermarking system
# (docs/research_plan.md, section 4b; related work: docs/related_work_check.md).
#
# The owner publishes C = V(O, m_V) (V = the owner's watermarking system) and keeps the clean
# original O. A thief takes C, re-marks it with their own public embedder T and message m_T and
# the suspect S = T(C, m_T) circulates. Three tests, all computed from O, C, S and public models:
#
#   informed   Is the exact residual of C present in S? z = standardised correlation of clip(S - O)
#              with C - O against D never-issued copies of V (Cox et al. 1997 informed detection).
#   peel-off   Read the thief's message with the public decoders, regenerate that residual on S,
#              subtract it at several scales and read V again, blind.
#   order      Was T applied to C (owner first) or to O (thief on a leaked clean original)?
#              t = <S - B, A - B> / |A - B|^2 with A = T(C, m_T), B = T(O, m_T) and T's strength
#              fitted; z against D never-issued copies of V (t ~ 1: C first, t ~ 0: O).
#
# Generalises src/defence.py (ours only: InformedChecker, DefencePipeline._peel, order_proof),
# which stays as it was so earlier results are reproducible. Everything here works at the image's
# native resolution (<= 1024 px in the tests) because the baselines' residuals are high-frequency.

import io
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

Z_THRESHOLD = 4.0
N_DECOYS = 50
PEEL_SCALES = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)      # same as src.defence.PEEL_SCALES
SYSTEMS = ("ours_v3", "trustmark", "wam", "videoseal")


def to_u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0, 1) * 255).round().astype(np.uint8)


def jpeg(rgb: np.ndarray, quality: int = 70) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def delta_e(a: np.ndarray, b: np.ndarray) -> float:
    """Mean CIEDE2000 between two RGB images (same as src.metrics.visibility()['dE_mean'])."""
    from skimage.color import rgb2lab, deltaE_ciede2000
    return float(deltaE_ciede2000(rgb2lab(a), rgb2lab(b)).mean())


# ─── a uniform view of the four systems ───────────────────────────

class System:
    """
    One watermarking system behind the interface the tests need. level = multiple of the system's
    default strength. Messages: owner m_V(i), thief m_T(i) and never-issued decoys are disjoint.
    """
    name = "?"

    def message(self, i): raise NotImplementedError
    def thief_message(self, i): return self.message(500_000 + i)
    def decoy_message(self, k): return self.message(10_000_000 + k)
    def residual(self, rgb, msg, level=1.0): raise NotImplementedError   # float32 [0,1] units
    def foreign_read(self, rgb): raise NotImplementedError               # message or None
    def read_ok(self, rgb, i) -> bool: raise NotImplementedError         # owner's native blind decision
    def same(self, a, b) -> bool: raise NotImplementedError
    def key(self, msg) -> str: raise NotImplementedError

    def mark(self, rgb: np.ndarray, msg, level: float = 1.0) -> np.ndarray:
        return to_u8(rgb.astype(np.float32) / 255 + self.residual(rgb, msg, level))


class OursSystem(System):
    def __init__(self, checkpoint="checkpoints/hidden_v3_best.pt", label="ours_v3"):
        from src.baselines import Ours
        from src.defence import OursPeeler
        self.base = Ours(checkpoint, label=label)
        self.peeler = OursPeeler(self.base.model)
        self.name = label

    def message(self, i):
        return self.base.copy_id(i) if i < 500_000 else i       # copy ID (1000 + i), thief ID = 500000 + i
    def thief_message(self, i): return 500_000 + i
    def decoy_message(self, k): return (1 << 21) + 1_000_003 * k + 17      # outside the 20-bit ID space
    def residual(self, rgb, msg, level=1.0):
        return self.peeler.residual(rgb, msg, None if level == 1.0 else level)
    def foreign_read(self, rgb): return self.peeler.foreign_read(rgb)
    def read_ok(self, rgb, i): return bool(self.base.read_rgb(rgb, i)["ok"])
    def same(self, a, b): return int(a) == int(b)
    def key(self, msg): return str(int(msg))


class ClosedSystem(System):
    """
    A thief's PRIVATE watermarking model (A3): any checkpoint of ours, optionally with another key.
    The defender has no decoder and no embedder for it, so it is never in `systems`.
    """

    def __init__(self, checkpoint: str, label: str, key_seed: int | None = None):
        from src.defence import OursPeeler
        from src.hidden_v2_inference import HiddenV2Model
        self.model = HiddenV2Model(checkpoint)
        key = self.model.other_key(key_seed) if (key_seed is not None and self.model.key_dim) else None
        self.peeler = OursPeeler(self.model, key=key)
        self.name = label

    def message(self, i): return i
    def thief_message(self, i): return 700_000 + i
    def residual(self, rgb, msg, level=1.0):
        return self.peeler.residual(rgb, msg, None if level == 1.0 else level)
    def foreign_read(self, rgb): return None
    def read_ok(self, rgb, i): return False
    def same(self, a, b): return a == b
    def key(self, msg): return str(msg)


class _BitSystem(System):
    """TrustMark / WAM / Video Seal: messages come from the baseline, decisions are its native ones."""
    def __init__(self, base):
        self.base, self.name = base, base.name
    def message(self, i): return self.base.message(i)
    def residual(self, rgb, msg, level=1.0):
        return self.base.residual(rgb, msg, strength=self.base.default_strength * level)
    def foreign_read(self, rgb): return self.base.foreign_read(rgb)
    def read_ok(self, rgb, i): return bool(self.base.read_rgb(rgb, i)["ok"])


class TrustMarkSystem(_BitSystem):
    def same(self, a, b): return str(a) == str(b)
    def key(self, msg): return str(msg)


class BitsSystem(_BitSystem):
    """WAM (32 bits) and Video Seal (256 bits) have no ECC: 'same' tolerates a few bit errors."""
    def same(self, a, b): return float(np.mean(np.asarray(a) == np.asarray(b))) >= 0.9
    def key(self, msg): return "".join(map(str, np.asarray(msg).astype(int)))


_CACHE: dict = {}


def load_system(name: str) -> System:
    if name not in _CACHE:
        if name == "ours_v3":
            _CACHE[name] = OursSystem()
        elif name == "trustmark":
            from src.baselines import TrustMarkBaseline
            _CACHE[name] = TrustMarkSystem(TrustMarkBaseline())
        elif name == "wam":
            from src.baselines import WAMBaseline
            _CACHE[name] = BitsSystem(WAMBaseline())
        elif name == "videoseal":
            from src.baselines import VideoSealBaseline
            _CACHE[name] = BitsSystem(VideoSealBaseline())
        else:
            raise ValueError(name)
    return _CACHE[name]


# ─── 1. informed check ────────────────────────────────────────────

class InformedBank:
    """
    Built once per (original, victim): the true copy and D never-issued copies of the victim
    system on the same original. test(S) -> z of the true copy's residual in S.
    """

    def __init__(self, original: np.ndarray, copies: list):
        o = original.astype(np.float32)
        self.o = o
        refs = np.stack([(c.astype(np.float32) - o).reshape(-1) for c in copies])
        self.clip = 4 * float(refs[0].std()) + 1e-6
        refs -= refs.mean(axis=1, keepdims=True)
        refs /= np.linalg.norm(refs, axis=1, keepdims=True) + 1e-12
        self.refs = refs

    def scores(self, suspect: np.ndarray) -> np.ndarray:
        """Correlation of the suspect's change with every copy in the bank."""
        d = np.clip(suspect.astype(np.float32) - self.o, -self.clip, self.clip).reshape(-1)
        d -= d.mean()
        nd = float(np.linalg.norm(d))
        return np.zeros(len(self.refs), dtype=np.float32) if nd < 1e-9 else self.refs @ d / nd

    def test(self, suspect: np.ndarray) -> dict:
        sc = self.scores(suspect)
        if not sc.any():
            return {"z": 0.0, "score": 0.0}
        null = sc[1:]
        return {"z": float((sc[0] - null.mean()) / (null.std(ddof=1) + 1e-12)), "score": float(sc[0])}


# ─── 2. peel-off ──────────────────────────────────────────────────

def find_marks(suspect: np.ndarray, systems: list, victim: System, i: int, skip: set | None = None) -> list:
    """Messages the public decoders read from the suspect, except the victim's own registered one."""
    marks = []
    for s in systems:
        if skip and s.name in skip:
            continue
        msg = s.foreign_read(suspect)
        if msg is None:
            continue
        if s.name == victim.name and victim.same(msg, victim.message(i)):
            continue
        marks.append((s, msg))
    return marks


def peel(suspect: np.ndarray, victim: System, i: int, systems: list, marks: list | None = None,
         rounds: int = 2) -> dict:
    """
    Peel-off. marks = find_marks(...) of the unpeeled suspect (computed if None). Subtract the sum
    of the regenerated residuals at each PEEL_SCALES, read the victim blind; if nothing reads, peel
    once at scale 1 and look for marks that were hidden underneath. peel_ok also counts a blind
    read of the suspect itself (zero marks to remove).
    """
    if victim.read_ok(suspect, i):
        return {"ok": True, "blind": True, "scale": None, "round": 0, "found": []}
    base, tried, found = suspect, set(), []
    for rnd in range(rounds):
        cur = marks if (rnd == 0 and marks is not None) else find_marks(base, systems, victim, i, skip=tried)
        if not cur:
            break
        res = []
        for s, msg in cur:
            tried.add(s.name)
            found.append((s.name, s.key(msg)))
            res.append(s.residual(base, msg))
        r = sum(res)
        x = base.astype(np.float32) / 255
        for scale in PEEL_SCALES:
            if victim.read_ok(to_u8(x - scale * r), i):
                return {"ok": True, "blind": False, "scale": scale, "round": rnd + 1, "found": found}
        base = to_u8(x - r)
    return {"ok": False, "blind": False, "scale": None, "round": 0, "found": found}


# ─── 3. order proof ───────────────────────────────────────────────

ORDER_Z = 4.0
PRESENCE_K = 24       # decoy thief messages for the presence test (deviation 5)


class ThiefBank:
    """T's residual on the original and on each of the victim copies for one thief message
    (residuals kept as float16: they are ~1e-2, the relative error is below 1e-3)."""

    def __init__(self, thief: System, original: np.ndarray, copies: list, msg):
        self.system, self.msg = thief, msg
        self.o = original.astype(np.float32) / 255
        self.r_o = thief.residual(original, msg)
        self.copies = copies
        self.res = [thief.residual(c, msg).astype(np.float16) for c in copies]

    def ts(self, suspect: np.ndarray) -> tuple:
        """t for every copy in the bank (index 0 first) and the fitted thief strength."""
        s = suspect.astype(np.float32) / 255
        denom = float((self.r_o * self.r_o).sum()) + 1e-12
        a = float(np.clip(((s - self.o) * self.r_o).sum() / denom, 0.25, 8.0))   # thief strength (fitted)
        B = np.clip(self.o + a * self.r_o, 0, 1)
        sb = s - B
        ts = []
        for c, rc in zip(self.copies, self.res):
            d = np.clip(c.astype(np.float32) / 255 + a * rc.astype(np.float32), 0, 1) - B
            ts.append(float((sb * d).sum() / (float((d * d).sum()) + 1e-12)))
        return ts, a

    def test(self, suspect: np.ndarray) -> dict:
        ts, a = self.ts(suspect)
        t, null = ts[0], np.array(ts[1:])
        return {"t": t, "z": float((t - null.mean()) / (null.std(ddof=1) + 1e-12)), "scale": a}


ROBUST_POSTS = ("id", "jpeg90", "jpeg70", "jpeg50", "blur0.6", "blur1.0")


def _post(x: np.ndarray, name: str) -> np.ndarray:
    """x float32 [0,1] -> the same after a candidate post-processing (the thief's, or the platform's)."""
    if name == "id":
        return x
    if name.startswith("jpeg"):
        return jpeg(to_u8(x), int(name[4:])).astype(np.float32) / 255
    import cv2
    return cv2.GaussianBlur(x, (0, 0), float(name[4:]))


def test_robust(bank: "ThiefBank", suspect: np.ndarray, posts: tuple = ROBUST_POSTS, a_grid=(1.0, 1.5, 2.0, 3.0)) -> dict:
    """
    Post-processing tolerant order proof (not pre-registered; reported next to the original, A1/A2).
    For every candidate post-processing P the regenerated candidates are processed the same way before
    comparing: t_P = <s - P(B), P(A_k) - P(B)> / |P(A_k) - P(B)|^2, with the thief strength re-fitted per P
    (grid around the least-squares value, minimum |s - P(B)|). z_P is taken against the decoys under the
    same P; the proof is max_P z_P. The decoys see the same max-over-P procedure only through the null
    runs (empirical false-claim rate), not analytically.
    """
    s = suspect.astype(np.float32) / 255
    a_ls = float(np.clip(((s - bank.o) * bank.r_o).sum() / (float((bank.r_o * bank.r_o).sum()) + 1e-12), 0.25, 8.0))
    best = {"z": -1e9}
    for p in posts:
        fits = []
        for g in ((1.0,) if p == "id" else a_grid):      # "id" is exactly the plain proof, so the max is never below it
            a = min(a_ls * g, 8.0)
            B = _post(np.clip(bank.o + a * bank.r_o, 0, 1), p)
            fits.append((float(((s - B) ** 2).sum()), a, B))
        _, a, B = min(fits, key=lambda f: f[0])
        sb = s - B
        ts = []
        for c, rc in zip(bank.copies, bank.res):
            d = _post(np.clip(c.astype(np.float32) / 255 + a * rc.astype(np.float32), 0, 1), p) - B
            ts.append(float((sb * d).sum() / (float((d * d).sum()) + 1e-12)))
        null = np.array(ts[1:])
        z = float((ts[0] - null.mean()) / (null.std(ddof=1) + 1e-12))
        if z > best["z"]:
            best = {"z": z, "t": ts[0], "scale": a, "post": p}
    return best


@dataclass
class Evidence:
    informed_z: float
    informed: bool
    peel_ok: bool
    peel_blind: bool
    peel_scale: float | None
    marks: list
    raw_marks: list
    order_read_z: float | None
    order_read_t: float | None
    order_claim_z: float | None
    order_claim_t: float | None
    order_scale: float | None
    order_read_raw_z: float | None = None     # order proof alone (before the presence gate)
    order_claim_raw_z: float | None = None
    presence_read_z: float | None = None      # thief-presence / message-specificity z of the best mark
    presence_claim_z: float | None = None

    @property
    def order_read(self) -> bool:
        return self.order_read_z is not None and self.order_read_z >= ORDER_Z

    @property
    def order_claim(self) -> bool:
        return self.order_claim_z is not None and self.order_claim_z >= ORDER_Z

    @property
    def restored(self) -> bool:                      # primary (pre-registered): informed or peel-off or order (decoder-read)
        return self.informed or self.peel_ok or self.order_read

    @property
    def restored_claim(self) -> bool:                # with the thief's disclosed message
        return self.informed or self.peel_ok or self.order_claim


class Case:
    """Everything the owner holds for one photo: O, C and the decoy copies of the victim system."""

    def __init__(self, victim: System, i: int, original: np.ndarray, systems: list, n_decoys: int = N_DECOYS):
        self.victim, self.i, self.systems, self.o = victim, i, systems, original
        self.copy = victim.mark(original, victim.message(i))
        self.copies = [self.copy] + [victim.mark(original, victim.decoy_message(k)) for k in range(n_decoys)]
        self.informed_bank = InformedBank(original, self.copies)
        self._banks: dict = {}
        self._pres_key, self._pres = None, None

    def bank(self, thief: System, msg) -> ThiefBank:
        k = (thief.name, thief.key(msg))
        if k not in self._banks:
            if len(self._banks) >= 2:                 # memory: keep the most recent banks only
                self._banks.pop(next(iter(self._banks)))
            self._banks[k] = ThiefBank(thief, self.o, self.copies, msg)
        return self._banks[k]

    def presence(self, thief: System, msg, suspect: np.ndarray) -> float:
        """
        Thief presence and message specificity (deviation 5). a(m) = <S - C, w_C(m)> / |w_C(m)|^2 with
        w_C(m) = T(C, m) - C: how strongly the claimed thief residual explains what changed from the owner's
        published copy C. z = (a(claimed) - mean a(decoy)) / sd a(decoy) over PRESENCE_K never-used messages of
        the same system. ~0 when nothing was added to C (S = C) or the message is wrong; large when T(., m)
        was applied. Needs only C, S and T (the owner holds C).
        """
        if self._pres_key != thief.name:
            c = self.copy.astype(np.float32) / 255
            self._pres = (c, [thief.residual(self.copy, thief.decoy_message(k)).astype(np.float16) for k in range(PRESENCE_K)])
            self._pres_key = thief.name
        c, dec = self._pres
        d = suspect.astype(np.float32) / 255 - c

        def a(r):
            r = r.astype(np.float32)
            return float((d * r).sum() / (float((r * r).sum()) + 1e-12))

        null = np.array([a(r) for r in dec])
        return float((a(thief.residual(self.copy, msg)) - null.mean()) / (null.std(ddof=1) + 1e-9))

    def robust_order(self, suspect: np.ndarray, ev: "Evidence", thief_claim: tuple | None = None) -> tuple:
        """(read_z, claim_z) of the post-processing tolerant order proof for the marks `ev` found / the thief's claim;
        min(order z, presence z) as in `evidence`."""
        cache, rz, cz = {}, None, None

        def one(sy, msg):
            k = (sy.name, sy.key(msg))
            if k not in cache:
                cache[k] = min(test_robust(self.bank(sy, msg), suspect)["z"], self.presence(sy, msg, suspect))
            return cache[k]

        for sy, msg in ev.raw_marks:
            z = one(sy, msg)
            rz = z if rz is None or z > rz else rz
        if thief_claim is not None:
            cz = one(*thief_claim)
        return rz, cz

    def evidence(self, suspect: np.ndarray, thief_claim: tuple | None = None, run_order: bool = True,
                 peel_rounds: int = 2, presence_gate: bool = True) -> Evidence:
        """
        thief_claim = (System, message) the thief discloses (used for the 'claim' variant).
        presence_gate: the order proof fires only if the thief's residual is also present and message-specific
        (min of the two z values; deviation 5). False = the pre-registered proof alone (E1 pilot).
        """
        inf = self.informed_bank.test(suspect)
        marks = find_marks(suspect, self.systems, self.victim, self.i)
        pl = peel(suspect, self.victim, self.i, self.systems, marks=marks, rounds=peel_rounds)
        rz = rt = rraw = rpres = None
        scale = None
        if run_order:
            for s, msg in marks:
                r = self.bank(s, msg).test(suspect)
                pz = self.presence(s, msg, suspect)
                zc = min(r["z"], pz) if presence_gate else r["z"]
                if rz is None or zc > rz:
                    rz, rt, scale, rraw, rpres = zc, r["t"], r["scale"], r["z"], pz
        cz = ct = craw = cpres = None
        if run_order and thief_claim is not None:
            r = self.bank(*thief_claim).test(suspect)
            cpres = self.presence(thief_claim[0], thief_claim[1], suspect)
            cz, ct, craw = (min(r["z"], cpres) if presence_gate else r["z"]), r["t"], r["z"]
            scale = scale if scale is not None else r["scale"]
        return Evidence(informed_z=inf["z"], informed=inf["z"] >= Z_THRESHOLD, peel_ok=pl["ok"],
                        peel_blind=pl["blind"], peel_scale=pl["scale"],
                        marks=[(s.name, s.key(m)) for s, m in marks], raw_marks=marks,
                        order_read_z=rz, order_read_t=rt, order_claim_z=cz, order_claim_t=ct,
                        order_scale=scale, order_read_raw_z=rraw, order_claim_raw_z=craw,
                        presence_read_z=rpres, presence_claim_z=cpres)
