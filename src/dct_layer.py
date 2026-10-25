# src/dct_layer.py
# Second, secret-keyed watermark layer: DCT spread spectrum in LUMINANCE.
#
# Why: HiDDeN v2 carries its message entirely in colour (a luminance-only
# version of its residual reads 0/12). A thief who re-protects our published
# image with the same v2 model overwrites the colour channel for the blind
# decoder (tests/test_rewatermark.py). A second mark in an independent channel
# (brightness, frequency domain) with a carrier only our server can generate
# (HMAC of the copy ID with the server key) survives that and serves as
# ownership evidence.
#
# Method: Cox, Kilian, Leighton & Shamoon (1997), multiplicative spread
# spectrum on the n largest full-image DCT coefficients of luminance:
#     v'_i = v_i (1 + alpha x_i),   x ~ N(0, 1) seeded by HMAC(key, copy_id)
# Unlike src/dct_watermark.py (kept for reproducibility) the change is applied
# as a pure luminance delta added equally to R, G and B, so the colour channel
# (where v2 lives) is not touched, and the carrier is keyed, not a plain hash.
#
# Detection is informed (non-blind): it needs the reference image the layer was
# embedded onto (stored by the server) and a suspect in the same frame (align
# first, src/defence.align). Under "no watermark", the Cox similarity is ~N(0,1)
# (x is i.i.d. N(0,1)); calibrated empirically against never-issued carriers.

import hashlib
import hmac
import os

import numpy as np
from scipy.fft import dctn, idctn

_KEY = os.environ.get("WATERMARK_KEY", "thesis-demo-key").encode()
ALPHA = 0.02
N_COEFFS = 1000


def _luma(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float64)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def carrier(copy_id: int, n: int = N_COEFFS, key: bytes = _KEY) -> np.ndarray:
    seed = hmac.new(key, f"dct-layer:{copy_id}".encode(), hashlib.sha256).digest()
    return np.random.default_rng(int.from_bytes(seed[:8], "big")).standard_normal(n)


def _positions(y: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    coeffs = dctn(y, norm="ortho").ravel()
    mag = np.abs(coeffs)
    mag[0] = 0                                  # never the DC (mean brightness)
    idx = np.argsort(mag)[-n:]
    return idx, coeffs[idx]


def embed(reference_rgb: np.ndarray, copy_id: int, alpha: float = ALPHA,
          n: int = N_COEFFS) -> np.ndarray:
    """reference image (uint8 RGB) -> image with the keyed luminance layer."""
    y = _luma(reference_rgb)
    idx, values = _positions(y, n)
    delta = np.zeros(y.size)
    delta[idx] = values * alpha * carrier(copy_id, n)
    dy = idctn(delta.reshape(y.shape), norm="ortho")
    out = reference_rgb.astype(np.float64) + dy[..., None]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def similarity(suspect_rgb: np.ndarray, reference_rgb: np.ndarray, copy_ids,
               n: int = N_COEFFS) -> np.ndarray:
    """
    Cox similarity  sim = x . w* / |w*|,  w*_i = v*_i / (g v_i) - 1,  for one or
    many copy IDs at once (the extraction is shared). suspect and reference must
    be the same size (same frame).
    """
    if suspect_rgb.shape != reference_rgb.shape:
        raise ValueError("suspect must be aligned to the reference frame")
    idx, values = _positions(_luma(reference_rgb), n)
    v_star = dctn(_luma(suspect_rgb), norm="ortho").ravel()[idx]
    safe = np.where(np.abs(values) > 1e-8, values, 1e-8)
    # Global brightness / contrast scales every coefficient by about the same
    # gain g; estimate it robustly and divide it out, otherwise the constant
    # (g - 1) term swamps the correlation (brightness 1.2: sim 3.7 -> 5.6).
    gain = float(np.median(v_star / safe))
    gain = gain if 0.2 < gain < 5 else 1.0
    w = v_star / (gain * safe) - 1
    norm = np.linalg.norm(w)
    ids = np.atleast_1d(copy_ids)
    if norm < 1e-12:
        return np.zeros(len(ids))
    return np.array([carrier(int(c), n) @ w / norm for c in ids])


def detect(suspect_rgb: np.ndarray, reference_rgb: np.ndarray, copy_id: int,
           n_null: int = 500, threshold: float = 6.0, n: int = N_COEFFS) -> dict:
    """
    Similarity for copy_id plus an empirical null from n_null never-issued
    carriers (IDs above the 20-bit ID space, so they can never be real copies).
    Theoretical null ~N(0,1); threshold 6 is Cox et al.'s choice (p ~ 1e-9).
    """
    null_ids = range(1 << 21, (1 << 21) + n_null)
    sims = similarity(suspect_rgb, reference_rgb, [copy_id, *null_ids], n)
    s, null = float(sims[0]), sims[1:]
    z = (s - null.mean()) / (null.std() + 1e-12)
    return {"sim": round(s, 3), "z": round(float(z), 2),
            "p_empirical": (1 + int((null >= s).sum())) / (n_null + 1),
            "null_mean": round(float(null.mean()), 3), "null_std": round(float(null.std()), 3),
            "detected": s > threshold}
