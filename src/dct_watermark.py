# src/dct_watermark.py
# DCT spread-spectrum watermarking — Cox et al. 1997 + Barni et al. 1998
#
# References:
#   Cox, I.J. et al. (1997). Secure spread spectrum watermarking for multimedia.
#   IEEE Trans. Image Processing, 6(12), 1673-1687. doi:10.1109/83.650120
#
#   Barni, M. et al. (1998). A DCT-domain system for robust image watermarking.
#   Signal Processing, 66(3), 357-372. doi:10.1016/S0165-1684(98)00015-2

import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn
import hashlib


def _make_carrier(seed: str, n_coeffs: int) -> np.ndarray:
    """
    Generate i.i.d. Gaussian carrier vector seeded by seller ID.
    Cox et al. 1997, Section IV: watermark xi ~ N(0,1).
    Using a deterministic seed allows blind re-generation at detection time.
    """
    seed_int = int(hashlib.sha256(seed.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed_int)
    return rng.standard_normal(n_coeffs)


def _get_luminance(img_array: np.ndarray) -> np.ndarray:
    """Convert RGB to YCbCr and return Y (luminance) channel."""
    pil = Image.fromarray(img_array)
    ycbcr = pil.convert("YCbCr")
    return np.array(ycbcr, dtype=np.float64)


def embed(image_path: str, seller_id: str, output_path: str,
          alpha: float = 0.015, n_coeffs: int = 300) -> dict:
    """
    Embed seller fingerprint using DCT spread-spectrum watermarking.

    Algorithm (Cox et al. 1997, Section III-A, Equation 2):
      1. Compute full-image DCT of luminance channel
      2. Select top-n coefficients by magnitude (excluding DC)
      3. Generate Gaussian carrier X seeded by seller_id
      4. Embed: v'_i = v_i * (1 + alpha * x_i)     [Cox Eq. 2]
      5. Inverse DCT and reconstruct RGB image

    Args:
        image_path  : path to original PNG cover image
        seller_id   : unique seller identifier (the fingerprint)
        output_path : path to save watermarked stego image
        alpha       : embedding strength (default 0.1 — Cox et al. recommendation)
        n_coeffs    : number of DCT coefficients to modify (default 1000)

    Returns:
        dict with embedding metadata for detection
    """
    img_array = np.array(Image.open(image_path).convert("RGB"))
    ycbcr = _get_luminance(img_array)
    Y = ycbcr[:, :, 0].copy()

    # Step 1: Full-image 2D DCT of luminance channel
    dct_coeffs = dctn(Y, norm="ortho")

    # Step 2: Select top-n coefficients by magnitude, exclude DC [0,0]
    flat = dct_coeffs.flatten()
    flat_abs = np.abs(flat)
    flat_abs[0] = 0  # exclude DC component (Cox et al. footnote 1)
    top_indices = np.argsort(flat_abs)[-n_coeffs:]

    # Step 3: Generate Gaussian carrier seeded by seller ID
    carrier = _make_carrier(seller_id, n_coeffs)

    # Step 4: Embed using multiplicative rule — Cox et al. 1997 Equation (2)
    # v'_i = v_i * (1 + alpha * x_i)
    original_values = flat[top_indices].copy()
    flat[top_indices] = original_values * (1.0 + alpha * carrier)

    # Step 5: Inverse DCT → reconstruct luminance
    watermarked_dct = flat.reshape(dct_coeffs.shape)
    Y_watermarked = idctn(watermarked_dct, norm="ortho")
    Y_watermarked = np.clip(Y_watermarked, 0, 255)

    # Reconstruct RGB image
    ycbcr_w = ycbcr.copy()
    ycbcr_w[:, :, 0] = Y_watermarked
    watermarked_rgb = np.array(
        Image.fromarray(ycbcr_w.astype(np.uint8), "YCbCr").convert("RGB")
    )
    Image.fromarray(watermarked_rgb).save(output_path, format="PNG")

    return {
        "seller_id": seller_id,
        "alpha": alpha,
        "n_coeffs": n_coeffs,
        "top_indices": top_indices,
        "original_values": original_values,
    }


def detect(suspect_path: str, original_path: str,
           seller_id: str, embed_meta: dict,
           threshold: float = 6.0) -> dict:
    """
    Detect watermark using normalised correlation (Cox et al. 1997, Eq. 4).

    sim(X, W*) = X · W* / |W*|

    If sim > threshold, watermark is present. Cox et al. show that
    threshold=6 gives extremely low false-positive probability.

    Args:
        suspect_path  : path to image being tested
        original_path : path to original unwatermarked image
        seller_id     : seller ID to test against
        embed_meta    : metadata returned from embed()
        threshold     : detection threshold (default 6.0, Cox et al. 1997)

    Returns:
        dict with similarity score, detection result, and confidence
    """
    # Load both images as luminance
    orig  = np.array(Image.open(original_path).convert("RGB"))
    susp  = np.array(Image.open(suspect_path).convert("RGB"))

    orig_Y = _get_luminance(orig)[:, :, 0].astype(np.float64)
    susp_Y = _get_luminance(susp)[:, :, 0].astype(np.float64)

    # DCT of both
    orig_dct = dctn(orig_Y, norm="ortho").flatten()
    susp_dct = dctn(susp_Y, norm="ortho").flatten()

    # Extract watermark estimate at known coefficient positions
    top_indices     = embed_meta["top_indices"]
    original_values = embed_meta["original_values"]

    # Recover approximate watermark signal: W* = (v* - v) / v
    v_star = susp_dct[top_indices]
    v      = original_values
    W_star = np.where(np.abs(v) > 1e-8, (v_star - v) / v, 0.0)

    # Regenerate carrier
    carrier = _make_carrier(seller_id, len(top_indices))

    # Normalised correlation — Cox et al. 1997 Equation (4)
    norm_W = np.linalg.norm(W_star)
    if norm_W < 1e-10:
        sim = 0.0
    else:
        sim = float(np.dot(carrier, W_star) / norm_W)

    detected = sim > threshold

    return {
        "seller_id"  : seller_id,
        "similarity" : round(sim, 4),
        "threshold"  : threshold,
        "detected"   : detected,
        "confidence" : "HIGH" if sim > threshold * 2 else ("LOW" if sim > threshold else "NONE")
    }

