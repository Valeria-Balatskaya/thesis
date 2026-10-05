from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
from PIL import Image


def compute(original_path: str, stego_path: str) -> dict:
    """
    Compute PSNR and SSIM between original and stego image.
    Target values from thesis plan: PSNR > 38 dB, SSIM > 0.98.
    Chan & Cheng 2004 Table 1: k=1 gives PSNR ~51 dB on standard images.
    """
    orig = np.array(Image.open(original_path).convert("RGB"))
    stego = np.array(Image.open(stego_path).convert("RGB"))

    psnr_val = psnr(orig, stego, data_range=255)
    ssim_val = ssim(orig, stego, channel_axis=2, data_range=255)

    return {"PSNR_dB": round(psnr_val, 4), "SSIM": round(ssim_val, 6)}

def visibility(original_path: str, stego_path: str) -> dict:
    """
    Perceptual colour difference (CIEDE2000) between original and watermarked
    image. PSNR treats all changes alike; the eye does not: a colour tint on a
    flat white background is far more visible than the same energy hidden in
    texture. Rule of thumb: dE < 1 not noticeable, 1-2 noticeable on close
    inspection, > 2 noticeable at a glance.
      dE_mean  average over the image
      dE_p99   99th percentile (the worst spots)
      dE_flat  average over flat, bright areas (backgrounds), or None if there are none
    Reference: Sharma, Wu & Dalal (2005), The CIEDE2000 color-difference formula.
    """
    from scipy.ndimage import gaussian_filter
    from skimage.color import rgb2lab, deltaE_ciede2000

    orig = np.array(Image.open(original_path).convert("RGB"))
    stego = np.array(Image.open(stego_path).convert("RGB"))
    de = deltaE_ciede2000(rgb2lab(orig), rgb2lab(stego))

    gray = orig.astype(np.float32).mean(axis=2) / 255
    mean = gaussian_filter(gray, 4)
    std = np.sqrt(np.maximum(gaussian_filter(gray * gray, 4) - mean * mean, 0))
    flat = (std < 0.01) & (gray > 0.6)
    return {"dE_mean": round(float(de.mean()), 3),
            "dE_p99": round(float(np.percentile(de, 99)), 3),
            "dE_flat": round(float(de[flat].mean()), 3) if flat.sum() > 100 else None}
