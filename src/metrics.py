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