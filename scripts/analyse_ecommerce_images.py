# scripts/analyse_ecommerce_images.py
# How do e-commerce product photos differ from "regular" photos?
#
# Measures every image in two or more folders and summarises per set. The
# properties are the ones that matter for an invisible watermark:
#
#   flat_share       share of pixels in flat regions (local std of brightness
#                    < 1% at sigma 4 px) — a watermark cannot hide there, and a
#                    colour change on a flat area is what the eye notices first
#   bg_share         share of pixels within a small colour distance of the
#                    median border colour — the plain studio background
#   white_share      share of near-white pixels (all channels >= 235)
#   pure_white_share share of pixels at pure white (all channels >= 254), the
#                    Amazon-style #FFFFFF background: an additive watermark is
#                    clipped there (it can only darken), so it loses signal
#   edge_density     mean Sobel gradient magnitude of brightness (0-255 scale)
#                    — texture/detail available to mask the watermark
#   laplacian_var    variance of the Laplacian — fine-detail energy
#   saturation       mean HSV saturation (0-1)
#   colourfulness    Hasler & Suesstrunk (2003) colourfulness metric
#   luma_range       5th-95th percentile spread of brightness (0-255)
#   width, height, megapixels, aspect (w/h), is_square
#   jpeg_quality     estimated libjpeg quality from the quantisation table
#                    (None for PNG/HEIC-converted files)
#
# All pixel measures are computed after resizing the longer side to 1024 px, so
# that resolution differences between the sets do not change the statistics.
#
# Run:
#   python scripts/fetch_reference_images.py            # once: ABO + COCO samples
#   python scripts/analyse_ecommerce_images.py \
#       "Own product photos=data/products_png" \
#       "ABO product photos=data/abo_sample" \
#       "COCO regular photos=data/coco_sample"
# Output: output/ecommerce_analysis/{per_image.csv, summary.csv, *.png}

import csv
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

OUT = Path("output/ecommerce_analysis")
WORK_SIDE = 1024
SETS_DEFAULT = [
    "Own product photos=data/products_png",
    "ABO product photos=data/abo_sample",
    "COCO regular photos=data/coco_sample",
]
# Standard libjpeg luminance quantisation table (quality 50), Annex K
_STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], float)


def jpeg_quality(img: Image.Image) -> int | None:
    """Estimate libjpeg quality by matching the luminance table to scaled standard tables."""
    q = getattr(img, "quantization", None)
    if not q or 0 not in q:
        return None
    table = np.array(q[0], float)
    best, best_err = None, np.inf
    for quality in range(1, 101):
        scale = 5000 / quality if quality < 50 else 200 - 2 * quality
        ref = np.clip(np.floor((_STD_LUMA * scale + 50) / 100), 1, 255)
        err = np.abs(np.sort(ref) - np.sort(table)).mean()   # order-free: zigzag vs natural
        if err < best_err:
            best, best_err = quality, err
    return best


def measure(path: str) -> dict:
    src = Image.open(path)
    w, h = src.size
    jq = jpeg_quality(src)
    img = src.convert("RGB")
    scale = WORK_SIDE / max(w, h)
    if scale < 1:
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    rgb = np.asarray(img, dtype=np.float32)
    gray = rgb @ np.array([0.299, 0.587, 0.114], np.float32)

    g = gray / 255
    mean = ndimage.gaussian_filter(g, 4)
    std = np.sqrt(np.maximum(ndimage.gaussian_filter(g * g, 4) - mean * mean, 0))
    flat = std < 0.01

    border = np.concatenate([rgb[:8].reshape(-1, 3), rgb[-8:].reshape(-1, 3),
                             rgb[:, :8].reshape(-1, 3), rgb[:, -8:].reshape(-1, 3)])
    bg_colour = np.median(border, axis=0)
    bg = np.linalg.norm(rgb - bg_colour, axis=2) < 18

    gx, gy = ndimage.sobel(gray, 1), ndimage.sobel(gray, 0)
    lap = ndimage.laplace(gray)

    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    colourfulness = np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())

    return {
        "file": os.path.basename(path),
        "width": w, "height": h, "megapixels": round(w * h / 1e6, 2),
        "aspect": round(w / h, 3), "is_square": int(abs(w / h - 1) < 0.02),
        "flat_share": round(float(flat.mean()), 4),
        "bg_share": round(float(bg.mean()), 4),
        "white_share": round(float((rgb.min(axis=2) >= 235).mean()), 4),
        "pure_white_share": round(float((np.asarray(src.convert("RGB")).min(axis=2) >= 254).mean()), 4),
        "edge_density": round(float(np.hypot(gx, gy).mean() / 4), 2),   # /4: Sobel kernel gain
        "laplacian_var": round(float(lap.var()), 1),
        "saturation": round(float(sat.mean()), 4),
        "colourfulness": round(float(colourfulness), 2),
        "luma_range": round(float(np.percentile(gray, 95) - np.percentile(gray, 5)), 1),
        "jpeg_quality": jq,
    }


SUMMARY_COLS = ["flat_share", "bg_share", "white_share", "pure_white_share", "edge_density", "laplacian_var",
                "saturation", "colourfulness", "luma_range", "megapixels", "aspect",
                "is_square", "jpeg_quality"]


def summarise(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for col in SUMMARY_COLS:
        vals = np.array([r[col] for r in rows if r[col] is not None], float)
        if len(vals) == 0:
            out.update({f"{col}_mean": None, f"{col}_median": None, f"{col}_iqr": None})
            continue
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        out.update({f"{col}_mean": round(vals.mean(), 4), f"{col}_median": round(med, 4),
                    f"{col}_iqr": f"{q1:.3g}-{q3:.3g}"})
    return out


def charts(sets: dict[str, list[dict]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ["#2a78d6", "#eb6834", "#1baf7a"]   # dataviz reference categorical slots 1-3
    ink, quiet, grid = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": quiet, "axes.labelcolor": ink,
                         "xtick.color": quiet, "ytick.color": quiet, "axes.titlesize": 11})
    names = list(sets)

    def boxes(ax, col, title, ylabel, scale=1.0):
        data = [[r[col] * scale for r in sets[n] if r[col] is not None] for n in names]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=True,
                        medianprops={"color": ink, "linewidth": 2},
                        whiskerprops={"color": quiet}, capprops={"color": quiet},
                        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": quiet,
                                    "markeredgecolor": "none", "alpha": 0.6})
        for patch, c in zip(bp["boxes"], colours):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
            patch.set_edgecolor(c)
            patch.set_linewidth(1.5)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels([f"{n.replace(' photos', '')}\n(n={len(sets[n])})" for n in names])
        ax.set_title(title, loc="left", color=ink, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    boxes(axes[0], "flat_share", "Flat area (no texture to hide in)", "% of pixels", 100)
    boxes(axes[1], "bg_share", "Plain background", "% of pixels", 100)
    boxes(axes[2], "edge_density", "Texture / edge density", "mean gradient (0-255)")
    fig.suptitle("Product photos have far more flat background and less texture than regular photos",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "flat_background_texture.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    boxes(axes[0], "saturation", "Colour saturation", "mean HSV saturation (0-1)")
    boxes(axes[1], "pure_white_share", "Pure-white pixels (#FFFFFF)", "% of pixels", 100)
    boxes(axes[2], "megapixels", "Resolution", "megapixels")
    fig.suptitle("Marketplace photos use pure-white backgrounds; regular photos are more colourful",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "colour_white_resolution.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    bins = np.linspace(0.4, 2.0, 33)
    for n, c in zip(names, colours):
        ax.hist([r["aspect"] for r in sets[n]], bins=bins, histtype="step", linewidth=2,
                color=c, label=f"{n} (n={len(sets[n])})")
    ax.axvline(1.0, color=quiet, linestyle="--", linewidth=1)
    ax.text(1.01, ax.get_ylim()[1] * 0.92, "square", color=quiet, fontsize=9)
    ax.set_xlabel("aspect ratio (width / height)")
    ax.set_ylabel("images")
    ax.set_title("Aspect ratios: marketplaces favour square images", loc="left",
                 fontweight="bold", color=ink)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "aspect_ratio.png", dpi=200)
    plt.close(fig)


def main(specs: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sets: dict[str, list[dict]] = {}
    for spec in specs:
        name, folder = spec.split("=", 1)
        files = sorted(f for f in os.listdir(folder)
                       if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
        sets[name] = [measure(os.path.join(folder, f)) for f in files]
        print(f"{name}: {len(files)} images")

    with open(OUT / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["set"] + list(next(iter(sets.values()))[0].keys()))
        w.writeheader()
        for name, rows in sets.items():
            for r in rows:
                w.writerow({"set": name, **r})

    summary = [{"set": name, **summarise(rows)} for name, rows in sets.items()]
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print(f"\n{'median (IQR)':<16}" + "".join(f"{s['set'][:22]:>26}" for s in summary))
    for col in SUMMARY_COLS:
        print(f"{col:<16}" + "".join(
            f"{str(s[col + '_median']) + ' (' + str(s[col + '_iqr']) + ')':>26}" for s in summary))
    charts(sets)
    print(f"\nWrote {OUT}/per_image.csv, summary.csv and 3 charts")


if __name__ == "__main__":
    main(sys.argv[1:] or SETS_DEFAULT)
