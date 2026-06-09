# scripts/convert_images.py
# One-time utility: convert downloaded SIPI TIFF images to PNG
# Run once from project root: python scripts/convert_images.py

from PIL import Image
import os

INPUT_DIR  = "data/sipi"
OUTPUT_DIR = "data/sipi"

images = ["baboon", "airplane", "peppers", "splash", "house"]

for name in images:
    tiff_path = os.path.join(INPUT_DIR, f"{name}.tiff")
    png_path  = os.path.join(OUTPUT_DIR, f"{name}.png")

    if not os.path.exists(tiff_path):
        print(f"[SKIP] {tiff_path} not found")
        continue

    img = Image.open(tiff_path).convert("RGB")
    img.save(png_path, format="PNG")
    print(f"[OK] {name}.tiff → {name}.png  size={img.size}")

print("\nDone. You can now use the PNG files in main.py.")