import sys, os, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dct_watermark import embed, detect
from src.metrics import compute

IMAGES    = ["baboon", "airplane", "peppers", "splash", "house"]
SELLER_ID = "SELLER_ID:0042"
PARAMS = [
    (0.015, 300),
    (0.02, 300),
    (0.02, 500),
    (0.03, 300),
    (0.03, 500),
]

os.makedirs("results/dct", exist_ok=True)
os.makedirs("output", exist_ok=True)

print(f"\n{'alpha':<8} {'n_coeffs':<10} {'Image':<12} {'PSNR (dB)':<12} {'SSIM':<10} {'Similarity':<12} {'Detected':<10} Result")
print("-" * 90)
rows = []

for alpha, n_coeffs in PARAMS:
    for name in IMAGES:
        orig   = f"data/sipi/{name}.png"
        stego  = f"results/dct/dct_{name}_a{alpha}_n{n_coeffs}.png"
        meta   = embed(orig, SELLER_ID, stego, alpha=alpha, n_coeffs=n_coeffs)
        result = detect(stego, orig, SELLER_ID, meta, threshold=6.0)
        m      = compute(orig, stego)
        ok = result["detected"] and m["PSNR_dB"] > 38 and m["SSIM"] > 0.98
        rows.append([alpha, n_coeffs, name, m["PSNR_dB"], m["SSIM"], result["similarity"], result["detected"], ok])
        print(f"{alpha:<8} {n_coeffs:<10} {name:<12} {m['PSNR_dB']:<12} {m['SSIM']:<10} {result['similarity']:<12} {str(result['detected']):<10} {'PASS' if ok else 'FAIL'}")

with open("output/dct_tuning_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["alpha", "n_coeffs", "image", "psnr_db", "ssim", "similarity", "detected", "pass"])
    w.writerows(rows)

print("\nCSV saved to output/dct_tuning_results.csv")