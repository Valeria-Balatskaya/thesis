# tests/test_overhead.py
# Part C1 — computational overhead (registered objective 1.2.4: minimal
# computational overhead of embedding and extraction).
#
#   per image, at 512 / 1024 / 2048 px (longer side), on CPU and on MPS (Apple GPU):
#     embed            HiDDeN v2 watermark + PNG write (embed_product)
#     blind_hit        blind read of a watermarked image (one decode)
#     blind_miss       blind read of an unmarked image (decode + 8 aspect retries)
#     pipeline_rotate  full Retrieve-Align-Verify on a rotated copy (blind fails,
#                      SIFT retrieval among 20 products, alignment, verify)
#   ownership tools (CPU, 1024 px): DCT layer embed / detect (500 null carriers),
#     informed checker build (500 decoys) / test
#   retrieval scaling (CPU, 1024 px): register N products (SIFT fingerprint) and
#     analyse a rotated copy against N = 10 ... 1000 registered products
#
# Times are medians of repeated runs after a warm-up; run with nothing else
# using the machine. Hardware is printed with the results.
# Run: python tests/test_overhead.py

import sys, os, csv, time, platform, subprocess
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from PIL import Image

from src.hidden_v2_inference import HiddenV2Model
from src.defence import DefencePipeline, InformedChecker
from src import modern_attacks as ma, dct_layer

OUT = "results/overhead"
os.makedirs(OUT, exist_ok=True)
os.makedirs("output", exist_ok=True)
REPEATS = 5
SIZES = [512, 1024, 2048]
DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
N_PRODUCTS = [10, 50, 100, 250, 500, 1000]


def timed(fn, repeats=REPEATS, sync=None):
    fn()                                   # warm-up (model load, kernels, caches)
    if sync:
        sync()
    out = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        if sync:
            sync()
        out.append(time.perf_counter() - t)
    return float(np.median(out))


def resized(src, side, dst):
    img = Image.open(src).convert("RGB")
    s = side / max(img.size)
    img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS).save(dst)
    return dst


try:
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                         text=True).stdout.strip()
except Exception:
    cpu = platform.processor()
print(f"Hardware: {cpu}, {os.cpu_count()} cores, torch {torch.__version__}, devices {DEVICES}\n")

source = "data/abo_sample/" + sorted(os.listdir("data/abo_sample"))[1]
plain = "data/abo_negatives/" + sorted(os.listdir("data/abo_negatives"))[0]
rows = []

# ─── per-image costs by size and device ───────────────────────────
reg_files = sorted(os.listdir("data/abo_negatives"))[1:20]
for device in DEVICES:
    model = HiddenV2Model("checkpoints/hidden_v2_best.pt", device=device)
    sync = torch.mps.synchronize if device == "mps" else None
    for side in SIZES:
        orig = resized(source, side, f"{OUT}/orig_{side}.png")
        neg = resized(plain, side, f"{OUT}/neg_{side}.png")
        wm = f"{OUT}/wm_{side}.png"
        t_embed = timed(lambda: model.embed_product(orig, 4242, wm, strength=0.7, mask_floor=0.3), sync=sync)
        t_hit = timed(lambda: model.read_product(wm), sync=sync)
        t_miss = timed(lambda: model.read_product(neg), sync=sync)
        pipe = DefencePipeline(model)
        pipe.register(0, orig, [4242])
        for k, f in enumerate(reg_files):       # 20 registered products in total
            pipe.register(k + 1, resized(f"data/abo_negatives/{f}", min(side, 1024), f"{OUT}/reg_{k}.png"), [])
        rot = f"{OUT}/rot_{side}.png"
        ma.rotate(wm, rot, degrees=8)
        assert pipe.analyse(rot)["copy_id"] == 4242, "rotated copy not traced"
        t_pipe = timed(lambda: pipe.analyse(rot), repeats=3, sync=sync)
        for op, t in [("embed", t_embed), ("blind_hit", t_hit), ("blind_miss", t_miss),
                      ("pipeline_rotate", t_pipe)]:
            rows.append({"group": "per_image", "device": device, "size_px": side, "operation": op,
                         "seconds": round(t, 4)})
        print(f"{device:<4} {side:>5}px  embed {t_embed:6.3f}s  blind hit {t_hit:6.3f}s  "
              f"miss {t_miss:6.3f}s  pipeline (rotate, 20 products) {t_pipe:6.3f}s", flush=True)

# ─── ownership tools (CPU, 1024 px) ───────────────────────────────
model = HiddenV2Model("checkpoints/hidden_v2_best.pt", device="cpu")
o = np.asarray(Image.open(f"{OUT}/orig_1024.png").convert("RGB"))
w = np.asarray(Image.open(f"{OUT}/wm_1024.png").convert("RGB"))
t_dct_e = timed(lambda: dct_layer.embed(w, 4242))
d = dct_layer.embed(w, 4242)
t_dct_d = timed(lambda: dct_layer.detect(d, w, 4242, n_null=500))
t_inf_b = timed(lambda: InformedChecker(model, o, 4242), repeats=3)
chk = InformedChecker(model, o, 4242)
t_inf_t = timed(lambda: chk.test(d))
for op, t in [("dct_layer_embed", t_dct_e), ("dct_layer_detect_500null", t_dct_d),
              ("informed_build_500decoys", t_inf_b), ("informed_test", t_inf_t)]:
    rows.append({"group": "ownership", "device": "cpu", "size_px": 1024, "operation": op,
                 "seconds": round(t, 4)})
print(f"\nownership (cpu, 1024px): DCT embed {t_dct_e:.3f}s  DCT detect {t_dct_d:.3f}s  "
      f"informed build {t_inf_b:.2f}s  informed test {t_inf_t:.4f}s", flush=True)

# ─── retrieval scaling (CPU) ──────────────────────────────────────
pool = [f"data/abo_negatives/{f}" for f in sorted(os.listdir("data/abo_negatives"))[20:]] + \
       [f"data/coco_sample/{f}" for f in sorted(os.listdir("data/coco_sample"))] + \
       [f"data/abo_sample/{f}" for f in sorted(os.listdir("data/abo_sample"))[2:]]
pipe = DefencePipeline(model)
pipe.register(0, f"{OUT}/orig_1024.png", [4242])
rot = f"{OUT}/rot_1024.png"
registered = 1
print("\nretrieval scaling (cpu, 1024px, rotated copy of product 0 -> full pipeline)")
for n in N_PRODUCTS:
    t = time.perf_counter()
    while registered < n:
        pipe.register(registered, resized(pool[registered - 1], 1024, f"{OUT}/scale_reg_{registered}.png"), [])
        registered += 1
    t_reg = (time.perf_counter() - t)
    res = pipe.analyse(rot)
    t_an = timed(lambda: pipe.analyse(rot), repeats=3)
    rows.append({"group": "retrieval", "device": "cpu", "size_px": 1024,
                 "operation": f"analyse_rotated_n{n}", "seconds": round(t_an, 4), "n_products": n,
                 "traced": int(res["copy_id"] == 4242)})
    print(f"  N={n:>5}: analyse {t_an:6.2f}s  traced {res['copy_id'] == 4242}  "
          f"(registering the new products took {t_reg:.1f}s)", flush=True)

with open("output/overhead.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=["group", "device", "size_px", "operation", "seconds",
                                       "n_products", "traced"])
    wr.writeheader()
    wr.writerows(rows)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.6))
    pts = [r for r in rows if r["group"] == "retrieval"]
    ax.plot([r["n_products"] for r in pts], [r["seconds"] for r in pts], marker="o",
            color="#2a78d6", linewidth=2)
    ax.set_xlabel("registered products")
    ax.set_ylabel("seconds per suspect")
    ax.set_title("Full-pipeline time grows linearly with the catalogue (SIFT retrieval)",
                 loc="left", fontweight="bold", fontsize=10)
    ax.grid(color="#e4e3df")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig("output/overhead_retrieval.png", dpi=200)
except Exception as e:
    print("chart skipped:", e)
print(f"\nCSV: output/overhead.csv, chart: output/overhead_retrieval.png  (hardware: {cpu})")
