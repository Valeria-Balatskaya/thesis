# tests/test_in_the_wild.py
# Two checks that move the evaluation closer to how a marketplace really
# handles and displays product photos.
#
# A. Marketplace CDN variants (simulated). A marketplace re-encodes every
#    uploaded photo into several fixed sizes (gallery, search grid, thumbnail),
#    as WebP or JPEG. How small can a variant get before the copy is lost?
#
# B. Real browser captures (NOT simulated). A listing page containing the
#    protected photo is rendered by headless Chrome and captured at several
#    display sizes and pixel densities. The capture is what a thief gets from
#    "screenshot the listing": real browser scaling and colour handling, the
#    photo surrounded by page layout and text.
#
# Run: python tests/test_in_the_wild.py [image_folder] [n_images] [strength] [mask_floor]

import sys, os, csv, shutil, subprocess, html
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.hidden_v2_inference import HiddenV2Model
from src.defence import DefencePipeline
from src import modern_attacks as ma

CHECKPOINT = "checkpoints/hidden_v2_best.pt"
IMAGE_DIR = sys.argv[1] if len(sys.argv) > 1 else "data/products_png"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
STRENGTH = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
MASK_FLOOR = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3

CDN_SIZES = [1024, 720, 512, 360, 192, 128, 64]
# (label, CSS width of the photo, window width, window height, device pixel ratio)
CAPTURES = [
    ("desktop 1x, photo 480px",  480, 1280, 900, 1),
    ("retina 2x, photo 480px",   480, 1280, 900, 2),
    ("desktop 1x, photo 320px",  320, 1280, 800, 1),
    ("phone 3x, photo 340px",    340,  390, 844, 3),
    ("search grid 1x, 200px",    200, 1280, 700, 1),
]
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    shutil.which("google-chrome") or "", shutil.which("chromium") or "", shutil.which("chrome") or "",
]

OUT = os.path.abspath("results/in_the_wild")
os.makedirs(OUT, exist_ok=True)
os.makedirs("output", exist_ok=True)
if not os.path.exists(CHECKPOINT):
    sys.exit(f"Checkpoint not found: {CHECKPOINT}")

files = sorted(f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg")))
step = max(1, len(files) // N)
files = files[::step][:N]

model = HiddenV2Model(CHECKPOINT)
pipeline = DefencePipeline(model)
products = {}
for i, fname in enumerate(files):
    pid, copy_id = i + 1, 6000 + i
    orig = os.path.join(IMAGE_DIR, fname)
    pub = f"{OUT}/pub_{pid}.png"
    pipeline.register(pid, orig, [copy_id])
    model.embed_product(orig, copy_id, pub, strength=STRENGTH, mask_floor=MASK_FLOOR)
    products[pid] = {"pub": pub, "copy_id": copy_id}
print(f"{len(products)} protected photos from {IMAGE_DIR} (strength {STRENGTH}, mask floor {MASK_FLOOR})\n")


def tally(path, copy_id):
    res = pipeline.analyse(path)
    if res["copy_id"] == copy_id:
        return "blind" if res["verdict"] == "watermark" else "aligned"
    if res["copy_id"] is not None:
        return "WRONG"
    return "picture" if res["verdict"] == "fingerprint_only" else "lost"


def show(label, outcomes):
    n = len(outcomes)
    c = {k: outcomes.count(k) for k in ("blind", "aligned", "picture", "lost", "WRONG")}
    print(f"{label:<28}{(c['blind'] + c['aligned']) / n:>8.0%}{c['blind']:>8}{c['aligned']:>9}"
          f"{c['picture']:>14}{c['lost']:>6}{c['WRONG']:>7}")
    return c


HEADER = f"{'':<28}{'traced':>8}{'blind':>8}{'aligned':>9}{'picture only':>14}{'lost':>6}{'wrong':>7}"
rows = []

# ─── A. CDN variants ──────────────────────────────────────────────

print("A. MARKETPLACE CDN VARIANTS (simulated re-encode)")
print(HEADER)
for fmt in ("webp", "jpeg"):
    for size in CDN_SIZES:
        outcomes = []
        for pid, p in products.items():
            path = f"{OUT}/cdn_tmp.png"
            ma.marketplace_variant(p["pub"], path, size=size, fmt=fmt, quality=80)
            outcomes.append(tally(path, p["copy_id"]))
        c = show(f"{fmt} {size}px", outcomes)
        rows.append({"part": "cdn", "condition": f"{fmt}_{size}", "n": len(outcomes), **c})

# ─── B. Real browser captures ─────────────────────────────────────

chrome = next((c for c in CHROME_CANDIDATES if c and os.path.exists(c)), None)
print("\nB. REAL BROWSER CAPTURES (headless Chrome renders a listing page)")
if not chrome:
    print("   skipped: Chrome / Chromium not found on this machine")
else:
    print(HEADER)
    page_tpl = """<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;font-family:Arial,sans-serif;background:#f4f4f4;color:#222}}
header{{background:#ff5a00;color:#fff;padding:14px 20px;font-size:20px;font-weight:bold}}
.wrap{{display:flex;flex-wrap:wrap;gap:24px;padding:20px;background:#fff;margin:16px}}
.wrap img{{width:{w}px;height:auto;display:block}}
h1{{font-size:20px;margin:0 0 10px}}.price{{font-size:28px;font-weight:bold;margin:10px 0}}
button{{background:#ff5a00;color:#fff;border:0;padding:12px 28px;font-size:16px}}
</style></head><body><header>marketplace &nbsp; <span style="font-weight:normal;font-size:14px">szukaj produktu...</span></header>
<div class="wrap"><img src="{src}"><div><h1>{title}</h1><div>Stan: nowy &middot; Dostawa jutro</div>
<div class="price">1 299,00 z&#322;</div><button>KUP TERAZ</button>
<p style="max-width:300px;color:#666">Oryginalny produkt, gwarancja 24 miesi&#261;ce. Darmowy zwrot do 30 dni.</p></div></div>
</body></html>"""
    for label, css_w, win_w, win_h, dpr in CAPTURES:
        outcomes = []
        for pid, p in products.items():
            page = f"{OUT}/page.html"
            with open(page, "w") as f:
                f.write(page_tpl.format(w=css_w, src="file://" + html.escape(os.path.abspath(p["pub"])),
                                        title=f"Smartfon — oferta {pid}"))
            shot = f"{OUT}/capture_{pid}_{css_w}_{dpr}x.png"
            subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                            f"--window-size={win_w},{win_h}", f"--force-device-scale-factor={dpr}",
                            "--allow-file-access-from-files", f"--screenshot={shot}", "file://" + page],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
            outcomes.append(tally(shot, p["copy_id"]) if os.path.exists(shot) else "lost")
        c = show(label, outcomes)
        rows.append({"part": "browser", "condition": label, "n": len(outcomes), **c})

with open("output/in_the_wild.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("\nCSV: output/in_the_wild.csv   captures: results/in_the_wild/")
sys.exit(1 if any(r["WRONG"] for r in rows) else 0)
