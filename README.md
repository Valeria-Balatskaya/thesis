# Watermarking for E-commerce Image Traceability

Bachelor's thesis project: steganography and digital watermarking to trace stolen
product images back to the seller/product they came from.

## Setup (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Setup (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
# TraceMark web app -> http://127.0.0.1:8000
uvicorn app.main:app --reload

# Benchmarks are standalone scripts (not pytest)
python tests/test_hidden_v2_eval.py                       # watermark alone, classic attacks
python tests/test_defence_benchmark.py data/products_png  # full pipeline, modern + AI attacks
python tests/test_in_the_wild.py data/products_png        # real browser captures, CDN sizes
python scripts/field_test.py data/field                   # score images you collected by hand
```

Optional AI attacks (background replacement, diffusion-VAE regeneration):
`pip install -r requirements-ai.txt`.

## The web app (TraceMark)

- **Protect an image** — upload an original, get back a copy carrying an invisible
  63-bit ID; compare slider, amplified watermark view, PSNR/SSIM.
- **Catalogue** — issue extra copies per partner/channel so a leak can be traced to
  its source; list the URLs where a product may appear.
- **Attack Lab** — apply what image thieves do (crop, screenshot, banner, AI edits),
  stack attacks, and watch the four detection stages try to trace the result.
- **Scan a suspect** — check any image; produces a printable evidence report.
- **Web Monitor** — crawls the pages on a watchlist, checks every image on them, and
  flags sightings outside the authorised URLs. "Build demo shop" creates a fake
  marketplace page with stolen copies so the monitor can be demonstrated live.

App data (uploads, database, crawled images) lives in `app/data/` and is git-ignored.

## Model checkpoints

Checkpoints are git-ignored and must be copied into `checkpoints/` manually;
without them the benchmarks and the web app stop with `FileNotFoundError`.

- `checkpoints/hidden_final.pt` — HiDDeN v1 (only needed for the older v1 benchmarks)
- `checkpoints/hidden_v2_best.pt` — HiDDeN v2 (used by the app). Train it with
  `notebooks/train_hidden_v2.ipynb` in Google Colab, then evaluate it with
  `python tests/test_hidden_v2_eval.py`
