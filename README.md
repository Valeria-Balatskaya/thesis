# TraceMark — invisible watermarks to trace stolen product images

Bachelor's thesis project: *A System for Invisible Digital Watermark Embedding for Content
Protection in E-Commerce Systems.*

A seller uploads a product photo. TraceMark returns a copy that looks the same but carries an
invisible 63-bit ID: a 20-bit copy ID, a 10-bit HMAC tag made with the server key, and BCH error
correction. When the photo turns up elsewhere (downloaded, screenshotted, photographed off a screen,
forwarded in a messenger, cropped, overlaid, AI-edited), TraceMark reads the ID back. It reports which
seller, which product and which issued copy (channel) it came from.

How it works:
- **Embedding:** a HiDDeN-style neural encoder, retrained for product photos (v3). It puts the signal
  in textured areas and leaves white backgrounds untouched. v4 adds a secret key.
- **Detection: Retrieve–Align–Verify.**
  1. Read blind.
  2. If that fails, find the registered original by SIFT/RANSAC.
  3. Warp the suspect back into the original's frame.
  4. Read again.
- **Ownership disputes:**
  - a calibrated informed check finds our exact signal under someone else's watermark
    (verdict `watermark_overwritten`);
  - optional keyed DCT layer;
  - a signed, hash-chained registration log (Ed25519, optional RFC 3161 timestamps);
  - content binding against copy attacks.

Results, limits and the numbers behind every claim: `docs/supervisor_questions.md` and `CLAUDE.md`.

## Install (macOS / Linux, Python 3.12+; developed on 3.14)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-ai.txt        # optional: AI attacks (SD VAE regeneration, rembg)
bash scripts/get_baselines.sh             # optional: TrustMark + Watermark Anything baselines
```

On Windows, use `.venv\Scripts\activate`. Everything else is the same.

**Checkpoints** are git-ignored. Copy them into `checkpoints/`:

| File | Model | Needed for |
|---|---|---|
| `hidden_v3_best.pt` | v3, product-photo model (default) | the app, all current tests |
| `hidden_v4_best.pt` | v4, secret-key model | the app prefers it when present |
| `hidden_v2_best.pt` | v2 | history / comparison rows |
| `hidden_final.pt` | v1 | oldest benchmarks, ablation (residual embedding) |

The app takes the first file that exists **and loads**, in the order v4 → v3 → v2.
`TRACEMARK_CHECKPOINT=checkpoints/hidden_v3_best.pt` pins one model. Copies protected with one model
can't be read by another.

**Data** for the evaluation (git-ignored):
- `python scripts/fetch_reference_images.py` downloads 60 ABO (Amazon Berkeley Objects, CC BY 4.0) and
  60 COCO photos. The `manifest` and `train` sub-commands build the product-level training split.
- Your own product photos go in `data/products_png/` (convert HEIC with `sips -s format png -Z 1024`).

## Run the app

```bash
uvicorn app.main:app --port 8000          # http://127.0.0.1:8000, data in app/data/
TRACEMARK_DATA=/tmp/tm-demo uvicorn app.main:app --port 8765   # separate demo/test data folder
```

| Page | What it does |
|---|---|
| **Protect** | Upload an original. The upload is refused (HTTP 409) if it already carries a TraceMark ID or matches a registered picture. You get the protected copy, a compare slider, the amplified watermark view, and PSNR/SSIM. |
| **Catalogue** | Issue extra copies per partner/channel (leak tracing). Authorise the URLs where the product may appear. |
| **Attack Lab** | Play the thief (crop, screenshot, banner, rotation, AI edits, stacked) and watch the four stages trace the result. |
| **Scan** | Check any image. Verdicts: `watermark`, `watermark_aligned`, `watermark_overwritten` (ownership dispute), `watermark_unbound` (possible copy attack), `fingerprint_only`, `none`. Each comes with a printable evidence report `/report/{id}`, which includes the signed registration record. |
| **Web Monitor** | Crawl a watchlist on a schedule. "Build demo shop" makes a fake thief page to crawl. |

Environment variables:
- `WATERMARK_KEY` / `WATERMARK_MODEL_KEY`: secrets. Change them before real use.
- `TRACEMARK_DCT_LAYER=1`: second watermark layer.
- `TRACEMARK_TSA_URL`: RFC 3161 timestamp authority.
- `TRACEMARK_DATA`: data folder.
- `TRACEMARK_CHECKPOINT`: pin the model.

## Run the evaluation

Benchmarks are standalone scripts (`python tests/<file>.py`, not pytest). Run them **one at a
time**: two torch processes in parallel slow each other down 10–50×.

```bash
bash scripts/reproduce_all.sh                      # every thesis table and figure, ~13-15 h on an M1 Pro
bash scripts/reproduce_all.sh v3_vs_v4 tables      # selected steps only
python scripts/thesis_tables.py                    # rebuild tables (Markdown, 95% Wilson CIs) from existing CSVs
```

Outputs:
- `output/thesis/*.md`: tables;
- `thesis/figures/`: figures;
- `output/thesis/logs/`: console logs.

The main individual tests:

| Test | Question |
|---|---|
| `tests/test_v3_vs_v2.py 30 [v4=...]` | visibility (PSNR, SSIM, ΔE) and robustness under 23 attacks, own / ABO / COCO |
| `tests/test_overwrite_cost.py 20 20 [v4=...]` | how much visible damage a thief must cause to overwrite our mark |
| `tests/test_overwritten_verdict.py` | hits and false positives of the `watermark_overwritten` verdict |
| `tests/test_v4_keys.py checkpoints/hidden_v4_best.pt` | v4: wrong keys, other-key and same-key overwriting |
| `tests/test_baselines.py 30` | ours vs TrustMark and Watermark Anything, default and matched visibility, our pipeline around each |
| `tests/test_ablation.py` | contribution of each component |
| `tests/test_real_world.py` | real screenshots, phone photos and messenger round trips (`docs/real_world_protocol.md`) |
| `tests/test_rewatermark.py`, `test_copy_attack.py`, `test_overhead.py`, `test_in_the_wild.py` | re-watermarking, copy attack, run time, real browser captures |

To evaluate a newly trained v4 against v3 in one go: `bash scripts/evaluate_v4.sh` (~3–4 h).

## Train in Google Colab

Step-by-step guide: [docs/colab_v3.md](docs/colab_v3.md).
1. Open `notebooks/train_hidden_v2.ipynb` from GitHub in Colab.
2. Pick a T4 GPU.
3. Set `RUN = 'v3'` or `'v4'`.
4. Run all.

Data and checkpoints are kept in Google Drive, and training resumes after a disconnect. Times:
v3 ≈ 5.5–6 h (40,000 steps), v4 ≈ 3.5 h (20,000 steps, warm start from v3). Copy the downloaded
`hidden_v*_best.pt` into `checkpoints/`.

## Repository map

| Path | Content |
|---|---|
| `src/hidden_v2.py`, `src/training/train_v2.py` | model (v2/v3/v4 options) and training loop |
| `src/hidden_v2_inference.py` | `embed_product` / `read_product` (blind, aspect-ratio retry, key override) |
| `src/payload.py`, `src/bch.py` | ID + HMAC tag + BCH(63,30,6), false-positive maths |
| `src/defence.py` | Retrieve–Align–Verify, `InformedChecker`, `ownership_check`, `mark_overwritten` |
| `src/dct_layer.py`, `src/dct_watermark.py`, `src/lsb.py` | keyed DCT layer; classical baselines (Cox DCT, LSB) |
| `src/attacks.py`, `src/ecommerce_attacks.py`, `src/modern_attacks.py`, `src/attack_suite.py` | attack simulations; the 23-attack thesis suite |
| `src/baselines.py` | TrustMark and Watermark Anything behind one interface |
| `src/metrics.py`, `src/stats.py` | PSNR/SSIM/CIEDE2000; Wilson intervals |
| `app/` | TraceMark web app (FastAPI + SQLite + single-page UI) |
| `scripts/` | data download, real-world capture tools, reproduction, tables |
| `docs/` | supervisor answers, Colab guide, capture protocol, design notes |
| `thesis/` | outline, chapter notes, `references.bib`, UML sources, figures |
| `app/service.py`, `app/database.py`, `src/hidden_inference.py`, `src/ensemble.py` | legacy v1 code, kept so earlier results stay reproducible |
