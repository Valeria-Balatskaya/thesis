# Project context for Claude

Bachelor's thesis: **steganography and digital watermarking to protect product
images in e-commerce** — trace a stolen image (screenshots, re-uploads, filters,
AI edits) back to the seller/product it came from. The author develops on a Mac
(Apple Silicon) and previously on Windows; HiDDeN training happens in Google Colab.

## Codebase map
- `src/lsb.py` — LSB baseline (fails every attack; used as negative result)
- `src/dct_watermark.py` — Cox et al. 1997 DCT spread-spectrum. **Non-blind**:
  detection needs the reference image + exact coefficient indices, so it breaks
  under crop/perspective/aspect changes. Very low false-positive rate (0/300).
- `src/hidden_watermark.py`, `src/training/train.py` — HiDDeN v1 (48-bit, 128px) + training loop
- `src/hidden_v2.py`, `src/training/train_v2.py` — HiDDeN v2: residual-style training,
  fixed PSNR budget (residual rescaled to target, no image loss), differentiable
  distortions (real JPEG, perspective, colour, crop, rescale, screenshot, print-photo),
  63-bit payload. Train in Colab with `notebooks/train_hidden_v2.ipynb`
- `src/bch.py`, `src/payload.py` — BCH(63,30,t=6); payload = 20-bit product ID +
  10-bit HMAC tag, XOR mask. Blind detection, theoretical FPR 8.6e-6 per image
- `src/hidden_v2_inference.py` — `embed_product` / `read_product` (blind, no original needed)
- `src/hidden_inference.py` — embed/decode. `mode="residual"` (default) adds only
  the upsampled watermark residual (~40 dB); `mode="legacy"` = old behaviour (~20-29 dB)
- `src/ensemble.py` — v4 **layered**: HiDDeN residual then DCT in ONE published file.
  `embed_dual_storage` = legacy v3 (two files, HiDDeN copy never published)
- `src/attacks.py`, `src/ecommerce_attacks.py` — attack suites (incl. screenshot,
  print-photograph, native screenshot simulations)
- `src/defence.py` — **Retrieve-Align-Verify** pipeline (`DefencePipeline.analyse`):
  1 blind read -> 2 SIFT+RANSAC retrieval against registered originals (also mirrored)
  -> 3 warp suspect into the original's frame (uncovered pixels filled from the
  UN-watermarked original) -> 4 read again; with one candidate product the bits may
  also be matched to its known codewords (<= ~11 errors at FPR 1e-6 instead of BCH's 6).
  Verdicts: `watermark`, `watermark_aligned`, `fingerprint_only`, `none`.
  `informed_score()` = non-blind check for one copy's exact residual (ownership disputes)
  A product has several *copies*, each with its own watermark ID (leak tracing)
- `src/modern_attacks.py` — off-centre crop, rotate, mirror, perspective, page
  screenshot, phone photo of screen, messenger recompress, promo overlay, inpaint,
  upscale; AI: `regenerate_vae` (SD VAE), `replace_background` (rembg) — optional,
  `pip install -r requirements-ai.txt`
- `app/` — **TraceMark** web app (FastAPI + SQLite, single-page UI in `app/static/index.html`):
  `main.py` routes, `guard.py` service (protect / analyse / attack lab, `STRENGTH = 0.5`;
  `precheck()` refuses uploads that are already watermarked or registered -> HTTP 409),
  `store.py` DB (`app/data/tracemark.db`, git-ignored), `crawler.py` web monitor
  (watchlist crawl + scheduler; `/demo/shop` is a generated thief page to crawl),
  `/report/{id}` printable evidence report. `service.py`, `database.py`, `thesis.db`
  are the legacy v1 app, kept but no longer served
- `tests/` — standalone benchmark scripts (run with `python tests/<file>.py`, not pytest)
- `checkpoints/hidden_final.pt` — trained v1 model, git-ignored, copy manually
  (benchmarks fail with FileNotFoundError without it)
- `checkpoints/hidden_v2_best.pt` — v2 model once trained; evaluate with
  `python tests/test_hidden_v2_eval.py` (real PIL attacks + false-positive regression)

## Key findings so far (tests/test_deployed_benchmark.py, 5 SIPI images x 12 attacks)
| System | Correct seller | PSNR | False positives |
|---|---|---|---|
| v3 dual storage (as actually deployed) | 57% | 41.7 dB | 24/60 |
| v4 layered (current) | 82% | 37.2 dB | 18/60 |

The earlier "91%" (tests/test_ensemble_benchmark.py) attacked both files separately,
which doesn't match what gets published. **All false positives come from HiDDeN**,
and the main cause is the decision rule, not the checkpoint: with 48 bits,
"BER < 0.40" (<= 19 errors) is met by pure random bits with probability 9.7% per
candidate, i.e. ~40% with 5 sellers — exactly the observed 16-24/60. A 48-bit rule
safe at 1e-6 would need BER <= 0.17, which the v1 checkpoint (clean BER 0.19-0.44 in
residual mode) cannot meet. False-positive counts vary a little between runs because
the noise attacks are unseeded.

## Retrieve-Align-Verify results (tests/test_defence_benchmark.py)
35 registered product photos (2 copies each), 12 held out, 17 attacks, 549 attacked images.
| Strength | PSNR | blind only | full pipeline | false watermark claims | wrong copy |
|---|---|---|---|---|---|
| 1.0 | 40.0 dB | 62% | 98% | 0 / 410 | 0 |
| 0.5 (app default) | 45.8 dB | 61% | 96% | 0 / 410 | 0 |

- Blind read fails completely (0%) on off-centre crop, rotation, mirror, perspective and
  page screenshots; alignment recovers 97-100% of them.
- Weak spots at 0.5: 50% off-centre crop (25% of the area) 43% traced (80% at 1.0);
  AI background replacement 92% (n=12); page screenshot 97%.
- AI VAE regeneration: 12/12 read blind at both strengths (n=12 only; full diffusion
  img2img not tested).
- The watermark lives entirely in colour: a luminance-only version of the residual
  reads 0/12. At strength 1.0 a green/purple tint is visible on white backgrounds,
  hence the 0.5 default.
- "Same picture" check: intensity correlation let 31/60 *different photos of the same
  product* through. Gradient correlation at 256px with threshold 0.65 -> 0/60, at the
  cost of a few true matches falling to "none". Sibling photos reach 0.61 at most.
- Leak tracing between two copies of one product: 74/75 correct, 0 confused.
- Run times: ~5-10 min on CPU. Results at 1.0 predate the stricter structure check
  (its robustness columns are unaffected; control B was 31/60 then).

## Visibility (measured 2026-10-05, 12 product photos, CIEDE2000; ~1 = just noticeable)
| Model / setting | PSNR | dE mean | dE on flat areas | notes |
|---|---|---|---|---|
| v2, strength 1.0 | 40.0 | 3.2 | 3.2 | green/purple tint clearly visible on white |
| v2, 0.5 | 45.8 | 1.6 | 1.6 | still visible to the author |
| v2, 0.7 + flat mask 0.3 (app default) | 47.5 | 1.2 | 0.85 | full pipeline 95% (584 imgs), 50% crop 20% |
| v3 trial (chroma_weight 8, mask_floor 0.3) | 45.2 | 0.40 | 0.29 | 20-min local trial on synthetic data |

- v2 carries its signal entirely in low-frequency colour; no strength/mask setting is both
  invisible and robust. PSNR does not capture this — use `src.metrics.visibility` (dE).
- v3 = same network trained with `--chroma-weight 8 --mask-floor 0.3` (colour changes cost
  8x in the budget; 30% amplitude on flat areas). The trial read 12/12 clean, JPEG 30,
  messenger, CDN 360 but 0/12 native screenshot and screen photo — the same geometric
  gap the first v2 trial had before full training. **Needs the full Colab run**
  (`notebooks/train_hidden_v2.ipynb`, RUN='v3'); the app prefers
  `checkpoints/hidden_v3_best.pt` when present. Not yet proven robust.

## In-the-wild checks (tests/test_in_the_wild.py, 12 photos, v2 0.7/0.3)
- Real headless-Chrome captures of a listing page: photo shown at 480px (1x, 2x) 12/12,
  340px on a 3x phone 12/12, 320px 11/12, **200px search-grid 0/12**. All via alignment.
- Simulated CDN re-encodes (WebP/JPEG q80): >= 360px 12/12; 192px 10-12/12; **<= 128px lost**.
- `scripts/field_test.py <folder>` scores images collected by hand (marketplace
  downloads, OS screenshots, phone photos, messenger forwards) against the app's catalogue.
- Allegro: listing images are served by its CDN in fixed sizes; crawling listing pages is
  blocked by bot protection and against the terms — use the official REST API
  (developer.allegro.pl, OAuth) to list offers and their image URLs, then feed those to
  `app/crawler.py`. Not implemented (needs the user's API credentials).

## E-commerce vs regular images (2026-10-20, docs/supervisor_questions.md)
Supervisor Q1. `scripts/fetch_reference_images.py` downloads 60 ABO (Amazon Berkeley Objects,
real listings, CC BY 4.0) + 60 COCO test2017 photos (seed 0) into git-ignored `data/abo_sample`,
`data/coco_sample`. `scripts/analyse_ecommerce_images.py` -> `output/ecommerce_analysis/`.
| median | own 47 | ABO 60 | COCO 60 |
|---|---|---|---|
| flat area | 66% | 50% | 5% |
| pure white #FFFFFF | 0% | 37% | 0% |
| edge density | 4.4 | 8.4 | 16.8 |
| saturation | 0.09 | 0.11 | 0.31 |
| square (mean) / JPEG q | 0% / PNG | 40% / 76 | 7% / 96 |
`tests/test_ecommerce_vs_regular.py` (30/set x 12 attacks, v2, 0 FP everywhere):
| set | app 0.7/0.3 blind / full | dE flat | full 1.0 blind |
|---|---|---|---|
| own | 100% / 100% | 0.86 | 100% |
| ABO | **81.7% / 91.4%** | 0.62 | 95.8% |
| COCO | 95.8% / 98.6% | 0.78 | 98.9% |
- Real marketplace photos are the hardest case; own photos (grey backdrop, no pure white)
  overestimate robustness -> report ABO in the thesis. ABO weak spots at 0.7/0.3: native
  screenshot 17/30, brightness 18/30 (white clips at 255), instagram 21/30, JPEG30 22/30.
  Failures vs flat share Spearman 0.51 (p .004), vs edge density -0.41 (p .026).
- All 5 ABO marketplace_square blind misses were tall photos (aspect 0.85 = 1879x2200, 0.52,
  0.32) not in `ASPECT_RATIOS`; alignment recovered 5/5.
- Proposed, not implemented: train v3 on COCO+ABO incl. pure-white backgrounds; pre-darken
  #FFFFFF by 1-2 levels before embedding (headroom against clipping); add 6:7 and 1:2 aspect
  ratios (costs FP budget); brightness/levels in training.

## Re-watermarking (tests/test_rewatermark.py, 20 own + 20 ABO, v2 at 0.7/0.3)
Supervisor Q2. Baseline 39/40 read blind (one busy ABO photo fails, pipeline recovers it).
- Other systems on top (DCT, LSB, HiDDeN v1, stacked 2-3) and visible marks (small corner,
  large tiled): ours still read blind 36-39/40 (clean/JPEG q70), pipeline 38-40/40, thief ID
  never; control (same marks on un-watermarked originals) 0/40 false positives each.
- **Our own v2 on top overwrites ours for the blind decoder**: ours 0/40; thief's ID read
  28-37/40 (same strength), 40/40 (thief at full strength); 2-3 v2 layers -> mostly nothing.
  Our residual is still present: `defence.informed_score` (correlation of suspect-original with
  our exact residual, large changes clipped) beats all 8 never-issued decoy copies in 40/40
  (same strength), 30-32/40 (thief full strength), 38-40/40 (stacked). Evidence, not a
  calibrated threshold; needs alignment.
- Fix: `guard.precheck()` before protecting. API test in a separate TRACEMARK_DATA folder,
  11 scenarios all correct (copy, +JPEG, +native screenshot, +rotate 8, +promo overlay, leaked
  clean original, same seller again, foreign valid ID -> 409; unrelated photos -> accepted).
- Ownership argument: HMAC key (can't forge our IDs) + we hold the earlier clean original;
  Craver et al. 1997/1998. Not covered: an attacker running their own copy of the model with
  another key outside our server (pre-check doesn't apply there).

## Working rules learned the hard way
- The user runs the app on port 8000 with their own data in `app/data/`. Never delete or
  seed it. Test with `TRACEMARK_DATA=<other folder> uvicorn app.main:app --port 8765`.

## Agreed roadmap
1. ✅ Fix architecture (single layered image) + honest benchmark with false positives
2. ✅ HiDDeN v2 trained in Colab, evaluated, and integrated into the app
3. ✅ Geometric alignment (Retrieve-Align-Verify, `src/defence.py`)
4. Compare with pretrained robust watermarks (Watermark Anything, TrustMark, StegaStamp)
5. Partly done: AI attacks (VAE regeneration, background replacement). Still open:
   SD img2img at several strengths, generative inpainting, neural upscaling
6. Partly done: SIFT fingerprint as second layer. Open: global embeddings
   (DINOv2/CLIP) + ANN index so retrieval scales beyond a few hundred products; C2PA
7. Real screenshot / phone-photo / social-platform round-trip dataset (all attacks are
   still simulations)
8. Web monitor discovery: plug a reverse-image-search API (Google Vision Web Detection,
   TinEye) into `app/crawler.py`; today it only crawls a watchlist
9. ✅ Supervisor Q1/Q2: e-commerce image analysis + re-watermarking test + upload pre-check
   (docs/supervisor_questions.md). v3 training should now include ABO product photos
10. **Next:** full Colab run of v3 (colour charge + flat-area mask), then re-run
   `test_hidden_v2_eval.py`, `test_defence_benchmark.py`, `test_in_the_wild.py` on it

## HiDDeN v2 training notes
- HiDDeN's plain design (spatially constant message map + pooled decoder) stalled at
  ~60% bit accuracy in local trials. What fixed it: a per-bit learnable spatial carrier
  in the encoder (added to the output) plus a shallow full-resolution "matched filter"
  branch in the decoder -> 100% bit accuracy at 40 dB within ~300 steps (no distortions).
- 20-minute local trial (`checkpoints/hidden_v2_trial.pt`: 3,600 steps, batch 16, 405
  synthetic images) on the same 5 SIPI images x 12 real attacks: **55/60 = 92% correct
  product ID, blind, at 40.0 dB, 0 false positives in 1,560 clean images**. Only
  `native_screenshot` fails (0/5). In training validation, strong perspective (10%),
  screenshot and print-photo distortions were still near chance, and crop accuracy
  swung between checkpoints (99% -> 20%), so training was not yet stable.
  These are trial numbers, not thesis results — the Colab run replaces them.
- **Colab model** (`checkpoints/hidden_v2_best.pt`, step 28,000 of 40,000, COCO val2017,
  2.1 it/s on a T4): on the 5 SIPI photos x 12 real attacks **58/60 = 97% correct
  product ID, blind, 40.0 dB, 0 false positives in 1,560 clean images** (v4 layered:
  82%, 37.2 dB, 16/60). Only misses: peppers and splash under `native_screenshot`.
  Colab validation at full severity: perspective 44% decodable, screenshot 69%,
  print-photo 79%, combo 70%, everything else 99-100%.
  **Generalisation gap:** on the 25 procedural pattern images (flat saturated shapes)
  it reads only 13%, even clean (BER ~0.19).
- **Product photos** (`data/products_png/`, 47 iPhone shots of phones/boxes on white or
  grey, 1024x768, git-ignored; convert HEIC with `sips -s format png -Z 1024`):
  `python tests/test_hidden_v2_eval.py checkpoints/hidden_v2_best.pt data/products_png`
  -> **564/564 = 100% across 12 attacks, 0 false positives in 2,064 clean images**.
  Without the aspect-ratio fallback `marketplace_square` was 2/47: the watermark is
  position-dependent, so a centre crop of a 4:3 photo to a square misaligns it.
  `read_product` now pads a failed read back to common aspect ratios (4:3, 3:4, 3:2,
  2:3, 16:9, 9:16) and retries -> 47/47. Limits: only centre crops from those ratios;
  off-centre or arbitrary crops are not handled (roadmap 3). The 47 photos are one
  session, one camera, a handful of products — not independent samples.
- Local trials: `--device mps` with `PYTORCH_ENABLE_MPS_FALLBACK=1` (~3 it/s); CPU is ~10x slower.

## Conventions
- Report numbers honestly, including failures — they go into the thesis discussion.
- Keep legacy code paths so earlier results stay reproducible.
- Don't commit large generated files (`results/`, `output/` are git-ignored).
- Work happens on branch `claude/keen-gauss-dwwse7` (PR #1).
