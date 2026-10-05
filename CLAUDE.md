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
- `app/` — FastAPI + SQLite service (sellers, products, scan, monitor, dashboard)
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

## Agreed roadmap
1. ✅ Fix architecture (single layered image) + honest benchmark with false positives
2. **Mostly done:** HiDDeN v2 trained in Colab and evaluated (see training notes).
   Remaining: integrate v2 into `src/ensemble.py` / `app/`, re-run the deployed benchmark
3. Geometric alignment of suspect to original (SIFT/ORB) before DCT detection
4. Compare with pretrained robust watermarks (Watermark Anything, TrustMark, StegaStamp)
5. AI-edit attacks: SD img2img at several strengths, inpainting, background
   removal/replacement, AI upscaling; discuss regeneration attacks (Zhao et al. 2023)
6. Second defence layer: perceptual hash / DINOv2-CLIP fingerprinting; mention C2PA
7. Real screenshot / phone-photo dataset instead of only simulations

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
