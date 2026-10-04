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
- `src/hidden_watermark.py`, `src/training/` — HiDDeN (48-bit, 128px) + training loop
- `src/hidden_inference.py` — embed/decode. `mode="residual"` (default) adds only
  the upsampled watermark residual (~40 dB); `mode="legacy"` = old behaviour (~20-29 dB)
- `src/ensemble.py` — v4 **layered**: HiDDeN residual then DCT in ONE published file.
  `embed_dual_storage` = legacy v3 (two files, HiDDeN copy never published)
- `src/attacks.py`, `src/ecommerce_attacks.py` — attack suites (incl. screenshot,
  print-photograph, native screenshot simulations)
- `app/` — FastAPI + SQLite service (sellers, products, scan, monitor, dashboard)
- `tests/` — standalone benchmark scripts (run with `python tests/<file>.py`, not pytest)
- `checkpoints/hidden_final.pt` — trained model, git-ignored, copy manually

## Key findings so far (tests/test_deployed_benchmark.py, 5 SIPI images x 12 attacks)
| System | Correct seller | PSNR | False positives |
|---|---|---|---|
| v3 dual storage (as actually deployed) | 57% | 41.7 dB | 24/60 |
| v4 layered (current) | 82% | 37.2 dB | 18/60 |

The earlier "91%" (tests/test_ensemble_benchmark.py) attacked both files separately,
which doesn't match what gets published. **All false positives come from HiDDeN**:
the current checkpoint decodes even clean images with BER 0.19-0.44, and the
BER<0.40 rule overlaps with unwatermarked images. Likely cause: Phase 4C
`lambda_image=3.0` over-weighted imperceptibility.

## Agreed roadmap
1. ✅ Fix architecture (single layered image) + honest benchmark with false positives
2. **Next:** retrain HiDDeN in Colab — residual-style training, noise layers for
   perspective / colour shift / screenshot rescale+crop / real JPEG / combos,
   payload = product ID with error correction (BCH) so detection is blind and
   false-positive rate is controllable; add a BER/FPR regression test
3. Geometric alignment of suspect to original (SIFT/ORB) before DCT detection
4. Compare with pretrained robust watermarks (Watermark Anything, TrustMark, StegaStamp)
5. AI-edit attacks: SD img2img at several strengths, inpainting, background
   removal/replacement, AI upscaling; discuss regeneration attacks (Zhao et al. 2023)
6. Second defence layer: perceptual hash / DINOv2-CLIP fingerprinting; mention C2PA
7. Real screenshot / phone-photo dataset instead of only simulations

## Conventions
- Report numbers honestly, including failures — they go into the thesis discussion.
- Keep legacy code paths so earlier results stay reproducible.
- Don't commit large generated files (`results/`, `output/` are git-ignored).
- Work happens on branch `claude/keen-gauss-dwwse7` (PR #1).
