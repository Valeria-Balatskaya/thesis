# Answers to the supervisor's questions (meeting 1)

Measured 2026-10-20 with the existing `checkpoints/hidden_v2_best.pt` (no retraining).
App setting = strength 0.7 + flat-area mask 0.3 (what TraceMark uses).

---

## Question 1 — Do e-commerce images differ from regular images, and is the system built for them?

### 1a. How they differ (measured)

`scripts/analyse_ecommerce_images.py` on three sets. Every pixel measure is computed at 1024 px.
- **Own:** 47 own product photos (iPhone, one session).
- **ABO:** 60 Amazon listing photos from Amazon Berkeley Objects.
- **COCO:** 60 regular photos from COCO test2017, the same kind of photos HiDDeN was trained on.

| Median (n) | Own (47) | ABO (60) | COCO (60) | What it means for a watermark |
|---|---|---|---|---|
| Flat area (no texture) | 66% | 50% | 5% | Nowhere to hide the signal; a change there is the first thing the eye sees |
| Plain background (border colour) | 36% | 48% | 10% | Same |
| **Pure-white pixels (#FFFFFF)** | **0%** | **37%** | 0% | An added signal is **clipped** at 255: it can only darken, so half of it is lost |
| Texture / edge density | 4.4 | 8.4 | 16.8 | Product photos have 2–4× less texture to mask the watermark |
| Colour saturation | 0.09 | 0.11 | 0.31 | Product photos are close to grey/white; a colour tint stands out |
| Resolution | 0.8 MP | 2.9 MP | 0.3 MP | Marketplaces store large images, then serve resized copies |
| Square images | 0% | 40% (mean) | 7% (mean) | Marketplaces crop or pad to square; a third of ABO is tall portrait (0.85 = 1879×2200) |
| JPEG quality | n/a (PNG) | 76 (every file) | 96 | Marketplaces re-encode everything at fixed quality |

Charts: `output/ecommerce_analysis/flat_background_texture.png`, `colour_white_resolution.png`,
`aspect_ratio.png`.

**Conclusion:** product photos are mostly plain background with little texture and little colour. Real
marketplace photos are additionally pure white, square or tall, and re-compressed. Each of these
makes an invisible watermark both more visible and less robust than on regular photos.

### 1b. Does the watermark behave differently on them? (`tests/test_ecommerce_vs_regular.py`)

30 images per set × 12 attacks (360 attacked images per row). 0 false positives on clean originals in every set.

| Set | Setting | PSNR | ΔE mean / flat | Blind read | Full pipeline |
|---|---|---|---|---|---|
| Own products | app 0.7/0.3 | 47.3 dB | 1.24 / 0.86 | 100% | 100% |
| **ABO products** | app 0.7/0.3 | 46.2 dB | 1.18 / 0.62 | **81.7%** | **91.4%** |
| COCO regular | app 0.7/0.3 | 44.4 dB | 1.63 / 0.78 | 95.8% | 98.6% |
| Own products | full 1.0 | 40.0 dB | 3.27 / 3.19 | 100% | — |
| ABO products | full 1.0 | 40.6 dB | 2.40 / 2.36 | 95.8% | — |
| COCO regular | full 1.0 | 40.0 dB | 2.80 / 2.76 | 98.9% | — |

- **Real marketplace photos are the hardest case.** At the app setting, ABO fails most on native screenshot (17/30),
  brightness +20% (18/30; a white background clips at 255), Instagram filter (21/30) and JPEG q30 (22/30).
- Failures rise with flat area (Spearman ρ = 0.51, p = 0.004) and fall with texture (ρ = −0.41,
  p = 0.026). Photos that are at least 40% white failed 24% of blind reads; the others failed 13%.
- **Our own 47 photos overestimate robustness.** Their backdrop is light grey, not pure white, so they never
  hit the clipping problem. Thesis results should report ABO alongside them.
- The flat-area mask does what it should for visibility: on flat areas ΔE is 0.62 (ABO), below the
  "just noticeable" level of ~1.

### 1c. Where the design is already e-commerce-specific

| E-commerce property | Design decision | Checked? |
|---|---|---|
| Seller always keeps the original | Retrieve–Align–Verify uses it to undo crops, rotation, screenshots | Yes: ABO 81.7% → 91.4% |
| Large flat / white backgrounds | Flat-area mask; ΔE (not only PSNR) as the visibility metric | Yes: ΔE flat 0.62–0.86 |
| Marketplace square crop | Aspect-ratio retry in `read_product` (4:3, 3:2, 16:9 and their portraits) | Partly: all 5 ABO misses were tall photos (0.85, 0.52, 0.32) outside that list; alignment recovered 5/5 |
| CDN resizing / thumbnails | `tests/test_in_the_wild.py` (≥ 360 px 12/12, ≤ 128 px lost) | Yes, a known limit |
| Re-encoding at fixed quality | JPEG and WebP in training and in tests | Yes |
| Resellers / leaks | One watermark ID per copy (leak tracing) | Yes: 74/75 |
| Theft scenarios | Attack suite: square crop, CDN, messenger, banners, page screenshot, AI background | Yes |

### 1d. Where it is NOT yet tailored (proposals, not implemented)

1. **The model is trained on COCO, not product photos.** Proposal: train v3 on a mix of COCO and ABO product
   photos, with pure-white backgrounds, so the network learns to hide its signal in the product itself.
2. **Pure-white clipping.** Proposal (no retraining): before embedding, lower pure-white pixels by
   1–2 levels (255 → 253, invisible), so the watermark can push both up and down.
3. **Tall Amazon formats.** Add 6:7 (0.85) and 1:2 to `ASPECT_RATIOS`. Cost: each extra ratio
   adds one blind-read attempt to the false-positive budget.
4. **Brightness on white backgrounds** clips the signal. Proposal: add brightness/levels changes to the
   v3 training distortions.
5. **Evaluation set.** Report ABO (real listings) in the thesis, not only the 47 own photos.

---

## Question 2 — What happens if someone puts another watermark on top of ours?

`tests/test_rewatermark.py`: 40 product photos (20 own + 20 ABO), protected with our watermark at the app
setting, then a second watermark is added. Each case is read clean and after JPEG q70.
- **Baseline:** 39/40 are read blind before anything is added. One busy ABO photo fails, and the full pipeline recovers it.
- **Informed check:** the test also asks whether our exact signal is still there, by comparing against 8 decoy copies that were never issued.

| Second watermark | PSNR vs ours | Our ID, blind (clean / JPEG) | Our ID, full pipeline | Thief's ID read | Informed: ours found |
|---|---|---|---|---|---|
| DCT (Cox) | 41.9 dB | 39 / 39 | 40 / 40 | never | 40/40 |
| LSB | 92.5 dB | 39 / 37 | 40 / 39 | never | 40/40 |
| HiDDeN v1 | 41.1 dB | 39 / 39 | 39 / 39 | never | 40/40 |
| DCT + v1 (+ LSB) stacked | 37.4 dB | 39 / 39 | 39 / 39 | never | 40/40 |
| Small visible "© thief-shop" | 31.4 dB | 39 / 37 | 40 / 39 | never | 40/40 |
| Large tiled visible watermark | 25.1 dB | 39 / 36 | 40 / 38 | never | 40/40 |
| **Our system again** (thief re-protects, same strength) | 46.7 dB | **0 / 0** | **0 / 0** | 28–37 of 40 | **40/40** |
| Our system again, full strength | 40.3 dB | 0 / 0 | 0 / 0 | 40/40 | 30–32 of 40 |
| Our system 2–3 more times | 42–43 dB | 0 / 0 | 0 / 0 | 6–37 of 40 | 38–40 of 40 |

Negative control: the non-TraceMark watermarks applied to **un-watermarked** originals never read as one of our IDs (0/40 in every case).

**What this means:**
- **A watermark from another system does not remove ours.** It acts like a small amount of noise, which the BCH
  error correction absorbs. Visible logos and banners leave the hidden signal readable underneath.
- **A second watermark from our own system overwrites ours for the blind decoder.** The thief's v2 encoder
  is conditioned on the image, so it makes the decoder output *its* message. Our signal is still physically present,
  as the informed check finds it in 40/40 cases at the same strength. But at full strength the informed check drops to 30–32 of 40.
- **Fix, now implemented:** `guard.protect()` runs the detection pipeline on every upload *before*
  watermarking (`guard.precheck`). It refuses (HTTP 409, with a clear message) any image that carries one of our copy IDs, that
  matches a registered original, or that carries a valid TraceMark ID not in the catalogue. Tested
  through the API, in a separate data folder, on 11 scenarios:
  - **Refused:** the protected copy; that copy after JPEG, native screenshot, 8° rotation and promo overlay; the leaked clean original; the same seller uploading it again; a foreign valid ID.
  - **Accepted:** unrelated photos.

### Ownership: who is the rightful owner?

This is the "rightful ownership" problem of Craver, Memon, Yeo & Yeung: *Can invisible watermarks
resolve rightful ownerships?* (Proc. SPIE 3022, 1997), extended as *Resolving rightful ownerships with
invisible watermarking techniques* (IEEE J. Selected Areas in Communications 16(4), 1998): a thief adds their own mark and claims the image. In our
system ownership can be proven:

1. **The thief cannot forge our IDs.** Every ID carries a 10-bit HMAC tag made with our secret key. Without the
   key, a forged ID passes the check with probability 2⁻¹⁰, and with BCH the per-image false-positive rate is about 9 × 10⁻⁶.
2. **We hold the earlier, clean original** with a registration timestamp. The thief's "original" is
   our published copy: it matches our original under SIFT/RANSAC, and it still contains our exact embedded
   signal (informed check 40/40 at the same strength). Our clean original contains nothing of theirs.
   This asymmetry is the standard way to resolve the dispute.
3. **The pre-check stops the attack inside TraceMark itself**, because a protected image cannot be registered twice.

**Limits:**
- The informed check is supporting evidence. It compares against 8 decoy copies, not a calibrated threshold.
- It needs the suspect to be aligned with the original.
- It weakens when a thief overwrites at full strength and then re-compresses (30/40).
- An attacker who runs their *own* copy of our model, with a different key, outside our server is not
  stopped by the pre-check. Then only points 1 and 2 apply.
