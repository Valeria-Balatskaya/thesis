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

---

# Meeting 2 — v3 preparation, texture-only embedding, overwrite defences

Measured 2026-10-25/26 with the existing `checkpoints/hidden_v2_best.pt` (no retraining;
v3 is trained in Colab separately). App setting = strength 0.7 + flat mask 0.3.
Each result is tagged with the registered objective it serves (1.2.1–1.2.5, 1.4).

## Request 1 — use the e-commerce analysis to improve the model (v3 prep, 1.2.1)

**Training data (A1).**
- v3 trains on COCO plus **2,500 Amazon (ABO) product photos**, half and half per batch.
- The ABO split is made **at product level** using the ABO listings metadata: any product with an
  image in the evaluation set is excluded from training together with all its other views.
  Result: **0 shared images and 0 shared products**. (The 60 evaluation images belong to 90 products, because variants share photos.)
- No single category may exceed 10% of the training set (ABO is about 30% phone cases): 274 categories.
- The exact images are listed in `data_manifests/abo_train.csv`, `abo_eval.csv` and `coco_eval.csv`, so
  the split is fixed and can be audited. COCO evaluation images are from test2017; training uses val2017.

**New training distortions (A2).**
- `levels`: brightness, black point, white point, so a white background clips at 255. This was the weakest attack on ABO.
- **Overwrite augmentation:** in 15% of steps someone re-protects the image with the same model and a
  random message, and the decoder must still return the *first* message.

**Embedding-side fixes (A3).**
- *White headroom* (255 → 253 before embedding). With v2 at the app setting it raised ABO blind
  reads from 81.9% to 87.8% (full pipeline 91.4% → 94.2%), but the 2-level shift of the
  background itself raises background ΔE from 0.62 to 0.93. v3 embeds nothing on pure white,
  so headroom is redundant there and is **not** used in the v3 run.
- *Aspect ratios 6:7 and 1:2* added to the blind read's square-crop retry:
  - Gain: square-crop reads on ABO 51/60 → 53/60.
  - Cost: 0/1,667 false positives on clean images with either list (95% upper bound 0.18%). The theoretical
    worst case per image rises from 6.0 × 10⁻⁵ to 7.7 × 10⁻⁵ (+29%).
  - Remaining misses: very wide banners (2.8–3.7:1) and photos where the watermark is weak anyway.

**Notebook (A5).** `RUN='v3'` uses colour charge 8, the NVF texture mask (floor 0.1, nothing on pure white,
energy moved into texture), the levels distortion, overwrite augmentation and the COCO+ABO mix.
Data is cached in Google Drive, with a checkpoint every 2,000 steps and resume. Step-by-step guide:
[docs/colab_v3.md](colab_v3.md). Estimated time: about 5.5–6 h on a free T4.

## Request 2 — embed only where there is texture (1.2.2, 1.2.3, 1.4)

Implemented two masks: the existing local-variance mask ("flat"), and the **noise visibility function** (NVF,
Voloshynovskiy et al. 1999): NVF = 1 / (1 + θσ²), θ = D/σ²max. The amplitude is `floor` on flat areas and
full in texture. Optionally nothing at all on pure white, and optionally the removed energy is moved into texture.

`tests/test_texture_masks.py`, existing v2 model (mask applied at embedding only), strength 0.7,
30 own + 30 ABO photos × 12 attacks:

| Mask | ABO PSNR | ABO ΔE / ΔE on background | ABO blind / full pipeline | Own blind / full |
|---|---|---|---|---|
| none | 43.7 | 1.70 / 1.67 | 93.9% / 97.2% | 100% / 100% |
| flat 0.3 (current app) | 46.3 | 1.17 / 0.62 | 81.9% / 91.4% | 100% / 100% |
| flat 0.1 | 46.9 | 1.00 / 0.26 | 71.1% / 85.8% | 95.8% / 99.7% |
| flat 0.0 | 47.1 | 0.95 / 0.15 | 67.5% / 82.5% | 79.2% / 97.2% |
| NVF 0.3 | 47.7 | 0.97 / 0.47 | 75.3% / 88.1% | 98.6% / 100% |
| NVF 0.1 | 49.0 | 0.74 / 0.07 | 53.9% / 69.7% | 37.8% / 74.4% |
| NVF 0.0 (texture only) | 49.3 | 0.68 / 0.005 | 44.2% / 63.1% | 10.6% / 39.7% |
| NVF 0.0 + nothing on white | 49.9 | 0.63 / 0.002 | 42.2% / 61.1% | 11.1% / 39.4% |
| **NVF 0.1 + energy into texture** | 43.4 | 1.35 / **0.29** | **82.8% / 92.8%** | 100% / 100% |
| flat 0.3 + white headroom | 42.9 | 1.37 / 0.93 | 87.8% / 94.2% | 100% / 100% |

- **Texture-only embedding does remove the watermark from the background:** ΔE on flat areas drops to about 0.005.
- **With a model not trained for the mask, robustness collapses:** 63% on ABO, 40% on own photos. Own
  photos suffer more because smooth phones and boxes have almost no texture.
- **NVF 0.1 with energy moved into texture keeps today's robustness** (92.8% vs 91.4%) while halving the
  background change (ΔE 0.29 vs 0.62). The price is more change in textured areas (PSNR 43.4).
  This is the setting v3 is trained with, so the network can learn to work inside the mask.
- **Which photos fail** at the current setting: all 9 failing photos are ABO. The worst three are **small products on a
  large pure-white background** (8–31% product, 68–91% pure white): 4–7 of 12 attacks traced. Own photos never fail.

## Request 3 — defences against overwriting, beyond the upload pre-check (1.2.3, 1.2.5)

**B1. Second, secret-keyed layer in luminance** (`src/dct_layer.py`).
- How it works: Cox spread spectrum on the 1,000 largest DCT coefficients of brightness, with a carrier made from HMAC(server key, copy ID). It is added equally to R, G and B, so v2's colour channel is untouched.
- Test: `tests/test_overwrite_defences.py`, 20 own + 20 ABO photos.

| | PSNR | SSIM | ΔE / background | v2 read, ABO (blind / full) |
|---|---|---|---|---|
| v2 only | 46.97 | 0.9978 | 1.15 / 0.75 | 188 / 215 of 240 |
| v2, then DCT layer | 44.40 | 0.9953 | 1.20 / 0.78 | 183 / 213 of 240 |
| DCT layer, then v2 | 44.35 | 0.9952 | 1.25 / 0.87 | 196 / 218 of 240 |

The layer costs about 2.6 dB PSNR, but almost no visible change (ΔE +0.05). It doesn't hurt v2; own photos stay at 240/240.

When a thief re-protects our published image with v2 (40 photos, v2 + DCT layer):

| Thief | Our v2 read blind | Calibrated colour check | DCT layer | Ownership proven |
|---|---|---|---|---|
| v2 same strength | 0/40 | 32–34/40 | 38–39/40 | **39–40/40** |
| v2 full strength | 0/40 | 0/40 | 34–37/40 | **34–37/40** (v2 alone: 0/40) |
| v2 twice | 0/40 | 4–5/40 | 36–38/40 | **36–38/40** (v2 alone: 5/40) |

(clean and after JPEG q70)

- **Weakness:** the DCT layer is lost under crop 10% (0/40), print-and-photograph (0/40), native screenshot
  (0/40) and the Instagram filter (2–3/40). It is partly lost under brightness +20% (about 50%) and square crop (about 60%).
  Full-image DCT detection is very sensitive to small misalignment and non-linear tone curves.
- **The layers complement each other:** v2 with alignment covers geometry and tone, the DCT layer covers overwriting.
- **Order:** "v2, then DCT" gives the DCT layer larger margins (median similarity 29 vs 19 on clean copies). "DCT, then v2" keeps
  v2 slightly stronger on ABO.
- **In the app:** opt-in with `TRACEMARK_DCT_LAYER=1`. When a scan returns "same picture, watermark unreadable", the
  ownership check runs and the evidence report shows it. Tested: a copy overwritten by a full-strength v2 mark returned
  "same picture", with ownership **present** (DCT similarity 11.1, v2 informed z 2.0).

**B2. Calibrated informed detection.**
- The 8-decoy comparison is replaced by a null distribution of **500 never-issued copies** per photo, giving a z-score
  and empirical p-value. Rule: z ≥ 4.
- **False positives:** 0/240 on images without the tested copy (clean original, a sibling copy with
  another ID, a thief-only image; clean and JPEG). The highest null z was 3.07. The DCT layer had 0/480, highest
  similarity 2.8 against a threshold of 6.
- Re-running `test_rewatermark.py` with the calibrated check is stricter than the 8-decoy version:

| Second watermark on top of ours | 8 decoys (meeting 1) | Calibrated (z ≥ 4) |
|---|---|---|
| Other systems, visible marks, stacked | 40/40 | 40/40 (median z ≈ 7.5) |
| Our v2 again, same strength | 40/40 | 33–34/40 |
| Our v2 again, full strength | 30–32/40 | **0/40** |
| Our v2 two or three more times | 38–40/40 | 0–5/40 |

The earlier "our signal is still underneath" claim holds for marks from other systems and for a same-strength
overwrite. It does **not** hold for a full-strength or repeated overwrite. That is why the DCT layer matters.

**B3. Tamper-evident registration log** (`app/registry_log.py`).
- On every issued copy the log stores the SHA-256 of the original and the copy, the copy, product and seller IDs, and the UTC time. Each record is
  chained to the previous record's hash and signed with an Ed25519 server key. There is an optional RFC 3161 timestamp from an
  external authority (tested with freetsa.org).
- Verification catches a changed record ("contents changed after signing"), a deleted record ("sequence
  gap") and a back-dated record.
- The evidence report shows the record, both file hashes, the signature, the chain and the
  timestamp check. The public key is at `/api/registry/public-key`.
- **Limit:** deleting only the *newest* record can't be seen
  from the chain itself. The external timestamps, or publishing the newest record hash, cover that.

**B4. Copy attack** (Kutter et al. 2000; `tests/test_copy_attack.py`, 40 photos):
the attacker estimates our watermark from a protected copy and adds it to a *different* image, to frame
an innocent seller or fake provenance.

| Residual estimate added to 40 unrelated photos | Our decoder reads the victim's ID | Framed (verdict "watermark") | Flagged "unbound" |
|---|---|---|---|
| exact residual (needs the leaked clean original), ×1 / ×2 | 39/40 / 40/40 | **0/40** | 39/40 / 40/40 |
| median-filter estimate, ×1 / ×2 | 0/40 / 0/40 | 0/40 | 0 |
| high-pass estimate, ×1 / ×2 | 0/40 / 1/40 | 0/40 | 1/40 / 5/40 |

- **The watermark itself can be transplanted** if the attacker has the exact residual: the blind decoder reads it 39–40/40.
  Simple filter estimates don't extract v2's low-frequency colour signal.
- **Defence implemented: content binding** (`DefencePipeline.analyse(verify_content=True)`). After any read,
  blind or aligned, the suspect must show the registered picture of that ID (SIFT/RANSAC plus a gradient structure check,
  ignoring black padding). Transplants become "watermark_unbound": **0 of 240 framed**.
  - The first version checked only blind reads, and 5 of 80 high-pass fakes were framed through the alignment path, because the
    high-pass estimate also carries ghost edges of the victim's photo. Now both paths are checked.
- **Cost:** 1 of 480 genuine attacked copies is flagged (a square crop).
- **Image-bound HMAC tag (the requested design), not implemented.** Binding the tag to a perceptual hash of the
  *registered* original doesn't stop a transplant: the verifier would recompute the tag from the stored hash, and the
  transplanted bits still match. Binding it to the *suspect's* hash breaks genuine reads: a 64-bit pHash
  matched exactly in only 195 of 480 attacked genuine copies (crop 0/40, native screenshot 0/40, square 4/40).
  So 285/480 genuine reads would fail. Content binding gives the same protection without changing the payload.

**B5. Out of scope, worth discussing in the thesis.**
- *Key-conditioned models:* the encoder and decoder take a secret key as an extra input, so each key writes
  and reads its own channel. A thief running the public model with another key could not overwrite our
  channel for *our* decoder, and our decoder could not be fooled by theirs. This needs retraining and key
  management. It is the principled fix for "someone with the same model re-protects the image", which today we
  handle only inside TraceMark (pre-check) and with the second layer.
- *C2PA Content Credentials:* a signed provenance manifest (creator, time, edits) attached to the file, with
  an invisible watermark as a "soft binding" that finds the manifest again after metadata is stripped.
  Marketplaces and camera vendors are adopting it. Our registration log is a server-side version of the
  same idea. Combining the two (a TraceMark ID as the soft binding of a C2PA manifest) would make the evidence
  verifiable by third parties without trusting our server.

## Registered objectives not covered before (1.2.4, 1.4)

**C1. Computational overhead** (`tests/test_overhead.py`):
Apple M1 Pro (10 cores) laptop, median of 5 runs after a warm-up, nothing else running.

| Per image | 512 px CPU | 1024 px CPU | 2048 px CPU | 512 px MPS | 1024 px MPS | 2048 px MPS |
|---|---|---|---|---|---|---|
| Embed (incl. PNG write) | 0.04 s | 0.12 s | 0.52 s | 0.02 s | 0.06 s | 0.21 s |
| Blind read, watermark found | 0.02 s | 0.04 s | 0.07 s | 0.01 s | 0.02 s | 0.07 s |
| Blind read, nothing found (+8 aspect retries) | 0.19 s | 0.22 s | 0.28 s | 0.04 s | 0.06 s | 0.16 s |
| Full pipeline on a rotated copy (20 products) | 0.26 s | 0.46 s | 0.68 s | 0.10 s | 0.30 s | 0.59 s |

| Ownership tools (CPU, 1024 px) | Time |
|---|---|
| DCT layer embed / detect (500 null carriers) | 0.09 s / 0.10 s |
| Calibrated informed check: build (500 decoys) / test | 16 s once per original / 0.01 s |

| Registered products | 10 | 50 | 100 | 250 | 500 | 1,000 |
|---|---|---|---|---|---|---|
| Full pipeline on a rotated copy (CPU) | 0.43 s | 0.57 s | 0.82 s | 1.44 s | 2.89 s | 5.39 s |

Chart: `output/overhead_retrieval.png`.

- **Embedding and blind reading cost well under a second, even at 2048 px on a laptop CPU.** That is negligible next to a
  product upload. The Apple GPU (MPS) is 2–5× faster for the network parts.
- **The bottleneck is retrieval:** SIFT matching against every registered original grows linearly, about 5 ms per
  product (5.4 s at 1,000 products). Registering a product costs about 50 ms. Fine for a shop's catalogue, but a
  marketplace with millions of products needs a global embedding index (DINOv2/CLIP plus approximate nearest neighbours) to
  pick the top-k candidates first. This is still on the roadmap.
- The informed check's 16 s build is a one-off per original and is only needed in a dispute.

**C2. Parameter study on ABO** (`tests/test_parameter_study.py`):
30 ABO photos × 12 attacks per setting. 0/30 false positives on the clean originals (blind and pipeline).

| Mask | Strength | PSNR | SSIM | ΔE / background | Blind | Full pipeline |
|---|---|---|---|---|---|---|
| none | 0.3 | 50.6 | 0.9969 | 0.70 / 0.70 | 66.7% | 80.6% |
| none | 0.5 | 46.5 | 0.9938 | 1.22 / 1.21 | 87.8% | 93.6% |
| none | 0.7 | 43.7 | 0.9907 | 1.70 / 1.68 | 93.6% | 97.5% |
| none | 1.0 | 40.6 | 0.9861 | 2.40 / 2.36 | 95.8% | 98.3% |
| flat 0.3 | 0.3 | 53.3 | 0.9990 | 0.43 / 0.17 | 35.8% | 56.4% |
| flat 0.3 | 0.5 | 49.1 | 0.9977 | 0.82 / 0.40 | 70.0% | 82.5% |
| **flat 0.3** | **0.7 (app)** | 46.3 | 0.9962 | 1.18 / 0.62 | 82.2% | 91.4% |
| flat 0.3 | 1.0 | 43.3 | 0.9937 | 1.69 / 0.94 | 89.2% | 96.7% |
| flat 0.0 | 0.3 | 54.0 | 0.9994 | 0.37 / 0.03 | 22.5% | 37.8% |
| flat 0.0 | 0.5 | 49.9 | 0.9987 | 0.67 / 0.09 | 50.8% | 72.8% |
| flat 0.0 | 0.7 | 47.1 | 0.9979 | 0.95 / 0.16 | 67.5% | 83.1% |
| flat 0.0 | 1.0 | 44.1 | 0.9964 | 1.36 / 0.26 | 77.5% | 90.3% |

Chart: `output/parameter_study_tradeoff.png` (read rate against ΔE and against PSNR).

- **Strength is the main dial:** each step up costs about 3 dB PSNR and gains robustness, with diminishing returns
  above 0.7.
- **At the same overall visibility, masking does not buy robustness.** At ΔE ≈ 1.2, no mask at 0.5 reads 93.6%
  and flat 0.3 at 0.7 reads 91.4%. What masking changes is *where* the change sits: the background ΔE is 0.62
  instead of 1.21, which matters for white product backgrounds.
- **The full pipeline adds 5–20 points over the blind read**, and the most when the watermark is weak.
- **PSNR hides where the change is:** flat 0.0 at 1.0 and no mask at 0.7 differ by only 0.4 dB PSNR (44.1 vs 43.7),
  yet the background change differs six-fold (ΔE 0.26 vs 1.68). Visibility must be judged with ΔE on flat
  areas, as in meeting 1.
