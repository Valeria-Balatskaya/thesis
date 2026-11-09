# Thesis outline — structure and notes (no prose)

**Registered title:** *A System for Invisible Digital Watermark Embedding for Content Protection in
E-Commerce Systems.* Deadline: February 2027.

> **Objectives — check the wording.** The registered text of 1.2.1–1.2.5 and 1.4 is not in the
> repository. The mapping below is **inferred** from how the meeting notes tag results. Replace each
> line with the registered wording and tell me if any mapping is wrong.
>
> | Code | Inferred meaning | Where it is answered |
> |---|---|---|
> | 1.2.1 | Analyse e-commerce images and derive requirements; adapt the method to them | Ch. 3; v3 training data/options (Ch. 5) |
> | 1.2.2 | Invisibility / perceptual quality of the embedded mark | Ch. 6.2 (PSNR, SSIM, ΔE, background ΔE) |
> | 1.2.3 | Robustness to common and malicious manipulations | Ch. 6.3–6.6 |
> | 1.2.4 | Computational overhead / practicality | Ch. 6.8 |
> | 1.2.5 | Ownership verification, traceability, security against forgery and overwriting | Ch. 4.5, 6.5–6.7 |
> | 1.4 | Research methods: experimental evaluation, comparison, statistics | Ch. 6.1, 6.9, 6.10 |

Figures go in `thesis/figures/` (`bash scripts/reproduce_all.sh` regenerates them from `output/thesis/`).
Tables come from `output/thesis/*.md`. Every percentage needs n and a 95% Wilson interval (`src/stats.py`).

---

## 1 Introduction and aims
- Problem: product photos are the seller's work and capital. Copied by competitors, fake shops and
  resellers. Taken by right-click save, screenshot, phone photo, messenger forward, AI edit.
- Why visible watermarks / metadata are not enough: cropped, inpainted, stripped by every CDN.
- Goal: trace a found image back to the seller **and to the copy (channel) that leaked**.
- Aims and objectives: the table above, with the registered wording.
- Scope / non-goals: no legal standard of proof; no marketplace-scale search index; attacks are mostly
  simulated (real-capture dataset, Ch. 6.7).
- Contributions (bullets, to phrase yourself):
  1. e-commerce image analysis (ABO vs COCO vs own photos), turned into model requirements;
  2. HiDDeN-based model retrained for product photos (v3), plus a key-conditioned v4;
  3. Retrieve–Align–Verify, a pipeline around any blind decoder (shown with TrustMark/WAM too);
  4. ownership dispute support: calibrated informed check, `watermark_overwritten` verdict,
     signed registration log;
  5. a complete web application (TraceMark), and an honest benchmark with false-positive rates.
- Thesis structure paragraph.
- **Open:** confirm the contributions list with the supervisor.

## 2 Literature review
- 2.1 Terms: watermarking vs steganography vs fingerprinting. Blind / non-blind / informed detection.
  Robust vs fragile. Capacity–imperceptibility–robustness triangle. Cox et al. 2008 (book).
- 2.2 Spatial domain: LSB (van Schyndel 1994; Chan & Cheng 2004) → own negative result (`src/lsb.py`).
- 2.3 Transform domain:
  - DCT spread spectrum (Cox et al. 1997), own implementation `src/dct_watermark.py`;
  - DWT (Barni et al. 2001);
  - SVD (Liu & Tan 2002, and the ownership problem it caused, cf. Craver);
  - perceptual masking: NVF (Voloshynovskiy et al. 1999), amplitude modulation (Kutter 1998).
- 2.4 Deep learning:
  - HiDDeN (Zhu 2018), StegaStamp (Tancik 2020), TrustMark (Bui 2023), Watermark Anything (Sander 2025);
  - generative-model watermarks (Stable Signature, Tree-Ring) as context;
  - FiLM conditioning (Perez 2018) for v4.
- 2.5 Attacks:
  - signal processing, geometric, capture (screenshot, print–scan);
  - copy attack (Kutter 2000); ambiguity / ownership attacks (Craver 1997/1998);
  - regeneration attacks (Zhao 2024).
- 2.6 Content protection in e-commerce: reverse image search, perceptual hashing / fingerprinting
  (SIFT, Lowe 2004; RANSAC, Fischler & Bolles 1981); marketplace practice (CDN re-encoding,
  fixed sizes, Allegro API note in CLAUDE.md).
- 2.7 Provenance: C2PA (spec 2.2) hard and soft bindings; RFC 3161 timestamps; signatures (Ed25519).
- 2.8 Gap: no system combines a product-photo-specific invisible mark, copy-level tracing, geometric
  recovery with the seller's original, and dispute evidence.
- **Open:** do you need more DWT/SVD papers (e.g. hybrid DWT-SVD)? Your supervisor may expect 30–50
  references in total. `references.bib` has 36.

## 3 Analysis of e-commerce images and requirements (1.2.1)
- 3.1 Data: own 47 photos, ABO 60 (eval split, CC BY 4.0), COCO 60. Product-level split
  (`data_manifests/`, 0 shared products).
- 3.2 Measured properties table → docs/supervisor_questions.md §1a; charts
  `output/ecommerce_analysis/*.png` → `thesis/figures/ecommerce_*.png`.
- 3.3 Consequences for a watermark:
  - flat areas: visibility;
  - pure white: clipping;
  - square / tall crops: aspect retry;
  - JPEG q76;
  - CDN sizes: ≤ 128 px limit.
- 3.4 Threat model: copy, screenshot, phone photo, messenger, crop/rotate/mirror, banner overlay, AI
  background swap / regeneration, re-protection by a thief, copy attack, insider.
- 3.5 Requirements:
  - functional (protect, issue copies per channel, scan, monitor, evidence report);
  - non-functional:
    - invisibility (ΔE < 1 on background);
    - robustness;
    - FPR ≤ 1e-5 per read;
    - < 1 s embed;
    - ownership evidence.
- Table: requirement → design decision → where verified (extend §1c of supervisor_questions.md).
- **Open:** pick ΔE thresholds you can justify (≈1 JND; Sharma 2005 for the formula only).

## 4 System design
- 4.1 Architecture overview → `thesis/figures/uml_component.png`.
- 4.2 Use cases (seller, thief, investigator, admin) → `uml_use_case.png`.
- 4.3 Protect flow (pre-check → embed → register → sign) → `uml_activity_protect.png`.
- 4.4 Detection: Retrieve–Align–Verify + ownership → `uml_activity_detect.png`, `uml_sequence_scan.png`.
- 4.5 Payload design: 20-bit ID, 10-bit HMAC tag, BCH(63,30,6), XOR mask; FPR derivation 8.6e-6;
  candidate matching limit (~11 errors at 1e-6).
- 4.6 Model design:
  - v2: residual, fixed PSNR budget, carrier, matched filter;
  - v3: NVF mask, zero-white, renorm, chroma weight, levels, overwrite augmentation;
  - v4: key conditioning.
- 4.7 Ownership layer: informed check (null of 500 never-issued copies, z ≥ 4), DCT layer, signed
  hash-chained log, verdict set incl. `watermark_overwritten`, `watermark_unbound`.
- 4.8 Data model (SQLite tables) → optional class/ER diagram.
- **Open:** UML notation level your faculty expects (UML 2.5 strict, or informal?).

## 5 Implementation
- Tech stack: Python 3.14, PyTorch 2.13, OpenCV, FastAPI, SQLite; Colab T4 training.
- Training pipeline (`src/training/train_v2.py`, notebook RUN='v3'/'v4'), differentiable distortions,
  data mix, checkpoints; training curves (ask me to export them from the Colab logs).
- Inference API (`src/hidden_v2_inference.py`), aspect retry.
- Pipeline (`src/defence.py`), ownership (`InformedChecker`, `mark_overwritten`).
- Web app TraceMark (`app/`): screenshots of Protect, Scan, Attack Lab, report → take them yourself
  with `TRACEMARK_DATA=<demo folder>`, port 8765.
- Testing approach: standalone benchmark scripts, fixed seeds, `scripts/reproduce_all.sh`.
- **Open:** how much code to show (listings vs pseudo-code). Suggest pseudo-code for embedding, RAV and
  the informed check.

## 6 Experimental evaluation
- 6.1 Method:
  - data and the product-level split;
  - attack suite (23, `src/attack_suite.py`);
  - metrics (PSNR, SSIM, ΔE whole/background, exact ID, FPR);
  - Wilson intervals;
  - hardware (M1 Pro);
  - what is simulated.
- 6.2 Visibility (1.2.2): v2 vs v3 vs v4 table (`output/thesis/visibility.md`), signal maps
  `output/v2_vs_v3_signal.png`, parameter study `output/parameter_study_tradeoff.png`.
- 6.3 Robustness (1.2.3):
  - blind vs full pipeline per set and attack (`output/thesis/robustness.md`);
  - weak spots (screenshot blind, 50% crop, ≤ 128 px).
- 6.4 Alignment contribution: blind 0% on geometric → 97–100% (defence benchmark); same pipeline around
  TrustMark / WAM (`output/baselines/`).
- 6.5 Re-watermarking and overwriting (1.2.5):
  - other systems on top (`test_rewatermark`);
  - same model (`test_overwrite_cost`): thief damage ΔE vs ownership proven, chart `overwrite_cost.png`;
  - v4 other-key results;
  - `watermark_overwritten` verdict FPR (`test_overwritten_verdict`).
- 6.6 Security: copy attack + content binding; forged IDs (HMAC); leak tracing between copies (74/75).
- 6.7 Real captures: `output/real_world/` (after you collect them) vs the matching simulations.
- 6.8 Overhead (1.2.4): `test_overhead` tables, retrieval scaling chart.
- 6.9 Comparison with pretrained watermarks: main table + chart (`output/baselines/baselines.png`).
  Default and matched-ΔE settings.
- 6.10 Ablation: `output/thesis/ablation.md` (one component removed at a time).
- 6.11 Threats to validity:
  - 47 own photos = one session;
  - n = 30 per set;
  - simulated attacks;
  - informed check needs the original;
  - FPR measured on ≤ a few thousand images, the rest is theory.
- **Open:**
  - Do you report v2 at all in the main tables, or only as history? Suggest an appendix.
  - Final model choice (v3 or v4) after Part 1.

## 7 Conclusions and future work
- Answer each objective in one or two bullets with the key number and its CI.
- Limitations: as in 6.11, plus an attacker with their own model copy and key outside the server.
- Future work:
  - spatial redundancy (`docs/spatial_redundancy.md`);
  - global embedding index (DINOv2/CLIP + ANN);
  - C2PA soft binding;
  - reverse image search API;
  - full diffusion img2img attacks;
  - marketplace API integration (Allegro REST).

## Appendices
- A: the full per-attack tables.
- B: reproduction guide (README + `scripts/reproduce_all.sh`).
- C: the capture protocol (`docs/real_world_protocol.md`).
- D: UML sources (`thesis/uml/*.puml`).
