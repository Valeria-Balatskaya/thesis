# Research plan: from a strong thesis to a new result

Written 2026-11-24. Goal: turn the project from a well-built system into a result that adds
something new to the field, without new model training, before the thesis deadline (February 2027).

## 1. Where the field stands (literature check, November 2026)

| Work | What it shows | What it leaves open |
|---|---|---|
| Bulychev et al., *Watermarks Attack Watermarks: Re-Watermarking as a Generic Removal Strategy* (arXiv 2605.16796, 2026) | Re-watermarking an already watermarked image removes the original. Tested on 96 dataset × victim × attacker combinations, bit accuracy drops by 25–48%. No gradients, keys or surrogate models needed. | Attack paper only. **No defence proposed.** |
| *Yours or Mine? Overwriting Attacks Against Neural Audio Watermarking* (arXiv 2509.05835, 2025) | Overwriting lets an attacker hijack ownership of audio. | Audio only, no recovery of the original claim. |
| Dong et al., *Residual Transferability in Neural Image Watermarking* (arXiv 2609.32241, 2026) | Forgery: a watermark residual can be copied onto other images. Proposes CoverLock. | Forgery, not overwriting. |
| Wang et al., *ResGuard* (arXiv 2604.03693, 2026) | An attacker who holds original/watermarked pairs can strip the mark. Defence: make the residual depend on the host image. | Defends the embedder; says nothing about disputes. |
| Nemecek et al., *Authenticated Contradictions* (CVPR 2026 Workshops, best paper) | C2PA (Coalition for Content Provenance and Authenticity) manifests and watermarks are verified separately, so they can contradict each other. | Cross-layer audit; ordering is not addressed. |
| Li et al., *Image Watermarking With Memory Across Reissuance* (arXiv 2611.23363, 2026) | "Generational stratigraphy": a publisher re-issues its own images and recovers which generation each region belongs to. | Same publisher controls every layer; **no adversarial third party**. |
| *Watermarks and Fingerprints as Soft Bindings* benchmark (arXiv 2610.04151, 2026); WARP (arXiv 2609.40031); WAVES (2024) | Large simulated benchmarks; cropping to a quarter of the image sends TrustMark and Video Seal to chance. | Simulated attacks only; no product photos; no real captures. |
| TrustMark (ICCV 2025) | Removal and re-watermarking are **built in as a feature** (TrustMark-RM). | Ownership after re-marking is not considered. |
| Screen-shooting papers (NiMark, TIACam, …) | Real phone recaptures, but only for StegaStamp / HiDDeN-type models. | No TrustMark / WAM / Video Seal; no messengers; no product photos. |

**The gap.** Re-watermarking is now a documented, generic attack (2026), and the open industry
watermarks (Adobe TrustMark, Meta WAM, Meta Video Seal) allow it by design. Nobody has published a
way for the **rightful owner to get the claim back** after someone re-marks the image: show that the
first mark is still there, read it again, and show which mark came first.

TraceMark already does all three for its own watermark:

| Mechanism | Result |
|---|---|
| Informed check | 40/40 at 1× |
| Peel-off | 35–37/40 at 1.5–2× |
| Order proof | 628/640 correct, 0/640 false claims |

The step that makes this a field-level result: make it **independent of the watermarking system**,
then **attack it adaptively**.

> These works were found through search snippets; arxiv.org was not reachable from the cloud session.
> Read the full PDFs before citing them, and check nothing newer appeared. Search alerts:
> "re-watermarking defense", "watermark ownership dispute image", "watermark embedding order".

## 2. The claim

**Blind watermarks cannot prove ownership once open watermarking models exist: anyone can re-mark
an image invisibly. Owner-side forensics can restore the claim for any open watermarking system.**

Working title: *Stolen, Re-marked, Reclaimed: Owner-Side Forensics Against Re-Watermarking of Open
Image Watermarks* (alternative: *Who Marked First?*).

### Contributions

1. **Measurement: ownership hijack.** For all four systems (ours v3, TrustMark, WAM, Video Seal),
   re-marking by the same system or by another system transfers the readable claim to the thief, and
   costs little:

   | Thief | Original message kept | Visible cost (ΔE) |
   |---|---|---|
   | TrustMark over TrustMark | 0/40 | 0.57 (invisible) |
   | Video Seal over Video Seal | 0/40 | 0.22 (invisible) |

   This extends Bulychev et al. from removal to **hijack**, and shows the consequence for C2PA soft
   bindings: a lookup returns the thief's manifest.
2. **Defence: an owner-side forensic protocol that works with any watermarking system.** Its inputs:
   - the clean original O;
   - the issued copy C;
   - a signed, time-stamped registration;
   - the thief's public embedder.

   Its outputs:
   - (a) the **informed check**: is C's signal present in the suspect, calibrated against decoy copies;
   - (b) **peel-off**: re-create the thief's residual, subtract it, read again;
   - (c) the **order proof**: was the thief's mark applied to C (after us) or to O?

   Each output comes with a calibrated false-claim rate. A neutral third party can verify it: anyone
   can rerun the public embedders on O and C.
3. **Adaptive attacks on the defence** (section 4). This is what makes a security claim credible.
4. **System and benchmark.**
   - Retrieve–Align–Verify improves every watermark tested by 11–30 points.
   - Product photos are a harder domain: flat areas and pure white.
   - A real-capture benchmark: screenshots, phone photos and messengers, for all four systems.

### Why it works (one paragraph for the paper)

The thief only has C = O + r_ours. Their output is S = E_t(C, m), and every operation they can
apply is computed **from C**. Without O they cannot produce E_t(O, m). So, whatever the thief's
system, the projection of S − E_t(O, m) onto E_t(C, m) − E_t(O, m) is close to 1 when the thief
started from C, and close to 0 when they started from O.

It fails only when:
- the thief has O (a leak of the clean original, which is then not our copy, and is correctly not
  claimed);
- the thief removes our residual well enough to approximate O (tested as attack A4 below);
- the thief's embedder cannot be run (closed system; then only the informed check remains).

## 3. Experiments (no training; code already has `foreign_read` / `residual` for every baseline)

**E1 – Any system as the victim.** Today the defence protects only ours. Run it on all 4 × 4
combinations:
- victim ∈ {ours v3, TrustMark, WAM, Video Seal};
- thief ∈ {same system, each other system};
- strength 1 / 1.5 / 2 / 3×, clean and JPEG q70.

Report:
- victim message read blind;
- informed check (decoys = never-issued messages of the victim system);
- peel-off;
- order proof;
- false claims.

**E2 – Scale.** 300 victim photos (100 own/ABO/COCO each, or ABO from `data/abo_negatives`) and at
least 1,000 null cases for the false-claim rate, so the 95% upper bound is ≤ 0.3%.

**E3 – Leak attribution survives re-marking.** 5 issued copies per product. The thief re-marks
one of them. Does the order proof / informed check name the right copy (sales channel)? Report the
wrong-copy rate.

**E4 – Third-party verification.** Rerun the order proof from the saved (O, C, suspect, thief system,
message) on a clean machine. It must give the same result, which proves reproducibility.

## 4. Adaptive attacks on the defence (the credibility section)

| ID | Thief's move | Expected | What to report |
|---|---|---|---|
| A1 | post-process after re-marking: JPEG 50–90, resize 0.5–0.9, 5% crop, screenshot | handled by alignment | order-proof accuracy |
| A2 | random, unknown strength 0.5–4× | strength fitted | accuracy vs ΔE |
| A3 | closed / private watermark (simulate: retrained or unknown-seed model, embedder not given to us) | peel-off and order proof unavailable; informed check still works | informed rate |
| A4 | **estimated-original attack**: denoise or regenerate C to approximate O (median / BM3D-like denoiser, SD VAE, img2img 0.1–0.3), then re-mark | the main threat | ownership and false claims vs the thief's ΔE |
| A5 | stacked re-marking ×2, ×3 (same or mixed systems), with iterative peel-off | currently 2 → 5/40; try iterative peel | rate |
| A6 | fake original (Craver): thief subtracts a pattern from S and claims the result as their original | our signed RFC 3161 timestamp predates; the order proof shows their "original" contains our C signal | argument + test |
| A7 | thief marks the leaked **clean** original | must NOT claim ours (already 0/640) | false-claim rate at scale |

**Pre-registered success criteria.** Decide these before running, and report pass or fail.

| Criterion | Target |
|---|---|
| Ownership restored, thief ΔE ≤ 2 | ≥ 90% for each victim system |
| False claims | 95% Wilson upper bound ≤ 0.5% on ≥ 1,000 null cases |
| Wrong copy (E3) | 0 |
| A4 | report the ΔE at which the claim is lost, so the thief must visibly damage the image to win |

## 5. Real-world benchmark (with the capture protocol)

Add 10 photos each protected by TrustMark, WAM and Video Seal to `data/real_world`, next to the v3
set. Capture them with the same protocol: screenshots, phone photos, WhatsApp, Telegram. Score each
system with its own decoder, blind and with Retrieve–Align–Verify. To check against the literature:
the first real-capture comparison of open industry watermarks on product photos.

## 6. Timeline

| When | What |
|---|---|
| Nov 24 – Nov 30 | E1 pilot on 40 photos; full-text reading of the related work; fix the success criteria |
| Dec 1 – Dec 7 | E2 at scale + E3; adaptive attacks A1, A2, A4, A7 |
| Dec 8 – Dec 14 | A3, A5, A6, E4; real captures (you) incl. 4 systems |
| Dec 15 – Jan 31 | Thesis chapters (this becomes the core of chapters 6.5 and 7); paper draft in parallel |
| Feb 2027 | Defence. Then: arXiv preprint + submission (with supervisor, after P&G / patent check) |

## 7. How this changes the thesis

The thesis story changes from:

> "I built a watermarking system and it works well"

to:

> "Open watermarks can be hijacked invisibly. I show that for four systems and give the owner a
> calibrated way to win the dispute."

Chapter 6 gets one headline table: victim × thief × defence. Chapter 7 states the limit honestly:
- a thief with the clean original, or with a closed system, can only be answered by the informed
  check and the signed timestamp.
