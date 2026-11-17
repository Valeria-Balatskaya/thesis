# Overwrite defences — summary (2026-11-17)

Threat: a thief takes our protected copy and re-protects it with a watermark of their own
(our model file, TrustMark, WAM, Video Seal), so a blind read shows their message, or nothing.
Victim: ours v3 at the app setting. Counts are of 40 photos (20 own + 20 ABO) unless noted;
"×" = thief strength relative to that system's default. Generated tables with Wilson
intervals: `output/thesis/overwrite_defences.md`, `overwrite_cost.md`, `baselines.md`.

| Defence | What it stops | Works up to (same-model thief) | False positives | Cost | Status |
|---|---|---|---|---|---|
| Upload pre-check (`guard.precheck`) | re-protecting a copy (or leaked original) **through our app** | any strength; 11/11 API scenarios | unrelated photos accepted (11 scenarios) | 1 blind read + retrieval per upload | app |
| Informed check (calibrated, z ≥ 4) | proves ownership when the blind read fails | 1x **40/40**, 1.5x 34, 2x 13, 3x 3 | 0/240 (max z 3.07); `watermark_overwritten` 0/520 | ~15 s per issued copy; needs alignment + original | app |
| DCT layer (`TRACEMARK_DCT_LAYER=1`) | second, luminance mark for the ownership check | v3+DCT 40 / 37 / 26 / 6 / 5 at 1/1.5/2/3/4x | 0/480 (max sim 2.8) | −2.6 dB PSNR, ΔE +0.05; lost under crop, print, native screenshot (0–3/40) | optional |
| v4 key-conditioned model | harder to overwrite (overwrite training) | same key 39 / 39 / 34 / 27 at 1/1.5/2/3x | 0/90 | needs training; normal robustness 98.9 → 96.3% (ABO 97.7 → 92.5); key does not separate | not deployed |
| **Peel-off** (`DefencePipeline(peelers=)`) | **our ID read again** after subtracting the foreign mark | blind 1x 14, 1.5x **35**, 2x **35**, 3x 26–32, 4x 6–14; pipeline up to 37 | **0/1,167** never-protected, **0/320** originals with only a foreign mark | only when a foreign mark is read: 6 scales x marks found, 2 rounds | app |
| **Order proof** (`defence.order_proof`) | **who marked first** (our copy vs the clean original) | positives 628/640 = 98.1% (1x and 2x, 4 thief systems) | **0/640** "ours first" when the thief marked a leaked clean original (max z 3.62) | ~8 s per test (25 decoys, 512 px) | app + report |
| Tiles 2x2 (`src/tiles.py`, no training) | whole-image thief with our model file | whole-image thief 60/60 at 1–3x, 59/60 at 4x (plain v3: 17 → 1/60); **tiled thief 27 / 7 / 4 / 3 / 2 of 60** | 0/1,000 | 17 blind attempts (vs 9); blind native screenshot 36 → 5/60; 50% crop 55/60 < 94% → **rule failed** | not adopted |

Notes
- Foreign industry marks never hide ours: TrustMark / WAM / Video Seal / TrustMark+WAM on top at 1–4x
  → ours read 33–40/40 blind before peeling, 39–40/40 after, pipeline 40/40.
- Informed check and peel-off cover complementary ranges: at 1x the thief's own ID is unreadable
  (8/40), so there is nothing to peel, but the informed check proves ownership (40/40); from 1.5x
  the thief's ID is readable (33–39/40) and peel-off restores ours (35/40 at 1.5–2x). A joint rate on
  identical thief images was not measured (different test scripts).
- Not solved: two layers of ours on top (peel-off 2 → 5/40); a thief at ≥ 4x (ΔE 4.2, clearly
  visible); a thief who knows the tiling (tiles); a thief running the model outside our server
  with another key (pre-check does not apply; v4 key does not separate).
- Other systems re-protected by themselves (A1): TrustMark, WAM, Video Seal keep their original
  message 0/40 at every strength; TrustMark and Video Seal are replaced at 1x for ΔE 0.22–0.57.
