# Design note: spatial redundancy (tiles with independent keys)

Status: **future work, not implemented.** Part 2.3 of the final phase. Objectives 1.2.3 (robustness),
1.2.5 (ownership / security).

## Why

Two weaknesses remain after v3/v4 and Retrieve–Align–Verify:

1. **Large crops.** The model sees the whole image squeezed to 128×128 px
   (`HiDDeNv2.watermark` resizes, embeds, and upsamples the residual back). The 63 bits are
   spread over the entire frame, so a crop that keeps 25% of the area (`offcentre_crop_50`)
   keeps only part of every bit's carrier. Measured: v2 29/90, v3 80/90 (full pipeline).
2. **Overwriting.** One mark covers the whole image, so a thief only has to beat it once,
   everywhere at the same time (`tests/test_overwrite_cost.py`).

## Idea

Split the image into a grid of K tiles, for example 2×2 or 3×3. Each tile gets its **own
complete 63-bit codeword**: the same copy ID, but a different key per tile,
`key_t = HMAC(server_key, "tile", t)`. Use the v4 key conditioning for this, or for v2/v3 a
per-tile XOR mask on the payload. Reading:

- **Blind read:** decode each tile on its own. Accept the copy ID if at least one tile passes
  BCH and HMAC. Better: add the tiles' soft logits for the same ID before BCH (soft
  combining). This lowers the bit error rate when every tile is weak.
- **Aligned read:** after alignment the tile grid is known exactly. So every tile that is at
  least partly covered can be read, and tiles filled from the clean original carry nothing.

## Expected trade-offs (estimates, to be measured)

| Aspect | Expected effect | Why |
|---|---|---|
| Large crops | **Better**: one surviving tile is enough | 50% crop keeps ≥ 1 of 4 tiles in a 2×2 grid almost always |
| Overwrite cost | **Higher**: the thief must beat every tile; with per-tile keys a v4 thief can't target one channel | Damage adds up over all tiles |
| Per-tile robustness | **Worse**: each tile is embedded at 128 px from a smaller area, so it has 1/K of the pixels per bit | JPEG, noise and resizing hit a single tile harder; soft combining wins some of this back |
| Small thumbnails | **Worse**: a 200 px search-grid image gives 100 px per tile in 2×2 | Already the limit today (≤ 128 px lost, `tests/test_in_the_wild.py`) |
| Blind false positives | **Worse by up to K×** if any tile may decide | 8.6×10⁻⁶ per read → ≤ 3.4×10⁻⁵ (2×2) or 7.7×10⁻⁵ (3×3), before the 8 aspect retries. Requiring 2 agreeing tiles makes it negligible but loses the single-tile crop benefit |
| Visibility | **Risk of seams** at tile borders | Overlapping tiles with a raised-cosine blend avoid hard edges. ΔE stays the same if the PSNR budget is the same per tile |
| Compute | K forward passes (embed 0.12 s → about 0.5 s at 1024 px, 2×2) | Still small next to an upload (`tests/test_overhead.py`) |
| Training | Not needed for a first test with v3. Needed for per-tile keys (v4 is already key-conditioned) | Tiles are just smaller images to the network |

## Cheapest experiment, if time allows (≈ 1 day, no training)

1. Embed with v3 per tile (2×2, 10% overlap blended), same ID, payload XOR-masked per tile.
2. Run the 23-attack suite of `tests/test_v3_vs_v2.py` on 30 ABO + 30 own photos.
3. Read each tile separately, and also all tiles soft-combined.
4. Compare with whole-image v3: crop 50%, JPEG q30, CDN 360 px, 200 px grid, and ΔE.
5. Count clean-image false positives with the any-tile rule on 1,000 ABO negatives
   (`data/abo_negatives`).

Decision rule: worth adopting if crop 50% improves clearly (for example ≥ 85/90), JPEG q30
and CDN 360 do not drop by more than 3 points, and false positives stay 0/1,000.

## Related work to cite

- Kutter, Jordan & Bossen (1998), *Digital watermarking of color images using amplitude
  modulation*, J. Electronic Imaging 7(2): repeating the mark and estimating it locally.
- Tiling with independent seeds is the common answer to cropping in classical spread
  spectrum (Cox et al. 2008, *Digital Watermarking and Steganography*, 2nd ed.,
  discussion of geometric distortions).
- Watermark Anything (Sander et al., 2024/2025) predicts a message per pixel. That is the
  extreme version of this idea: it can read several local messages and survives splicing.
