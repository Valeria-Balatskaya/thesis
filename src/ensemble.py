# src/ensemble.py
# Ensemble watermarking: embed both DCT and HiDDeN into the same image.
# At detection time, run both decoders and use whichever gives higher confidence.
# This covers each method's weaknesses:
#   DCT survives JPEG, noise, mild resize — but dies on brightness/crop
#   HiDDeN survives brightness/crop/filters — but is quality-lossy and scale-limited
# Together they cover the union of attack classes.

from pathlib import Path

import numpy as np
from PIL import Image

from src.dct_watermark import embed as dct_embed, detect as dct_detect
from src.hidden_inference import HiddenModel, _seller_to_bits, _ber

DCT_ALPHA = 0.02
DCT_N_COEFFS = 500
DCT_THRESHOLD = 6.0

# HiDDeN identification confidence threshold: we consider a HiDDeN match
# "confident" only when BER is below this value. Empirically HiDDeN achieves
# ~0.15-0.35 BER on our tests; random baseline is 0.5. We use 0.40 as a
# conservative "did the watermark survive at all" gate.
HIDDEN_CONFIDENT_BER = 0.40


class EnsembleWatermarker:
    """
    Embed both DCT and HiDDeN watermarks; identify via best-surviving detector.

    Design:
      - embed(): DCT first, then HiDDeN on top of the DCT-watermarked image.
        Order matters: DCT is applied to luminance top-N DCT coefficients,
        HiDDeN adds a pixel-space residual. HiDDeN's changes could disturb
        DCT coefficients, so we let HiDDeN "see" the DCT signal and adapt.
      - identify(): runs BOTH detectors against every registered seller.
        Returns the identification with higher confidence.
    """

    def __init__(self, hidden_checkpoint: str):
        self.hidden = HiddenModel(hidden_checkpoint)

    def embed(self, image_path: str, seller_id: str,
              output_path: str,
              tmp_dir: str = "app/watermarked") -> dict:
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        tmp_dct = f"{tmp_dir}/_tmp_dct_{abs(hash(seller_id)) % 10**8}.png"

        # Step 1: DCT embedding
        dct_meta = dct_embed(image_path, seller_id, tmp_dct,
                             alpha=DCT_ALPHA, n_coeffs=DCT_N_COEFFS)

        # Step 2: HiDDeN embedding on top of DCT-watermarked image
        self.hidden.embed(tmp_dct, seller_id, output_path)

        # Clean up intermediate
        try:
            Path(tmp_dct).unlink()
        except FileNotFoundError:
            pass

        return {
            "seller_id": seller_id,
            "dct_meta": dct_meta,
            "algorithm": "ensemble_dct+hidden",
        }

    def identify(self, suspect_path: str,
                 candidates: list[dict]) -> dict:
        """
        candidates: list of dicts with keys:
            seller_id       (str)
            original_path   (str)  — needed for DCT informed detector
            dct_meta        (dict) — from embed()
        Returns best identification across both detectors.
        """
        results = []
        for c in candidates:
            # DCT detection
            dct_result = dct_detect(
                suspect_path,
                c["original_path"],
                c["seller_id"],
                c["dct_meta"],
                threshold=DCT_THRESHOLD,
            )
            dct_confidence = max(0.0, dct_result["similarity"] / DCT_THRESHOLD)
            #   Interpretation: >1.0 means detection fires; higher = more confident

            results.append({
                "seller_id":    c["seller_id"],
                "dct_sim":      dct_result["similarity"],
                "dct_detected": dct_result["detected"],
                "dct_confidence": dct_confidence,
            })

        # HiDDeN identification (single pass across all candidates)
        seller_ids = [c["seller_id"] for c in candidates]
        hidden_result = self.hidden.identify(suspect_path, seller_ids)
        # Build lookup: seller_id -> ber
        hidden_by_seller = {r["seller_id"]: r["ber"]
                            for r in hidden_result["ranking"]}
        for r in results:
            ber = hidden_by_seller.get(r["seller_id"], 0.5)
            r["hidden_ber"] = ber
            r["hidden_detected"] = ber < HIDDEN_CONFIDENT_BER
            #   HiDDeN "confidence": lower BER = higher confidence, normalized
            #   so BER=0 → confidence 1.0, BER=0.5 (random) → confidence 0
            r["hidden_confidence"] = max(0.0, 1.0 - 2 * ber)

        # ── Fusion: pick the best signal per candidate ──
        for r in results:
            r["ensemble_confidence"] = max(r["dct_confidence"],
                                           r["hidden_confidence"])
            r["detected"] = r["dct_detected"] or r["hidden_detected"]
            # Which detector "voted" for this match?
            if r["dct_confidence"] >= r["hidden_confidence"]:
                r["winning_detector"] = "dct"
            else:
                r["winning_detector"] = "hidden"

        # Rank by ensemble confidence (higher = more likely this seller)
        results.sort(key=lambda x: x["ensemble_confidence"], reverse=True)

        top = results[0]
        match = top if top["detected"] else None

        return {
            "match": match,
            "ranking": results[:10],
            "n_candidates": len(candidates),
        }