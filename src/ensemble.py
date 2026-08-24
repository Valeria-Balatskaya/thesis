# src/ensemble.py
# Ensemble watermarking v3 — parallel independent watermarks with OR-detection.
#
# Design (clean version):
#   Embedding: DCT and HiDDeN are embedded INDEPENDENTLY into two SEPARATE
#              output files. The seller keeps both. When distributing product
#              images they use the DCT version (better quality) as the primary,
#              and keep the HiDDeN version as a fallback for post-attack scans.
#
#   Detection: Given a suspect image, we DON'T know which watermarked version
#              it originated from. So we run BOTH detectors. Whichever fires
#              with higher confidence wins. This works because DCT and HiDDeN
#              are trained on different domains — one usually survives when
#              the other doesn't.
#
# This is architecturally cleaner than trying to stack watermarks in one file.

from pathlib import Path

from src.dct_watermark import embed as dct_embed, detect as dct_detect
from src.hidden_inference import HiddenModel

DCT_ALPHA = 0.02
DCT_N_COEFFS = 500
DCT_THRESHOLD = 6.0

# HiDDeN "detected" gate: BER below this = we consider the watermark present
HIDDEN_DETECTED_BER = 0.40


class EnsembleWatermarker:
    """
    Parallel-embedding ensemble: produces two watermarked versions per image.
    Identification runs both detectors and returns the more confident match.
    """

    def __init__(self, hidden_checkpoint: str):
        self.hidden = HiddenModel(hidden_checkpoint)

    def embed(self, image_path: str, seller_id: str,
              dct_output: str, hidden_output: str) -> dict:
        """
        Produces TWO watermarked outputs. Both encode the same seller_id.
        Seller distributes whichever suits their use case (or both).
        """
        dct_meta = dct_embed(image_path, seller_id, dct_output,
                             alpha=DCT_ALPHA, n_coeffs=DCT_N_COEFFS)
        self.hidden.embed(image_path, seller_id, hidden_output)
        return {
            "seller_id": seller_id,
            "dct_output": dct_output,
            "hidden_output": hidden_output,
            "dct_meta": dct_meta,
            "algorithm": "ensemble_v3_parallel",
        }

    def identify(self, suspect_path: str, candidates: list[dict]) -> dict:
        """
        candidates: [{seller_id, original_path, dct_meta}, ...]

        Strategy: for each candidate seller, compute BOTH a DCT confidence and
        a HiDDeN confidence, in normalized [0, 1] units. Take the MAX per
        candidate. Rank by max confidence. Winner is top if its winning
        detector actually fired above threshold.
        """
        # DCT scores against every candidate
        dct_scores = {}
        for c in candidates:
            r = dct_detect(suspect_path, c["original_path"],
                           c["seller_id"], c["dct_meta"],
                           threshold=DCT_THRESHOLD)
            dct_scores[c["seller_id"]] = r  # {similarity, detected, ...}

        # HiDDeN scores (one pass)
        seller_ids = [c["seller_id"] for c in candidates]
        h_result = self.hidden.identify(suspect_path, seller_ids)
        h_ber = {r["seller_id"]: r["ber"] for r in h_result["ranking"]}

        # Fuse per candidate: max of (dct_conf, hidden_conf)
        rows = []
        for c in candidates:
            sid = c["seller_id"]
            dct_r = dct_scores[sid]
            dct_conf = dct_r["similarity"] / DCT_THRESHOLD   # >1.0 means detected
            h_b = h_ber.get(sid, 0.5)
            h_conf = max(0.0, (0.5 - h_b) / 0.5)  # BER 0 → 1.0, BER 0.5 → 0

            if dct_conf >= h_conf:
                winner = "dct"
                confidence = dct_conf
                detected = dct_r["detected"]
            else:
                winner = "hidden"
                confidence = h_conf
                detected = h_b < HIDDEN_DETECTED_BER

            rows.append({
                "seller_id":  sid,
                "dct_sim":    dct_r["similarity"],
                "dct_detected": dct_r["detected"],
                "hidden_ber": h_b,
                "hidden_detected": h_b < HIDDEN_DETECTED_BER,
                "winning_detector": winner,
                "confidence": confidence,
                "detected": detected,
            })

        rows.sort(key=lambda r: r["confidence"], reverse=True)
        top = rows[0]
        match = top if top["detected"] else None

        return {
            "match": match,
            "ranking": rows[:10],
            "n_candidates": len(candidates),
        }