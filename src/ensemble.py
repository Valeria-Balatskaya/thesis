# src/ensemble.py
# Ensemble watermarking v4 — layered watermarks in ONE published image.
#
# Design:
#   Embedding: HiDDeN (residual mode) is embedded first, then DCT spread-spectrum
#              on top of that, producing a SINGLE file that is the one published.
#              Both watermarks therefore travel with every copy of the image.
#              The HiDDeN-only intermediate is kept server-side as the
#              reference image the (non-blind) DCT detector needs.
#
#   Detection: Run BOTH detectors against the suspect. Whichever gives higher
#              confidence per candidate wins. Then rank candidates by their
#              winning confidence.
#
# v3 (dual storage) embedded the two watermarks into two separate files and only
# published the DCT one, so the HiDDeN watermark never reached a stolen copy.
# See tests/test_deployed_benchmark.py for the honest comparison.

from pathlib import Path

from src.dct_watermark import embed as dct_embed, detect as dct_detect
from src.hidden_inference import HiddenModel

DCT_ALPHA = 0.02
DCT_N_COEFFS = 500
DCT_THRESHOLD = 6.0

HIDDEN_DETECTED_BER = 0.40


class EnsembleWatermarker:
    def __init__(self, hidden_checkpoint: str):
        self.hidden = HiddenModel(hidden_checkpoint)

    def embed(self, image_path: str, seller_id: str,
              output_path: str, reference_path: str) -> dict:
        """
        Layered embed: HiDDeN residual -> reference_path (server-side),
        then DCT on top -> output_path (the one image that gets published).
        Detection must use reference_path as the DCT "original".
        """
        self.hidden.embed(image_path, seller_id, reference_path, mode="residual")
        dct_meta = dct_embed(reference_path, seller_id, output_path,
                             alpha=DCT_ALPHA, n_coeffs=DCT_N_COEFFS)
        return {
            "seller_id": seller_id,
            "output_path": output_path,
            "reference_path": reference_path,
            "dct_meta": dct_meta,
            "algorithm": "ensemble_layered",
        }

    def embed_dual_storage(self, image_path: str, seller_id: str,
                           dct_output: str, hidden_output: str) -> dict:
        """
        LEGACY v3 behaviour (two separate files, legacy HiDDeN mode).
        Kept only so earlier benchmarks remain reproducible for the thesis.
        """
        dct_meta = dct_embed(image_path, seller_id, dct_output,
                             alpha=DCT_ALPHA, n_coeffs=DCT_N_COEFFS)
        self.hidden.embed(image_path, seller_id, hidden_output, mode="legacy")
        return {
            "seller_id": seller_id,
            "dct_output": dct_output,
            "hidden_output": hidden_output,
            "dct_meta": dct_meta,
            "algorithm": "ensemble_dual_storage",
        }

    def identify(self, suspect_path: str, candidates: list[dict]) -> dict:
        # DCT scores against every candidate
        dct_scores = {}
        for c in candidates:
            r = dct_detect(suspect_path, c.get("reference_path", c["original_path"]),
                           c["seller_id"], c["dct_meta"],
                           threshold=DCT_THRESHOLD)
            dct_scores[c["seller_id"]] = r

        # HiDDeN scores (one pass)
        seller_ids = [c["seller_id"] for c in candidates]
        h_result = self.hidden.identify(suspect_path, seller_ids)
        h_ber = {r["seller_id"]: r["ber"] for r in h_result["ranking"]}

        rows = []
        for c in candidates:
            sid = c["seller_id"]
            dct_r = dct_scores[sid]
            dct_conf = dct_r["similarity"] / DCT_THRESHOLD
            h_b = h_ber.get(sid, 0.5)
            h_conf = max(0.0, (0.5 - h_b) / 0.5)

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