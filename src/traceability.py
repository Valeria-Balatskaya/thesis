# src/traceability.py
# Multi-seller identification: given a suspect image and a database of
# known sellers, identify which seller (if any) uploaded the original.
#
# This is the real e-commerce problem. Marketplaces have a closed set of
# registered sellers and want to answer: "who does this stolen image
# belong to?" — not the open-world "is there any watermark here" question.

import numpy as np
from src.dct_watermark import embed, detect


class SellerRegistry:
    """
    In-memory registry mapping seller_id -> embedding metadata + original path.
    In production this would be a database; for the thesis a dict is fine
    and easier to reason about.
    """

    def __init__(self):
        self._sellers: dict[str, dict] = {}

    def register(self, seller_id: str, original_path: str,
                 watermarked_path: str, alpha: float = 0.02,
                 n_coeffs: int = 500) -> dict:
        """
        Embed a watermark for this seller and store everything needed
        to later identify their content.
        """
        meta = embed(original_path, seller_id, watermarked_path,
                     alpha=alpha, n_coeffs=n_coeffs)
        self._sellers[seller_id] = {
            "original_path": original_path,
            "watermarked_path": watermarked_path,
            "meta": meta,
            "alpha": alpha,
            "n_coeffs": n_coeffs,
        }
        return meta

    def identify(self, suspect_path: str, threshold: float = 6.0) -> dict:
        """
        Test the suspect image against every registered seller and return
        the best match (highest similarity above threshold), or None.

        Returns a full ranked list so the thesis can show confusion behaviour.
        """
        scores = []
        for seller_id, record in self._sellers.items():
            result = detect(
                suspect_path,
                record["original_path"],
                seller_id,
                record["meta"],
                threshold=threshold,
            )
            scores.append({
                "seller_id": seller_id,
                "similarity": result["similarity"],
                "detected": result["detected"],
            })

        # Sort by similarity descending
        scores.sort(key=lambda s: s["similarity"], reverse=True)

        top = scores[0] if scores else None
        match = top if (top and top["detected"]) else None

        return {
            "match": match,          # best seller above threshold, or None
            "ranking": scores,       # full ranking for analysis
            "n_candidates": len(scores),
        }

    def __len__(self):
        return len(self._sellers)