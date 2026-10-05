# app/service.py
# Bridges FastAPI routes with the ensemble watermarking system.
#
# Architecture: layered ensemble (see src/ensemble.py).
#   - Seller uploads original → HiDDeN residual + DCT are layered into ONE image
#   - That single image is what the seller downloads and publishes, so both
#     watermarks travel with every stolen copy
#   - The HiDDeN-only intermediate is kept server-side as the DCT reference
#   - Scan runs both detectors on the suspect; whichever fires wins

from pathlib import Path

from src.ensemble import EnsembleWatermarker
from src.dct_watermark import embed as dct_embed
from src.hidden_inference import HiddenModel, _seller_to_bits, _ber
from app import database as db

HIDDEN_CHECKPOINT = "checkpoints/hidden_final.pt"
_ENSEMBLE = None


def _get_ensemble() -> EnsembleWatermarker:
    global _ENSEMBLE
    if _ENSEMBLE is None:
        _ENSEMBLE = EnsembleWatermarker(HIDDEN_CHECKPOINT)
    return _ENSEMBLE


DCT_ALPHA = 0.02
DCT_N_COEFFS = 500
DCT_THRESHOLD = 6.0
HIDDEN_DETECTED_BER = 0.40
ALGORITHM_NAME = "ensemble_layered"


def _reference_path_for(watermarked_path: str) -> str:
    """Server-side DCT reference (HiDDeN-only intermediate) for a published image."""
    p = Path(watermarked_path)
    return str(p.with_name(p.stem + "_ref" + p.suffix))


def watermark_product(seller_id: int, title: str, sku: str | None,
                      original_path: str, watermarked_path: str) -> dict:
    ensemble = _get_ensemble()

    product = db.add_product(
        seller_id=seller_id,
        title=title,
        sku=sku,
        original_path=original_path,
        watermarked_path=watermarked_path,
        algorithm=ALGORITHM_NAME,
        alpha=DCT_ALPHA,
        n_coeffs=DCT_N_COEFFS,
    )

    seller_payload = f"PRODUCT:{product['id']}"
    ensemble.embed(original_path, seller_payload,
                   output_path=watermarked_path,
                   reference_path=_reference_path_for(watermarked_path))

    return product


def identify_product(suspect_path: str) -> dict:
    """
    Identify the product by trying BOTH DCT detector and HiDDeN decoder
    against every registered product. Whichever fires wins.
    """
    products = db.list_all_products()
    if not products:
        return {"match": None, "ranking": [], "n_candidates": 0}

    ensemble = _get_ensemble()

    # Reconstruct candidate list (same format as ensemble.identify expects)
    candidates = []
    for p in products:
        seller_payload = f"PRODUCT:{p['id']}"
        # Deterministic re-embed on the reference to recover DCT metadata
        reference = _reference_path_for(p["watermarked_path"])
        if not Path(reference).exists():
            # Product created before the layered design: DCT was embedded on the original
            reference = p["original_path"]
        temp_out = f"app/watermarked/_probe_{p['id']}.png"
        dct_meta = dct_embed(
            reference, seller_payload, temp_out,
            alpha=p["alpha"], n_coeffs=p["n_coeffs"],
        )
        candidates.append({
            "seller_id": seller_payload,
            "original_path": p["original_path"],
            "reference_path": reference,
            "dct_meta": dct_meta,
            "_product_row": p,
        })

    result = ensemble.identify(suspect_path, candidates)

    # Enrich for the UI
    def _enrich(row):
        for c in candidates:
            if c["seller_id"] == row["seller_id"]:
                p = c["_product_row"]
                return {
                    **{k: v for k, v in row.items()},
                    "product_id": p["id"],
                    "product_title": p["title"],
                    "seller_name": p["seller_name"],
                }
        return row

    return {
        "match": _enrich(result["match"]) if result["match"] else None,
        "ranking": [_enrich(r) for r in result["ranking"]],
        "n_candidates": result["n_candidates"],
        "threshold_note": "DCT sim > 6 OR HiDDeN BER < 0.40",
    }