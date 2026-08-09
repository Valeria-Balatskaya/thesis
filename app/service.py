# app/service.py
# Business logic: bridges FastAPI routes with the watermarking modules.
# Routes stay thin; anything to do with watermarking lives here.

from src.dct_watermark import embed as dct_embed, detect as dct_detect
from app import database as db

DCT_ALPHA = 0.02
DCT_N_COEFFS = 500
DETECTION_THRESHOLD = 6.0
ALGORITHM_NAME = "dct_cox1997"


def watermark_product(seller_id: int, title: str, sku: str | None,
                      original_path: str, watermarked_path: str) -> dict:
    """
    Embed a DCT watermark for this product and persist metadata.
    The watermark carries the *product's database ID* — that's what a scanner recovers.
    """
    # First insert with a placeholder so we get a product ID
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
    # Now embed using the actual product ID as the watermark payload
    seller_payload = f"PRODUCT:{product['id']}"
    dct_embed(original_path, seller_payload, watermarked_path,
              alpha=DCT_ALPHA, n_coeffs=DCT_N_COEFFS)
    return product


def identify_product(suspect_path: str) -> dict:
    """
    Run the suspect image against every registered product across every seller.
    Returns the best match above threshold, plus a ranked list.
    """
    products = db.list_all_products()
    if not products:
        return {"match": None, "ranking": [], "n_candidates": 0}

    scores = []
    for p in products:
        # Reconstruct embed metadata deterministically to run the informed detector
        seller_payload = f"PRODUCT:{p['id']}"
        temp_out = f"app/watermarked/_probe_{p['id']}.png"
        meta = dct_embed(
            p["original_path"], seller_payload, temp_out,
            alpha=p["alpha"], n_coeffs=p["n_coeffs"],
        )
        result = dct_detect(
            suspect_path, p["original_path"], seller_payload, meta,
            threshold=DETECTION_THRESHOLD,
        )
        scores.append({
            "product_id":   p["id"],
            "product_title": p["title"],
            "seller_id":    p["seller_id"],
            "seller_name":  p["seller_name"],
            "similarity":   result["similarity"],
            "detected":     result["detected"],
        })

    scores.sort(key=lambda x: x["similarity"], reverse=True)
    top = scores[0]
    match = top if top["detected"] else None

    return {
        "match": match,
        "ranking": scores[:10],
        "n_candidates": len(scores),
        "threshold": DETECTION_THRESHOLD,
    }