# app/main.py
# FastAPI entrypoint for the traceability service.
#
# Endpoints:
#   POST /api/sellers                         — create a seller account
#   GET  /api/sellers                         — list all sellers (with product counts)
#   POST /api/sellers/{id}/products           — upload a product image (multipart)
#   GET  /api/sellers/{id}/products           — list a seller's catalog
#   POST /api/scan                            — identify a suspect image
#   GET  /api/watermarked/{filename}          — serve a watermarked image
#   GET  /api/products/{id}/urls              — list a product's authorized URLs
#   POST /api/products/{id}/urls              — add an authorized URL to a product
#   DELETE /api/products/urls/{url_id}        — remove an authorized URL
#   GET  /api/products/{id}/findings          — historical monitoring findings for a product
#   POST /api/monitor/scan                    — batch scan candidate images
#   GET  /                                    — frontend

import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import database as db
from app import service

app = FastAPI(title="Thesis Traceability Service")

Path("app/uploads").mkdir(parents=True, exist_ok=True)
Path("app/watermarked").mkdir(parents=True, exist_ok=True)
db.init_db()


def _save_upload(upload: UploadFile, dest_path: str) -> None:
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)


# ─── Seller endpoints ─────────────────────────────────────────────

@app.post("/api/sellers")
async def api_create_seller(
    email: str = Form(...),
    business_name: str = Form(...),
):
    email = email.strip().lower()
    business_name = business_name.strip()
    if not email or not business_name:
        raise HTTPException(400, "email and business_name are required")
    if db.get_seller_by_email(email):
        raise HTTPException(409, f"Seller with email '{email}' already exists")
    seller = db.create_seller(email, business_name)
    return JSONResponse({"ok": True, "seller": seller})


@app.get("/api/sellers")
async def api_list_sellers():
    return JSONResponse({"sellers": db.list_sellers()})


# ─── Product endpoints ────────────────────────────────────────────

@app.post("/api/sellers/{seller_id}/products")
async def api_add_product(
    seller_id: int,
    title: str = Form(...),
    sku: str = Form(""),
    image: UploadFile = File(...),
):
    seller = db.get_seller_by_id(seller_id)
    if not seller:
        raise HTTPException(404, "seller not found")
    title = title.strip()
    if not title:
        raise HTTPException(400, "title is required")

    existing = db.list_products_for_seller(seller_id)
    next_idx = len(existing) + 1
    filename_base = f"seller{seller_id}_product{next_idx}"
    original_path = f"app/uploads/{filename_base}.png"
    watermarked_path = f"app/watermarked/{filename_base}.png"
    _save_upload(image, original_path)

    product = service.watermark_product(
        seller_id=seller_id,
        title=title,
        sku=sku.strip() or None,
        original_path=original_path,
        watermarked_path=watermarked_path,
    )
    return JSONResponse({
        "ok": True,
        "product": product,
        "download_url": f"/api/watermarked/{filename_base}.png",
    })


@app.get("/api/sellers/{seller_id}/products")
async def api_list_products(seller_id: int):
    seller = db.get_seller_by_id(seller_id)
    if not seller:
        raise HTTPException(404, "seller not found")
    products = db.list_products_for_seller(seller_id)
    return JSONResponse({"seller": seller, "products": products})


# ─── Scan endpoint ────────────────────────────────────────────────

@app.post("/api/scan")
async def api_scan(image: UploadFile = File(...)):
    suspect_path = "app/uploads/_suspect.png"
    _save_upload(image, suspect_path)

    result = service.identify_product(suspect_path)
    return JSONResponse({
        "ok": True,
        "match":         result["match"],
        "ranking":       result["ranking"],
        "n_candidates":  result["n_candidates"],
        "threshold_note": result.get("threshold_note"),
    })


# ─── Static file serving ──────────────────────────────────────────

@app.get("/api/watermarked/{filename}")
async def api_watermarked(filename: str):
    path = Path("app/watermarked") / filename
    if not path.exists():
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="image/png", filename=f"watermarked_{filename}")


# ─── Product-centric monitoring ───────────────────────────────────
# Each product has a list of "authorized URLs" — the seller's own legitimate
# places the image should appear (their Shopify, Etsy, etc). Monitor runs a
# scan across a set of candidate images (each tagged with a supposed source
# URL). Matches are checked against the authorized list:
#   authorized=True  → image found where it's supposed to be
#   authorized=False → image found somewhere UNAUTHORIZED → likely stolen
# All findings are recorded to the database for a persistent dashboard.

@app.get("/api/products/{product_id}/urls")
async def api_list_authorized_urls(product_id: int):
    if not db.get_product(product_id):
        raise HTTPException(404, "product not found")
    return JSONResponse({"urls": db.list_authorized_urls(product_id)})


@app.post("/api/products/{product_id}/urls")
async def api_add_authorized_url(product_id: int, url: str = Form(...)):
    if not db.get_product(product_id):
        raise HTTPException(404, "product not found")
    url = url.strip()
    if not url:
        raise HTTPException(400, "url required")
    row = db.add_authorized_url(product_id, url)
    return JSONResponse({"ok": True, "url": row})


@app.delete("/api/products/urls/{url_id}")
async def api_delete_authorized_url(url_id: int):
    ok = db.delete_authorized_url(url_id)
    if not ok:
        raise HTTPException(404, "url not found")
    return JSONResponse({"ok": True})


@app.get("/api/products/{product_id}/findings")
async def api_list_findings(product_id: int):
    if not db.get_product(product_id):
        raise HTTPException(404, "product not found")
    return JSONResponse({"findings": db.list_findings_for_product(product_id)})


@app.post("/api/monitor/scan")
async def api_monitor_scan(
    images: list[UploadFile] = File(...),
    source_urls: str = Form(""),
):
    """
    Batch scan a set of candidate images (simulating a reverse-image-search
    crawler having fetched them). For each match found:
      1. Look up which product it matched
      2. Check if the supposed source URL is on that product's authorized list
      3. Record the finding as authorized=True or authorized=False
    Returns a summary broken down by authorized vs unauthorized.
    """
    urls = [u.strip() for u in source_urls.split(",")] if source_urls else []

    authorized_findings = []
    unauthorized_findings = []
    scanned = 0

    for i, upload in enumerate(images):
        suspect_path = f"app/uploads/_monitor_{i}.png"
        _save_upload(upload, suspect_path)
        scanned += 1

        result = service.identify_product(suspect_path)
        source_url = (urls[i] if i < len(urls) and urls[i]
                      else f"(file: {upload.filename})")

        if not result["match"]:
            continue

        product_id = result["match"]["product_id"]
        auth_urls = [r["url"] for r in db.list_authorized_urls(product_id)]
        # Simple substring match — real system would normalize/parse URLs properly
        is_authorized = any(auth_url in source_url or source_url in auth_url
                            for auth_url in auth_urls)

        confidence = float(result["match"].get("confidence", 0))
        detector = result["match"].get("winning_detector", "unknown")

        db.record_finding(
            product_id=product_id, source_url=source_url,
            source_file=upload.filename, confidence=confidence,
            detector=detector, authorized=is_authorized,
        )

        finding = {
            "source_url": source_url,
            "source_file": upload.filename,
            "match": result["match"],
            "confidence": confidence,
            "authorized": is_authorized,
        }
        (authorized_findings if is_authorized else unauthorized_findings).append(finding)

    return JSONResponse({
        "ok": True,
        "scanned": scanned,
        "matches_found": len(authorized_findings) + len(unauthorized_findings),
        "authorized": authorized_findings,
        "unauthorized": unauthorized_findings,
    })


# Frontend at /
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")