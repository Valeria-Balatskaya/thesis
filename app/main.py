# app/main.py
# FastAPI entrypoint for the traceability service.
#
# Endpoints:
#   POST /api/sellers                     — create a seller account
#   GET  /api/sellers                     — list all sellers (with product counts)
#   POST /api/sellers/{id}/products       — upload a product image (multipart)
#   GET  /api/sellers/{id}/products       — list a seller's catalog
#   POST /api/scan                        — identify a suspect image
#   GET  /api/watermarked/{filename}      — serve a watermarked image
#   GET  /                                 — frontend

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

    # Save under a safe path derived from seller + product count
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


# Frontend at /
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")