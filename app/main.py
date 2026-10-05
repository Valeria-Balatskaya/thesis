# app/main.py
# TraceMark — FastAPI entrypoint.
#
#   uvicorn app.main:app --reload        then open http://127.0.0.1:8000
#
# Endpoints (all JSON unless noted):
#   GET    /api/overview                    counters, model info, recent detections
#   GET    /api/sellers        POST         seller accounts
#   GET    /api/products       POST         list / protect a new image (multipart)
#   GET    /api/products/{id}  DELETE       product with copies, URLs, detections
#   POST   /api/products/{id}/copies        issue another traceable copy (per channel)
#   POST   /api/products/{id}/urls          authorise a URL;  DELETE /api/urls/{id}
#   GET    /api/lab/attacks                 Attack Lab catalogue
#   POST   /api/lab/run                     attack a copy, then try to trace it
#   POST   /api/scan                        analyse an uploaded suspect image
#   GET    /api/watch          POST DELETE  monitor watchlist
#   POST   /api/watch/crawl                 crawl one URL or the whole watchlist
#   GET    /api/monitor/schedule  POST      automatic re-crawl interval
#   GET    /api/detections     DELETE       detection log
#   GET    /report/{id}                     printable evidence report (HTML)
#   POST   /api/demo/build                  build the demo "thief shop"
#   GET    /demo/shop                       demo page for the monitor to crawl
#   GET    /media/...                       stored images
#
# Routes are plain `def` (FastAPI runs them in a thread pool) so the monitor can
# crawl pages served by this same process without deadlocking.
#
# The previous v1 app (app/service.py, app/database.py, app/thesis.db) is kept
# for reproducibility but is no longer served.

import html
import random
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import crawler, guard, store
from src.distractor_gen import generate_distractor
from src.payload import blind_false_positive_rate, max_errors_for_fpr, PAYLOAD_BITS, ECC_T

app = FastAPI(title="TraceMark")
store.init_db()
crawler.start_scheduler()


def _need(obj, what: str):
    if not obj:
        raise HTTPException(404, f"{what} not found")
    return obj


# ─── overview ─────────────────────────────────────────────────────

@app.get("/api/overview")
def api_overview():
    try:
        model = guard.model_info()
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    return {"stats": store.stats(), "model": model,
            "recent": [guard.detection_json(d) for d in store.list_detections(limit=8)],
            "schedule": crawler.scheduler_status()}


# ─── sellers ──────────────────────────────────────────────────────

@app.get("/api/sellers")
def api_sellers():
    return {"sellers": store.list_sellers()}


@app.post("/api/sellers")
def api_create_seller(name: str = Form(...), email: str = Form(...)):
    name, email = name.strip(), email.strip().lower()
    if not name or "@" not in email:
        raise HTTPException(400, "a name and a valid email are required")
    if store.get_seller_by_email(email):
        raise HTTPException(409, f"a seller with email {email} already exists")
    return {"seller": store.create_seller(name, email)}


# ─── products ─────────────────────────────────────────────────────

@app.get("/api/products")
def api_products():
    out = []
    for p in store.list_products():
        copies = store.list_copies(p["id"])
        out.append({**p, "original_url": guard.media_url(p["original_path"]),
                    "thumb_url": guard.media_url(copies[0]["file_path"]) if copies else None,
                    "copies": [{"id": c["id"], "label": c["label"]} for c in copies]})
    return {"products": out}


@app.post("/api/products")
def api_protect(seller_id: int = Form(...), title: str = Form(...),
                authorized_url: str = Form(""), image: UploadFile = File(...)):
    _need(store.get_seller(seller_id), "seller")
    if not title.strip():
        raise HTTPException(400, "title is required")
    try:
        return {"product": guard.protect(seller_id, title.strip(), image.file,
                                         authorized_url.strip() or None)}
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except OSError:
        raise HTTPException(400, "that file could not be read as an image")


@app.get("/api/products/{product_id}")
def api_product(product_id: int):
    return {"product": _need(guard.product_json(product_id), "product")}


@app.delete("/api/products/{product_id}")
def api_delete_product(product_id: int):
    _need(store.get_product(product_id), "product")
    guard.remove_product(product_id)
    return {"ok": True}


@app.post("/api/products/{product_id}/copies")
def api_new_copy(product_id: int, label: str = Form(...)):
    _need(store.get_product(product_id), "product")
    if not label.strip():
        raise HTTPException(400, "give the copy a label, e.g. the partner it is sent to")
    return {"copy": guard.create_copy(product_id, label.strip())}


@app.post("/api/products/{product_id}/urls")
def api_add_url(product_id: int, url: str = Form(...)):
    _need(store.get_product(product_id), "product")
    if not url.strip():
        raise HTTPException(400, "url is required")
    store.add_authorized_url(product_id, url.strip())
    return {"authorized_urls": store.list_authorized_urls(product_id)}


@app.delete("/api/urls/{url_id}")
def api_delete_url(url_id: int):
    store.delete_authorized_url(url_id)
    return {"ok": True}


# ─── attack lab ───────────────────────────────────────────────────

@app.get("/api/lab/attacks")
def api_lab_attacks():
    return {"attacks": guard.lab_catalogue()}


@app.post("/api/lab/run")
def api_lab_run(copy_id: int = Form(...), attack: str = Form(...),
                value: float | None = Form(None), stacked_on: str = Form("")):
    copy = _need(store.get_copy(copy_id), "copy")
    if attack not in guard.LAB_ATTACKS:
        raise HTTPException(400, "unknown attack")
    try:
        attacked = guard.lab_apply(guard.lab_source(copy_id, stacked_on or None), attack, value)
    except ImportError as e:
        raise HTTPException(501, f"this attack needs an optional package: {e}")
    result = guard.analyse(attacked)
    traced = result["copy"] is not None and result["copy"]["id"] == copy_id
    wrong = result["copy"] is not None and result["copy"]["id"] != copy_id
    return {"attack": guard.LAB_ATTACKS[attack][0], "attacked_url": guard.media_url(attacked),
            "expected_copy_id": copy_id, "expected_product_id": copy["product_id"],
            "traced": traced, "wrong": wrong, "result": result}


# ─── scan ─────────────────────────────────────────────────────────

@app.post("/api/scan")
def api_scan(image: UploadFile = File(...), page_url: str = Form(""), record: bool = Form(False)):
    path = guard.DATA / "scans" / f"{uuid.uuid4().hex}.png"
    try:
        guard._save_normalised(image.file, path)
    except OSError:
        raise HTTPException(400, "that file could not be read as an image")
    try:
        return {"result": guard.analyse(str(path), page_url=page_url.strip() or None,
                                        image_url=None, record=record)}
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))


# ─── web monitor ──────────────────────────────────────────────────

@app.get("/api/watch")
def api_watch():
    return {"watch": store.list_watch_urls(), "schedule": crawler.scheduler_status()}


@app.post("/api/watch")
def api_add_watch(url: str = Form(...), label: str = Form("")):
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "the URL must start with http:// or https://")
    store.add_watch_url(url, label.strip() or None)
    return {"watch": store.list_watch_urls()}


@app.delete("/api/watch/{watch_id}")
def api_delete_watch(watch_id: int):
    store.delete_watch_url(watch_id)
    return {"ok": True}


@app.post("/api/watch/crawl")
def api_crawl(url: str = Form("")):
    url = url.strip()
    results = [crawler.crawl_page(url)] if url else crawler.run_now()
    slim = [{**r, "matches": [{"image_url": m["image_url"], "verdict": m["verdict"],
                               "authorized": m["authorized"], "product": m["product"],
                               "copy": m["copy"], "detection_id": m["detection_id"]}
                              for m in r["matches"]]} for r in results]
    return {"results": slim}


@app.get("/api/monitor/schedule")
def api_schedule():
    return crawler.scheduler_status()


@app.post("/api/monitor/schedule")
def api_set_schedule(minutes: int = Form(...)):
    store.set_setting("crawl_minutes", str(max(0, minutes)))
    return crawler.scheduler_status()


@app.get("/api/detections")
def api_detections():
    return {"detections": [guard.detection_json(d) for d in store.list_detections()]}


@app.delete("/api/detections")
def api_clear_detections():
    store.clear_detections()
    return {"ok": True}


# ─── evidence report ──────────────────────────────────────────────

_VERDICT_TEXT = {
    "watermark": ("Watermark read directly",
                  "The invisible watermark was read from the found image without any reference "
                  "to the original. The embedded ID and its keyed check tag both validated."),
    "watermark_aligned": ("Watermark read after alignment",
                          "The found image had been cropped, warped or placed inside another image. "
                          "It was matched to the registered original by its visual features, warped back "
                          "into place, and the watermark was then read. Areas the found image does not "
                          "cover were filled from the un-watermarked original, which cannot contribute "
                          "watermark signal."),
    "fingerprint_only": ("Same picture, watermark not readable",
                         "The found image shows the same photograph as the registered original "
                         "(matched by visual features and structure), but the watermark could not be read. "
                         "This establishes that the picture is the seller's; it does NOT establish which "
                         "copy was taken."),
}


@app.get("/report/{detection_id}", response_class=HTMLResponse)
def report(detection_id: int):
    d = _need(store.get_detection(detection_id), "detection")
    e = html.escape
    title, explanation = _VERDICT_TEXT[d["verdict"]]
    stages = d["details"].get("stages", [])
    verify = next((s for s in stages if s["stage"] == "align_verify" and s.get("ok")), None)
    blind = next((s for s in stages if s["stage"] == "blind" and s.get("ok")), None)
    retrieve = next((s for s in stages if s["stage"] == "retrieve" and s.get("ok")), None)

    facts = []
    if blind:
        facts.append(f"Bit errors corrected: {blind['bit_errors']} of {PAYLOAD_BITS} "
                     f"(the code corrects up to {ECC_T}).")
        facts.append(f"Chance that an unmarked image produces a valid ID by accident: about "
                     f"{blind_false_positive_rate():.1e} per read.")
    if retrieve:
        facts.append(f"Visual match: {retrieve['inliers']} geometrically consistent feature points, "
                     f"structural similarity {retrieve['similarity']:.2f}, "
                     f"found image covers {retrieve['coverage']:.0%} of the original"
                     + (" (mirrored)." if retrieve.get("flipped") else "."))
    if verify:
        if verify["method"] == "bch":
            facts.append(f"After alignment the ID decoded with {verify['bit_errors']} corrected bit errors.")
        else:
            facts.append(f"After alignment {verify['bit_errors']} of {PAYLOAD_BITS} bits differed from this "
                         f"copy's code; up to {verify['limit']} are accepted at a false-positive rate of 1e-6.")

    def img(url, caption):
        return (f'<figure><img src="{e(url)}"><figcaption>{e(caption)}</figcaption></figure>'
                if url else "")

    copy_line = (f"<tr><th>Traced copy</th><td>#{d['copy_id']} — {e(d['copy_label'] or '')} "
                 f"(issued {e(d['copy_created_at'] or '')})</td></tr>" if d["copy_id"] else
                 "<tr><th>Traced copy</th><td>not determined</td></tr>")
    status = ("Authorised location" if d["authorized"] else "NOT an authorised location")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Evidence report #{d['id']}</title>
<style>
 body{{font:15px/1.5 -apple-system,Segoe UI,sans-serif;color:#111;max-width:860px;margin:32px auto;padding:0 20px}}
 h1{{font-size:24px;margin:0}} h2{{font-size:16px;margin:28px 0 8px;border-bottom:1px solid #ccc;padding-bottom:4px}}
 table{{border-collapse:collapse;width:100%}} th{{text-align:left;width:190px;vertical-align:top;color:#555;font-weight:500}}
 td,th{{padding:5px 0}} .badge{{display:inline-block;padding:3px 10px;border-radius:99px;font-size:13px;
 background:{'#e7f6ec' if d['authorized'] else '#fde8e8'};color:{'#17663a' if d['authorized'] else '#a01919'}}}
 .imgs{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}} figure{{margin:0}}
 img{{width:100%;border:1px solid #ccc}} figcaption{{font-size:12px;color:#555}} code{{font-size:12px;word-break:break-all}}
 .note{{font-size:13px;color:#555}} @media print{{button{{display:none}}}}
</style></head><body>
<button onclick="print()" style="float:right">Print / save PDF</button>
<h1>Evidence report #{d['id']}</h1>
<p class="note">Generated by TraceMark (research prototype). Times are UTC.</p>
<h2>Finding</h2>
<table>
<tr><th>Result</th><td><b>{e(title)}</b> &nbsp;<span class="badge">{status}</span></td></tr>
<tr><th>Product</th><td>{e(d['product_title'])} (product #{d['product_id']})</td></tr>
<tr><th>Rights holder</th><td>{e(d['seller_name'])} &lt;{e(d['seller_email'])}&gt;</td></tr>
{copy_line}
<tr><th>Found at (page)</th><td>{e(d['page_url'] or 'uploaded manually')}</td></tr>
<tr><th>Image address</th><td><code>{e(d['image_url'] or '—')}</code></td></tr>
<tr><th>First / last seen</th><td>{e(d['first_seen'])} / {e(d['last_seen'])} ({d['times_seen']}×)</td></tr>
<tr><th>SHA-256 of found file</th><td><code>{e(d['details'].get('sha256', '—'))}</code></td></tr>
</table>
<h2>How it was established</h2>
<p>{e(explanation)}</p>
<ul>{''.join(f'<li>{e(f)}</li>' for f in facts)}</ul>
<h2>Images</h2>
<div class="imgs">
{img(guard.media_url(d['original_path']), 'Registered original (held by the service)')}
{img(guard.media_url(d['snapshot_path']), 'Image as found')}
{img(guard.media_url(d['aligned_path']), 'Found image aligned to the original')}
</div>
<h2>Limits of this evidence</h2>
<p class="note">This is the output of a research prototype, evaluated on a limited image set with simulated
attacks. The false-positive figures are calculated from the code design and were not exceeded in testing;
they are not a legal standard of proof. A "same picture" result without a watermark read cannot tell the
seller's own copy apart from a stolen one.</p>
</body></html>"""


# ─── demo pages for the monitor ───────────────────────────────────
# A fake marketplace with stolen (attacked) copies of your protected products
# mixed with unrelated images, served by this app so the monitor can crawl a
# real HTTP page during a demo without touching anyone else's site.

_DEMO_ATTACKS = [("jpeg", 35), ("crop", 0.6), ("page", 0.6), ("overlay", None), ("mirror", None),
                 ("social", None), ("perspective", 0.1), ("screenshot", None), ("rotate", 6),
                 ("square", None)]


@app.post("/api/demo/build")
def api_demo_build(request: Request):
    products = store.list_products()
    if not products:
        raise HTTPException(400, "protect at least one product first")
    demo = guard.DATA / "demo"
    for f in demo.glob("*"):
        f.unlink()
    rng = random.Random(42)
    for i, p in enumerate(products[:10]):
        copies = store.list_copies(p["id"])
        if not copies:
            continue
        attack, value = _DEMO_ATTACKS[i % len(_DEMO_ATTACKS)]
        attacked = guard.lab_apply(rng.choice(copies)["file_path"], attack, value)
        Path(attacked).rename(demo / f"stolen_{p['id']}_{attack}.png")
    for i in range(5):
        generate_distractor(f"demo_decoy_{i}", size=512, output_path=str(demo / f"decoy_{i}.png"))
    base = str(request.base_url).rstrip("/")
    store.add_watch_url(f"{base}/demo/shop", "Demo: MegaDeals (stolen images)")
    return {"shop_url": f"{base}/demo/shop", "images": len(list(demo.glob('*.png')))}


@app.get("/demo/shop", response_class=HTMLResponse)
def demo_shop():
    files = sorted((guard.DATA / "demo").glob("*.png"))
    random.Random(1).shuffle(files)
    cards = "".join(
        f'<div class="card"><img src="/media/demo/{f.name}"><b>Super Deal #{i + 1}</b>'
        f'<span>${19 + 7 * i}.99</span><button>Buy now</button></div>' for i, f in enumerate(files))
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>MegaDeals</title><style>
body{{font-family:sans-serif;margin:0;background:#f3f3f3}}header{{background:#232f3e;color:#fff;padding:16px 24px;font-size:22px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:16px;padding:24px}}
.card{{background:#fff;padding:12px;border-radius:6px;display:flex;flex-direction:column;gap:6px}}
.card img{{width:100%;height:170px;object-fit:contain}}.card span{{color:#b12704;font-size:18px}}
button{{background:#ffd814;border:0;padding:8px;border-radius:99px}}</style></head><body>
<header>MegaDeals — unbeatable prices <small style="opacity:.6">(demo page generated by TraceMark)</small></header>
<div class="grid">{cards or '<p>Nothing here yet. Build the demo from the Web Monitor page.</p>'}</div></body></html>"""


# ─── media + frontend ─────────────────────────────────────────────

@app.get("/media/{path:path}")
def media(path: str):
    full = (guard.DATA / path).resolve()
    if guard.DATA.resolve() not in full.parents or not full.is_file() or full.suffix == ".db":
        raise HTTPException(404, "not found")
    return FileResponse(full)


@app.exception_handler(FileNotFoundError)
def _missing_checkpoint(request, exc):
    return JSONResponse({"detail": str(exc)}, status_code=503)


app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
