# app/crawler.py
# Web monitor: fetch a page, pull out every image on it, and run each one
# through the detection pipeline. Standard library only.
#
# Scope, stated honestly: this checks pages you point it at (a watchlist of
# marketplaces, competitor shops, search-result pages). Discovering unknown
# pages anywhere on the web needs a reverse-image-search provider (e.g. Google
# Cloud Vision Web Detection, TinEye, Bing Visual Search); their result URLs
# can be fed straight into crawl_page().

import hashlib
import io
import threading
import time
import urllib.request
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

from PIL import Image

from app import guard, store

USER_AGENT = "TraceMarkMonitor/1.0 (thesis research prototype)"
TIMEOUT = 12
MAX_HTML = 3 * 1024 * 1024
MAX_IMAGE = 20 * 1024 * 1024
MAX_IMAGES_PER_PAGE = 40
MIN_SIDE = 96
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")


class _ImageExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("img", "source"):
            for key in ("src", "data-src", "data-original", "data-lazy-src"):
                if a.get(key):
                    self.urls.append(a[key])
            for key in ("srcset", "data-srcset"):
                if a.get(key):      # take the last (usually largest) candidate
                    self.urls.append(a[key].split(",")[-1].strip().split(" ")[0])
        elif tag == "meta" and a.get("property") in ("og:image", "twitter:image") and a.get("content"):
            self.urls.append(a["content"])
        elif tag == "a" and a.get("href", "").lower().split("?")[0].endswith(IMAGE_EXT):
            self.urls.append(a["href"])


def _fetch(url: str, limit: int) -> bytes:
    if urlparse(url).scheme not in ("http", "https"):
        raise ValueError("only http(s) URLs can be crawled")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        data = resp.read(limit + 1)
    if len(data) > limit:
        raise ValueError("response too large")
    return data


def extract_image_urls(page_url: str) -> list[str]:
    html = _fetch(page_url, MAX_HTML).decode("utf-8", errors="replace")
    parser = _ImageExtractor()
    parser.feed(html)
    seen, out = set(), []
    for raw in parser.urls:
        if raw.startswith("data:"):
            continue
        full = urljoin(page_url, raw)
        if full not in seen:
            seen.add(full)
            out.append(full)
    return out[:MAX_IMAGES_PER_PAGE]


def crawl_page(page_url: str) -> dict:
    """Scan every image on one page. Matches are recorded as detections."""
    summary = {"page_url": page_url, "images": 0, "scanned": 0, "matches": [], "error": None}
    try:
        image_urls = extract_image_urls(page_url)
    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
        store.mark_crawled(page_url, f"error: {summary['error']}")
        return summary

    summary["images"] = len(image_urls)
    for image_url in image_urls:
        try:
            data = _fetch(image_url, MAX_IMAGE)
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            continue
        if min(img.size) < MIN_SIDE:
            continue
        name = hashlib.sha256(image_url.encode()).hexdigest()[:20]
        path = guard.DATA / "crawl" / f"{name}.png"
        img.save(path, format="PNG")
        summary["scanned"] += 1
        result = guard.analyse(str(path), page_url=page_url, image_url=image_url, record=True)
        if result["verdict"] != "none":
            summary["matches"].append({"image_url": image_url, **result})
        else:
            path.unlink(missing_ok=True)

    alerts = sum(1 for m in summary["matches"] if not m["authorized"])
    store.mark_crawled(page_url, f"{summary['scanned']} images, {len(summary['matches'])} matches, "
                                 f"{alerts} unauthorised")
    return summary


def crawl_watchlist() -> list[dict]:
    return [crawl_page(row["url"]) for row in store.list_watch_urls()]


# ─── scheduler ────────────────────────────────────────────────────
# A daemon thread re-crawls the watchlist every N minutes (0 = off).

_state = {"last_run": None, "running": False}


def schedule_minutes() -> int:
    return int(store.get_setting("crawl_minutes", "0"))


def scheduler_status() -> dict:
    return {"minutes": schedule_minutes(), **_state}


def _loop():
    while True:
        time.sleep(20)
        minutes = schedule_minutes()
        if minutes <= 0 or _state["running"]:
            continue
        if _state["last_run"] and time.time() - _state["last_run"] < minutes * 60:
            continue
        run_now()


def run_now() -> list[dict]:
    _state["running"] = True
    try:
        return crawl_watchlist()
    finally:
        _state["running"] = False
        _state["last_run"] = time.time()


def start_scheduler() -> None:
    threading.Thread(target=_loop, daemon=True, name="tracemark-monitor").start()
