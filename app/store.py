# app/store.py
# SQLite storage for the TraceMark app (app/main.py).
#
#   sellers          seller accounts
#   products         one row per protected product image (original kept server-side)
#   copies           one row per watermarked copy of a product. copies.id IS the
#                    watermark ID embedded in the image. A product has a "Public
#                    listing" copy and may have more (one per partner / channel),
#                    which is what lets a leak be traced to its source
#   authorized_urls  where a product is allowed to appear
#   watch_urls       pages the web monitor crawls
#   detections       every time a protected image was found somewhere
#
# Separate from the legacy app/thesis.db (app/database.py), which the earlier
# v1 experiments used. Plain sqlite3, no ORM.

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path("app/data")
DB_PATH = DATA_DIR / "tracemark.db"

# Watermark IDs start here so that tiny integers (which a biased decoder could
# plausibly emit) are never valid copies.
FIRST_COPY_ID = 4097


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _rows(sql: str, args=()) -> list[dict]:
    conn = _connect()
    rows = [dict(r) for r in conn.execute(sql, args).fetchall()]
    conn.close()
    return rows


def _one(sql: str, args=()) -> dict | None:
    rows = _rows(sql, args)
    return rows[0] if rows else None


def _exec(sql: str, args=()) -> int:
    conn = _connect()
    cur = conn.execute(sql, args)
    conn.commit()
    rowid = cur.lastrowid
    conn.close()
    return rowid


def init_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = _connect()
    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS sellers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL, email TEXT UNIQUE NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            seller_id INTEGER NOT NULL REFERENCES sellers(id) ON DELETE CASCADE,
            title TEXT NOT NULL, original_path TEXT NOT NULL,
            width INTEGER, height INTEGER, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS copies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
            label TEXT NOT NULL, file_path TEXT NOT NULL,
            psnr REAL, ssim REAL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS authorized_urls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
            url TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS watch_urls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE NOT NULL, label TEXT, created_at TEXT NOT NULL,
            last_crawled_at TEXT, last_status TEXT);
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
            copy_id INTEGER, verdict TEXT NOT NULL,
            image_url TEXT, page_url TEXT, snapshot_path TEXT, aligned_path TEXT,
            authorized INTEGER NOT NULL, details TEXT,
            first_seen TEXT NOT NULL, last_seen TEXT NOT NULL, times_seen INTEGER NOT NULL DEFAULT 1);
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
        INSERT OR IGNORE INTO sqlite_sequence (name, seq)
            SELECT 'copies', {FIRST_COPY_ID - 1}
            WHERE NOT EXISTS (SELECT 1 FROM sqlite_sequence WHERE name = 'copies');
    """)
    conn.commit()
    conn.close()


# ─── sellers ──────────────────────────────────────────────────────

def create_seller(name: str, email: str) -> dict:
    return get_seller(_exec("INSERT INTO sellers (name, email, created_at) VALUES (?,?,?)",
                            (name, email, _now())))


def get_seller(seller_id: int) -> dict | None:
    return _one("SELECT * FROM sellers WHERE id = ?", (seller_id,))


def get_seller_by_email(email: str) -> dict | None:
    return _one("SELECT * FROM sellers WHERE email = ?", (email,))


def list_sellers() -> list[dict]:
    return _rows("""SELECT s.*, COUNT(p.id) AS product_count FROM sellers s
                    LEFT JOIN products p ON p.seller_id = s.id GROUP BY s.id ORDER BY s.id""")


# ─── products and copies ──────────────────────────────────────────

def create_product(seller_id: int, title: str, original_path: str,
                   width: int, height: int) -> int:
    return _exec("""INSERT INTO products (seller_id, title, original_path, width, height, created_at)
                    VALUES (?,?,?,?,?,?)""", (seller_id, title, original_path, width, height, _now()))


def set_product_original(product_id: int, original_path: str) -> None:
    _exec("UPDATE products SET original_path = ? WHERE id = ?", (original_path, product_id))


def get_product(product_id: int) -> dict | None:
    return _one("""SELECT p.*, s.name AS seller_name, s.email AS seller_email
                   FROM products p JOIN sellers s ON s.id = p.seller_id WHERE p.id = ?""",
                (product_id,))


def list_products() -> list[dict]:
    return _rows("""SELECT p.*, s.name AS seller_name,
                      (SELECT COUNT(*) FROM detections d
                        WHERE d.product_id = p.id AND d.authorized = 0) AS alerts
                    FROM products p JOIN sellers s ON s.id = p.seller_id ORDER BY p.id DESC""")


def delete_product(product_id: int) -> None:
    _exec("DELETE FROM products WHERE id = ?", (product_id,))


def reserve_copy(product_id: int, label: str) -> int:
    """Insert the row first: its id is the watermark ID to embed."""
    return _exec("INSERT INTO copies (product_id, label, file_path, created_at) VALUES (?,?,?,?)",
                 (product_id, label, "", _now()))


def finish_copy(copy_id: int, file_path: str, psnr: float, ssim: float) -> None:
    _exec("UPDATE copies SET file_path = ?, psnr = ?, ssim = ? WHERE id = ?",
          (file_path, psnr, ssim, copy_id))


def get_copy(copy_id: int) -> dict | None:
    return _one("SELECT * FROM copies WHERE id = ?", (copy_id,))


def list_copies(product_id: int | None = None) -> list[dict]:
    if product_id is None:
        return _rows("SELECT * FROM copies WHERE file_path != '' ORDER BY id")
    return _rows("SELECT * FROM copies WHERE product_id = ? AND file_path != '' ORDER BY id",
                 (product_id,))


# ─── authorised URLs ──────────────────────────────────────────────

def add_authorized_url(product_id: int, url: str) -> int:
    return _exec("INSERT INTO authorized_urls (product_id, url, created_at) VALUES (?,?,?)",
                 (product_id, url, _now()))


def list_authorized_urls(product_id: int) -> list[dict]:
    return _rows("SELECT * FROM authorized_urls WHERE product_id = ? ORDER BY id", (product_id,))


def delete_authorized_url(url_id: int) -> None:
    _exec("DELETE FROM authorized_urls WHERE id = ?", (url_id,))


# ─── watchlist ────────────────────────────────────────────────────

def add_watch_url(url: str, label: str | None) -> None:
    _exec("INSERT OR IGNORE INTO watch_urls (url, label, created_at) VALUES (?,?,?)",
          (url, label, _now()))


def list_watch_urls() -> list[dict]:
    return _rows("SELECT * FROM watch_urls ORDER BY id")


def delete_watch_url(watch_id: int) -> None:
    _exec("DELETE FROM watch_urls WHERE id = ?", (watch_id,))


def mark_crawled(url: str, status: str) -> None:
    _exec("UPDATE watch_urls SET last_crawled_at = ?, last_status = ? WHERE url = ?",
          (_now(), status, url))


# ─── detections ───────────────────────────────────────────────────

def record_detection(product_id: int, copy_id: int | None, verdict: str,
                     image_url: str | None, page_url: str | None,
                     snapshot_path: str | None, aligned_path: str | None,
                     authorized: bool, details: dict) -> int:
    """The same image URL found again updates the existing row instead of adding one."""
    existing = None
    if image_url:
        existing = _one("SELECT id FROM detections WHERE image_url = ? AND product_id = ?",
                        (image_url, product_id))
    if existing:
        _exec("""UPDATE detections SET last_seen = ?, times_seen = times_seen + 1, verdict = ?,
                 copy_id = ?, authorized = ?, details = ?, snapshot_path = ?, aligned_path = ?
                 WHERE id = ?""",
              (_now(), verdict, copy_id, int(authorized), json.dumps(details),
               snapshot_path, aligned_path, existing["id"]))
        return existing["id"]
    return _exec("""INSERT INTO detections (product_id, copy_id, verdict, image_url, page_url,
                    snapshot_path, aligned_path, authorized, details, first_seen, last_seen)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                 (product_id, copy_id, verdict, image_url, page_url, snapshot_path, aligned_path,
                  int(authorized), json.dumps(details), _now(), _now()))


def _decode(row: dict) -> dict:
    row["details"] = json.loads(row["details"]) if row.get("details") else {}
    row["authorized"] = bool(row["authorized"])
    return row


def list_detections(product_id: int | None = None, limit: int = 200) -> list[dict]:
    where, args = ("WHERE d.product_id = ?", (product_id,)) if product_id else ("", ())
    rows = _rows(f"""SELECT d.*, p.title AS product_title, s.name AS seller_name,
                            c.label AS copy_label
                     FROM detections d JOIN products p ON p.id = d.product_id
                     JOIN sellers s ON s.id = p.seller_id
                     LEFT JOIN copies c ON c.id = d.copy_id
                     {where} ORDER BY d.last_seen DESC LIMIT ?""", (*args, limit))
    return [_decode(r) for r in rows]


def get_detection(detection_id: int) -> dict | None:
    rows = _rows("""SELECT d.*, p.title AS product_title, p.original_path, s.name AS seller_name,
                           s.email AS seller_email, c.label AS copy_label, c.file_path AS copy_path,
                           c.created_at AS copy_created_at
                    FROM detections d JOIN products p ON p.id = d.product_id
                    JOIN sellers s ON s.id = p.seller_id
                    LEFT JOIN copies c ON c.id = d.copy_id WHERE d.id = ?""", (detection_id,))
    return _decode(rows[0]) if rows else None


def clear_detections() -> None:
    _exec("DELETE FROM detections")


def stats() -> dict:
    row = _one("""SELECT
        (SELECT COUNT(*) FROM sellers) AS sellers,
        (SELECT COUNT(*) FROM products) AS products,
        (SELECT COUNT(*) FROM copies WHERE file_path != '') AS copies,
        (SELECT COUNT(*) FROM watch_urls) AS watched,
        (SELECT COUNT(*) FROM detections) AS detections,
        (SELECT COUNT(*) FROM detections WHERE authorized = 0) AS alerts,
        (SELECT COUNT(*) FROM detections WHERE authorized = 0 AND verdict != 'fingerprint_only') AS traced""")
    return row


def get_setting(key: str, default: str | None = None) -> str | None:
    row = _one("SELECT value FROM settings WHERE key = ?", (key,))
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    _exec("INSERT INTO settings (key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
          (key, value))
