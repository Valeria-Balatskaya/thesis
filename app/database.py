# app/database.py
# SQLite schema for the traceability service.
#
# Two tables:
#   sellers  — seller accounts (id, email, business_name, created_at)
#   products — one row per watermarked product image (belongs to a seller)
#
# We keep it plain sqlite3 (no ORM) so the thesis stays legible and
# reviewers can see exactly what's happening.

import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = "app/thesis.db"


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = _connect()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sellers (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            email          TEXT UNIQUE NOT NULL,
            business_name  TEXT NOT NULL,
            created_at     TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS products (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            seller_id         INTEGER NOT NULL,
            title             TEXT NOT NULL,
            sku               TEXT,
            original_path     TEXT NOT NULL,
            watermarked_path  TEXT NOT NULL,
            algorithm         TEXT NOT NULL,
            alpha             REAL,
            n_coeffs          INTEGER,
            created_at        TEXT NOT NULL,
            FOREIGN KEY(seller_id) REFERENCES sellers(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_products_seller ON products(seller_id);

        CREATE TABLE IF NOT EXISTS authorized_urls (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id   INTEGER NOT NULL,
            url          TEXT NOT NULL,
            created_at   TEXT NOT NULL,
            FOREIGN KEY(product_id) REFERENCES products(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_authurls_product ON authorized_urls(product_id);

        CREATE TABLE IF NOT EXISTS monitoring_findings (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id   INTEGER NOT NULL,
            source_url   TEXT NOT NULL,
            source_file  TEXT,
            confidence   REAL NOT NULL,
            detector     TEXT NOT NULL,
            authorized   INTEGER NOT NULL,
            found_at     TEXT NOT NULL,
            FOREIGN KEY(product_id) REFERENCES products(id) ON DELETE CASCADE
        );
    """)
    conn.commit()
    conn.close()


# ─── Seller operations ──────────────────────────────────────────────

def create_seller(email: str, business_name: str) -> dict:
    conn = _connect()
    try:
        cur = conn.execute(
            "INSERT INTO sellers (email, business_name, created_at) VALUES (?, ?, ?)",
            (email, business_name, datetime.utcnow().isoformat())
        )
        conn.commit()
        return get_seller_by_id(cur.lastrowid)
    finally:
        conn.close()


def get_seller_by_id(seller_id: int) -> dict | None:
    conn = _connect()
    row = conn.execute("SELECT * FROM sellers WHERE id = ?", (seller_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_seller_by_email(email: str) -> dict | None:
    conn = _connect()
    row = conn.execute("SELECT * FROM sellers WHERE email = ?", (email,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_sellers() -> list[dict]:
    conn = _connect()
    rows = conn.execute("""
        SELECT s.*, COUNT(p.id) AS product_count
        FROM sellers s
        LEFT JOIN products p ON p.seller_id = s.id
        GROUP BY s.id
        ORDER BY s.created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─── Product operations ─────────────────────────────────────────────

def add_product(seller_id: int, title: str, sku: str | None,
                original_path: str, watermarked_path: str,
                algorithm: str, alpha: float | None,
                n_coeffs: int | None) -> dict:
    conn = _connect()
    cur = conn.execute("""
        INSERT INTO products
        (seller_id, title, sku, original_path, watermarked_path,
         algorithm, alpha, n_coeffs, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (seller_id, title, sku, original_path, watermarked_path,
          algorithm, alpha, n_coeffs, datetime.utcnow().isoformat()))
    conn.commit()
    product_id = cur.lastrowid
    conn.close()
    return get_product(product_id)


def get_product(product_id: int) -> dict | None:
    conn = _connect()
    row = conn.execute("SELECT * FROM products WHERE id = ?", (product_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_products_for_seller(seller_id: int) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM products WHERE seller_id = ? ORDER BY created_at DESC",
        (seller_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_all_products() -> list[dict]:
    """Return every product across every seller — used by the scanner."""
    conn = _connect()
    rows = conn.execute("""
        SELECT p.*, s.business_name AS seller_name, s.email AS seller_email
        FROM products p
        JOIN sellers s ON s.id = p.seller_id
        ORDER BY p.created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ─── Authorized URL operations ──────────────────────────────────────

def add_authorized_url(product_id: int, url: str) -> dict:
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO authorized_urls (product_id, url, created_at) VALUES (?, ?, ?)",
        (product_id, url.strip(), datetime.utcnow().isoformat())
    )
    conn.commit()
    row_id = cur.lastrowid
    row = conn.execute("SELECT * FROM authorized_urls WHERE id = ?", (row_id,)).fetchone()
    conn.close()
    return dict(row)


def list_authorized_urls(product_id: int) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM authorized_urls WHERE product_id = ? ORDER BY created_at",
        (product_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_authorized_url(url_id: int) -> bool:
    conn = _connect()
    cur = conn.execute("DELETE FROM authorized_urls WHERE id = ?", (url_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


# ─── Monitoring findings operations ─────────────────────────────────

def record_finding(product_id: int, source_url: str, source_file: str,
                   confidence: float, detector: str, authorized: bool) -> dict:
    conn = _connect()
    cur = conn.execute("""
        INSERT INTO monitoring_findings
        (product_id, source_url, source_file, confidence, detector, authorized, found_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (product_id, source_url, source_file, confidence, detector,
          int(authorized), datetime.utcnow().isoformat()))
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return {"id": row_id}


def list_findings_for_product(product_id: int) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM monitoring_findings WHERE product_id = ? ORDER BY found_at DESC",
        (product_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]