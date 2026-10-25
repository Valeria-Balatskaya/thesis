# app/registry_log.py
# Tamper-evident registration log (Part B3: ownership evidence beyond the
# watermark itself, objective 1.2.5).
#
# Every time a copy is issued, one record is appended:
#     seq, created_at (UTC), copy_id, product_id, seller_id,
#     sha256 of the original, sha256 of the watermarked copy,
#     prev_hash = record_hash of the previous record          (hash chain)
#     record_hash = SHA-256 of the canonical JSON of the fields above
#     signature = Ed25519 signature of record_hash (server key)
# Changing, deleting, inserting or back-dating any record breaks the chain or a
# signature from that point on, so the log proves WHEN the service first saw
# an image and for WHOM - the basis for resolving an ownership dispute when
# someone else claims the same picture (Craver et al. 1997).
#
# Optional: an RFC 3161 trusted timestamp of each record_hash from an external
# timestamp authority, so even the server operator cannot back-date the whole
# chain. Enable with TRACEMARK_TSA_URL=https://freetsa.org/tsr (needs the
# `openssl` command line tool and network access; off by default).
#
# The server key lives in <TRACEMARK_DATA>/keys/ (git-ignored with app/data).

import base64
import hashlib
import json
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from app import store

GENESIS = "0" * 64
KEY_DIR = store.DATA_DIR / "keys"
TSA_URL = os.environ.get("TRACEMARK_TSA_URL")
TSA_FILES = {"cacert.pem": "https://freetsa.org/files/cacert.pem",
             "tsa.crt": "https://freetsa.org/files/tsa.crt"}
FIELDS = ["seq", "created_at", "copy_id", "product_id", "seller_id",
          "sha256_original", "sha256_copy", "prev_hash"]


def init_table() -> None:
    conn = store._connect()
    conn.execute("""CREATE TABLE IF NOT EXISTS registrations (
        seq INTEGER PRIMARY KEY,
        created_at TEXT NOT NULL, copy_id INTEGER NOT NULL, product_id INTEGER NOT NULL,
        seller_id INTEGER NOT NULL, sha256_original TEXT NOT NULL, sha256_copy TEXT NOT NULL,
        prev_hash TEXT NOT NULL, record_hash TEXT NOT NULL, signature TEXT NOT NULL,
        tsa_token TEXT)""")
    conn.commit()
    conn.close()


# ─── keys ─────────────────────────────────────────────────────────

def _private_key() -> Ed25519PrivateKey:
    path = KEY_DIR / "ed25519_private.pem"
    if not path.exists():
        KEY_DIR.mkdir(parents=True, exist_ok=True)
        key = Ed25519PrivateKey.generate()
        path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                           serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
        os.chmod(path, 0o600)
    return serialization.load_pem_private_key(path.read_bytes(), password=None)


def public_key_pem() -> str:
    return _private_key().public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()


def _public_key() -> Ed25519PublicKey:
    return _private_key().public_key()


# ─── records ──────────────────────────────────────────────────────

def _sha256_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _record_hash(rec: dict) -> str:
    canonical = json.dumps({k: rec[k] for k in FIELDS}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def append(copy_id: int) -> dict:
    """Append a signed, chained record for a freshly issued copy."""
    init_table()
    copy = store.get_copy(copy_id)
    product = store.get_product(copy["product_id"])
    last = store._one("SELECT seq, record_hash FROM registrations ORDER BY seq DESC LIMIT 1")
    rec = {"seq": (last["seq"] + 1) if last else 1,
           "created_at": store._now(),
           "copy_id": copy_id, "product_id": product["id"], "seller_id": product["seller_id"],
           "sha256_original": _sha256_file(product["original_path"]),
           "sha256_copy": _sha256_file(copy["file_path"]),
           "prev_hash": last["record_hash"] if last else GENESIS}
    rec["record_hash"] = _record_hash(rec)
    rec["signature"] = base64.b64encode(_private_key().sign(bytes.fromhex(rec["record_hash"]))).decode()
    rec["tsa_token"] = _timestamp(rec["record_hash"]) if TSA_URL else None
    store._exec(f"INSERT INTO registrations ({', '.join(rec)}) VALUES ({', '.join('?' * len(rec))})",
                tuple(rec.values()))
    return rec


def records() -> list[dict]:
    init_table()
    return store._rows("SELECT * FROM registrations ORDER BY seq")


def record_for_copy(copy_id: int) -> dict | None:
    init_table()
    return store._one("SELECT * FROM registrations WHERE copy_id = ? ORDER BY seq LIMIT 1", (copy_id,))


# ─── verification ─────────────────────────────────────────────────

def verify_chain(upto_seq: int | None = None) -> dict:
    """
    Walk the log from the first record: every record_hash must match its fields,
    every prev_hash the previous record, every signature the server key.
    Returns {"ok", "checked", "first_error"}.
    """
    pub, prev = _public_key(), GENESIS
    rows = records()
    for i, r in enumerate(rows):
        if upto_seq is not None and r["seq"] > upto_seq:
            break
        problem = None
        if r["seq"] != i + 1:
            problem = f"record {r['seq']}: sequence gap (expected {i + 1})"
        elif r["prev_hash"] != prev:
            problem = f"record {r['seq']}: chain broken (prev_hash does not match record {r['seq'] - 1})"
        elif _record_hash(r) != r["record_hash"]:
            problem = f"record {r['seq']}: contents changed after signing"
        else:
            try:
                pub.verify(base64.b64decode(r["signature"]), bytes.fromhex(r["record_hash"]))
            except (InvalidSignature, ValueError):
                problem = f"record {r['seq']}: invalid signature"
        if problem:
            return {"ok": False, "checked": i, "first_error": problem}
        prev = r["record_hash"]
    checked = len([r for r in rows if upto_seq is None or r["seq"] <= upto_seq])
    return {"ok": True, "checked": checked, "first_error": None}


def verify_copy(copy_id: int) -> dict | None:
    """Everything the evidence report shows about one copy's registration."""
    rec = record_for_copy(copy_id)
    if rec is None:
        return None
    copy = store.get_copy(copy_id)
    product = store.get_product(rec["product_id"]) if copy else None
    files = {
        "original_matches": bool(product) and Path(product["original_path"]).exists()
        and _sha256_file(product["original_path"]) == rec["sha256_original"],
        "copy_matches": bool(copy) and Path(copy["file_path"]).exists()
        and _sha256_file(copy["file_path"]) == rec["sha256_copy"],
    }
    out = {"record": rec, "chain": verify_chain(rec["seq"]), **files, "tsa": None}
    if rec.get("tsa_token"):
        out["tsa"] = verify_timestamp(rec["record_hash"], rec["tsa_token"])
    return out


# ─── optional RFC 3161 trusted timestamp ──────────────────────────

def _openssl() -> str:
    for c in ("/opt/homebrew/bin/openssl", "/usr/local/bin/openssl", "openssl"):
        if c == "openssl" or Path(c).exists():
            return c
    return "openssl"


def _timestamp(record_hash: str) -> str | None:
    """Ask the TSA to timestamp record_hash; base64 DER reply, or None on any failure."""
    try:
        with tempfile.TemporaryDirectory() as d:
            q = Path(d) / "q.tsq"
            subprocess.run([_openssl(), "ts", "-query", "-digest", record_hash, "-sha256",
                            "-cert", "-no_nonce", "-out", str(q)],
                           check=True, capture_output=True, timeout=10)
            req = urllib.request.Request(TSA_URL, data=q.read_bytes(),
                                         headers={"Content-Type": "application/timestamp-query"})
            with urllib.request.urlopen(req, timeout=15) as r:
                return base64.b64encode(r.read()).decode()
    except Exception:
        return None


def verify_timestamp(record_hash: str, token_b64: str) -> dict:
    tsa_dir = KEY_DIR / "tsa"
    try:
        tsa_dir.mkdir(parents=True, exist_ok=True)
        for name, url in TSA_FILES.items():
            if not (tsa_dir / name).exists():
                with urllib.request.urlopen(url, timeout=15) as r:
                    (tsa_dir / name).write_bytes(r.read())
        with tempfile.TemporaryDirectory() as d:
            tsr = Path(d) / "r.tsr"
            tsr.write_bytes(base64.b64decode(token_b64))
            v = subprocess.run([_openssl(), "ts", "-verify", "-digest", record_hash, "-in", str(tsr),
                                "-CAfile", str(tsa_dir / "cacert.pem"),
                                "-untrusted", str(tsa_dir / "tsa.crt")],
                               capture_output=True, text=True, timeout=15)
            text = subprocess.run([_openssl(), "ts", "-reply", "-in", str(tsr), "-text"],
                                  capture_output=True, text=True, timeout=15).stdout
        when = next((l.split(":", 1)[1].strip() for l in text.splitlines()
                     if l.startswith("Time stamp:")), None)
        return {"ok": "Verification: OK" in v.stdout, "time": when}
    except Exception as e:
        return {"ok": False, "time": None, "error": str(e)}
