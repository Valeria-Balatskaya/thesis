# src/payload.py
# Watermark payload: product ID -> 63-bit codeword -> product ID.
#
# Layout of the 30 information bits (protected by BCH(63,30), t = 6):
#     [ 20-bit product ID | 10-bit keyed tag ]
#
# The tag is a truncated HMAC of the product ID. It makes detection *blind*
# (the ID is read straight from the image, no comparison against the catalogue)
# with a false-positive rate that can be calculated instead of guessed:
#
#   an unwatermarked image yields ~random bits
#   -> P(within 6 errors of some codeword)  = 2^30 * V(63,6) / 2^63  ~ 0.9 %
#   -> P(tag also matches)                  = 2^-10
#   -> P(false positive)                    ~ 8.8e-6 per image,
#      and lower still because the ID must also exist in the database.
#
# The codeword is XOR-ed with a keyed pseudo-random mask so that a decoder
# biased towards all-zeros / all-ones can never land on a valid codeword
# (the all-zero word is a codeword of every linear code).

import hashlib
import hmac
import os
from math import comb

import numpy as np

from src.bch import BCH

ID_BITS = 20
TAG_BITS = 10
_CODE = BCH(m=6, t=6)
assert _CODE.k == ID_BITS + TAG_BITS

PAYLOAD_BITS = _CODE.n          # 63 — the message length the network embeds
ECC_T = _CODE.t                 # 6  — correctable bit errors
MAX_PRODUCT_ID = (1 << ID_BITS) - 1

# Demo key. In a real deployment this is a server-side secret.
_KEY = os.environ.get("WATERMARK_KEY", "thesis-demo-key").encode()


def _int_to_bits(value: int, length: int) -> list[int]:
    return [(value >> (length - 1 - i)) & 1 for i in range(length)]


def _bits_to_int(bits) -> int:
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def _tag(product_id: int) -> int:
    digest = hmac.new(_KEY, f"product:{product_id}".encode(), hashlib.sha256).digest()
    return int.from_bytes(digest[:4], "big") >> (32 - TAG_BITS)


def _mask() -> np.ndarray:
    digest = hmac.new(_KEY, b"mask", hashlib.sha256).digest()
    bits = np.unpackbits(np.frombuffer(digest, dtype=np.uint8))
    return bits[:PAYLOAD_BITS].astype(np.uint8)


def encode_product_id(product_id: int) -> np.ndarray:
    """Product ID -> 63 payload bits (uint8 array of 0/1)."""
    if not 0 <= product_id <= MAX_PRODUCT_ID:
        raise ValueError(f"product_id must be in [0, {MAX_PRODUCT_ID}]")
    info = _int_to_bits(product_id, ID_BITS) + _int_to_bits(_tag(product_id), TAG_BITS)
    return _CODE.encode(info) ^ _mask()


def decode_payload(bits) -> dict:
    """
    Blind decode of 63 recovered bits.
    Returns product_id=None when no valid payload is present.
    """
    bits = np.asarray(bits).astype(np.uint8).reshape(-1)
    info, n_errors = _CODE.decode(bits ^ _mask())
    if info is None:
        return {"product_id": None, "errors_corrected": None, "reason": "uncorrectable"}
    product_id = _bits_to_int(info[:ID_BITS])
    if _bits_to_int(info[ID_BITS:]) != _tag(product_id):
        return {"product_id": None, "errors_corrected": n_errors, "reason": "tag_mismatch"}
    return {"product_id": product_id, "errors_corrected": n_errors, "reason": "ok"}


# ─── false-positive maths (used in tests and in the thesis) ───────

def binom_tail(n_bits: int, max_errors: int) -> float:
    """P(Binomial(n_bits, 0.5) <= max_errors): chance random bits match a
    given codeword with at most max_errors errors."""
    return sum(comb(n_bits, i) for i in range(max_errors + 1)) / 2 ** n_bits


def blind_false_positive_rate() -> float:
    """Theoretical P(random bits decode to a valid payload)."""
    decodable = 2 ** _CODE.k * binom_tail(_CODE.n, _CODE.t)
    return decodable * 2 ** -TAG_BITS


def max_errors_for_fpr(n_bits: int, n_candidates: int, fpr: float = 1e-6) -> int:
    """
    Largest error count that may be accepted when matching recovered bits
    against n_candidates known codewords while keeping the overall
    false-positive rate below fpr (union bound). Returns -1 if none is safe.
    """
    k = -1
    while k + 1 <= n_bits and n_candidates * binom_tail(n_bits, k + 1) <= fpr:
        k += 1
    return k
