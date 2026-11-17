# src/tiles.py
# Spatial redundancy experiment (docs/spatial_redundancy.md, Part B3). No training: the
# v3 network embeds its 63-bit codeword separately in each tile of a 2x2 grid.
#
#   embed    tile t gets encode_product_id(id) XOR mask_t (mask_t = HMAC(key, "tile:t")),
#            embedded by v3 on the tile crop alone; tiles overlap by OVERLAP of their size
#            and are blended with raised-cosine weights that sum to 1 (no seams)
#   read     soft combining: the decoder's logits of every tile, sign-flipped by mask_t,
#            are added before the hard decision (one "combined" word); plus every tile on
#            its own; plus the whole image read as if it were tile t (a crop that kept about
#            one tile). Any of them passing BCH + HMAC decides (blind read)
#
# Every extra read is another chance of a false positive: blind attempts here are
# 1 combined + 4 tiles + 4 whole-as-tile + aspect retries of the combined word only
# (8) = 17, against 9 for the plain v3 read (tests/test_tiles.py counts them on 1,000
# negatives). TiledModel behaves like HiddenV2Model for DefencePipeline: after alignment
# decode_variants() hands it the combined word and the 4 tile words to verify.

import hashlib
import hmac

import numpy as np
import torch
from PIL import Image

from src.hidden_v2_inference import HiddenV2Model, ASPECT_RATIOS
from src.payload import encode_product_id, decode_payload, PAYLOAD_BITS, _KEY

GRID = 2
OVERLAP = 0.10


def tile_mask(t: int) -> np.ndarray:
    d = hmac.new(_KEY, f"tile:{t}".encode(), hashlib.sha256).digest()
    return np.unpackbits(np.frombuffer(d, dtype=np.uint8))[:PAYLOAD_BITS].astype(np.uint8)


MASKS = [tile_mask(t) for t in range(GRID * GRID)]
SIGNS = [torch.from_numpy(1.0 - 2.0 * m.astype(np.float32)) for m in MASKS]   # -1 where mask bit is 1


def _spans(n: int):
    """Two overlapping 1-D intervals covering [0, n) and their blend weights."""
    half, ov = n / 2, OVERLAP * n / 2
    a0, a1 = 0, int(round(half + ov))
    b0, b1 = int(round(half - ov)), n
    pos = np.arange(n, dtype=np.float32)
    # raised cosine across the overlap [b0, a1): weight of the first interval goes 1 -> 0
    ramp = np.clip((pos - b0) / max(1, a1 - b0), 0, 1)
    w_a = 0.5 * (1 + np.cos(np.pi * ramp))
    w_a[pos >= a1] = 0
    return [(a0, a1, w_a), (b0, b1, 1 - w_a)]


def boxes(h: int, w: int):
    """[(y0, y1, x0, x1, weight HxW)] for the GRID x GRID tiles, row-major."""
    out = []
    for y0, y1, wy in _spans(h):
        for x0, x1, wx in _spans(w):
            out.append((y0, y1, x0, x1, wy[:, None] * wx[None, :]))
    return out


class TiledModel:
    """v3 checkpoint used tile by tile. Same interface DefencePipeline needs."""

    def __init__(self, checkpoint: str, device: str = "cpu"):
        self.base = HiddenV2Model(checkpoint, device)
        self.device, self.key_dim, self.try_mirror = device, 0, False

    def key(self, batch=1, key=None):
        return None

    # ─── embed ────────────────────────────────────────────────────
    @torch.no_grad()
    def embed_product(self, image_path, product_id, output_path, strength=1.0, mask_floor=None, **_):
        x = self.base._load(image_path)
        _, _, h, w = x.shape
        code = encode_product_id(product_id)
        total = torch.zeros_like(x)
        for t, (y0, y1, x0, x1, wt) in enumerate(boxes(h, w)):
            crop = x[:, :, y0:y1, x0:x1]
            msg = torch.tensor(code ^ MASKS[t], dtype=torch.float32).view(1, -1)
            r = self.base.model.watermark(crop, msg, strength, mask_floor=mask_floor) - crop
            total[:, :, y0:y1, x0:x1] += r * torch.from_numpy(wt[y0:y1, x0:x1]).float()
        out = torch.clamp(x + total, 0, 1)
        arr = (out[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(arr).save(output_path, format="PNG")
        return {"product_id": product_id, "output_path": output_path}

    # ─── read ─────────────────────────────────────────────────────
    @torch.no_grad()
    def _logits(self, x):
        return self.base.model.decode(x, None).reshape(-1)

    @torch.no_grad()
    def _tile_logits(self, x):
        """Unmasked logits (sign = codeword bit) of each tile of x's own grid."""
        _, _, h, w = x.shape
        return [self._logits(x[:, :, y0:y1, x0:x1]) * SIGNS[t]
                for t, (y0, y1, x0, x1, _) in enumerate(boxes(h, w))]

    @staticmethod
    def _bits(logits):
        return (logits > 0).numpy().astype(np.uint8)

    def decode_variants(self, x):
        """[combined, tile 0..3] codeword estimates (unmasked) for candidate verification."""
        tl = self._tile_logits(x)
        return [self._bits(sum(tl))] + [self._bits(l) for l in tl]

    def _decode_tensor(self, x, key=None):
        return self.decode_variants(x)[0]

    def read_product(self, image_path, try_aspects=True, **_):
        x = self.base._load(image_path)
        tl = self._tile_logits(x)
        whole = self._logits(x)
        tries = [("combined", self._bits(sum(tl)))]
        tries += [(f"tile{t}", self._bits(l)) for t, l in enumerate(tl)]
        tries += [(f"whole_as_tile{t}", self._bits(whole * SIGNS[t])) for t in range(len(SIGNS))]
        for how, bits in tries:
            d = decode_payload(bits)
            if d["product_id"] is not None:
                return {**d, "bits": tries[0][1], "aspect": None, "how": how, "mirrored": False}
        if try_aspects:
            _, _, h, w = x.shape
            for aw, ah in ASPECT_RATIOS:
                nw, nh = max(w, round(h * aw / ah)), max(h, round(w * ah / aw))
                if (nw, nh) == (w, h):
                    continue
                canvas = torch.full((1, 3, nh, nw), 0.5)
                top, left = (nh - h) // 2, (nw - w) // 2
                canvas[:, :, top:top + h, left:left + w] = x
                d = decode_payload(self._bits(sum(self._tile_logits(canvas))))
                if d["product_id"] is not None:
                    return {**d, "bits": tries[0][1], "aspect": f"{aw}:{ah}", "how": "combined", "mirrored": False}
        return {**decode_payload(tries[0][1]), "bits": tries[0][1], "aspect": None, "how": None, "mirrored": False}
