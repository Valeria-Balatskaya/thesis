# src/defence.py
# Retrieve - Align - Verify: layered detection for images that have been
# cropped, warped, pasted into something else, or edited.
#
# The HiDDeN v2 watermark is read blind, but it is position-dependent: an
# off-centre crop, a perspective change or a screenshot inside a web page
# misaligns it and the blind read fails. The server, however, still holds the
# seller's original. So detection runs in stages:
#
#   1. BLIND      read the watermark directly                 (src/hidden_v2_inference)
#   2. RETRIEVE   find which registered original the suspect shows, by matching
#                 local image features (SIFT) with a RANSAC homography
#   3. ALIGN      warp the suspect back into the original's frame; pixels the
#                 suspect does not cover are filled from the UN-watermarked
#                 original (which carries no watermark, so it cannot create one)
#   4. VERIFY     read the watermark again from the aligned image. Because
#                 retrieval has narrowed the search to one product, the bits can
#                 also be compared with that product's known codewords, which
#                 tolerates ~12 bit errors instead of the 6 BCH corrects, at a
#                 calculated false-positive rate (src/payload.max_errors_for_fpr)
#
# Verdicts, strongest first:
#   watermark          blind read succeeded
#   watermark_aligned  read succeeded after alignment -> proves the suspect
#                      derives from the published, watermarked copy
#   fingerprint_only   same picture as a registered original, watermark not
#                      readable (e.g. AI regeneration). Proves visual identity,
#                      NOT which copy leaked
#   none
#
# References:
#   Lowe (2004). Distinctive image features from scale-invariant keypoints. IJCV.
#   Fischler & Bolles (1981). Random sample consensus. CACM.

import cv2
import numpy as np
import torch
from PIL import Image

from src.hidden_v2_inference import HiddenV2Model
from src.payload import encode_product_id, decode_payload, max_errors_for_fpr, PAYLOAD_BITS

FEATURE_SIDE = 800        # images are resized so the longer side is this, for SIFT
MIN_INLIERS = 25          # RANSAC inliers needed to call two images "the same picture"
MIN_SIMILARITY = 0.65     # gradient agreement of the aligned suspect with the original.
                          # Measured on 47 product photos: different shots of the same
                          # product reach at most 0.61, true copies under attack have a
                          # 5th percentile of 0.67 (see CLAUDE.md)
SIM_SIDE = 256
VERIFY_FPR = 1e-6         # false-positive budget of candidate verification
TOP_K = 5                 # fingerprint candidates that get aligned and verified

_sift = cv2.SIFT_create(nfeatures=3000)
_matcher = cv2.BFMatcher(cv2.NORM_L2)


def _load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def fingerprint(rgb: np.ndarray) -> dict:
    """SIFT keypoints/descriptors, keypoint coordinates in full-resolution pixels."""
    h, w = rgb.shape[:2]
    scale = FEATURE_SIDE / max(h, w)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if scale < 1:
        gray = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
    keypoints, desc = _sift.detectAndCompute(gray, None)
    pts = np.array([k.pt for k in keypoints], dtype=np.float32).reshape(-1, 2) / scale
    return {"pts": pts, "desc": desc, "size": (w, h)}


def match(suspect_fp: dict, ref_fp: dict) -> dict | None:
    """Homography mapping suspect pixels -> reference pixels, or None."""
    if suspect_fp["desc"] is None or ref_fp["desc"] is None:
        return None
    if len(suspect_fp["desc"]) < 8 or len(ref_fp["desc"]) < 8:
        return None
    pairs = _matcher.knnMatch(suspect_fp["desc"], ref_fp["desc"], k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < MIN_INLIERS:
        return None
    src = suspect_fp["pts"][[m.queryIdx for m in good]]
    dst = ref_fp["pts"][[m.trainIdx for m in good]]
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None:
        return None
    inliers = int(mask.sum())
    if inliers < MIN_INLIERS:
        return None
    # Reject degenerate warps (mirrored / collapsed / wildly scaled)
    det = np.linalg.det(H[:2, :2])
    if not 0.02 < det < 50:
        return None
    return {"H": H, "inliers": inliers, "matches": len(good)}


def align(suspect_rgb: np.ndarray, H: np.ndarray, original_rgb: np.ndarray):
    """
    Warp the suspect into the original's frame.
    Returns (aligned image, coverage fraction, structural similarity to original).
    """
    h, w = original_rgb.shape[:2]
    warped = cv2.warpPerspective(suspect_rgb, H, (w, h), flags=cv2.INTER_CUBIC)
    mask = cv2.warpPerspective(np.full(suspect_rgb.shape[:2], 255, np.uint8), H, (w, h),
                               flags=cv2.INTER_NEAREST) > 0
    aligned = np.where(mask[:, :, None], warped, original_rgb)

    # Structure check on the covered area: is this the same photograph, or the
    # same object photographed again? Correlate image gradients: they survive
    # colour, brightness and compression changes, but a second shot of the same
    # product (other pose, other hand position) does not line up edge for edge.
    # Plain intensity correlation is not enough: two studio shots on a white
    # background correlate at 0.7-0.85 even when they are different photos.
    size = (SIM_SIDE, SIM_SIDE)

    def grads(img):
        g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), size, interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.0)
        return cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)

    m = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    m = cv2.erode(m, np.ones((5, 5), np.uint8)) > 0
    if m.sum() < 200:
        return aligned, float(mask.mean()), 0.0
    ax, ay = grads(warped)
    bx, by = grads(original_rgb)
    denom = float(np.sqrt((ax ** 2 + ay ** 2)[m].sum() * (bx ** 2 + by ** 2)[m].sum()))
    similarity = float((ax * bx + ay * by)[m].sum() / denom) if denom > 1e-6 else 0.0
    return aligned, float(mask.mean()), similarity


class DefencePipeline:
    """
    Registry of protected products + staged detection.
    A product can have several *copies*, each with its own watermark ID
    (e.g. "public listing", "sent to reseller X"), so a leak can be traced
    to the channel it came from.
    """

    def __init__(self, model: HiddenV2Model):
        self.model = model
        self.products: dict[int, dict] = {}   # product_id -> {original_path, fp, copy_ids}
        self.copy_to_product: dict[int, int] = {}

    # ─── registry ─────────────────────────────────────────────────

    def register(self, product_id: int, original_path: str, copy_ids: list[int]) -> None:
        entry = self.products.get(product_id)
        if entry is None or entry["original_path"] != original_path:
            entry = {"original_path": original_path,
                     "fp": fingerprint(_load_rgb(original_path)), "copy_ids": []}
            self.products[product_id] = entry
        entry["copy_ids"] = list(copy_ids)
        for cid in copy_ids:
            self.copy_to_product[cid] = product_id

    def forget(self, product_id: int) -> None:
        entry = self.products.pop(product_id, None)
        if entry:
            for cid in entry["copy_ids"]:
                self.copy_to_product.pop(cid, None)

    # ─── detection ────────────────────────────────────────────────

    def _read(self, rgb: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
        return self.model._decode_tensor(x.to(self.model.device))

    def _verify(self, bits: np.ndarray, product_id: int) -> dict | None:
        """Blind decode first; otherwise compare with this product's codewords."""
        blind = decode_payload(bits)
        if blind["product_id"] in self.copy_to_product:
            return {"copy_id": blind["product_id"], "bit_errors": blind["errors_corrected"],
                    "method": "bch"}
        copy_ids = self.products[product_id]["copy_ids"]
        if not copy_ids:
            return None
        limit = max_errors_for_fpr(PAYLOAD_BITS, len(copy_ids) * TOP_K, VERIFY_FPR)
        errors = {cid: int((bits != encode_product_id(cid)).sum()) for cid in copy_ids}
        best = min(errors, key=errors.get)
        if errors[best] <= limit:
            return {"copy_id": best, "bit_errors": errors[best], "method": "candidate",
                    "limit": limit}
        return None

    def analyse(self, suspect_path: str, use_alignment: bool = True) -> dict:
        suspect = _load_rgb(suspect_path)
        stages = []
        out = {"verdict": "none", "copy_id": None, "product_id": None,
               "stages": stages, "aligned": None, "bits": None}

        # 1. blind
        blind = self.model.read_product(suspect_path)
        known = blind["product_id"] in self.copy_to_product
        stages.append({"stage": "blind", "ok": known,
                       "bit_errors": blind["errors_corrected"] if known else None,
                       "aspect": blind.get("aspect"),
                       "note": None if known or blind["product_id"] is None
                       else "decoded an ID that is not registered"})
        out["bits"] = blind["bits"]
        if known:
            out.update(verdict="watermark", copy_id=blind["product_id"],
                       product_id=self.copy_to_product[blind["product_id"]])
            return out
        if not use_alignment or not self.products:
            return out

        # 2. retrieve (also try the mirror image — flipping is a common evasion)
        candidates = []
        for flipped in (False, True):
            img = suspect[:, ::-1].copy() if flipped else suspect
            fp = fingerprint(img)
            for pid, entry in self.products.items():
                m = match(fp, entry["fp"])
                if m:
                    candidates.append({**m, "product_id": pid, "flipped": flipped, "img": img})
            if candidates:
                break
        candidates.sort(key=lambda c: c["inliers"], reverse=True)
        if not candidates:
            stages.append({"stage": "retrieve", "ok": False})
            return out

        # 3 + 4. align and verify the best few candidates
        best_visual = None
        for cand in candidates[:TOP_K]:
            entry = self.products[cand["product_id"]]
            aligned, coverage, similarity = align(cand["img"], cand["H"],
                                                  _load_rgb(entry["original_path"]))
            info = {"product_id": cand["product_id"], "inliers": cand["inliers"],
                    "flipped": cand["flipped"], "coverage": round(coverage, 3),
                    "similarity": round(similarity, 3)}
            bits = self._read(aligned)
            verified = self._verify(bits, cand["product_id"])
            if verified:
                out["bits"] = bits
                stages.append({"stage": "retrieve", "ok": True, **info})
                stages.append({"stage": "align_verify", "ok": True, **verified})
                out.update(verdict="watermark_aligned", copy_id=verified["copy_id"],
                           product_id=self.copy_to_product[verified["copy_id"]],
                           aligned=aligned)
                return out
            # Near-identical photos (same object, same session) can all match;
            # the one whose structure agrees best is the picture itself.
            if similarity >= MIN_SIMILARITY and (best_visual is None
                                                 or similarity > best_visual[0]["similarity"]):
                best_visual = (info, aligned)

        if best_visual:
            info, aligned = best_visual
            stages.append({"stage": "retrieve", "ok": True, **info})
            stages.append({"stage": "align_verify", "ok": False})
            out.update(verdict="fingerprint_only", product_id=info["product_id"], aligned=aligned)
        else:
            top = candidates[0]
            stages.append({"stage": "retrieve", "ok": False, "inliers": top["inliers"],
                           "note": "features matched but the pictures differ "
                                   "(same object, different photo?)"})
        return out
