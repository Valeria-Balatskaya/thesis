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
#   watermark_unbound  (only with verify_content=True) the ID decodes but the
#                      image does not show the registered picture: possible
#                      copy attack (watermark transplanted onto another image)
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


def align(suspect_rgb: np.ndarray, H: np.ndarray, original_rgb: np.ndarray,
          ignore_padding: bool = False):
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

    if ignore_padding:
        # Black padding added by the attacker (crop-and-pad, letterboxing) is
        # not part of the picture: leave it out where the original is not dark
        pad = (warped.max(axis=2) <= 3) & (original_rgb.max(axis=2) > 30)
        mask = mask & ~pad
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

    def content_bound(self, suspect: np.ndarray, product_id: int) -> dict:
        """
        Copy-attack check (Kutter et al. 2000): does the suspect actually show the
        registered picture of the product whose ID was read? A watermark
        transplanted onto another image decodes fine but fails this.
        """
        entry = self.products[product_id]
        for flipped in (False, True):
            img = suspect[:, ::-1].copy() if flipped else suspect
            m = match(fingerprint(img), entry["fp"])
            if m:
                _, coverage, similarity = align(img, m["H"], _load_rgb(entry["original_path"]),
                                                ignore_padding=True)
                if similarity >= MIN_SIMILARITY:
                    return {"ok": True, "inliers": m["inliers"], "similarity": round(similarity, 3)}
                return {"ok": False, "inliers": m["inliers"], "similarity": round(similarity, 3)}
        return {"ok": False, "inliers": 0, "similarity": None}

    def analyse(self, suspect_path: str, use_alignment: bool = True,
                verify_content: bool = False) -> dict:
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
            if verify_content:
                bound = self.content_bound(suspect, out["product_id"])
                stages.append({"stage": "content_bound", **bound})
                if not bound["ok"]:
                    # ID decodes, but the picture is not the registered one:
                    # a transplanted watermark, or the image is too altered to compare
                    out["verdict"] = "watermark_unbound"
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
                                                  _load_rgb(entry["original_path"]),
                                                  ignore_padding=verify_content)
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
                if verify_content and similarity < MIN_SIMILARITY:
                    # bits match after alignment, but the picture does not: e.g. a
                    # high-pass copy attack that also carries ghost edges of our photo
                    out["verdict"] = "watermark_unbound"
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


# ─── ownership dispute: informed (non-blind) check ────────────────

def informed_score(suspect_rgb: np.ndarray, original_rgb: np.ndarray,
                   watermarked_rgb: np.ndarray) -> float:
    """
    Correlation between what the suspect adds to the original and the exact
    residual WE embedded in one copy (watermarked - original). Used when the
    blind decoder fails because someone put another watermark on top: the
    server holds both the clean original and every copy it issued, so it can
    test for one specific copy's signal directly (informed detection, Cox et
    al. 1997). Large local changes (visible logos, banners) are clipped so they
    do not swamp the correlation. All three images must be in the same frame
    (align first if the suspect was cropped or warped).
    Compare against decoy residuals of copies never issued (see
    tests/test_rewatermark.py) to turn the score into a decision.
    """
    o = original_rgb.astype(np.float32)
    ref = (watermarked_rgb.astype(np.float32) - o).ravel()
    k = 4 * float(ref.std()) + 1e-6
    d = np.clip(suspect_rgb.astype(np.float32) - o, -k, k).ravel()
    d = d - d.mean()
    ref = ref - ref.mean()
    denom = float(np.linalg.norm(d) * np.linalg.norm(ref))
    return float(d @ ref / denom) if denom > 1e-6 else 0.0


class InformedChecker:
    """
    Calibrated version of informed_score (Part B2): is THIS copy's v2 residual
    present in a suspect? The score for copy_id is compared with a null
    distribution of n_null never-issued copies (random IDs outside the 20-bit
    space, so they can never be real), embedded the same way on the same original.

    Built once per original (the n_null decoy residuals cost one batched pass),
    then test() is a matrix product per suspect. Everything is computed at
    `side` px (longer side): the v2 residual is generated at 128 px and
    upsampled, so nothing is lost. Suspects must be in the original's frame
    (align first). Decision rule used in the thesis: z >= Z_THRESHOLD.
    """

    def __init__(self, model: HiddenV2Model, original_rgb: np.ndarray, copy_id: int,
                 n_null: int = 500, side: int = 256, embed_kwargs: dict | None = None,
                 seed: int = 0):
        kw = {"strength": 0.7, "mask_floor": 0.3, **(embed_kwargs or {})}
        h, w = original_rgb.shape[:2]
        scale = side / max(h, w)
        self.size = (max(1, round(w * scale)), max(1, round(h * scale)))
        o = cv2.resize(original_rgb, self.size, interpolation=cv2.INTER_AREA)
        self.o = o.astype(np.float32).transpose(2, 0, 1).reshape(-1) / 255
        x = torch.from_numpy(o.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
        x = x.to(model.device)
        rng = np.random.default_rng(seed)
        ids = [copy_id] + [int(v) for v in rng.integers(1 << 21, 1 << 30, n_null)]
        refs = []
        with torch.no_grad():
            for start in range(0, len(ids), 128):
                chunk = ids[start:start + 128]
                msgs = torch.tensor(np.stack([_codeword_any(c) for c in chunk]),
                                    dtype=torch.float32, device=model.device)
                wm = model.model.watermark(x.expand(len(chunk), -1, -1, -1), msgs,
                                           key=model.key(len(chunk)), **kw)
                refs.append((wm - x).cpu().numpy().reshape(len(chunk), -1))
        refs = np.concatenate(refs)
        self.clip = 4 * float(refs[0].std()) + 1e-6
        refs = refs - refs.mean(axis=1, keepdims=True)
        self.refs = refs / (np.linalg.norm(refs, axis=1, keepdims=True) + 1e-12)

    def test(self, suspect_rgb: np.ndarray) -> dict:
        s = cv2.resize(suspect_rgb, self.size, interpolation=cv2.INTER_AREA)
        d = s.astype(np.float32).transpose(2, 0, 1).reshape(-1) / 255 - self.o
        d = np.clip(d, -self.clip, self.clip)
        d = d - d.mean()
        nd = float(np.linalg.norm(d))
        if nd < 1e-9:                    # suspect identical to the clean original
            return {"score": 0.0, "z": 0.0, "p_empirical": 1.0, "null_mean": 0.0, "null_std": 0.0}
        scores = self.refs @ d / nd
        score, null = float(scores[0]), scores[1:]
        z = (score - null.mean()) / (null.std() + 1e-12)
        return {"score": round(score, 4), "z": round(float(z), 2),
                "p_empirical": (1 + int((null >= score).sum())) / (len(null) + 1),
                "null_mean": round(float(null.mean()), 4), "null_std": round(float(null.std()), 4)}


Z_THRESHOLD = 4.0     # calibrated in tests/test_overwrite_defences.py


def informed_test(model: HiddenV2Model, suspect_rgb: np.ndarray, original_rgb: np.ndarray,
                  copy_id: int, **kwargs) -> dict:
    """One-off InformedChecker(...).test(suspect)."""
    return InformedChecker(model, original_rgb, copy_id, **kwargs).test(suspect_rgb)


def _codeword_any(copy_id: int) -> np.ndarray:
    """63-bit message for any integer ID (decoys may exceed the 20-bit ID space)."""
    from src.payload import MAX_PRODUCT_ID
    if 0 <= copy_id <= MAX_PRODUCT_ID:
        return encode_product_id(copy_id)
    seed = np.random.default_rng(copy_id)
    return seed.integers(0, 2, PAYLOAD_BITS).astype(np.uint8)


def ownership_check(model: HiddenV2Model, suspect_rgb: np.ndarray, original_rgb: np.ndarray,
                    copy_id: int, dct_reference_rgb: np.ndarray | None = None,
                    checker: "InformedChecker | None" = None) -> dict:
    """
    Ownership evidence when the blind decoder cannot read our ID (e.g. someone
    re-protected the image with their own watermark). Two independent informed
    checks; the copy counts as present if either fires:
      v2 informed  calibrated z-score of our exact colour residual (InformedChecker)
      DCT layer    keyed luminance spread spectrum (src/dct_layer), if the copy has one
    The suspect is aligned to the original's frame first (SIFT/RANSAC); false
    positives measured 0/240 and 0/480 (tests/test_overwrite_defences.py).
    """
    from src import dct_layer
    suspect = suspect_rgb
    if suspect.shape != original_rgb.shape:
        fp = fingerprint(original_rgb)
        m = match(fingerprint(suspect), fp)
        if m is None:
            flipped = suspect[:, ::-1].copy()
            m = match(fingerprint(flipped), fp)
            suspect = flipped if m is not None else suspect
        if m is None:
            return {"aligned": False, "informed": None, "dct": None, "owned": False}
        suspect = align(suspect, m["H"], original_rgb)[0]
    checker = checker or InformedChecker(model, original_rgb, copy_id)
    informed = checker.test(suspect)
    dct = None
    if dct_reference_rgb is not None:
        dct = dct_layer.detect(suspect, dct_reference_rgb, copy_id)
    owned = informed["z"] >= Z_THRESHOLD or bool(dct and dct["detected"])
    return {"aligned": True, "informed": informed, "dct": dct, "owned": owned}
