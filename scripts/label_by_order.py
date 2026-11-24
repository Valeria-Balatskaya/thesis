# scripts/label_by_order.py
# Label real-world captures by the ORDER they were taken in, with the picture match as a check.
#
# Why: sort_captures.py identifies each capture by picture (SIFT + RANSAC). Several of our own
# product photos show the same product from similar angles, so the picture match can confuse
# siblings, and matches with few points can be wrong. A wrong label would later be counted as a
# watermark error. But every capture session follows a known order:
#   viewer   screenshot_*, phone_photo_*        to_capture/viewer.html item order
#   listing  page_screenshot, phone_screenshot   listing_01.html, listing_02.html, ...
#   send     whatsapp, telegram, messenger, ...  alphabetical order of the 30 PNGs (Finder drag)
# The captures (sorted by the time in their file names) are aligned to that order with dynamic
# programming that allows stray captures (junk), double captures and skipped items. The picture
# match scores every pairing, so the order and the picture must agree.
#
#   python scripts/label_by_order.py <capture_type> <inbox> [--order viewer|listing|send]
#       -> prints the proposed labels and writes data/real_world/_review/<capture_type>.html
#   python scripts/label_by_order.py <capture_type> <inbox> --apply [--device ..] [--app ..]
#       -> replaces this capture type's files in data/real_world/<capture_type>/ and its rows in
#          captures.csv (the inbox is never modified). Refuses while CONFLICT rows remain,
#          unless --force (only after you checked them in the review page).
#
# Statuses:
#   agree      order and picture point to the same image
#   order      picture match too weak to decide (< 25 points); label from the order
#   CONFLICT   the picture clearly points to another image (>= 60 points and >= 3x the
#              assigned image's points): check visually in the review page
#   junk       a stray capture (no picture match, not needed by the order): not used

import argparse
import csv
import datetime
import html
import os
import re
import shutil
from math import log1p

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

EXT = (".png", ".jpg", ".jpeg", ".heic", ".heif", ".webp")
FIELDS = ["file", "capture_type", "copy_id", "item", "gt_source", "inliers", "runner_up",
          "width", "height", "device", "app", "notes", "original_name", "added"]
FEATURE_SIDE = 800                      # as in src/defence.py
MIN_POINTS = 25                         # as src/defence.MIN_INLIERS
JUNK, DUP, SKIP = -1.0, -1.5, -3.0      # alignment penalties (log-point units)
_sift = cv2.SIFT_create(nfeatures=3000)
_matcher = cv2.BFMatcher(cv2.NORM_L2)


def load(path):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((1600, 1600), Image.LANCZOS)
    return np.asarray(img)


def fingerprint(rgb):
    h, w = rgb.shape[:2]
    scale = min(1.0, FEATURE_SIDE / max(h, w))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if scale < 1:
        gray = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    kp, desc = _sift.detectAndCompute(gray, None)
    pts = np.array([k.pt for k in kp], dtype=np.float32).reshape(-1, 2) / scale
    return pts, desc


def points(fa, fb):
    """RANSAC inliers between two fingerprints (0 if no plausible match)."""
    (pa, da), (pb, db) = fa, fb
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return 0
    pairs = _matcher.knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return 0
    H, mask = cv2.findHomography(pa[[m.queryIdx for m in good]], pb[[m.trainIdx for m in good]],
                                 cv2.RANSAC, 4.0)
    if H is None:
        return 0
    det = np.linalg.det(H[:2, :2])
    return int(mask.sum()) if 0.02 < det < 50 else 0


def natural_key(name):
    # "WhatsApp Image 2026-11-24 at 22.28.53 (10).jpeg": time digits, then the (n) suffix;
    # a name without a suffix comes before "(1)".
    return [int(x) for x in re.findall(r"\d+", name)], name


def expected_order(root, kind):
    tc = f"{root}/to_capture"
    if kind == "viewer":
        js = open(f"{tc}/viewer.html").read()
        m = re.search(r"const items=\[(.*?)\];", js)
        return [x.strip().strip('"') for x in m.group(1).split(",")]
    if kind == "listing":
        out, k = [], 1
        while os.path.exists(f"{tc}/listing_{k:02d}.html"):
            out.append(re.search(r'<img src="([^"]+)"', open(f"{tc}/listing_{k:02d}.html").read()).group(1))
            k += 1
        return out
    return sorted(f for f in os.listdir(tc) if f.endswith(".png") and "/" not in f)


def default_order(capture_type):
    if capture_type in ("page_screenshot", "phone_screenshot"):
        return "listing"
    if capture_type.startswith(("screenshot", "phone_photo", "shop_grid")):
        return "viewer"
    return "send"


def align(S):
    """S[i][k] = picture score of capture i vs expected position k. Returns a label per capture
    (expected index, or None for junk), maximising order-consistent score."""
    n, m = S.shape
    NEG = -1e9
    dp = np.full((n + 1, m + 1), NEG)        # dp[i][j]: i captures used, positions < j settled
    back = {}
    dp[0][0] = 0.0
    for j in range(1, m + 1):                # positions skipped before the first capture
        dp[0][j] = dp[0][j - 1] + SKIP
        back[(0, j)] = ("skip", 0, j - 1)
    for i in range(n):
        for j in range(m + 1):
            if dp[i][j] <= NEG / 2:
                continue
            cand = [(dp[i][j] + JUNK, ("junk", i, j), j)]
            if j >= 1:                          # duplicate of the last assigned position
                cand.append((dp[i][j] + S[i][j - 1] + DUP, ("dup", i, j), j))
            for skip in range(0, 3):            # next position, or skip 1-2 positions
                k = j + skip
                if k < m:
                    cand.append((dp[i][j] + S[i][k] + SKIP * skip, ("take", i, j, k), k + 1))
            for val, how, nj in cand:
                if val > dp[i + 1][nj]:
                    dp[i + 1][nj] = val
                    back[(i + 1, nj)] = how
    # positions left after the last capture are skipped
    best_j = max(range(m + 1), key=lambda j: dp[n][j] + SKIP * (m - j))
    labels = [None] * n
    i, j = n, best_j
    while i > 0:
        how = back[(i, j)]
        if how[0] == "junk":
            labels[i - 1] = None
            i, j = how[1], how[2]
        elif how[0] == "dup":
            labels[i - 1] = j - 1
            i, j = how[1], how[2]
        elif how[0] == "take":
            labels[i - 1] = how[3]
            i, j = how[1], how[2]
        else:                                   # leading skips (i == 0 never reaches here)
            break
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_type")
    ap.add_argument("inbox")
    ap.add_argument("--order", choices=["viewer", "listing", "send"])
    ap.add_argument("--root", default="data/real_world")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--force", action="store_true", help="apply even with CONFLICT rows (after checking them)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--app", default=None)
    ap.add_argument("--notes", default=None)
    a = ap.parse_args()
    root, ctype = a.root, a.capture_type
    order = a.order or default_order(ctype)
    manifest = list(csv.DictReader(open(f"{root}/manifest.csv")))
    by_file = {os.path.basename(r["file"]): r for r in manifest}
    expected = [f for f in expected_order(root, order) if f in by_file]
    names = sorted((f for f in os.listdir(a.inbox) if f.lower().endswith(EXT)
                    and not f.startswith(("x_", "."))), key=natural_key)
    if not names:
        raise SystemExit(f"no captures in {a.inbox}")
    print(f"{ctype}: {len(names)} captures, order '{order}' with {len(expected)} images")

    ref = {f: fingerprint(load(f"{root}/originals/{by_file[f]['item']}.png")) for f in set(expected)}
    files = sorted(set(expected))
    P = np.zeros((len(names), len(files)), dtype=int)
    for i, nm in enumerate(names):
        fp = fingerprint(load(os.path.join(a.inbox, nm)))
        for k, f in enumerate(files):
            P[i, k] = points(fp, ref[f])
    P[P < MIN_POINTS] = 0
    col = {f: k for k, f in enumerate(files)}
    S = np.array([[log1p(P[i, col[e]]) - 2.0 for e in expected] for i in range(len(names))])
    labels = align(S)

    rows, counts = [], {"agree": 0, "order": 0, "CONFLICT": 0, "junk": 0}
    for i, nm in enumerate(names):
        best_k = int(P[i].argmax())
        best_f, best_p = files[best_k], int(P[i, best_k])
        if labels[i] is None:
            status, f, ap_ = "junk", None, 0
        else:
            f = expected[labels[i]]
            ap_ = int(P[i, col[f]])
            if best_p >= MIN_POINTS and best_f == f:
                status = "agree"
            elif best_p >= 60 and best_p >= 3 * max(ap_, 1):
                status = "CONFLICT"
            else:
                status = "order"
        counts[status] += 1
        rows.append({"name": nm, "file": f, "status": status, "points": ap_,
                     "best": best_f, "best_points": best_p})
    missing = [e for e in expected if e not in {r["file"] for r in rows}]
    dups = sorted({r["file"] for r in rows if r["file"]
                   and sum(x["file"] == r["file"] for x in rows) > 1})

    print(f"{'capture':<46}{'label':<14}{'status':<10}{'points':>7}   picture says")
    for r in rows:
        lab = by_file[r["file"]]["item"] if r["file"] else "-"
        says = f"{by_file[r['best']]['item']} ({r['best_points']})" if r["best_points"] else "no match"
        print(f"{r['name'][:45]:<46}{lab:<14}{r['status']:<10}{r['points']:>7}   {says}")
    print(f"\nagree {counts['agree']}, label from order {counts['order']}, CONFLICT {counts['CONFLICT']}, "
          f"junk {counts['junk']}; doubles: {', '.join(by_file[d]['item'] for d in dups) or 'none'}; "
          f"not captured: {', '.join(by_file[e]['item'] for e in missing) or 'none'}")

    os.makedirs(f"{root}/_review", exist_ok=True)
    review = f"{root}/_review/{ctype}.html"
    def img(p):
        return f'<img src="file://{html.escape(os.path.abspath(p))}">'
    trs = "".join(
        f'<tr class="{r["status"]}"><td>{html.escape(r["name"])}<br><b>{r["status"]}</b></td>'
        f'<td>{img(os.path.join(a.inbox, r["name"]))}</td>'
        f'<td>{(by_file[r["file"]]["item"] + "<br>" + img(root + "/to_capture/" + r["file"])) if r["file"] else "junk"}</td>'
        f'<td>{(by_file[r["best"]]["item"] + " (" + str(r["best_points"]) + ")<br>" + img(root + "/to_capture/" + r["best"])) if r["best_points"] else "no match"}</td></tr>'
        for r in rows)
    open(review, "w").write(
        "<!doctype html><meta charset=utf-8><title>Review " + ctype + "</title><style>"
        "body{font:14px sans-serif;margin:16px}td{border-bottom:1px solid #ddd;padding:6px;vertical-align:top}"
        "img{max-width:220px;max-height:160px;display:block;margin-top:4px}"
        "tr.CONFLICT{background:#fde2e2}tr.order{background:#fff6d6}tr.junk{background:#eee;color:#777}"
        "</style><h2>" + ctype + "</h2><p>Check every yellow (order only) and red (CONFLICT) row: "
        "the capture (2nd column) must show the same photo as the label (3rd column).</p>"
        "<table><tr><th>capture</th><th></th><th>label (from order)</th><th>picture match says</th></tr>"
        + trs + "</table>")
    print(f"review page: {review}  (open it in a browser)")

    if not a.apply:
        print("dry run: nothing changed. Check the review page, then re-run with --apply.")
        return
    if counts["CONFLICT"] and not a.force:
        raise SystemExit("CONFLICT rows: check them in the review page; re-run with --apply --force "
                         "only if the order labels are right.")
    log = f"{root}/captures.csv"
    old = list(csv.DictReader(open(log))) if os.path.exists(log) else []
    prev = next((r for r in old if r["capture_type"] == ctype), {})
    keep = [r for r in old if r["capture_type"] != ctype]
    dest = f"{root}/{ctype}"
    if os.path.isdir(dest):
        shutil.rmtree(dest)                   # only copies live here; the inbox keeps the originals
    os.makedirs(dest)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    count, new = {}, []
    for r in rows:
        if r["file"] is None:
            continue
        m = by_file[r["file"]]
        cid = int(m["copy_id"])
        count[cid] = count.get(cid, 0) + 1
        ext = os.path.splitext(r["name"])[1].lower()
        out = f"{dest}/{cid}_{count[cid]}{ext}"
        shutil.copy2(os.path.join(a.inbox, r["name"]), out)
        w, h = Image.open(out).size
        new.append({"file": os.path.relpath(out, root), "capture_type": ctype, "copy_id": cid,
                    "item": m["item"], "gt_source": {"agree": "order+picture", "order": "order",
                                                     "CONFLICT": "order (checked by hand)"}[r["status"]],
                    "inliers": r["points"], "runner_up": r["best_points"] if r["best"] != r["file"] else "",
                    "width": w, "height": h,
                    "device": a.device if a.device is not None else prev.get("device", ""),
                    "app": a.app if a.app is not None else prev.get("app", ""),
                    "notes": a.notes if a.notes is not None else prev.get("notes", ""),
                    "original_name": r["name"], "added": now})
    for x in sorted(os.listdir(a.inbox)):     # unrelated photos added on purpose
        if x.startswith("x_") and x.lower().endswith(EXT):
            out = f"{dest}/{x}"
            shutil.copy2(os.path.join(a.inbox, x), out)
            w, h = Image.open(out).size
            new.append({"file": os.path.relpath(out, root), "capture_type": ctype, "copy_id": "",
                        "item": "unrelated", "gt_source": "unrelated", "inliers": "", "runner_up": "",
                        "width": w, "height": h, "device": prev.get("device", ""), "app": prev.get("app", ""),
                        "notes": prev.get("notes", ""), "original_name": x, "added": now})
    with open(log, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(keep + new)
    print(f"applied: {len(new)} captures of {ctype} in {dest}/ (replaced {len(old) - len(keep)} old rows)")


if __name__ == "__main__":
    main()
