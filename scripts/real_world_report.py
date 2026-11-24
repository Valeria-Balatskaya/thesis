# scripts/real_world_report.py
# output/thesis/real_world.md: real captures (tests/test_real_world.py -> output/real_world/rows.csv) next to
# their closest simulation from tests/test_v3_vs_v2.py (output/v3_vs_v2_robustness.csv, system v3, strength 1.0).
# Wilson 95% intervals. Run: python scripts/real_world_report.py

import csv
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.stats import pct_ci, wilson

ROWS = "output/real_world/rows.csv"
SIM = "output/v3_vs_v2_robustness.csv"
CAPT = "data/real_world/captures.csv"
OUT = "output/thesis/real_world.md"

# capture type -> (title, closest simulations in tests/test_v3_vs_v2.py, docs/real_world_protocol.md)
PAIRS = {
    "screenshot_100": ("Mac screenshot, Chrome 100 %", ["native_screenshot", "screenshot"]),
    "page_screenshot": ("Mac full-screen screenshot of a listing page", ["page_screenshot"]),
    "phone_photo_front": ("iPhone photo of the screen, straight on", ["screen_photo", "perspective"]),
    "whatsapp": ("WhatsApp Desktop, standard quality", ["social_recompress", "jpeg_q70"]),
    "telegram": ("Telegram, compressed", ["social_recompress", "jpeg_q70"]),
}


def ci(k, n):
    return pct_ci(k, n) if n else "n/a"


real = list(csv.DictReader(open(ROWS)))
sim = [r for r in csv.DictReader(open(SIM)) if r["system"] == "v3"]
meta = {(r["capture_type"], os.path.basename(r["file"]).split("_")[0]): r for r in csv.DictReader(open(CAPT))}

L = ["# Real-world captures vs simulations (v3, strength 1.0)", "",
     "Source: `tests/test_real_world.py --informed` on `data/real_world` (20 products of the 30-item set, one "
     "capture per product and type, 10 own + 10 ABO; 10 clean controls per type). Simulations: "
     "`tests/test_v3_vs_v2.py`, 30 own + 30 ABO + 30 COCO photos, same model and strength. "
     "Wilson 95 % intervals. Regenerate with `python scripts/real_world_report.py`.", ""]

# ---- 1. real captures
L += ["## 1. Real captures", "",
      "| Capture type | n | Blind read | Full pipeline | Wrong copy | Verdicts | False positives (10 controls) |",
      "|---|---|---|---|---|---|---|"]
tot = Counter()
for t, (title, _) in PAIRS.items():
    p = [r for r in real if r["capture_type"] == t and r["kind"] == "protected"]
    c = [r for r in real if r["capture_type"] == t and r["kind"] == "control"]
    n = len(p)
    kb, kf, kw = (sum(int(r[k]) for r in p) for k in ("blind", "full", "wrong"))
    v = Counter(r["verdict"] for r in p)
    fp = sum(int(r["false_positive"]) for r in c)
    tot.update(n=n, blind=kb, full=kf, wrong=kw, fp=fp, ctrl=len(c))
    L.append(f"| {t} ({title}) | {n} | {ci(kb, n)} | {ci(kf, n)} | {kw}/{n} | "
             + ", ".join(f"{k} {x}" for k, x in sorted(v.items())) + f" | {fp}/{len(c)} |")
lo, hi = wilson(tot["fp"], tot["ctrl"])
L.append(f"| **all** | {tot['n']} | {ci(tot['blind'], tot['n'])} | {ci(tot['full'], tot['n'])} | {tot['wrong']}/{tot['n']} | | "
         f"{tot['fp']}/{tot['ctrl']} (upper bound {100 * hi:.1f} %) |")
L += ["", "The same 20 products and the same 10 controls appear in every capture type, so the types are not independent "
      "samples and the 50 control scans are 10 distinct pictures scanned five times.", ""]

# ---- 2. comparison
def rel(k, n, ks, ns):
    """Real (k/n) vs simulation (ks/ns): 'below' / 'above' if the Wilson intervals do not overlap, else 'overlap'."""
    lo, hi = wilson(k, n)
    slo, shi = wilson(ks, ns)
    return "below" if hi < slo else "above" if lo > shi else "overlap"


L += ["## 2. Each type vs its closest simulation", "",
      "Interval relation: `overlap` = the real interval overlaps the simulation's (own + ABO, n = 60); `below` / `above` = the "
      "intervals are apart. Blind and full are judged separately.", "",
      "| Real capture | Real: blind / full | Simulation | Sim (own+ABO, n=60): blind / full | Sim (all 3 sets, n=90): blind / full | Real vs sim: blind | Real vs sim: full |",
      "|---|---|---|---|---|---|---|"]
for t, (title, sims) in PAIRS.items():
    p = [r for r in real if r["capture_type"] == t and r["kind"] == "protected"]
    n = len(p)
    kb, kf = sum(int(r["blind"]) for r in p), sum(int(r["full"]) for r in p)
    for i, a in enumerate(sims):
        s_all = [r for r in sim if r["attack"] == a]
        s_oa = [r for r in s_all if r["set"] in ("own", "abo")]
        b = lambda rr, k: sum(int(r[k]) for r in rr)
        first = i == 0
        L.append(f"| {t if first else ''} | {ci(kb, n) + ' / ' + ci(kf, n) if first else ''} | {a} | "
                 f"{ci(b(s_oa, 'blind'), len(s_oa))} / {ci(b(s_oa, 'full'), len(s_oa))} | "
                 f"{ci(b(s_all, 'blind'), len(s_all))} / {ci(b(s_all, 'full'), len(s_all))} | "
                 f"{rel(kb, n, b(s_oa, 'blind'), len(s_oa))} | {rel(kf, n, b(s_oa, 'full'), len(s_oa))} |")
L.append("")

# ---- 3. misses
L += ["## 3. Captures the full pipeline did not trace", ""]
miss = [r for r in real if r["kind"] == "protected" and r["full"] != "1"]
for r in miss:
    cid = os.path.basename(r["file"]).split("_")[0]
    m = meta.get((r["capture_type"], cid), {})
    L.append(f"- **{r['capture_type']}/{r['file']}** (copy {cid}, {r['set']} photo, capture {r['w']}x{r['h']}, verdict `{r['verdict']}`; "
             f"label from {m.get('gt_source', '?')}, SIFT inliers {m.get('inliers', '?')})")
L += ["", "Both are low-detail products shown small on the listing page (copy 7005 = a white box on a grey wall, 7012 = a "
      "red pen on white). SIFT found no match (0 inliers) so no alignment was possible and the blind read failed too: "
      "the retrieval limit already noted for the ABO fabric swatch. The labels were checked against the source image of the "
      "listing pages (`listing_03.html` -> `v3_06.png`, `listing_12.html` -> `v3_13.png`).", ""]
for r in miss:
    cid = os.path.basename(r["file"]).split("_")[0]
    others = [(x["capture_type"], x["full"]) for x in real if x["kind"] == "protected" and os.path.basename(x["file"]).split("_")[0] == cid
              and x["capture_type"] != r["capture_type"]]
    L.append(f"- Copy {cid} in the other capture types (full pipeline): " + ", ".join(f"{t} {'traced' if f == '1' else 'MISSED'}" for t, f in others))
L.append("")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
print("->", OUT)
