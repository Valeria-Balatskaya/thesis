# scripts/null_pilot_report.py
# Null pilot (tests/test_owner_checks.py MODE=null, output/owner_forensics/null_pilot/<victim>.csv):
# per victim the number of tests, the distribution of the null z values (informed z, order-proof z raw and
# gated, presence z), counts above 3 and 4, and the Wilson 95 % upper bound on the false-claim rate;
# one QQ plot of all null z values against N(0,1).  Run: python scripts/null_pilot_report.py
# -> output/thesis/null_pilot.md, output/thesis/null_pilot_qq.png

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scipy import stats as sps
from src.stats import wilson

D = "output/owner_forensics/null_pilot"
VICTIMS = ("ours_v3", "trustmark", "wam", "videoseal")
OUT = "output/thesis/null_pilot.md"
FIG = "output/thesis/null_pilot_qq.png"


def num(r, k):
    return float(r[k]) if r.get(k, "") != "" else None


def rows_of(v):
    p = f"{D}/{v}.csv"
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


# statistic -> (title, row filter, getter). Informed z of the clean original is the sentinel 0.00 (S = O, no change
# to correlate) and is left out.
STATS = [("informed z", lambda r: r["case"] != "clean", lambda r: num(r, "informed_z")),
         ("order z, read variant (raw)", lambda r: r["case"].startswith("thief_"), lambda r: num(r, "order_read_raw_z")),
         ("order z, claim variant (raw)", lambda r: r["case"] != "clean", lambda r: num(r, "order_claim_raw_z")),
         ("order z, claim variant (gated = min with presence)", lambda r: r["case"] != "clean", lambda r: num(r, "order_claim_z")),
         ("presence z (claim) - NOT a null statistic: the thief mark really is on the leaked original, so it should be high", lambda r: r["case"] != "clean", lambda r: num(r, "presence_claim_z"))]


def vals(rows, flt, get):
    return np.array([x for r in rows if flt(r) for x in [get(r)] if x is not None])


L = ["# Null pilot: false claims and null z distributions", "",
     "Source: `tests/test_owner_checks.py` (MODE=null) on the first 100 photos of the E2 null set per victim. Per photo 6 tests: "
     "the thief (each of the four systems, random strength 1/1.5/2/3 x, clean or JPEG q70) marks the **leaked clean original**; "
     "an unrelated photo marked by a random thief; the clean original. Nothing here is an owner-marked copy, so no defence may "
     "claim. Threshold 4 is unchanged. `Regenerate: python scripts/null_pilot_report.py`.", ""]
L += ["## 1. False claims", "", "| Victim | photos | tests (photos x conditions) | photos with a false claim | Wilson 95 % upper bound (photos) | "
      "tests firing informed / peel-off / order-read / order-claim | upper bound (tests, treats tests as independent) |", "|---|---|---|---|---|---|---|"]
allz = {s[0]: [] for s in STATS}
for v in VICTIMS:
    rows = rows_of(v)
    if not rows:
        L.append(f"| {v} | not run | | | | | |")
        continue
    ph = {}
    for r in rows:
        a = int(r["informed"]) or int(r["peel_ok"]) or int(r["order_read"]) or int(r["order_claim"])
        ph[r["image"]] = ph.get(r["image"], 0) or a
    k, n = sum(ph.values()), len(ph)
    fired = [sum(int(r[c]) for r in rows) for c in ("informed", "peel_ok", "order_read", "order_claim")]
    kt = sum(1 for r in rows if int(r["informed"]) or int(r["peel_ok"]) or int(r["order_read"]) or int(r["order_claim"]))
    L.append(f"| {v} | {n}{'' if n >= 100 else ' (unfinished)'} | {len(rows)} | {k}/{n} | {100 * wilson(k, n)[1]:.1f} % | "
             + " / ".join(map(str, fired)) + f" | {100 * wilson(kt, len(rows))[1]:.2f} % |")
L += ["", "With 100 photos the upper bound for 0 false claims is 3.7 %; it is a pilot, E2 uses >= 1,000 photos per victim.", ""]

L += ["## 2. Null z distributions", "", "N(0,1) reference: mean 0, SD 1, 0.13 % above 3, 0.003 % above 4. z is computed against decoys "
      "(D = 50 for informed / order, 24 for presence), so the true null tail is somewhat heavier than normal.", ""]
for title, flt, get in STATS:
    L += [f"**{title}**", "", "| Victim | n | mean | SD | max | > 3 | > 4 |", "|---|---|---|---|---|---|---|"]
    pool = []
    for v in VICTIMS:
        rows = rows_of(v)
        z = vals(rows, flt, get)
        if len(z) == 0:
            continue
        pool.append(z)
        L.append(f"| {v} | {len(z)} | {z.mean():.2f} | {z.std(ddof=1):.2f} | {z.max():.2f} | {(z > 3).sum()} | {(z > 4).sum()} |")
        allz[title].append((v, z))
    if pool:
        z = np.concatenate(pool)
        L.append(f"| all | {len(z)} | {z.mean():.2f} | {z.std(ddof=1):.2f} | {z.max():.2f} | {(z > 3).sum()} | {(z > 4).sum()} |")
    L.append("")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
cols = {"ours_v3": "#1b9e77", "trustmark": "#d95f02", "wam": "#7570b3", "videoseal": "#e7298a"}
for ax, title in zip(axes, ("informed z", "order z, claim variant (raw)")):
    lim = 4.5
    for v, z in allz[title]:
        z = np.sort(z)
        q = sps.norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z))
        ax.plot(q, z, ".", ms=3, color=cols[v], label=f"{v} (n={len(z)})", alpha=.7)
        lim = max(lim, float(z.max()) + .5)
    ax.plot([-4, 4], [-4, 4], "k-", lw=1)
    ax.axhline(4, color="r", ls="--", lw=1)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, lim)
    ax.set_xlabel("N(0,1) quantile")
    ax.set_ylabel("null z")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
fig.suptitle("Null pilot: observed null z against N(0,1); red = threshold 4", fontsize=10)
fig.tight_layout()
os.makedirs("output/thesis", exist_ok=True)
fig.savefig(FIG, dpi=150)
L += ["## 3. QQ plot", "", f"![QQ plot]({os.path.basename(FIG)}) (`{FIG}`)", ""]
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
print("->", OUT, FIG)
