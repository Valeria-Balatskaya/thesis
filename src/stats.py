# src/stats.py
# Confidence intervals for the percentages reported in the thesis tables.
#
# Wilson score interval (Wilson 1927; recommended over the normal "Wald" interval
# for small n and rates near 0 or 1 by Brown, Cai & DasGupta 2001). For k = 0 it
# still gives a useful upper bound (e.g. 0/1,667 -> 95% upper bound 0.23%).

import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for k successes out of n, as fractions (lo, hi)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def pct_ci(k: int, n: int, digits: int = 1) -> str:
    """'91.4% [88.1–93.8] (329/360)' — the format used in the thesis tables."""
    if n == 0:
        return "n/a (0)"
    lo, hi = wilson(k, n)
    f = f"{{:.{digits}f}}"
    return f"{f.format(100 * k / n)}% [{f.format(100 * lo)}–{f.format(100 * hi)}] ({k}/{n})"


def frac_ci(k: int, n: int) -> str:
    """'34/40 [70–93%]' — compact form for small samples."""
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k}/{n} [{100 * lo:.0f}–{100 * hi:.0f}%]"


if __name__ == "__main__":
    for k, n in [(0, 40), (34, 40), (329, 360), (0, 1667), (40, 40)]:
        print(k, n, pct_ci(k, n), frac_ci(k, n))
