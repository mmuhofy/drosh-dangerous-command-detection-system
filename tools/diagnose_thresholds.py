"""Exhaustive trace of choose_thresholds on a six-sample case small to verify by hand.

The randomised comparisons in test_thresholds.py say the search disagrees with
brute force but a 400-sample cloud hides which term is wrong. This runs a case
where every number can be checked on paper, and prints the full candidate table:
for each (warn index, block index) pair, what the suffix sums compute and what
_threshold_cost reports at the same operating point.

A row where the two differ names the offending term directly.

    PYTHONPATH=ml/src .venv/bin/python tools/diagnose_thresholds.py
"""

from __future__ import annotations

import numpy as np

from drosh_ml.labels import Risk
from drosh_ml.train import (
    FALSE_NEGATIVE_COST,
    MISSED_RISK_COST,
    _threshold_cost,
    choose_thresholds,
)

# Hand-checkable: three safe well below, one risky in the middle, two destructive
# on top, with one ambiguous pair to exercise tie handling.
TRUTH = np.array(
    [
        Risk.SAFE.value,
        Risk.SAFE.value,
        Risk.SAFE.value,
        Risk.RISKY.value,
        Risk.DESTRUCTIVE.value,
        Risk.DESTRUCTIVE.value,
    ]
)
RISK = np.array([0.05, 0.10, 0.20, 0.80, 1.60, 1.90])


def main() -> None:
    order = np.argsort(RISK, kind="stable")
    sorted_risk = RISK[order]
    is_d = (TRUTH == Risk.DESTRUCTIVE.value)[order]
    is_r = (TRUTH == Risk.RISKY.value)[order]
    is_s = (TRUTH == Risk.SAFE.value)[order]

    top = float(sorted_risk[-1])
    candidates = np.concatenate(
        [
            [top + 1.0],
            sorted_risk[:-1] + (sorted_risk[1:] - sorted_risk[:-1]) / 2.0,
            [top + 1e-9],
        ]
    )
    safe_suffix = np.concatenate([np.cumsum(is_s[::-1])[::-1], [0.0]])
    risky_prefix = np.concatenate([[0.0], np.cumsum(is_r)])
    destr_suffix = np.concatenate([np.cumsum(is_d[::-1])[::-1], [0.0]])

    n = len(RISK)
    print("truth   :", [int(t) for t in TRUTH[order]], " (0=safe 1=risky 2=destructive)")
    print("risk    :", [round(float(r), 3) for r in sorted_risk])
    print("is_safe :", [bool(x) for x in is_s])
    print("is_risky:", [bool(x) for x in is_r])
    print("is_dest :", [bool(x) for x in is_d])
    print()
    print(f"{'k':>3} {'candidate':>10} {'safeSuf':>8} {'riskPre':>8} {'destSuf':>8}")
    print("-" * 42)
    for k in range(n + 1):
        print(
            f"{k:>3} {candidates[k]:>10.4f} {safe_suffix[k]:>8} "
            f"{risky_prefix[k]:>8} {destr_suffix[k]:>8}"
        )

    print()
    print("=== pair scan ===")
    print(f"{'i':>3} {'j':>3} {'warn':>8} {'block':>8} {'search':>9} {'direct':>9} {'delta':>8}")
    print("-" * 60)
    search_best = (float("inf"), None, None)
    for i in range(n + 1):
        base = safe_suffix[i] + risky_prefix[i] * MISSED_RISK_COST
        for j in range(i, n + 1):
            cost = base + destr_suffix[j] * FALSE_NEGATIVE_COST
            direct = _threshold_cost(TRUTH, RISK, float(candidates[i]), float(candidates[j]))
            if i == 0 or j == i or (i, j) == (n, n) or (i, j) == (2, 2):
                print(
                    f"{i:>3} {j:>3} {candidates[i]:>8.4f} {candidates[j]:>8.4f} "
                    f"{cost:>9.1f} {direct:>9.1f} {cost - direct:>8.1f}"
                )
            if cost < search_best[0]:
                search_best = (cost, i, j)

    cost, i, j = search_best
    print()
    print(
        f"search argmin    : i={i} j={j} warn={candidates[i]:.4f} "
        f"block={candidates[j]:.4f} cost={cost:.1f}"
    )

    grid = np.unique(
        np.concatenate(
            [RISK, [RISK.min() - 1, RISK.max() + 1], (np.sort(RISK)[:-1] + np.sort(RISK)[1:]) / 2]
        )
    )
    best = (float("inf"), 0.0, 0.0)
    for w in grid:
        for b in grid:
            if b < w:
                continue
            c = _threshold_cost(TRUTH, RISK, float(w), float(b))
            if c < best[0] - 1e-12:
                best = (c, float(w), float(b))
    print(
        f"brute force best : warn={best[1]:.4f} block={best[2]:.4f} cost={best[0]:.1f}"
    )

    got = choose_thresholds(TRUTH, RISK)
    got_cost = _threshold_cost(TRUTH, RISK, got["warn"], got["block"])
    print(f"choose_thresholds: warn={got['warn']:.4f} block={got['block']:.4f} cost={got_cost:.1f}")
    print()
    print("MATCH" if abs(got_cost - best[0]) < 1e-9 else "MISMATCH")


if __name__ == "__main__":
    main()