"""Compare choose_thresholds against the direct cost at matching candidates.

test_thresholds.py says the two disagree (2619 vs 43 on synthetic data) but not
where. This walks the candidate indices the search uses and prints, for a handful
of them, the cost the search computed from its suffix sums next to the cost
_threshold_cost reports at the same operating point. A row where the two differ
localises the bug to a specific term.

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


def main() -> None:
    rng = np.random.default_rng(0)
    n = 400
    truth = np.concatenate(
        [np.zeros(n), np.ones(n // 2), np.full(n // 2, Risk.DESTRUCTIVE.value)]
    )
    centres = np.concatenate(
        [np.full(n, 0.1), np.full(n // 2, 1.0), np.full(n // 2, 1.9)]
    )
    risk = np.clip(centres + rng.normal(0.0, 0.35, len(centres)), 0.0, 2.0)

    order = np.argsort(risk, kind="stable")
    sorted_risk = risk[order]
    is_destructive = (truth == Risk.DESTRUCTIVE.value)[order]
    is_risky = (truth == Risk.RISKY.value)[order]
    is_safe = (truth == Risk.SAFE.value)[order]

    safe_suffix = np.concatenate([np.cumsum(is_safe[::-1])[::-1], [0.0]])
    risky_suffix = np.concatenate([np.cumsum(is_risky[::-1])[::-1], [0.0]])
    destr_suffix = np.concatenate([np.cumsum(is_destructive[::-1])[::-1], [0.0]])
    cuts = np.concatenate([[np.min(sorted_risk) - 1.0], sorted_risk])

    print(f"samples           : {len(truth)}")
    print(f"cuts length       : {len(cuts)}   suffix lengths: {len(safe_suffix)}")
    print()
    print(f"{'i':>6} {'warn':>8} {'safeSuf':>8} {'riskySuf':>9} {'destrSuf':>9}"
          f" {'searchCost':>11} {'directCost':>11} {'delta':>9}")
    print("-" * 82)

    for i in range(0, len(cuts), max(1, len(cuts) // 18)):
        j = i  # warn == block
        search_cost = (
            safe_suffix[i] * 1.0
            + risky_suffix[i] * MISSED_RISK_COST
            + destr_suffix[j] * FALSE_NEGATIVE_COST
        )
        direct = _threshold_cost(truth, risk, float(cuts[i]), float(cuts[j]))
        delta = search_cost - direct
        print(
            f"{i:>6} {cuts[i]:>8.4f} {safe_suffix[i]:>8} {risky_suffix[i]:>9} "
            f"{destr_suffix[j]:>9} {search_cost:>11.1f} {direct:>11.1f} {delta:>9.1f}"
        )

    print()
    got = choose_thresholds(truth, risk)
    got_cost = _threshold_cost(truth, risk, got["warn"], got["block"])
    print(f"search returned   : warn={got['warn']:.4f} block={got['block']:.4f} cost={got_cost:.1f}")

    best = float("inf")
    best_pair = (0.0, 0.0)
    grid = np.unique(np.concatenate([risk, [risk.min() - 1, risk.max() + 1]]))
    for w in grid:
        for b in grid:
            if b < w:
                continue
            c = _threshold_cost(truth, risk, float(w), float(b))
            if c < best:
                best = c
                best_pair = (float(w), float(b))
    print(f"brute force best  : warn={best_pair[0]:.4f} block={best_pair[1]:.4f} cost={best:.1f}")
    print()
    print("If the delta column is non-zero, the suffix sums do not describe the cost.")
    print("If the delta column is all zero but the search still loses, the argmin is wrong.")


if __name__ == "__main__":
    main()