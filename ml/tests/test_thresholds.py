"""The threshold search must agree with an independent brute force.

``train.choose_thresholds`` optimises a weighted cost over candidate operating
points. ``train._threshold_cost`` states that cost directly, and is simple enough
to trust by inspection. If the search does not return the minimum of that
function, the model ships thresholds nobody reasoned about — and the symptom is
subtle: validation MAE stays fine while the shipped thresholds sit above every
score the model can produce, so a whole risk class silently never fires.

That is not hypothetical. The first shipped model had warn=1.746 and
block=1.999 against a maximum observed score of 1.9993, catching 5 of 10750
destructive commands, while block=0.50 achieved a 10000x lower cost. Accuracy
was never the problem; the search was.

This test pins the two implementations together on randomised inputs and on the
degenerate cases that produce silly optima.
"""

from __future__ import annotations

import numpy as np
import pytest

from drosh_ml.labels import Risk
from drosh_ml.train import (
    FALSE_NEGATIVE_COST,
    MISSED_RISK_COST,
    _threshold_cost,
    choose_thresholds,
)


def _brute_force(truth: np.ndarray, risk: np.ndarray) -> tuple[float, float, float]:
    """Minimum-cost (warn, block) pair, found by evaluating every combination."""
    candidates = np.unique(
        np.concatenate(
            [
                risk,
                [risk.min() - 1.0, risk.max() + 1.0],
                # Midpoints matter too: the optimum of a threshold rule can sit
                # between two observed values rather than on one of them.
                (risk[:-1] + risk[1:]) / 2.0 if len(risk) > 1 else risk,
            ]
        )
    )
    best = (float("inf"), 0.0, 0.0)
    for warn in candidates:
        for block in candidates:
            if block < warn:
                continue
            cost = _threshold_cost(truth, risk, float(warn), float(block))
            if cost < best[0]:
                best = (cost, float(warn), float(block))
    return best


def _synthesised(seed: int, n: int = 400, separation: float = 1.4) -> tuple[np.ndarray, np.ndarray]:
    """Three bands, like the real model: tight, tight, tight, with some overlap."""
    rng = np.random.default_rng(seed)
    truth = np.concatenate(
        [np.zeros(n), np.ones(n // 2), np.full(n // 2, Risk.DESTRUCTIVE.value)]
    )
    centres = np.concatenate(
        [np.full(n, 0.1), np.full(n // 2, 1.0), np.full(n // 2, 1.9)]
    )
    risk = np.clip(centres + rng.normal(0.0, separation / 4.0, len(centres)), 0.0, 2.0)
    return truth, risk


@pytest.mark.parametrize("seed", range(8))
def test_search_matches_brute_force(seed: int) -> None:
    truth, risk = _synthesised(seed)
    best_cost, best_warn, best_block = _brute_force(truth, risk)

    got = choose_thresholds(truth, risk)
    got_cost = _threshold_cost(truth, risk, got["warn"], got["block"])

    assert got_cost == pytest.approx(best_cost, rel=1e-9), (
        f"search returned cost {got_cost} but the minimum is {best_cost}; "
        f"got warn={got['warn']} block={got['block']}, "
        f"best warn={best_warn} block={best_block}"
    )


def test_warn_threshold_is_below_every_destructive_score() -> None:
    """A threshold above the observed maximum can never fire.

    This is the exact defect the shipped model had. Asserted directly rather
    than inferred from a cost comparison, because the failure is silent: nothing
    raises, the model just never blocks.
    """
    truth, risk = _synthesised(3)
    got = choose_thresholds(truth, risk)

    destructive = risk[truth == Risk.DESTRUCTIVE.value]
    assert got["block"] < destructive.max(), (
        f"block={got['block']} is above the highest destructive score "
        f"{destructive.max()}, so destructive commands cannot be blocked"
    )


def test_block_catches_destructive_and_spares_safe() -> None:
    """Recall on the top class, without drowning the user in warnings.

    The bound on false alarms is deliberately loose. An earlier version asserted
    zero, and that assertion was about the test data rather than about the code:
    _synthesised spreads each band with a standard deviation large enough that
    the classes genuinely overlap, so a threshold that catches every destructive
    command necessarily catches a few safe ones. The real corpus separates far
    more cleanly (see ml/artifacts/eval_report.json: 100% destructive recall at
    0.4% false alarms), but the search cannot know that in advance, and a
    threshold that refuses to trade is not a threshold.

    What must hold is that the trade is not catastrophic: catching everything
    while warning on a quarter of all safe traffic would be a model failure, not
    a threshold failure.
    """
    truth, risk = _synthesised(4)
    got = choose_thresholds(truth, risk)

    is_d = truth == Risk.DESTRUCTIVE.value
    is_s = truth == Risk.SAFE.value

    recall = float(((risk >= got["block"]) & is_d).sum() / max(1, is_d.sum()))
    false_alarm_rate = float(((risk >= got["warn"]) & is_s).sum() / max(1, is_s.sum()))

    assert recall >= 0.90, f"destructive recall only {recall:.1%}"
    assert false_alarm_rate <= 0.25, (
        f"{false_alarm_rate:.1%} of safe commands warned; the threshold is "
        "trading away the whole safe class to avoid missing anything"
    )


def test_missed_risky_warning_is_actually_charged() -> None:
    """Without this term the middle class never warns.

    A warning only costs anything on a SAFE command, so a cost function that
    forgets "you should have warned on this RISKY one" has no reason to move the
    warn threshold off the floor. The model then collapses to two classes.
    """
    truth = np.concatenate([np.zeros(100), np.ones(50), np.full(50, 2.0)])
    # RISKY sits just below DESTRUCTIVE, so block alone would cover both.
    risk = np.concatenate([np.full(100, 0.05), np.full(50, 1.7), np.full(50, 1.95)])

    got = choose_thresholds(truth, risk)
    risky = risk[truth == Risk.RISKY.value]

    assert got["warn"] < risky.max(), (
        "warn collapsed above every risky score; the middle class cannot warn"
    )
    assert MISSED_RISK_COST > 0


def test_thresholds_are_ordered_and_finite() -> None:
    for seed in range(4):
        truth, risk = _synthesised(seed, n=120)
        got = choose_thresholds(truth, risk)
        assert got["warn"] <= got["block"], got
        assert np.isfinite(got["warn"]) and np.isfinite(got["block"]), got
        assert got["block"] <= 3.0, "block threshold escaped the risk range"


def test_degenerate_all_safe_is_never_blocked() -> None:
    """Nothing to block, so nothing should be blocked."""
    truth = np.zeros(200)
    risk = np.full(200, 0.1)
    got = choose_thresholds(truth, risk)
    assert not (risk >= got["block"]).any()


def test_degenerate_everything_destructive_warns() -> None:
    """Everything is real, so everything warns."""
    truth = np.full(200, Risk.DESTRUCTIVE.value)
    risk = np.full(200, 1.5)
    got = choose_thresholds(truth, risk)
    assert (risk >= got["warn"]).all()