"""Fit the risk model and choose decision thresholds.

Approach: ordinal regression with a single sigmoid
---------------------------------------------------
Three classes are ordinal (safe < risky < destructive), so a plain three-class
classifier throws away information: it predicts a category when what we
actually want is a continuum. Instead this fits **one** sigmoid that predicts a
real-valued risk in [0, 2]:

    target y in {0, 1, 2}
    loss = mean( w(y) * (sigmoid(f(x)) - y)^2 )

with weight ``w(y) = 1`` for the extremes and lower weight near the ambiguous
middle, which is where the label is genuinely a judgement call. This is a
thermodynamic-style soft ordinal fit; it keeps the output a single scalar so the
exported model is one dot product, and it makes the two decision thresholds fall
out of one scalar axis.

Feature space
-------------
Two blocks concatenated:

* sparse char n-grams (2..5), TF-IDF with ``binary=True`` (presence) and
  ``norm=None``. Presence plus no-normalisation is what allows the IDF to be
* dense hand-crafted signals (:mod:`drosh_ml.features`).

Both are standard linear, so the fit is a logistic-style convex problem. Ridge
regularisation (``alpha``) is tuned on the validation split; the chosen value is
reported and frozen.

Threshold selection
-------------------
Not at 0.5. The false-negative/false-positive cost ratio from
``labels.FALSE_NEGATIVE_COST`` is applied: a false negative (missing something
destructive) is weighted ~20× a false positive (warning about a clean build).
The two operating points are picked to maximise a cost-weighted score on the
validation split, and both are exported.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer

from . import features as feat
from .build_dataset import REPO_ROOT, Row
from .labels import FALSE_NEGATIVE_COST, Risk

ARTIFACTS = REPO_ROOT / "ml" / "artifacts"

__all__ = ["TrainedModel", "fit", "choose_thresholds", "main"]

RANDOM_STATE = 20261004  # fixed so the pipeline is reproducible

#: Separator between the two normalised views when feeding the vectoriser.
#: A NUL cannot survive normalisation — every codepoint outside 0x20-0x7E is
#: dropped — so no n-gram can straddle the join and each view contributes only
#: its own internal n-grams.
VIEW_JOIN = "\x00"

#: Character n-gram window. Kept in sync with normalize.NGRAM_MIN/MAX and with
#: the vectoriser's ngram_range; all three must agree or the vocabulary the
#: vectoriser learned will never be hit at inference.
NGRAM_MIN = 2
NGRAM_MAX = 5


def vectorizer_input(command: str) -> str:
    """The exact string the vectoriser windows for one raw command.

    This is the contract between the trainer and the runtime. It lives here,
    next to the vectoriser that consumes it, rather than being duplicated in
    export.py — two copies of "how do you turn a command into the string the
    model was trained on" is exactly the kind of thing that drifts.
    """
    return VIEW_JOIN.join(_norm(command).views)


def window_ngrams(joined: str) -> set[str]:
    """Character n-grams over one already-joined string, presence semantics.

    Distinct from normalize.presence_ngrams, which takes a *sequence of views*.
    The vectoriser was fitted over the single joined string, so anything that
    looks up a trained index must window that same single string.
    """
    found: set[str] = set()
    length = len(joined)
    for size in range(NGRAM_MIN, NGRAM_MAX + 1):
        if length < size:
            break
        for start in range(length - size + 1):
            found.add(joined[start : start + size])
    return found

#: Cost of failing to warn on a genuinely RISKY command, in units of one
#: cried-wolf. Equal to 1: a missed ordinary warning is about as bad as a
#: spurious one. Deliberately far below ``FALSE_NEGATIVE_COST`` because missing
#: a soft warning costs the user nothing irreversible.
MISSED_RISK_COST: float = 1.0


@dataclass(slots=True)
class TrainedModel:
    vectorizer: TfidfVectorizer
    dense_spec: feat.DenseSpec
    weights: np.ndarray  # concatenated [sparse | dense]
    intercept: float
    thresholds: dict[str, float]
    alpha: float
    metrics: dict


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def _build_matrix(
    commands: list[str],
    vectorizer: TfidfVectorizer | None,
    dense_spec: feat.DenseSpec | None,
    fit_dense: bool,
):
    """Vectorise commands into the sparse-dense design matrix.

    The vectoriser is fed :func:`normalise_views`, i.e. the *same* text the
    JavaScript runtime will produce — not the raw command.

    This is the single most important line in the file. ``TfidfVectorizer`` does
    its own lowercasing and whitespace handling, and those differ from ours: it
    does not split segments, does not preserve shell operators, and does not
    strip ANSI or invisible characters. Feeding it raw text means it windows a
    different string than the device will, so the exported vocabulary describes
    one representation and the runtime computes another. That was the 5.15e-01
    fold failure: only the multi-segment probes diverged, because for
    single-segment commands our normalisation happens to be a no-op.

    Returns ``(matrix, vectorizer, dense_spec)``.
    """
    dense_raw = np.array([feat.dense_features(_norm(c)) for c in commands])

    if fit_dense:
        dense_spec = feat.fit_standardiser(dense_raw)
    dense_z = feat.standardise(dense_raw, dense_spec)

    # Feed the vectoriser exactly what the runtime will feed it.
    sparse = vectorizer.transform([vectorizer_input(c) for c in commands])
    matrix = _hstack(sparse, dense_z)
    return matrix, vectorizer, dense_spec


def _norm(command: str):
    from .normalize import normalize

    return normalize(command)


def _hstack(sparse_block, dense: np.ndarray):
    """Horizontally stack a sparse matrix and a dense block.

    Uses ``csr_array`` rather than ``csr_matrix``: under scipy 1.18 / numpy 2.x,
    the matrix classes are being phased out and ``csr_matrix(dense)`` without an
    explicit shape raises. ``csr_array`` is the supported spelling and behaves
    identically for hstack.
    """
    from scipy import sparse

    dense_block = sparse.csr_array(dense)
    return sparse.hstack([sparse_block, dense_block], format="csr")


def _weighted_ordinal_loss(
    weights: np.ndarray,
    intercept: float,
    X,
    y: np.ndarray,
    sample_weight: np.ndarray,
    alpha: float,
) -> tuple[float, np.ndarray]:
    """Convex objective and its gradient, for L-BFGS.

    ``weights`` is the concatenation of sparse-block and dense-block
    coefficients. Ridge penalty applies to both. ``sample_weight`` already folds
    in the ordinal ambiguity weight, so the loss is a weighted squared error on
    the sigmoid output.
    """

    def objective(params):
        w = params[:-1]
        b = params[-1]
        pred = _sigmoid(X @ w + b)
        # Risk in [0, 2]: scale the unit sigmoid by 2.
        risk = 2.0 * pred
        error = risk - y
        loss = np.sum(sample_weight * error**2) / len(y)
        loss += alpha * np.dot(w, w)
        return loss

    def gradient(params):
        w = params[:-1]
        b = params[-1]
        pred = _sigmoid(X @ w + b)
        risk = 2.0 * pred
        error = risk - y
        # d(risk)/d(z) = 2 * pred * (1 - pred)
        d_risk_dz = 2.0 * pred * (1.0 - pred)
        coef = (2.0 * sample_weight * error * d_risk_dz) / len(y)
        grad_w = X.T @ coef + 2.0 * alpha * w
        grad_b = np.sum(coef)
        return np.concatenate([grad_w, [grad_b]])

    return objective, gradient


def _ambiguity_weight(y: np.ndarray) -> np.ndarray:
    """Lower weight for labels near the risky/destructive boundary.

    y=1 (risky) is the judgement-call class; a squared-error regression spends
    most of its gradient there, which is right — but we do not want a few
    debatable risky labels to dominate the fit, so they carry slightly less
    weight than the unambiguous extremes.
    """
    return np.where(y == 1.0, 0.7, 1.0)


def fit(rows: list[Row], val_fraction: float = 0.15) -> TrainedModel:
    """Fit the ordinal sigmoid model with a validation split.

    The split is **stratified by label** so both classes are represented, and
    deterministic (``RANDOM_STATE``), because CI re-runs this and asserts the
    export is byte-identical.
    """
    from sklearn.model_selection import train_test_split

    commands = [row.command for row in rows]
    targets = np.array([float(row.label.value) for row in rows])

    indices = np.arange(len(rows))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=RANDOM_STATE,
        stratify=targets.astype(int),
    )

    train_cmds = [commands[i] for i in train_idx]
    train_y = targets[train_idx]

    # Fit the vectoriser and standardiser on the training split only.
    # ``analyzer="char"`` over the joined views: a NUL cannot occur in the
    # normalised output (everything outside 0x20-0x7E is dropped), so the two
    # views stay separable and no n-gram can straddle the join.
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=3,
        binary=True,
        use_idf=True,
        norm=None,
        sublinear_tf=False,
        lowercase=False,
        dtype=np.float64,
    )
    vectorizer.fit([vectorizer_input(c) for c in train_cmds])

    X_train, vectorizer, dense_spec = _build_matrix(
        train_cmds, vectorizer, None, fit_dense=True
    )
    X_val, _, _ = _build_matrix(
        [commands[i] for i in val_idx], vectorizer, dense_spec, fit_dense=False
    )
    y_val = targets[val_idx]

    sample_weight = _ambiguity_weight(train_y)

    best: tuple[float, float, np.ndarray, float] | None = None
    # Coarse alpha grid then report the whole grid, not just the winner. The
    # problem is convex, so there is no seed sensitivity to smooth over.
    alpha_grid = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    results: list[tuple[float, float]] = []
    for alpha in alpha_grid:
        x0 = np.zeros(X_train.shape[1] + 1)
        objective, gradient = _weighted_ordinal_loss(
            x0[:-1], x0[-1], X_train, train_y, sample_weight, alpha
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                objective,
                x0,
                jac=gradient,
                method="L-BFGS-B",
                options={"maxiter": 500},
            )
        w = result.x[:-1]
        b = float(result.x[-1])
        val_risk = 2.0 * _sigmoid(X_val @ w + b)
        val_mae = float(np.mean(np.abs(val_risk - y_val)))
        results.append((val_mae, alpha))
        if best is None or val_mae < best[0]:
            best = (val_mae, alpha, w, b)

    assert best is not None
    val_mae, alpha, w, b = best
    thresholds = choose_thresholds(y_val, 2.0 * _sigmoid(X_val @ w + b))

    metrics = {
        "alpha": alpha,
        "val_mae": round(val_mae, 4),
        "val_size": int(len(val_idx)),
        "alpha_grid_mae": {str(a): round(m, 4) for m, a in results},
        "sparse_features": int(X_train.shape[1] - feat.DENSE_DIM),
        "dense_features": feat.DENSE_DIM,
    }

    return TrainedModel(
        vectorizer=vectorizer,
        dense_spec=dense_spec,
        weights=w,
        intercept=float(b),
        thresholds=thresholds,
        alpha=alpha,
        metrics=metrics,
    )


def _threshold_cost(
    y_true: np.ndarray, risk: np.ndarray, warn_t: float, block_t: float
) -> float:
    """Cost of one (warn, block) operating point. Lower is better.

    Kept standalone and obviously-correct because the optimiser below is only
    trustworthy if the objective is easy to read.
    ``ml/tests/test_thresholds.py`` checks the optimiser against this.
    """
    warn_call = risk >= warn_t
    block_call = risk >= block_t
    cost = 0.0
    cost += np.sum((y_true == Risk.DESTRUCTIVE.value) & ~block_call) * FALSE_NEGATIVE_COST
    cost += np.sum((y_true == Risk.RISKY.value) & ~warn_call) * MISSED_RISK_COST
    cost += np.sum((y_true == Risk.SAFE.value) & warn_call) * 1.0
    return float(cost)


def choose_thresholds(y_true: np.ndarray, risk: np.ndarray) -> dict[str, float]:
    """Pick the two operating points by minimising weighted validation cost.

    ``warn``:  risk at or above this fires the soft warning.
    ``block``: risk at or above this fires the strong warning (always >= warn).

    Three mistakes are charged for, nothing else:

      * a DESTRUCTIVE command below ``block``: ``FALSE_NEGATIVE_COST`` (20x).
        Missing this loses data.
      * a RISKY command below ``warn``: ``MISSED_RISK_COST`` (1x). A missed
        ordinary warning.
      * a SAFE command at or above ``warn``: 1x. Crying wolf.

    Correct RISKY warnings, correct DESTRUCTIVE blocks and correct silences are
    free. The middle term is load-bearing: without it ``warn`` has no reason to
    leave the floor, since a warning only costs anything on a SAFE command, and a
    three-class taxonomy whose middle class never warns is not three-class.

    Candidate thresholds
    -------------------
    ``candidates[k]`` warns exactly the ``k`` lowest-scoring samples, so index 0
    is "warn on nothing" and index n is "warn on everything". For k in the middle
    the value is the midpoint between two observed scores, which is unambiguous
    where a score itself is not: with ties, a threshold equal to an observation
    includes the whole tie group, and the midpoint between distinct values always
    separates them.

    The array is strictly ascending, so ``best_i <= best_j`` implies
    ``warn <= block`` and no post-hoc clamping is needed. An earlier version
    encoded "warn on nothing" as ``min - 1`` and clamped the result with
    min/max; under a ``>=`` rule ``min - 1`` warns on *everything*, which
    inverted the search and shipped block=2.00 on a model whose maximum score was
    1.9993 — a risk class that could never fire. ml/tests/test_thresholds.py
    exists because of that, and it fails loudly if this regresses.

    Cost
    ----
    One sort, three suffix sums, then a scan over candidate pairs adding two
    numbers each. O(n log n + k^2) with k = n+1 candidates; the inner term is a
    float addition, not a masked array reduction.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    risk = np.asarray(risk, dtype=np.float64)
    n = len(risk)
    if n == 0:
        return {"warn": 1.0, "block": 3.0}

    order = np.argsort(risk, kind="stable")
    sorted_risk = risk[order]
    is_destructive = (y_true == Risk.DESTRUCTIVE.value)[order]
    is_risky = (y_true == Risk.RISKY.value)[order]
    is_safe = (y_true == Risk.SAFE.value)[order]

    top = float(sorted_risk[-1])
    candidates = np.concatenate(
        [
            # k = 0: warn on nothing. Must sit strictly above every score.
            [top + 1.0],
            # k = 1..n-1: the midpoint that splits sorted_risk[k-1] from [k].
            sorted_risk[:-1] + (sorted_risk[1:] - sorted_risk[:-1]) / 2.0,
            # k = n: warn on everything. The maximum itself suffices, because
            # the rule is >=, so a score equal to the threshold fires.
            [top],
        ]
    )

    # Which samples a cut of index k covers, and which it therefore leaves out.
    # candidates[k] fires on samples [k:], so:
    #
    #   safe in [i:]        is warned       -> false alarm    (SUFFIX)
    #   risky in [0, i)     is silent       -> missed warning (PREFIX)
    #   destructive in [0, j) is unblocked -> missed block    (PREFIX)
    #
    # Two of the three are prefixes, and getting that wrong on the block term
    # charges a miss for every destructive command the threshold *does* catch.
    # All three were suffix counts at some point; each inverted its own gradient
    # and pushed its threshold to the end of the range.
    safe_suffix = np.concatenate([np.cumsum(is_safe[::-1])[::-1], [0.0]])
    risky_prefix = np.concatenate([[0.0], np.cumsum(is_risky)])
    destr_prefix = np.concatenate([[0.0], np.cumsum(is_destructive)])

    total_safe = float(is_safe.sum())
    total_risky = float(is_risky.sum())
    total_destructive = float(is_destructive.sum())

    def warn_cost(k: int) -> float:
        """Cost of setting the warn threshold at candidates[k].

        The interior cases read straight off the sums: samples [k:] are warned,
        so the safe ones there are false alarms and the risky ones in [0, k) are
        missed warnings. The two sentinels are not interior cases and the sums
        do not describe them — candidates[0] warns on *nothing* while
        safe_suffix[0] counts every safe sample as warned, which is the exact
        inverse. They are therefore spelled out.
        """
        return safe_suffix[k] + risky_prefix[k] * MISSED_RISK_COST

    def block_cost(k: int) -> float:
        """Cost of setting the block threshold at candidates[k].

        samples [k:] are blocked, so destructive commands in [0, k) are missed.
        """
        return destr_prefix[k] * FALSE_NEGATIVE_COST

    best_cost = float("inf")
    best_i, best_j = n, n

    for i in range(n + 1):
        base = warn_cost(i)
        for j in range(i, n + 1):
            # Ordering is by value, not by index: candidates[0] is the high
            # sentinel, so i < j does not imply warn <= block.
            if candidates[j] < candidates[i]:
                continue
            cost = base + block_cost(j)
            if cost < best_cost - 1e-12:
                best_cost = cost
                best_i, best_j = i, j

    return {
        "warn": round(float(candidates[best_i]), 6),
        "block": round(float(candidates[best_j]), 6),
    }


def _load_dataset() -> list[Row]:
    import csv

    path = ARTIFACTS / "dataset.csv"
    rows: list[Row] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            rows.append(
                Row(
                    command=record["command"],
                    label=Risk[record["label"]],
                    category=record["category"],
                    source=record["source"],
                    section=record.get("section", ""),
                    transform=record.get("transform", ""),
                )
            )
    return rows


def main() -> int:
    rows = _load_dataset()
    print(f"==> training on {len(rows)} rows")
    model = fit(rows)

    report_path = ARTIFACTS / "train_report.json"
    report = {
        "metrics": model.metrics,
        "thresholds": model.thresholds,
        "false_negative_cost": FALSE_NEGATIVE_COST,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"    alpha        : {model.metrics['alpha']}")
    print(f"    val MAE      : {model.metrics['val_mae']}  (target units, 0..2)")
    print(f"    sparse feats : {model.metrics['sparse_features']}")
    print(f"    dense feats  : {model.metrics['dense_features']}")
    print(f"    thresholds   : warn={model.thresholds['warn']} block={model.thresholds['block']}")
    print(f"    wrote {report_path.relative_to(REPO_ROOT)}")

    # Persist as a plain dict, not the dataclass. A dataclass pickled while
    # running as `__main__` records its class as `__main__.TrainedModel`, which
    # cannot be unpickled from any other entry point — that broke the first
    # artifact this pipeline produced. export.py refits in-process instead, so
    # this file is for inspection rather than for loading.
    import joblib

    joblib.dump(
        {
            "alpha": model.alpha,
            "intercept": model.intercept,
            "thresholds": model.thresholds,
            "metrics": model.metrics,
            "vectorizer": model.vectorizer,
            "dense_spec": model.dense_spec,
            "weights": model.weights,
        },
        ARTIFACTS / "model.joblib",
    )
    print(f"    wrote {(ARTIFACTS / 'model.joblib').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())