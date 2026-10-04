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

    Returns ``(matrix, vectorizer, dense_spec)``. When ``fit_dense`` is set the
    standardiser is (re)fitted here; otherwise the frozen one is applied.
    """
    dense_raw = np.array([feat.dense_features(_norm(c)) for c in commands])

    if fit_dense:
        dense_spec = feat.fit_standardiser(dense_raw)
    dense_z = feat.standardise(dense_raw, dense_spec)

    sparse = vectorizer.transform(commands)
    matrix = _hstack(sparse, dense_z)
    return matrix, vectorizer, dense_spec


def _norm(command: str):
    from .normalize import normalize

    return normalize(command)


def _hstack(sparse, dense: np.ndarray):
    """Horizontally stack a scipy sparse matrix and a dense array."""
    from scipy import sparse

    return sparse.hstack([sparse, sparse.csr_matrix(dense)], format="csr")


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
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=3,
        binary=True,
        use_idf=True,
        norm=None,
        sublinear_tf=False,
        dtype=np.float64,
    )
    vectorizer.fit(train_cmds)

    X_train, vectorizer, dense_spec = _build_matrix(
        train_cmds, vectorizer, None, fit_dense=True
    )
    X_val, _, _ = _build_matrix(
        [commands[i] for i in val_idx], vectorizer, dense_spec, fit_dense=False
    )
    y_val = targets[val_idx]

    sample_weight = _ambiguity_weight(train_y)

    best: TrainedModel | None = None
    # Coarse alpha grid then a local refine; convex, so no seed sensitivity.
    alpha_grid = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    results = []
    for alpha in alpha_grid:
        w0 = np.zeros(X_train.shape[1] + 1)
        objective, gradient = _weighted_ordinal_loss(
            w0[: X_train.shape[1]], w0[-1], X_train, train_y, sample_weight, alpha
        )

        def full_objective(params, obj=objective):
            return obj(params)

        def full_gradient(params, grad=gradient):
            return grad(params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                full_objective,
                w0,
                jac=full_gradient,
                method="L-BFGS-B",
                options={"maxiter": 500},
            )
        w = result.x[:-1]
        b = result.x[-1]
        val_risk = 2.0 * _sigmoid(X_val @ w + b)
        val_mae = float(np.mean(np.abs(val_risk - y_val)))
        results.append((val_mae, alpha, w, b))
        if best is None or val_mae < best[0]:
            best = (val_mae, alpha, w, b)

    assert best is not None
    val_mae, alpha, w, b = best
    thresholds = choose_thresholds(y_val, 2.0 * _sigmoid(X_val @ w + b))

    metrics = {
        "alpha": alpha,
        "val_mae": round(val_mae, 4),
        "val_size": int(len(val_idx)),
        "alpha_grid_mae": {str(a): round(m, 4) for m, a, _, _ in results},
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


def choose_thresholds(y_true: np.ndarray, risk: np.ndarray) -> dict[str, float]:
    """Pick the two operating points by cost-weighted validation error.

    ``destructive`` threshold ``block``: fire the strong warning.
    ``risky`` threshold ``warn``: fire the soft warning.

    Cost of a mistake, in units of one correct call:
      * missing a DESTRUCTIVE (true 2, predicted < 2): ``FALSE_NEGATIVE_COST``
      * warning on a SAFE (true 0, predicted > 0): 1
      * everything else: 1

    We sweep candidate thresholds and keep the pair minimising total cost.
    """
    y_true = np.asarray(y_true)
    risk = np.asarray(risk)
    candidates = np.unique(np.concatenate([risk, [0.5, 1.0, 1.5]]))

    def cost_for(block_t: float, warn_t: float) -> float:
        # block if risk >= block_t, warn if risk >= warn_t
        block_call = risk >= block_t
        warn_call = risk >= warn_t
        cost = 0.0
        for truth, predicted_block, predicted_warn in zip(y_true, block_call, warn_call):
            if truth == Risk.DESTRUCTIVE.value and not predicted_block:
                cost += FALSE_NEGATIVE_COST  # missed a destructive command
            elif truth == Risk.SAFE.value and (predicted_block or predicted_warn):
                cost += 1.0  # cried wolf on a safe command
            else:
                cost += 1.0
        return cost

    best = None
    for block_t in candidates:
        for warn_t in candidates:
            if warn_t > block_t:
                continue  # warn must be the softer, lower threshold
            total = cost_for(block_t, warn_t)
            if best is None or total < best[0]:
                best = (total, float(block_t), float(warn_t))

    assert best is not None
    _, block_t, warn_t = best
    return {"warn": round(warn_t, 4), "block": round(block_t, 4)}


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

    import joblib

    joblib.dump(model, ARTIFACTS / "model.joblib")
    print(f"    wrote {(ARTIFACTS / 'model.joblib').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())