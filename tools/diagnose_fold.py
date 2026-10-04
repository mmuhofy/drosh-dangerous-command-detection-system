"""Temporary diagnostic: locate the IDF-fold discrepancy.

Kept in the repo rather than pasted into an ssh heredoc, because the codespace
SSH layer swallows multi-line stdin. Run from the codespace:

    PYTHONPATH=ml/src .venv/bin/python tools/diagnose_fold.py
"""

from __future__ import annotations

import numpy as np

from drosh_ml import features as feat
from drosh_ml import train as T
from drosh_ml.normalize import normalize, presence_ngrams
from drosh_ml.train import _load_dataset

PROBES = [
    "rm -rf /",
    "ls -la",
    "sudo dd if=/dev/zero of=/dev/sda",
    "git status",
]


def main() -> None:
    print("loading dataset", flush=True)
    rows = _load_dataset()
    print(f"  rows={len(rows)}", flush=True)

    model = T.fit(rows)
    print("fit complete", flush=True)

    vectorizer = model.vectorizer
    spec = model.dense_spec
    weights = model.weights
    bias = model.intercept
    sparse_dim = len(vectorizer.vocabulary_)
    dense_dim = feat.DENSE_DIM

    print(f"weights={len(weights)} sparse={sparse_dim} dense={dense_dim} "
          f"sum={sparse_dim + dense_dim}", flush=True)
    print(f"thresholds={model.thresholds}", flush=True)
    print(f"val_mae={model.metrics['val_mae']}", flush=True)

    folded_sparse = weights[:sparse_dim] * vectorizer.idf_
    dense_w = weights[sparse_dim:]

    for command in PROBES:
        norm = normalize(command)

        dense_z = feat.standardise(feat.dense_features(norm)[None, :], spec)
        dense_manual = float((dense_z @ dense_w)[0])

        sparse_manual = 0.0
        for gram in presence_ngrams(norm.views):
            index = vectorizer.vocabulary_.get(gram)
            if index is not None:
                sparse_manual += float(folded_sparse[index])

        folded_score = 2.0 / (1.0 + np.exp(-(dense_manual + sparse_manual + bias)))

        matrix, _, _ = T._build_matrix([command], vectorizer, spec, fit_dense=False)
        reference = float((2.0 / (1.0 + np.exp(-(matrix @ weights + bias))))[0])

        # The same score computed WITHOUT folding, to isolate which block drifts.
        unfused = 0.0
        for gram in presence_ngrams(norm.views):
            index = vectorizer.vocabulary_.get(gram)
            if index is not None:
                unfused += float(weights[index]) * float(vectorizer.idf_[index])
        unfused_score = 2.0 / (1.0 + np.exp(-(dense_manual + unfused + bias)))

        print(
            f"  {command[:34]:<36} folded={folded_score:.6f} "
            f"reference={reference:.6f} diff={abs(folded_score - reference):.2e} "
            f"| sparse_folded={sparse_manual:+.5f} sparse_unfused={unfused:+.5f} "
            f"dense={dense_manual:+.5f}",
            flush=True,
        )


if __name__ == "__main__":
    main()