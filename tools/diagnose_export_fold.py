"""Per-command breakdown of the export fold check.

Calls export._assert_fold_is_exact's own code path and prints, per probe, the
unfitted score, the folded score and the difference. tools/diagnose_fold.py
already showed the fold is exact (6.66e-16) using the same primitives, so if
export disagrees there is a difference between the two call sites and this
locates it by command rather than by inspection.

Run from the codespace:

    PYTHONPATH=ml/src .venv/bin/python -u tools/diagnose_export_fold.py
"""

from __future__ import annotations

import numpy as np

from drosh_ml import export, features as feat, train as train_mod
from drosh_ml.export import _probe_commands, _sigmoid
from drosh_ml.normalize import normalize, presence_ngrams
from drosh_ml.train import _load_dataset


def main() -> None:
    rows = _load_dataset()
    model = train_mod.fit(rows)

    vectorizer = model.vectorizer
    idf = vectorizer.idf_
    sparse_w = model.weights[: len(idf)]
    dense_w = model.weights[len(idf) :]
    folded = sparse_w * idf

    print(f"len(idf_)          = {len(idf_ := idf)}")
    print(f"len(vocabulary_)   = {len(vectorizer.vocabulary_)}")
    print(f"len(weights)       = {len(model.weights)}")
    print(f"len(sparse_w)      = {len(sparse_w)}")
    print(f"len(dense_w)       = {len(dense_w)}  (expected {feat.DENSE_DIM})")
    print(f"dense spec mean    = {model.dense_spec.mean.shape}")
    print(f"dense spec std     = {model.dense_spec.std.shape}")
    print(f"intercept          = {model.intercept:+.6f}")
    print(f"thresholds         = {model.thresholds}")
    print()

    probe = _probe_commands()

    matrix, _, _ = train_mod._build_matrix(
        probe, vectorizer, model.dense_spec, fit_dense=False
    )
    unfitted = 2.0 * _sigmoid(matrix @ model.weights + model.intercept)

    worst = 0.0
    for row, command in enumerate(probe):
        norm = normalize(command)

        dense_z = feat.standardise(feat.dense_features(norm)[None, :], model.dense_spec)
        dense_term = float((dense_z @ dense_w)[0])

        sparse_folded = 0.0
        sparse_unfused = 0.0
        hits = 0
        for gram in presence_ngrams(norm.views):
            index = vectorizer.vocabulary_.get(gram)
            if index is not None:
                hits += 1
                sparse_folded += float(folded[index])
                sparse_unfused += float(sparse_w[index]) * float(idf[index])

        folded_score = 2.0 / (1.0 + np.exp(-(dense_term + sparse_folded + model.intercept)))
        unfused_score = 2.0 / (1.0 + np.exp(-(dense_term + sparse_unfused + model.intercept)))
        reference = float(unfitted[row])
        diff = abs(folded_score - reference)
        worst = max(worst, diff)

        print(f"  {command[:38]:<40} ref={reference:.6f} folded={folded_score:.6f} "
              f"unfused={unfused_score:.6f} diff={diff:.2e} hits={hits} dense={dense_term:+.5f}")

    print()
    print(f"worst diff = {worst:.6e}   tolerance = {export.FOLD_TOLERANCE:.1e}")
    print("VERDICT:", "PASS" if worst <= export.FOLD_TOLERANCE else "FAIL")


if __name__ == "__main__":
    main()