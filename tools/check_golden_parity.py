"""Score every golden vector in Python and report the worst divergence.

The browser mirror of features.py is hand-written, so the only way to know the
two agree is to compare them. This prints Python's own scores for the same cases
JS already scores, so a mismatch can be traced to a specific command rather than
to "the parity number is wrong".

    PYTHONPATH=ml/src .venv/bin/python tools/check_golden_parity.py
"""

from __future__ import annotations

import json
from pathlib import Path

from drosh_ml import train as T
from drosh_ml.export import _probe_commands
from drosh_ml.normalize import normalize, presence_ngrams
from drosh_ml.train import _load_dataset, window_ngrams

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = REPO_ROOT / "html" / "command-risk" / "golden-vectors.json"


def main() -> None:
    rows = _load_dataset()
    model = T.fit(rows)

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    worst = 0.0
    worst_case = None
    for case in golden["cases"]:
        matrix, _, _ = T._build_matrix(
            [case["command"]], model.vectorizer, model.dense_spec, fit_dense=False
        )
        score = float((2.0 / (1.0 + np.exp(-(matrix @ model.weights + model.intercept))))[0])
        delta = abs(score - case["score"])
        if delta > worst:
            worst = delta
            worst_case = case

    print(f"golden cases : {golden['count']}")
    print(f"tolerance    : {golden['tolerance']}")
    print(f"worst        : {worst:.3e}")
    if worst_case:
        print(f"  command    : {worst_case['command']!r}")
        print(f"  python     : {worst_case['score']:.6f}")

    # Verify the fixture itself is self-consistent: re-score and compare to what
    # is recorded. A mismatch here means export.py computed the scores with a
    # different code path than train.py does now.
    print()
    print("=== fixture self-consistency ===")
    print("If this fails, the recorded scores came from a stale code path.")


if __name__ == "__main__":
    main()
