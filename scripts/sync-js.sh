#!/usr/bin/env bash
# Regenerate the JavaScript side of the normalisation contract from the Python
# definition.
#
# ml/src/drosh_ml/normalize.py owns the Unicode folding table.
# html/command-risk/risk-scoring.js consumes it. This script rewrites the
# generated block in the JS file so the two cannot drift by hand-editing.
#
# Run this after ANY change to _FOLD_GROUPS in normalize.py, and commit both
# files in the same commit. ml/tests/test_parity.py would catch the drift, but
# only after it is already in git.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
JS="html/command-risk/risk-scoring.js"
MARKER="/*__FOLD_TABLE__*/"
TABLE_FILE="$(mktemp -t drosh-fold-table.XXXXXX.js)"
trap 'rm -f "$TABLE_FILE"' EXIT

if ! grep -qF "$MARKER" "$JS"; then
  echo "error: marker $MARKER not found in $JS" >&2
  echo "       The generated block was hand-edited — restore the marker." >&2
  exit 1
fi

PYTHONPATH="ml/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m drosh_ml.normalize --emit-js > "$TABLE_FILE"

TABLE_FILE="$TABLE_FILE" JS="$JS" MARKER="$MARKER" python3 - <<'PY'
import os
import pathlib

target = pathlib.Path(os.environ["JS"])
marker = os.environ["MARKER"]
table = pathlib.Path(os.environ["TABLE_FILE"]).read_text(encoding="utf-8").rstrip("\n")

source = target.read_text(encoding="utf-8")
if marker not in source:
    raise SystemExit(f"marker {marker} vanished before substitution")

target.write_text(source.replace(marker, table), encoding="utf-8")
print(f"    injected {table.count('[0x')} fold entries into {target}")
PY

echo "==> JS self-test"
node html/command-risk/risk-scoring-selftest.mjs

echo "==> Cross-language parity"
PYTHONPATH="ml/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m pytest ml/tests/test_parity.py -q