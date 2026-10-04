#!/usr/bin/env bash
# Full pipeline: generate dataset -> train -> evaluate -> export.
#
# Deterministic by construction: fixed seeds throughout, and the dataset is a
# pure function of the files in ml/data/ plus the code in ml/src/drosh_ml/.
# Running this twice on the same commit must produce byte-identical
# html/command-risk/risk-model.js — CI asserts exactly that.

set -euo pipefail
cd "$(dirname "$0")/.."

source .venv/bin/activate
export PYTHONPATH="ml/src${PYTHONPATH:+:$PYTHONPATH}"

echo "==> 1/4 dataset"
python -m drosh_ml.build_dataset

echo "==> 2/4 train"
python -m drosh_ml.train

echo "==> 3/4 evaluate"
python -m drosh_ml.evaluate

echo "==> 4/4 export"
python -m drosh_ml.export

echo "==> done. Prototype: open html/command-risk/index.html"
