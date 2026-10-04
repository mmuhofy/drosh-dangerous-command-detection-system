#!/usr/bin/env bash
# Freeze the resolved dependency set into requirements.lock.
#
# Run this after `scripts/bootstrap.sh` has produced a working environment and
# the numbers in the evaluation report look right. The lock file is the only
# trustworthy record of which scikit-learn produced a given exported model —
# scikit-learn has changed estimator internals and dtype handling across minor
# releases, so "requirements.txt" alone cannot reproduce a model.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Current environment"
PYTHONPATH="ml/src${PYTHONPATH:+:$PYTHONPATH}" python -c "
import sklearn, numpy, joblib, platform, sys
print(f'    python     {sys.version.split()[0]}')
print(f'    sklearn    {sklearn.__version__}')
print(f'    numpy      {numpy.__version__}')
print(f'    joblib     {joblib.__version__}')
"

echo "==> Freezing"
.venv/bin/python -m pip freeze --exclude-editable > requirements.lock

echo "==> requirements.lock written:"
cat requirements.lock
echo
echo "Commit this alongside the model it produced."
