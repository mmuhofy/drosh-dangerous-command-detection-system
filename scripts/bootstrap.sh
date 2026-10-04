#!/usr/bin/env bash
# One-shot environment bootstrap for the devcontainer.
#
# Idempotent: safe to re-run on an existing codespace.
#
# Creates a virtualenv, installs the package in editable mode so that
# `import drosh_ml` works without PYTHONPATH gymnastics, then installs the
# library dependencies. Deliberately does NOT create requirements.lock —
# that is pinned explicitly via scripts/pin.sh once we are satisfied with the
# resolved versions.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Python: $(python3 --version)"
echo "==> Node:    $(node --version 2>/dev/null || echo 'not available')"

if [ ! -d .venv ]; then
  echo "==> Creating virtualenv at .venv"
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --quiet --upgrade pip
echo "==> Installing drosh_ml (editable) + dependencies"
python -m pip install --quiet -e ml

echo "==> Verifying import"
python -c "import drosh_ml, sklearn, numpy; print('    drosh_ml', drosh_ml.__version__); print('    sklearn ', sklearn.__version__); print('    numpy   ', numpy.__version__)"

echo "==> Bootstrap complete. Try: python -m drosh_ml.normalize --selftest"