"""Dangerous-command detection model for Drosh.

The training pipeline lives here; the runtime is a plain JavaScript file at
``html/command-risk/risk-scoring.js`` plus an exported model that loads in a
browser with no dependencies.
"""

__version__ = "0.1.0"

#: Bumped whenever the exported model format changes incompatibly. Written into
#: the export header and checked by the JavaScript loader, so a stale
#: risk-model.js is rejected loudly instead of silently mis-scoring.
MODEL_FORMAT_VERSION = 1

#: Bumped whenever the normalisation rules change. The export records the value
#: in force when it ran; the loader refuses a model whose normalisation version
#: differs, because a model scored against different rules is meaningless.
NORMALIZE_VERSION = "1"

__all__ = ["MODEL_FORMAT_VERSION", "NORMALIZE_VERSION", "__version__"]