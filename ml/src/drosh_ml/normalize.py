"""Command normalisation — THE shared specification for the risky-command model.

This module is the single source of truth for how a raw shell command line is
turned into the token stream the model consumes. It has a byte-for-byte twin in
``html/command-risk/risk-scoring.js``. If the two ever disagree, the model is
effectively a different model on the device than the one that was evaluated —
so every rule below is numbered, and any edit must be made here first, then
mirrored in JavaScript, then covered by a golden-vector test.

Why the pipeline looks like this
--------------------------------
A user typing ``rm -rf`` in Drosh may produce any of::

    rm -rf /
    rm     -rf     /
    r"m" -rf /
    Rm -Rf /
    rm -rf /          # silinmesi gereken dosyalar
    \\033[31mrm -rf /

Normalising all of those onto a small set of canonical strings is what lets a
model trained on synthetic text generalise to real input. The critical design
decision is that **the output is pure ASCII**. Once every character outside
0x20-0x7E has been removed, Python code points, JS UTF-16 code units and bytes
are all the same thing, so the n-gram windowing in :func:`presence_ngrams`
cannot silently diverge between the two implementations. That single property
removes the entire class of Turkish-locale and emoji slicing bugs — ``ğ`` alone
would otherwise be one Python code point but one JS surrogate half.

Pipeline order (do not reorder without reading the tests)
---------------------------------------------------------
1. :func:`strip_ansi`      remove escape sequences *before* touching control
                           characters, otherwise the printable tail of
                           ``ESC[31m`` (``31m``) would survive as text.
2. segment split           split on ``[;|&]`` runs and on newlines. A
                           newline-separated command block becomes separate
                           segments so a single dangerous line inside a
                           multi-line paste is still visible to the
                           per-segment dense features.
3. per segment             whitespace-ish controls to space, remaining controls
                           deleted, folded to ASCII, lowercased, whitespace
                           collapsed.
4. reassemble              ``' ; '.join(segments)`` — a distinctive separator
                           so n-grams spanning a segment boundary are
                           distinguishable from n-grams inside one command.

Two views are derived from the result, and n-grams are extracted from both:

``text``      the reassembled string, quotes intact
``unquoted``  the same with ``'`` and ``"`` removed

The second view exists so that quote-based obfuscation (``r"m" -rf /``) lands on
the same n-gram set as the plain form. Presence semantics are used throughout —
see :func:`presence_ngrams` for why counting is deliberately not used.
"""

from __future__ import annotations

import re
import string
import sys
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

__all__ = [
    "NGRAM_MIN",
    "NGRAM_MAX",
    "Normalized",
    "normalize",
    "presence_ngrams",
    "strip_ansi",
    "fold_ascii",
]

# N-gram window. (2, 5) is the band that carries shell-specific signal:
#   "rm", "dd", "sh"      the binary itself
#   " -r", "if=", "chmo"  the flag shapes, including truncated ones
#   "sudo", ":(){", "df "  whole destructive idioms
NGRAM_MIN = 2
NGRAM_MAX = 5

#: Separator used when re-joining segments. Trailing space matters — it keeps
#: n-grams at a boundary from being identical to n-grams inside a segment.
SEGMENT_SEP = " ; "

#: Characters that separate independent shell commands. ``&`` is included so
#: that ``a && rm -rf /`` and ``a; rm -rf /`` behave identically. ``\r``/``\n``
#: are included so a multi-line paste is treated as separate commands rather
#: than one long one. The cost of including ``&`` is that ``2>&1`` yields a
#: harmless empty trailing segment.
SEGMENT_SPLIT_RE = re.compile(r"[;|&\r\n]+")

# Escape sequences. Ordered alternation: CSI and OSC must be tried before the
# generic two-character escape branch, otherwise ESC[ would match as a
# two-character escape and leave "31m" behind as text.
#   [0-?]*      parameter bytes 0x30-0x3F
#   [ -/]*      intermediate bytes 0x20-0x2F
#   [@-~]       final byte
_ANSI_RE = re.compile(
    r"\x1b(?:"
    r"\[[0-?]*[ -/]*[@-~]"
    r"|\][^\x07\x1b]*(?:\x07|\x1b\\)"
    r"|[PX^_][^\x1b]*(?:\x1b\\)"
    r"|[@-Z\x5c-\x5f]"
    r"|[ -/]*[0-~]"
    r")"
)

# Whitespace-like control characters become a space; every other control
# character is deleted outright.
_WS_CONTROL_RE = re.compile(r"[\t\v\f\r\n\x0b]")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

_SPACE_RUN_RE = re.compile(r" {2,}")

# --- Unicode -> ASCII folding ------------------------------------------------
#
# NFKD decomposition plus combining-mark removal handles the Latin-with-accent
# cases (é -> e) but NOT the characters that have no decomposition form. The
# Turkish dotted/dotless I and s/g are exactly such characters and they are not
# optional here: Turkish filenames ("İndirilenler", "şifre.txt", "günlük.log")
# are among the most likely non-ASCII strings a Drosh user will actually type.
#
# NFKD does handle İ (U+0130 -> I + U+0307), so it appears in the table only as
# a safety net. Anything not listed here is silently dropped, which is why the
# table errs toward over-inclusion: a dropped character can fuse two tokens
# together, whereas a mapped one cannot.
_FOLD_GROUPS: tuple[tuple[str, str], ...] = (
    # Latin-1 Supplement, uppercase
    ("ÀÁÂÃÄÅ", "A"),
    ("ÈÉÊË", "E"),
    ("ÌÍÎÏ", "I"),
    ("ÒÓÔÕÖØ", "O"),
    ("ÙÚÛÜ", "U"),
    ("ÝŶŸ", "Y"),
    ("Ç", "C"),
    ("Ñ", "N"),
    ("Ð", "D"),
    ("Þ", "TH"),
    ("Æ", "AE"),
    # Latin-1 Supplement, lowercase
    ("àáâãäå", "a"),
    ("èéêë", "e"),
    ("ìíîï", "i"),
    ("òóôõöø", "o"),
    ("ùúûü", "u"),
    ("ýÿŷ", "y"),
    ("ç", "c"),
    ("ñ", "n"),
    ("ð", "d"),
    ("þ", "th"),
    ("æ", "ae"),
    # Turkish (Latin Extended-A, no NFKD decomposition)
    ("ĞĠĢ", "G"),
    ("ĝğġģ", "g"),
    ("ıİ", "i"),
    ("Ş", "S"),
    ("ş", "s"),
    # Remaining Latin Extended-A worth carrying
    ("ĄĆĈĊČ", "C"),
    ("ĎĐ", "D"),
    ("ĒĖĘĚ", "E"),
    ("ĜĞĠĢ", "G"),
    ("ĤĦ", "H"),
    ("Ĵ", "J"),
    ("Ķ", "K"),
    ("ĹĻĽĿŁ", "L"),
    ("ŃŅŇ", "N"),
    ("ŔŖŘ", "R"),
    ("ŚŜŞȘ", "S"),
    ("ŢŤȚ", "T"),
    ("ŨŪŬŮŰŲ", "U"),
    ("Ŵ", "W"),
    ("Ŷ", "Y"),
    ("ŹŻŽ", "Z"),
    ("Œ", "OE"),
    ("œ", "oe"),
    ("ſ", "s"),
    ("ʼ", "'"),
    ("ċč", "c"),
    ("ďđ", "d"),
    ("ēėęě", "e"),
    ("ĝğġģ", "g"),
    ("ĥħ", "h"),
    ("ĵ", "j"),
    ("ķ", "k"),
    ("ĺļľŀł", "l"),
    ("ńņň", "n"),
    ("ŕŗř", "r"),
    ("śśŝşș", "s"),
    ("ţťț", "t"),
    ("ũūŭůűų", "u"),
    ("ŵ", "w"),
    ("ŷ", "y"),
    ("źżž", "z"),
    ("ɑ", "a"),
    ("ɐ", "a"),
    ("ə", "e"),
    ("ɛ", "e"),
    ("ɜ", "e"),
    ("ɪ", "i"),
    ("ɩ", "i"),
    ("ɯ", "u"),
    ("ʏ", "y"),
    # Ligatures and long s
    ("ﬀ", "ff"),
    ("ﬁ", "fi"),
    ("ﬂ", "fl"),
    ("ﬃ", "ffi"),
    ("ﬄ", "ffl"),
    ("ſ", "s"),
    # Punctuation and symbols that survive a copy-paste out of a document
    ("–—‐‑‒", "-"),
    ("‘’‚‛′", "'"),
    ("“”„‟″", '"'),
    ("…", "..."),
    ("•·", "*"),
    ("×", "x"),
    ("÷", "/"),
    ("«", '"'),
    ("»", '"'),
    ("−", "-"),
    # Non-breaking / thin / hair spaces all collapse to a plain space.
    ("\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f", " "),
    # Zero-width and byte-order marks are deleted: they are invisible in a
    # terminal but would survive naive slicing and hide "rm" from the model.
    ("\u200b\u200c\u200d\ufeff", ""),
    ("\ufffd", ""),
)

_TRANSLATION: dict[int, str] = {}
for _source, _target in _FOLD_GROUPS:
    for _char in _source:
        _TRANSLATION[ord(_char)] = _target
del _source, _target, _char

_ASCII_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)

# Printable ASCII range the pipeline guarantees as output.
_ASCII_MIN = 0x20
_ASCII_MAX = 0x7E


@dataclass(frozen=True, slots=True)
class Normalized:
    """The canonical form of one raw command line.

    Attributes:
        raw:   the input exactly as received, never modified.
        text:  all segments joined with :data:`SEGMENT_SEP`, ASCII-only,
               lowercased, whitespace collapsed, quotes intact.
        unquoted: ``text`` with ``'`` and ``"`` removed.
        segments: individually normalised, non-empty command segments.
        dropped_non_ascii: how many characters the ASCII fold discarded.
            Non-zero is normal for Turkish input and is worth surfacing in
            evaluation, because it means the model is reasoning about a lossy
            rendering of what the user typed.
    """

    raw: str
    text: str
    unquoted: str
    segments: tuple[str, ...]
    dropped_non_ascii: int

    @property
    def views(self) -> tuple[str, ...]:
        """n-gram sources, in the order they are fed to the model."""
        return (self.text, self.unquoted)


def strip_ansi(value: str) -> str:
    """Remove ANSI/VT escape sequences (step 1).

    Must run before control characters are deleted. ``ESC[31m`` would otherwise
    degrade to the literal text ``31m``, injecting noise into the n-gram space.
    """
    return _ANSI_RE.sub("", value)


def fold_ascii(value: str) -> tuple[str, int]:
    """Fold ``value`` to printable ASCII (step 3a).

    Returns the folded string and the number of characters dropped because they
    had no mapping and were not already ASCII.

    NFKD first, then combining marks stripped, then the explicit table, then
    everything outside 0x20-0x7E dropped. Dropping rather than substituting a
    placeholder is deliberate: a placeholder would mint new n-grams that appear
    nowhere in training.
    """
    decomposed = unicodedata.normalize("NFKD", value)
    without_marks = "".join(ch for ch in decomposed if not unicodedata.category(ch).startswith("M"))
    translated = without_marks.translate(_TRANSLATION)

    kept: list[str] = []
    dropped = 0
    for ch in translated:
        code = ord(ch)
        if _ASCII_MIN <= code <= _ASCII_MAX:
            kept.append(ch)
        else:
            dropped += 1
    return "".join(kept), dropped


def _normalise_segment(value: str) -> str:
    """Apply steps 3a-3d to a single command segment."""
    spaced = _WS_CONTROL_RE.sub(" ", value)
    stripped = _CONTROL_RE.sub("", spaced)
    folded, _ = fold_ascii(stripped)
    lowered = folded.translate(_ASCII_LOWER)
    return _SPACE_RUN_RE.sub(" ", lowered).strip(" ")


def normalize(raw: str) -> Normalized:
    """Normalise one raw command line.

    >>> normalize("rm -rf /").text
    'rm -rf /'
    >>> normalize("rm\\t-rf  /").segments
    ('rm -rf /',)
    >>> normalize("ls\\nrm -rf /").segments
    ('ls', 'rm -rf /')
    >>> normalize("Ğünye").text
    'gunye'
    """
    without_escapes = strip_ansi(raw)
    segments = tuple(
        segment
        for segment in (_normalise_segment(part) for part in SEGMENT_SPLIT_RE.split(without_escapes))
        if segment
    )

    text = SEGMENT_SEP.join(segments)
    unquoted = text.replace("'", "").replace('"', "")

    # Counted on the pre-segmentation string so that a character dropped from
    # any segment is still reported.
    _, dropped = fold_ascii(without_escapes)

    return Normalized(
        raw=raw,
        text=text,
        unquoted=unquoted,
        segments=segments,
        dropped_non_ascii=dropped,
    )


def presence_ngrams(views: Sequence[str]) -> set[str]:
    """Extract the n-gram feature set for one command (steps 5-6).

    Presence semantics, not counts, and this is a deliberate choice rather than
    a simplification:

    * ``rm -rf / rm -rf / rm -rf /`` must score identically to ``rm -rf /``.
      With counts the score grows superlinearly and an attacker who cannot read
      the model can inflate or deflate it by repetition.
    * No length normalisation is needed, so ``|w . x|`` is an exact fold of the
      IDF weights into the coefficients and the exported model is a single flat
      float array.
    * A long compound command (``a && b && c && ...``) already reads as more
      dangerous through the dense features, which is where that signal belongs.
    """
    found: set[str] = set()
    for view in views:
        length = len(view)
        for size in range(NGRAM_MIN, NGRAM_MAX + 1):
            if length < size:
                break
            # Safe because fold_ascii guarantees 0x20-0x7E only, which makes
            # code points, UTF-16 code units and bytes interchangeable here.
            for start in range(length - size + 1):
                found.add(view[start : start + size])
    return found


def emit_js_fold_table(indent: str = "    ") -> str:
    """Render the Unicode folding table as JavaScript source.

    The table in this module is the canonical one. Emitting the JavaScript from
    it rather than maintaining a hand-written copy means the two runtimes
    cannot drift on the Turkish and Latin-Extended-A cases, which are exactly
    the characters most likely to be mistyped in one implementation and correct
    in the other.

    Callers write the result between the BEGIN/END GENERATED markers in
    ``html/command-risk/risk-scoring.js``:

        python -m drosh_ml.normalize --emit-js
    """
    lines = [
        f"{indent}// GENERATED from ml/src/drosh_ml/normalize.py — do not edit by hand.",
        f"{indent}// {len(_TRANSLATION)} codepoints. Regenerate with:",
        f"{indent}//   python -m drosh_ml.normalize --emit-js",
        f"{indent}const FOLD = new Map([",
    ]
    for code in sorted(_TRANSLATION):
        target = _TRANSLATION[code]
        lines.append(f"{indent}  [0x{code:04X}, {_js_string(target)}],")
    lines.append(f"{indent}]);")
    return "\n".join(lines)


def _js_string(value: str) -> str:
    """Render a Python string as a JavaScript double-quoted literal.

    Control characters are escaped numerically. Anything printable ASCII is
    emitted literally; non-ASCII source characters are emitted as \\uXXXX so
    the generated file stays 7-bit clean and cannot be corrupted by an editor
    that does not preserve Unicode.
    """
    out = ['"']
    for char in value:
        code = ord(char)
        if char == '"':
            out.append('\\"')
        elif char == "\\":
            out.append("\\\\")
        elif 0x20 <= code <= 0x7E:
            out.append(char)
        else:
            out.append(f"\\u{code:04X}")
    out.append('"')
    return "".join(out)


def _self_test() -> int:
    """Run the invariant checks that the JS mirror must also satisfy."""
    cases: list[tuple[str, str, object]] = [
        ("plain", "rm -rf /", "rm -rf /"),
        ("tab+spaces", "rm\t\t-rf   /", "rm -rf /"),
        ("newline segments", "ls\nrm -rf /", "ls ; rm -rf /"),
        ("ansi colour", "\x1b[31mrm -rf /\x1b[0m", "rm -rf /"),
        ("ansi 256", "\x1b[38;5;196mrm -rf /", "rm -rf /"),
        ("osc title", "\x1b]0;title\x07rm -rf /", "rm -rf /"),
        ("uppercase", "RM -RF /", "rm -rf /"),
        ("turkish", "ĞÜNYE", "gunye"),
        ("turkish dotted I", "İndirilenler", "indirilenler"),
        ("accented", "café", "cafe"),
        ("nbsp", "rm -rf\u00a0/", "rm -rf /"),
        ("quoted", 'r"m" -rf /', 'r"m" -rf /'),
        ("empty", "   ", ""),
    ]

    failures = 0
    print(f"{'case':<22} {'expected':<24} {'actual':<24} ok")
    print("-" * 78)
    for name, raw, expected in cases:
        actual = normalize(raw).text
        ok = actual == expected
        failures += not ok
        print(f"{name:<22} {expected!r:<24} {actual!r:<24} {'yes' if ok else 'NO'}")

    quoted = normalize('r"m" -rf /')
    folded = normalize("rm -rf /")
    overlap = len(presence_ngrams(quoted.views) & presence_ngrams(folded.views))
    overlap_ratio = overlap / max(1, len(presence_ngrams(folded.views)))
    print(f"\nquote-obfuscation n-gram overlap with plain form: {overlap_ratio:.1%}")
    print(f"total failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    if "--emit-js" in sys.argv:
        print(emit_js_fold_table("    "))
    else:
        sys.exit(_self_test())