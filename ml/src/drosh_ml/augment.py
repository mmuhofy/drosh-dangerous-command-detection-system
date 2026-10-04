"""Obfuscation transforms, split into what the model trains on and what it is
tested on.

An uncomfortable finding that shaped this file
----------------------------------------------
Most "classic" shell obfuscation is *already neutralised* by the normaliser, so
transforming with it teaches nothing:

  * ``RM -RF /``            -> ``normalize`` lowercases. Identical features.
  * ``rm  -r  -f  /``       -> whitespace collapses. Identical features.
  * ``r\\u200bm -rf /``     -> zero-width is stripped, recovering ``rm -rf /``.

Augmenting with those would inflate the corpus with duplicates and, worse,
create a false sense of robustness. So the transform list below is restricted to
changes that **survive normalisation** — they alter the feature set while
preserving what the shell actually does. Each entry says which.

The train/holdout split
-----------------------
``HOLDOUT_FAMILIES`` are never applied to the training set. They exist to answer
one question: *did the model learn shell semantics, or did it memorise the
surface forms in the grammar?* A model that has only ever seen
``rm -rf /`` spelled plainly will fail on ``base64``-wrapped input, and that
failure is invisible unless the input was held out.

Determinism
-----------
Every transform takes an explicit ``random.Random`` seeded from the command
itself, so the corpus is a pure function of the source. CI retrains and asserts
the exported model is byte-identical; a transform that consumed a global RNG
would break that.
"""

from __future__ import annotations

import base64
import hashlib
import random
import re
from collections.abc import Callable, Sequence

__all__ = ["TRAIN_FAMILIES", "HOLDOUT_FAMILIES", "augment", "holdout_variants"]

Transform = Callable[[str, random.Random], "str | None"]


def _seed_for(command: str) -> random.Random:
    """A per-command RNG, stable across runs and machines."""
    digest = hashlib.sha256(command.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


# ---------------------------------------------------------------------------
# Training transforms — all of these survive normalisation
# ---------------------------------------------------------------------------

_FLAG_LONGFORM = (
    (re.compile(r"(?<!\S)-rf(?!\S)"), "--recursive --force"),
    (re.compile(r"(?<!\S)-fr(?!\S)"), "--force --recursive"),
    (re.compile(r"(?<!\S)-r(?!\S)"), "--recursive"),
    (re.compile(r"(?<!\S)-f(?!\S)"), "--force"),
    (re.compile(r"(?<!\S)-i(?!\S)"), "--interactive"),
    (re.compile(r"\s-l\b"), " --dereference"),
    (re.compile(r"\s-L\b"), " --dereference"),
)


def shell_wrapper(command: str, rng: random.Random) -> str | None:
    """``bash -c "..."`` — survives: adds real tokens, same effect."""
    if '"' in command:
        return f"sh -c '{command}'"
    return f'bash -c "{command}"' if rng.random() < 0.5 else f"sh -c '{command}'"


def subshell(command: str, rng: random.Random) -> str | None:
    """``( cmd )`` — survives: extra bracket tokens, same effect."""
    return f"( {command} )"


def env_prefix(command: str, rng: random.Random) -> str | None:
    """``LC_ALL=C cmd`` — survives: adds tokens, same effect."""
    prefixes = ("LC_ALL=C", "LANG=C.UTF-8", "TERM=dumb", "PATH=$PATH")
    return f"{rng.choice(prefixes)} {command}"


def flag_longform(command: str, rng: random.Random) -> str | None:
    """``-rf`` -> ``--recursive --force`` — survives as new n-grams."""
    out = command
    changed = False
    for pattern, replacement in _FLAG_LONGFORM:
        new = pattern.sub(replacement, out, count=1)
        if new != out:
            out = new
            changed = True
    return out if changed else None


def redirect_noise(command: str, rng: random.Random) -> str | None:
    """``cmd >/dev/null 2>&1`` — survives: adds tokens, output discarded."""
    return rng.choice(
        (
            f"{command} 2>/dev/null",
            f"{command} >/dev/null 2>&1",
            f"{command} >/dev/null",
            f"{command} 1>/dev/null",
        )
    )


def segment_suffix(command: str, rng: random.Random) -> str | None:
    """``cmd && echo ok`` — survives: a second segment is added."""
    return rng.choice(
        (
            f"{command} && echo ok",
            f"{command} ; true",
            f"{command} || true",
            f"{command} && pwd",
            f"{command} ; date",
        )
    )


def path_prefix(command: str, rng: random.Random) -> str | None:
    """``cd /tmp && cmd`` — survives: an unrelated leading segment."""
    return rng.choice(
        (
            f"cd /tmp && {command}",
            f"cd ~ && {command}",
            f"cd /var/www && {command}",
            f"cd . && {command}",
        )
    )


def quote_split(command: str, rng: random.Random) -> str | None:
    """``r''m -rf /`` — survives in the text view, neutralised by ``unquoted``.

    Kept because it trains the model to not depend on quote placement. The
    unquoted view already collapses it, so the two views together should make
    the command score the same as the plain form; that is an assertion in
    ``ml/tests/test_dataset.py``.

    The token length threshold is 2, not 3: ``rm`` is two characters and is
    precisely the token that matters most.
    """
    match = re.match(r"^([^\s]{2,})", command)
    if not match:
        return None
    token = match.group(1)
    if not token[0].isalpha():
        return None
    quote = rng.choice(("'", '"'))
    return f"{token[0]}{quote}{quote}{token[1:]} {command[len(token):]}"


def comment_tail(command: str, rng: random.Random) -> str | None:
    """``rm -rf / # notdugumda`` — survives: trailing comment tokens."""
    comments = ("# duzelt", "# sonra bak", "# gerekli", "# cleanup", "# TODO")
    return f"{command} {rng.choice(comments)}"


def continue_flag(command: str, rng: random.Random) -> str | None:
    """``rm -r -f \\\\n /`` — survives: whitespace collapse leaves the token."""
    return re.sub(r"(\s-\w+)\s", r"\1 \\\n   ", command, count=1)


TRAIN_FAMILIES: dict[str, Transform] = {
    "shell_wrapper": shell_wrapper,
    "subshell": subshell,
    "env_prefix": env_prefix,
    "flag_longform": flag_longform,
    "redirect_noise": redirect_noise,
    "segment_suffix": segment_suffix,
    "path_prefix": path_prefix,
    "quote_split": quote_split,
    "comment_tail": comment_tail,
    "line_continuation": continue_flag,
}


# ---------------------------------------------------------------------------
# Holdout transforms — never seen during training
# ---------------------------------------------------------------------------


def holdout_zero_width(command: str, rng: random.Random) -> str | None:
    """Insert U+200B inside the longest token. The normaliser strips it back out.

    Targets the *longest* token rather than the first one: the first token is
    often two characters (``rm``) and a threshold that skipped it left this
    family silently producing nothing.
    """
    tokens = command.split()
    if not tokens:
        return None
    index = max(range(len(tokens)), key=lambda i: len(tokens[i]))
    token = tokens[index]
    if len(token) < 3:
        return None
    cut = max(2, len(token) // 2)
    tokens[index] = f"{token[:cut]}\u200b{token[cut:]}"
    return " ".join(tokens)


def holdout_nbsp(command: str, rng: random.Random) -> str | None:
    """Swap the first two spaces for NBSP. The normaliser maps them back."""
    if "  " not in command and " " not in command:
        return None
    return command.replace(" ", "\u00a0", 2)


def holdout_base64(command: str, rng: random.Random) -> str | None:
    """``echo <b64> | base64 -d | sh`` — same effect, none of the same tokens.

    Worth stating plainly what this family tests, because it is not obvious:
    the normaliser lowercases, and base64 is case-sensitive, so the payload the
    model sees is **corrupt**. ``b64encode("rm -rf /")`` is ``cm0gLXJmIC8=``,
    which normalises to ``cm0glxjmic8=`` and decodes to ``b'rm \\x97\\x18\\xe6
    \\x89\\xcf'`` — not the original command.

    The shell still receives the real, working payload; only the model's view is
    damaged. So the model is structurally incapable of reading the payload and
    has to learn the wrapper instead. ``base64 -d | sh`` must be dangerous on its
    own, whatever it carries. That is the correct lesson, and this family is the
    only place it gets tested.
    """
    payload = base64.b64encode(command.encode("utf-8")).decode("ascii")
    return f"echo {payload} | base64 -d | sh"


def holdout_eval(command: str, rng: random.Random) -> str | None:
    """``eval "$(echo cmd)"`` — an extra layer of indirection.

    Weaker than base64 because the command text survives verbatim inside the
    substitution, so the model still sees ``rm -rf``. Kept as a regression
    guard: if eval-wrapping ever *hides* the payload, the model must still fire.
    """
    return f'eval "$(echo {command})"'


def holdout_hex_escape(command: str, rng: random.Random) -> str | None:
    """``$'\\x72\\x6d' -rf /`` — the first token as hex escapes.

    Nothing in the pipeline decodes these, so the model sees a literal
    ``$'\\x72\\x6d'`` where the shell sees ``rm``. A model that has learned
    surface patterns will miss this entirely, which is exactly the point of
    holding it out.
    """
    tokens = command.split()
    if not tokens:
        return None
    token = tokens[0]
    if len(token) < 2 or not token[0].isalpha():
        return None
    escaped = "$'" + "".join(f"\\x{ord(ch):02x}" for ch in token) + "'"
    tokens[0] = escaped
    return " ".join(tokens)


def holdout_octal_escape(command: str, rng: random.Random) -> str | None:
    """``$'\\162\\155' -rf /`` — same idea in octal, different surface."""
    tokens = command.split()
    if not tokens:
        return None
    token = tokens[0]
    if len(token) < 2 or not token[0].isalpha():
        return None
    escaped = "$'" + "".join(f"\\0{ord(ch):03o}" for ch in token) + "'"
    tokens[0] = escaped
    return " ".join(tokens)


HOLDOUT_FAMILIES: dict[str, Transform] = {
    "zero_width": holdout_zero_width,
    "nbsp": holdout_nbsp,
    "base64": holdout_base64,
    "eval": holdout_eval,
    "hex_escape": holdout_hex_escape,
    "octal_escape": holdout_octal_escape,
}


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def augment(command: str, families: Sequence[str] | None = None) -> list[str]:
    """Return the training variants of one command.

    Each family is applied independently to the *original*, not chained, so the
    result does not grow combinatorially and every variant is attributable to
    exactly one transform. That attribution is what makes the per-family recall
    table in ``evaluate.py`` interpretable.
    """
    names = list(TRAIN_FAMILIES) if families is None else list(families)
    out: list[str] = []
    seen = {command}
    for name in names:
        transform = TRAIN_FAMILIES[name]
        try:
            variant = transform(command, _seed_for(command))
        except Exception:
            # A transform blowing up must never take the build down; it just
            # contributes nothing for this row.
            continue
        if variant is None:
            continue
        variant = variant.strip()
        if variant and variant not in seen:
            seen.add(variant)
            out.append(variant)
    return out


def holdout_variants(command: str) -> list[tuple[str, str]]:
    """Return ``(family_name, variant)`` pairs for the held-out transforms."""
    out: list[tuple[str, str]] = []
    seen = {command}
    for name, transform in HOLDOUT_FAMILIES.items():
        try:
            variant = transform(command, _seed_for(command))
        except Exception:
            continue
        if variant is None:
            continue
        variant = variant.strip()
        if variant and variant not in seen:
            seen.add(variant)
            out.append((name, variant))
    return out