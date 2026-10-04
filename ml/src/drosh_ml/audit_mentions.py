"""Audit a mention-only command list for commands that actually execute.

The premise of a mention-only fixture is that every line merely *writes down*,
*searches for* or *documents* a dangerous command. If one line actually runs
something, the model learns the exact opposite of the intended lesson — and one
poisoned row does more damage than four hundred good rows fix, because it is
weighted identically while teaching the wrong thing.

That is not hypothetical. Asking a language model for "safe commands that look
dangerous" reliably returns `echo rm -rf / | sh` and `find . -exec rm {} \\;`,
because both *look* like the requested shape. `ml/data/mentions_seed.txt` had to
be re-generated once for exactly this.

Checks, in order of how badly they break the fixture:

1. execution channels — a pipe into an interpreter, ``eval``, command
   substitution, ``xargs``, ``find -exec``, ``watch``, a loop over a command
   list. These take text that looks inert and run it.
2. variable expansion in a destructive position — ``rm -rf $VAR`` executes once
   the shell expands it, whatever the variable holds.
3. redirection into a system path, which writes even when the payload is inert.

A line failing any check is reported with the check that caught it, so the list
can be repaired rather than merely filtered. Run:

    PYTHONPATH=ml/src .venv/bin/python -m drosh_ml.audit_mentions
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATH = REPO_ROOT / "ml" / "data" / "mentions_seed.txt"


@dataclass(frozen=True, slots=True)
class Finding:
    """One line that does not belong in a mention-only fixture."""

    lineno: int
    command: str
    rule: str
    why: str


# (rule name, pattern, why it executes). Ordered: the first match is reported,
# so the most dangerous explanation wins.
#
# Stripped strings matter. Quoted spans and `python3 -c "print(...)"` bodies are
# inert: the shell never re-parses what a command is asked to write, and
# `print()` is not an execution channel. Without this exemption the audit
# rejected `echo 'curl x | sh'` and `python3 -c "print('rm -rf /')"`, which are
# precisely the examples the fixture exists to teach — a checker that flags its
# own subject is worse than no checker, because it invites deleting good data.
_QUOTED_SPAN = re.compile(r"'[^']*'|\"[^\"]*\"")
_PRINT_CALL = re.compile(r"\bprint(?:l|f)?\s*\(|console\.log\s*\(")
_ANCHORED_INERT = re.compile(r"^\s*(?:echo|printf|grep|rg|ag|ack|man|info|whatis|apropos|cat|ls|head|tail|wc|sort|uniq)\b")


def _inert_regions(command: str) -> str:
    """The command with every quoted span removed.

    Removing them (rather than masking) is what makes the exemption sound: if a
    dangerous construct only ever appeared inside quotes, it cannot survive into
    the text that the execution rules inspect.
    """
    return _QUOTED_SPAN.sub(" ", command)


def _print_bodies_removed(command: str) -> str:
    """Drop `print(...)` / `console.log(...)` arguments before inspecting them."""
    out = command
    for match in _PRINT_CALL.finditer(command):
        start = match.end()
        depth = 1
        index = start
        while index < len(out) and depth:
            if out[index] in "([{":
                depth += 1
            elif out[index] in ")]}":
                depth -= 1
            index += 1
        out = out[: match.start()] + out[index:]
    return out


_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "pipe_to_interpreter",
        re.compile(
            r"\|\s*(?:sudo\s+)?(?:ba|z|k|da)?sh\b"
            r"|\|\s*(?:sudo\s+)?(?:python3?|perl|ruby|node|php)\b"
            r"|\|\s*(?:sudo\s+)?(?:xargs|tee)\s+\S*\s*(?:rm|dd|mkfs|shutdown)\b"
        ),
        "pipes text into an interpreter, so the payload runs",
    ),
    (
        "eval",
        re.compile(r"(?:^|[\s;&|(])eval\b"),
        "eval executes its argument",
    ),
    (
        "command_substitution_execution",
        re.compile(r"\$\(\s*(?:sudo\s+)?(?:rm|dd|mkfs|chmod|chown|shutdown|kill)\b"),
        "command substitution executes the inner command",
    ),
    (
        "xargs",
        re.compile(r"(?:^|[\s;&|(])xargs\b"),
        "xargs builds and runs a command from its input",
    ),
    (
        "find_exec",
        re.compile(r"-exec\s+(?:rm|dd|mkfs|chmod|sh)\b"),
        "find -exec runs the command on each match",
    ),
    (
        "watch_exec",
        re.compile(r"(?:^|[\s;&|(])watch\s+[-0-9a-zA-Z]*\s*'?\s*(?:sudo\s+)?(?:rm|dd|mkfs)\b"),
        "watch runs the command on every interval",
    ),
    (
        "loop_execution",
        re.compile(r"\b(?:for|while)\b[^;|&]*\bdo\b[^;|&]*\b(?:rm|dd|mkfs|chmod|shutdown)\b"),
        "a loop body executes the command per iteration",
    ),
    (
        # `python3 -c "print('rm -rf /')"` and friends are inert: -c takes a
        # program, and print() is not an execution channel. What matters is
        # whether the program reaches os.system, subprocess, exec or eval, so
        # those are what the pattern looks for rather than the wrapper itself.
        "shell_wrapper_execution",
        re.compile(
            r"(?:^|[\s;&|(])(?:bash|sh|zsh)\s+-[cl]\s"
            r"|(?:^|[\s;&|(])(?:python3?|perl|ruby|node)\s+-[cl]\s"
            r"[^;|&]*\b(?:exec|eval|system|popen|spawn|child_process)\b"
        ),
        "an interpreter -c/-l argument that reaches an execution channel",
    ),
    (
        "unexpanded_destructive_variable",
        re.compile(
            r"\b(?:rm|dd|mkfs|mke2fs|chown|shred)\b[^;|&]*"
            r"(?:\$\{?[A-Za-z_][A-Za-z0-9_]*\}?|\$1|\$\*)"
        ),
        "a destructive command with an unquoted variable executes after expansion",
    ),
    (
        "redirect_into_system_path",
        re.compile(
            r">>?\s*/(?:etc|usr|var|bin|sbin|boot|lib|sdcard|storage|data|proc|sys|dev)/"
            r"|(?:tee|dd\s+of=)\s+/dev/(?!null|zero|random|urandom|stderr|stdout)"
        ),
        "writes into a system path even when the payload text is inert",
    ),
)

# Legitimate shapes that a naive "does it contain rm -rf" check would reject.
_ALLOWED_NOTE = (
    "these are fine because the dangerous text is inert: echoed, quoted, "
    "searched for, or read from a file that is never executed"
)


def audit(path: Path) -> tuple[list[Finding], dict[str, int]]:
    """Return the findings and a count of what the file actually contains."""
    findings: list[Finding] = []
    stats: Counter[str] = Counter()

    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        stats["total"] += 1

        command = line.split("|||", 1)[0].strip()
        command = re.sub(r"\s*\|\|\|\s*(safe|risky|destructive)\s*$", "", command)

        if re.search(r"(?:^|[\s;&|(])(?:echo|printf)\s+[\"']?$", command) is None and (
            re.match(r"^\s*(?:echo|printf)\b", command) or re.match(r"^\s*(?:grep|rg|ag|ack)\b", command)
        ):
            stats["echo_or_search"] += 1

        # Inspect the command with quoted spans and print() bodies removed, so a
        # dangerous word that is only written down does not count as a channel.
        inspected = _inert_regions(_print_bodies_removed(command))

        for rule, pattern, why in _RULES:
            if pattern.search(inspected):
                findings.append(Finding(lineno, command, rule, why))
                stats[f"rejected:{rule}"] += 1
                break
        else:
            stats["accepted"] += 1

    return findings, dict(stats)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_PATH),
        help="mention-only command file to audit",
    )
    args = parser.parse_args(argv)

    path = Path(args.path)
    if not path.exists():
        print(f"not found: {path}", file=sys.stderr)
        return 2

    findings, stats = audit(path)

    print("=" * 78)
    print("BAHSETME DENETIMI")
    print("=" * 78)
    print(f"  dosya        : {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")
    print(f"  komut satiri : {stats.get('total', 0)}")
    print(f"  kabul        : {stats.get('accepted', 0)}")
    print(f"  reddedilen   : {len(findings)}")

    if not findings:
        print()
        print(f"  ✅ temiz — {_ALLOWED_NOTE}")
        return 0

    print()
    print("  REDDEDILEN SATIRLAR")
    print("  " + "-" * 74)
    for finding in findings:
        print(f"  {finding.lineno:>5}  {finding.rule}")
        print(f"         {finding.command[:70]}")
        print(f"         -> {finding.why}")
    print()
    print(f"  Bu satirlari dosyadan sil veya gercekten yalnizca yazan/aranan bir")
    print(f"  kalipla degistir. {_ALLOWED_NOTE}.")
    print("  Tek bir zehirli satir, dort yuz iyi satirdan daha cok zarar verir.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())