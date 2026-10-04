"""Assemble the training corpus.

Sources, in descending authority:

1. ``seed_from_muhofy.txt``      ground truth. Real commands, real labels.
2. ``negatives.py``              safe commands that look dangerous.
3. ``grammar.py``                destructive/risky commands, labels by construction.
4. ``augment.py``                obfuscated variants of all of the above.

Plus a held-out set built from ``augment.HOLDOUT_FAMILIES``, which never appears
in training.

The grader agreement report
---------------------------
Unlabelled seed entries need labels, and inventing them with an LLM would put
label noise into the one source of ground truth we have. So they go through a
transparent rule grader instead (:func:`grade`).

That grader is then run over the 1210 entries that *do* carry a label, and the
agreement rate is reported. This is the single most useful diagnostic in the
build: it says whether the synthetic half of the corpus is describing the same
world as the real half. Low agreement is not automatically a bug — the seed
contains judgement calls like ``rm -rf src`` labelled destructive while
``rm -rf node_modules`` is safe — but a large divergence means one of the two
halves is teaching the model something the other contradicts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from . import augment, grammar, negatives
from .labels import FALSE_NEGATIVE_COST, Risk
from .seed_loader import (
    SECTION_ACCIDENT,
    SECTION_FEARED,
    SECTION_LOOKS_SCARY,
    SECTION_ROUTINE,
    SeedEntry,
    load_seed,
)

__all__ = ["Row", "grade", "build", "main"]

REPO_ROOT = Path(__file__).resolve().parents[3]
SEED_PATH = REPO_ROOT / "ml" / "data" / "seed_from_muhofy.txt"
ARTIFACTS = REPO_ROOT / "ml" / "artifacts"


@dataclass(frozen=True, slots=True)
class Row:
    """One corpus row.

    ``category`` is kept alongside the label because evaluation groups by it: a
    single accuracy number hides the fact that a model can be excellent on
    ``fork_bomb`` and useless on ``raw_disk_write``.
    """

    command: str
    label: Risk
    category: str
    source: str
    section: str = ""
    transform: str = ""


# ---------------------------------------------------------------------------
# Rule grader
# ---------------------------------------------------------------------------

# Ordered: the first matching family wins, so the more specific patterns come
# first. Every entry is a named signal rather than a bare regex so the
# disagreement report can say *which* rule fired.
_SYSTEM_PATH_RE = re.compile(
    r"(?:^|[\s;&|])(/(?:etc|usr|var|bin|sbin|boot|lib|opt|srv|root|home|sys|proc|dev|sdcard"
    r"|storage|data)(?:/|\b)|\s~/?\s|\$\{?HOME\}?|\$\{?PWD\}?|\$\(pwd\))"
)
_WILDCARD_RE = re.compile(r"(?:^|[\s])\*(?:\.\*)?(?:\*)?(?:/|$)")
_BLOCK_DEV_RE = re.compile(r"/dev/(?:sd[a-z]|nvme\d|mmcblk|hd[a-z])")

# curl/wget alone are just downloaders — `curl -O file.zip` is routine. They are
# only dangerous when piped into an interpreter, which is checked separately in
# _PIPE_TO_SHELL_RE. The first draft of this grader flagged every `curl` as
# destructive and disagreed with the author on 31 rows; that was a grader bug,
# not a labelling disagreement.
_FETCH = ("curl", "wget", "fetch")
# Matches a pipe into an *interpreter invocation*, not merely any pipe into a
# word containing "node"/"python". The naive version flagged `ps aux | grep node`
# as a reverse shell, which is a grader bug the agreement report caught.
_PIPE_TO_SHELL_RE = re.compile(
    r"(?:curl|wget|fetch)[^;|&]*\|\s*(?:sudo\s+)?(?:ba|z|k|da)?sh\b"
    r"|\|\s*(?:sudo\s+)?(?:ba|z|k|da)?sh\b"
    r"|\|\s*(?:sudo\s+)?(?:python3?|perl|ruby|node|php)\b"
)

# Commands that destroy data or hand over control.
_DESTRUCTIVE_VERBS = (
    "mkfs", "mke2fs", "mkswap", "wipefs", "fdisk", "parted", "sgdisk", "gparted",
    "dd if=", "shutdown -h", "shutdown -r", "reboot", "poweroff", "halt",
    "init 0", "init 6", ":(){", ":(){:", "chmod 777", "chmod -R 777",
    "chmod -R a+w", "chmod a+rwx", "chmod 666", "chown -R root:root /",
    "authorized_keys", "nc -e", "ncat -e", "socat", "base64 -d |",
    "base64 --decode |", "eval \"$(echo",
)
_RISKY_VERBS = (
    "rm -rf", "rm -fr", "rm -r ", "rm -R ", "chmod -R", "chown -R", "kill -9",
    "killall", "pkill", "docker system prune", "docker image prune -a",
    "docker volume prune", "docker rm -f", "apt-get remove", "apt-get autoremove",
    "apt purge", "yum remove", "dnf remove", "git reset --hard", "git clean -f",
    "git push --force", "git push -f", "git checkout .", "git restore .",
    "git filter-branch", "git reflog expire", "kubectl delete", "truncate -s",
    "systemctl restart", "systemctl stop", "rm -f", "flush --noinput",
    "migrate reset", "db:reset", "force-reset", "--force-reset",
    "pip uninstall", "npm uninstall -g", "gem uninstall",
)

# Targets that make a recursive delete safe in practice: regenerable, disposable,
# and never something a user would miss. This is the rule that decides the
# highest-stakes case in the whole corpus — ``rm -rf node_modules`` must be
# silent and ``rm -rf /`` must not.
_SAFE_TARGETS = (
    "node_modules", "build/", "dist/", "target/", ".gradle", ".venv", "venv/",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".next", ".nuxt", ".cache",
    "vendor/", "pods/", "deriveddata", ".terraform", "coverage", ".tox",
    ".parcel-cache", ".turbo", "*.o", "*.class", "*.pyc", ".ds_store",
    "*.log", "*.tmp", "*.bak", "*~", ".cache/", "cmakecachefiles",
)


def grade(command: str) -> tuple[Risk, str]:
    """Assign a label to an unlabelled command.

    Returns:
        ``(risk, rule_name)``. The rule name is reported so a disagreement can
        be explained rather than merely counted.

    This grader is a *fallback*, not the source of truth. It exists so the
    unlabelled seed rows are not dropped, and it is deliberately conservative:
    anything it is not confident about lands on RISKY, because a grey-zone
    command that earns an extra warning is much cheaper than one that destroys
    something.
    """
    lowered = command.lower()

    # Download-into-interpreter is the highest-confidence destructive signal and
    # is checked before the generic verb lists, because `curl` by itself is a
    # routine downloader.
    if _PIPE_TO_SHELL_RE.search(lowered):
        return Risk.DESTRUCTIVE, "pipe_to_shell"

    for token in _DESTRUCTIVE_VERBS:
        if token in lowered:
            return Risk.DESTRUCTIVE, f"destructive_verb:{token.strip()}"

    # Recursive delete (rm -rf / -r / -R) with a target judgement. Checked
    # before generic risky verbs because the target inverts the verdict.
    if any(flag in lowered for flag in ("rm -rf", "rm -fr", "rm -r ", "rm -r", "rm -R")):
        if any(safe in lowered for safe in _SAFE_TARGETS):
            return Risk.SAFE, "delete_disposable_target"
        if _BLOCK_DEV_RE.search(lowered):
            return Risk.DESTRUCTIVE, "delete_block_device"
        if _WILDCARD_RE.search(lowered):
            return Risk.DESTRUCTIVE, "delete_wildcard"
        if _SYSTEM_PATH_RE.search(lowered):
            return Risk.DESTRUCTIVE, "delete_system_path"
        if "./" in lowered or lowered.rstrip().endswith(("/src", "/target", "src")):
            return Risk.RISKY, "delete_relative_path"
        return Risk.RISKY, "delete_unknown_target"

    # Plain `rm` (no -r) with a wildcard still deletes a whole class of files
    # and the author consistently treats it as destructive (`rm *.py`,
    # `rm -f *.md`). The flag check alone missed these.
    if re.search(r"(?:^|\s)rm(?:\s+-\w+)*\s+\*", lowered):
        return Risk.DESTRUCTIVE, "delete_wildcard_no_recursive"

    for token in _RISKY_VERBS:
        if token in lowered:
            return Risk.RISKY, f"risky_verb:{token.strip()}"

    if _SYSTEM_PATH_RE.search(lowered) and any(
        verb in lowered for verb in ("rm", "chmod", "chown", "dd", "mv", "cp")
    ):
        return Risk.RISKY, "touches_system_path"

    return Risk.SAFE, "default_safe"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

#: Seed sections imply a label even when the line itself was left unlabelled.
#: The unlabelled section in the seed is explicitly "grey zone", so it is not
#: overridden — those go to the grader.
_SECTION_DEFAULT: dict[str, Risk] = {
    SECTION_ROUTINE: Risk.SAFE,
    SECTION_LOOKS_SCARY: Risk.SAFE,
    SECTION_ACCIDENT: Risk.RISKY,
    SECTION_FEARED: Risk.RISKY,
}


def _seed_rows(corpus: Sequence[SeedEntry]) -> tuple[list[Row], list[Row], dict]:
    """Split the seed into labelled rows and freshly graded rows."""
    rows: list[Row] = []
    graded: list[Row] = []
    agreement = Counter()
    disagreements: list[dict] = []
    rules_used = Counter()

    for entry in corpus:
        if entry.label is not None:
            rows.append(
                Row(
                    command=entry.command,
                    label=entry.label,
                    category="seed",
                    source="seed",
                    section=entry.section,
                )
            )
            # Diagnostic only — never applied to the row itself.
            guessed, rule = grade(entry.command)
            if guessed is entry.label:
                agreement["agree"] += 1
            else:
                agreement["disagree"] += 1
                disagreements.append(
                    {
                        "command": entry.command,
                        "author": entry.label.name,
                        "grader": guessed.name,
                        "rule": rule,
                        "section": entry.section,
                    }
                )
            rules_used[rule] += 1
            continue

        label = _SECTION_DEFAULT.get(entry.section)
        if label is None:
            label, rule = grade(entry.command)
            rules_used[rule] += 1
        graded.append(
            Row(
                command=entry.command,
                label=label,
                category="seed",
                source="seed",
                section=entry.section,
                transform="grader",
            )
        )

    total = agreement["agree"] + agreement["disagree"]
    report = {
        "labelled_entries": total,
        "grader_agrees": agreement["agree"],
        "grader_disagrees": agreement["disagree"],
        "agreement_rate": round(agreement["agree"] / total, 4) if total else 0.0,
        "top_grader_rules": rules_used.most_common(12),
        "disagreement_samples": disagreements[:40],
    }
    return rows, graded, report


def build(*, verbose: bool = True) -> dict:
    """Assemble the full corpus. Pure function of the source tree."""
    corpus = load_seed(SEED_PATH)
    seed_rows, graded_rows, grader_report = _seed_rows(corpus)

    rows: list[Row] = []
    seen: set[tuple[str, Risk]] = set()

    def add(row: Row) -> None:
        """Deduplicate on (command, label), not on command alone.

        The same command with two different labels is not a duplicate to drop —
        it is a contradiction to resolve. Keeping both and letting the report
        count them is more honest than silently picking one.
        """
        key = (row.command, row.label)
        if key in seen:
            return
        seen.add(key)
        rows.append(row)

    for row in seed_rows:
        add(row)
    for row in graded_rows:
        add(row)
    for item in grammar.generate():
        add(
            Row(
                command=item.command,
                label=item.risk,
                category=item.category,
                source="grammar",
            )
        )
    for item in negatives.generate():
        add(
            Row(
                command=item.command,
                label=item.risk,
                category=item.category,
                source="negative",
            )
        )

    base_rows = list(rows)

    # Augmented variants. Only training families, applied to the base rows.
    for row in base_rows:
        for variant in augment.augment(row.command):
            add(
                Row(
                    command=variant,
                    label=row.label,
                    category=row.category,
                    source="augmented",
                    section=row.section,
                    transform="obfuscation",
                )
            )

    # Held-out set: same base commands, families the model never trains on.
    holdout: list[Row] = []
    holdout_seen: set[tuple[str, Risk, str]] = set()
    for row in base_rows:
        for family, variant in augment.holdout_variants(row.command):
            key = (variant, row.label, family)
            if key in holdout_seen:
                continue
            holdout_seen.add(key)
            holdout.append(
                Row(
                    command=variant,
                    label=row.label,
                    category=row.category,
                    source="holdout",
                    transform=family,
                )
            )

    stats = _statistics(rows, holdout, grader_report)
    stats["false_negative_cost"] = FALSE_NEGATIVE_COST

    if verbose:
        print(_render(stats))

    return {
        "rows": rows,
        "holdout": holdout,
        "stats": stats,
    }


def _statistics(rows: Sequence[Row], holdout: Sequence[Row], grader_report: dict) -> dict:
    by_label = Counter(row.label.name for row in rows)
    by_source = Counter(row.source for row in rows)
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_category[row.category][row.label.name] += 1
    by_section: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row.section:
            by_section[row.section][row.label.name] += 1

    holdout_by_family: dict[str, Counter[str]] = defaultdict(Counter)
    for row in holdout:
        holdout_by_family[row.transform][row.label.name] += 1

    commands = [row.command for row in rows]
    return {
        "total_rows": len(rows),
        "unique_commands": len(set(commands)),
        "holdout_rows": len(holdout),
        "by_label": dict(by_label),
        "by_source": dict(by_source),
        "by_category": {k: dict(v) for k, v in sorted(by_category.items())},
        "by_section": {k: dict(v) for k, v in sorted(by_section.items())},
        "holdout_by_family": {k: dict(v) for k, v in sorted(holdout_by_family.items())},
        "grader": grader_report,
    }


def _render(stats: dict) -> str:
    lines: list[str] = []
    add = lines.append
    add("=" * 74)
    add("DATASET")
    add("=" * 74)
    add(f"toplam satır          : {stats['total_rows']}")
    add(f"benzersiz komut       : {stats['unique_commands']}")
    add(f"holdout satır         : {stats['holdout_rows']}")

    add("")
    add("SINIF DAĞILIMI")
    total = stats["total_rows"]
    for label, count in sorted(stats["by_label"].items()):
        share = count / total * 100
        bar = "#" * int(share / 2)
        add(f"  {label:<14} {count:>6}  {share:>5.1f}%  {bar}")

    add("")
    add("KAYNAK")
    for source, count in sorted(stats["by_source"].items(), key=lambda kv: -kv[1]):
        add(f"  {source:<14} {count:>6}")

    add("")
    add("KATEGORİ (en çok satır)")
    for category, counts in sorted(
        stats["by_category"].items(), key=lambda kv: -sum(kv[1].values())
    )[:22]:
        detail = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        add(f"  {category:<34} {sum(counts.values()):>6}  {detail}")

    add("")
    add("SEED BÖLÜMLERİ")
    for section, counts in stats["by_section"].items():
        detail = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        add(f"  {section:<16} {sum(counts.values()):>6}  {detail}")

    add("")
    add("HOLDOUT AİLELERİ (eğitimde hiç görülmedi)")
    for family, counts in stats["holdout_by_family"].items():
        detail = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        add(f"  {family:<16} {sum(counts.values()):>6}  {detail}")

    grader = stats["grader"]
    add("")
    add("=" * 74)
    add("GRADER ANLAŞMASI (yalnızca tanısal — etiketlere dokunmaz)")
    add("=" * 74)
    add(
        f"  etiketli {grader['labelled_entries']} satırın "
        f"{grader['grader_agrees']} tanesinde kural-grader senin etiketinle aynı "
        f"({grader['agreement_rate'] * 100:.1f}%)"
    )
    add(f"  anlaşmazlık: {grader['grader_disagrees']}")
    add("")
    add("  en çok tetiklenen kurallar:")
    for rule, count in grader["top_grader_rules"][:10]:
        add(f"    {rule:<34} {count}")
    add("")
    add("  örnek anlaşmazlıklar:")
    for item in grader["disagreement_samples"][:20]:
        add(f"    sen={item['author']:<13} grader={item['grader']:<13} {item['command'][:58]}")
        add(f"      kural: {item['rule']}  (bölüm: {item['section']})")

    add("")
    add("=" * 74)
    return "\n".join(lines)


def _write(rows: Iterable[Row], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["command", "label", "category", "source", "section", "transform"])
        for row in rows:
            writer.writerow(
                [
                    row.command,
                    row.label.name,
                    row.category,
                    row.source,
                    row.section,
                    row.transform,
                ]
            )
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the training corpus.")
    parser.add_argument(
        "--quiet", action="store_true", help="suppress the human-readable report"
    )
    args = parser.parse_args()

    result = build(verbose=not args.quiet)
    rows = result["rows"]
    holdout = result["holdout"]

    train_path = ARTIFACTS / "dataset.csv"
    holdout_path = ARTIFACTS / "holdout.csv"
    stats_path = REPO_ROOT / "ml" / "artifacts" / "dataset_stats.json"

    _write(rows, train_path)
    _write(holdout, holdout_path)

    stats = dict(result["stats"])
    stats["generated_from"] = {
        "seed_entries": len(load_seed(SEED_PATH)),
        "grammar_rules": len(grammar.RULES),
    }
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print()
    print(f"yazıldı: {train_path.relative_to(REPO_ROOT)} ({len(rows)} satır)")
    print(f"yazıldı: {holdout_path.relative_to(REPO_ROOT)} ({len(holdout)} satır)")
    print(f"yazıldı: {stats_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())