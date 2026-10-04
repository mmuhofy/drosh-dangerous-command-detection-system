"""Reader for ``ml/data/seed_from_muhofy.txt``.

The seed is the only part of the dataset that comes from a real terminal rather
than from a generator, so it is treated as ground truth: when the synthetic
grammar and the seed disagree, the seed wins and the disagreement is reported
(see ``build_dataset.py --report``).

Format, as documented in the file's own header::

    # comments, and '# === SECTION ===' markers
    ls -la  |||  safe          <- explicit label
    ./gradlew assembleRelease   <- unlabelled: the grader's job

Sections carry meaning that the label alone does not. "Looks scary but is
completely harmless" is the single most valuable section in the file, because
those commands contain destructive vocabulary and must never warn; a model that
fails there is a model nobody keeps enabled.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .labels import Risk

__all__ = [
    "SeedEntry",
    "SeedCorpus",
    "Section",
    "load_seed",
    "SECTION_ROUTINE",
    "SECTION_ACCIDENT",
    "SECTION_FEARED",
    "SECTION_LOOKS_SCARY",
    "SECTION_UNLABELLED",
]

SECTION_ROUTINE = "routine"
SECTION_ACCIDENT = "accident"
SECTION_FEARED = "feared"
SECTION_LOOKS_SCARY = "looks_scary"
SECTION_UNLABELLED = "unlabelled"

_SECTION_BY_KEYWORD: tuple[tuple[str, str], ...] = (
    ("SIK SIK YAZDIKLARIM", SECTION_ROUTINE),
    ("YANLISLIKLA", SECTION_ACCIDENT),
    ("KORKTUĞUM", SECTION_FEARED),
    ("TEHLİKELİ GÖRÜNÜP MASUM", SECTION_LOOKS_SCARY),
    ("ETİKETSİZ", SECTION_UNLABELLED),
)

_SECTION_MARKER_RE = re.compile(r"^#\s*===\s*(.+?)\s*===$")
_LABEL_SPLIT = "|||"
_LABEL_NAMES: dict[str, Risk] = {
    "safe": Risk.SAFE,
    "risky": Risk.RISKY,
    "destructive": Risk.DESTRUCTIVE,
}


class SeedFormatError(ValueError):
    """Raised when the seed file cannot be parsed at all."""


@dataclass(frozen=True, slots=True)
class SeedEntry:
    """One command from the seed file.

    Attributes:
        command: exactly as written, whitespace trimmed but otherwise verbatim.
            Normalisation happens later and in one place, so that nothing can
            accidentally score a pre-normalised string.
        label: the author's label, or None if the line was left unlabelled.
        section: which heading the line sat under.
        line: 1-indexed line number, for error messages.
    """

    command: str
    label: Risk | None
    section: str
    line: int


@dataclass(frozen=True, slots=True)
class SeedCorpus:
    entries: tuple[SeedEntry, ...]
    unknown_labels: tuple[tuple[int, str], ...]

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[SeedEntry]:
        return iter(self.entries)

    def labelled(self) -> Iterator[SeedEntry]:
        return (e for e in self.entries if e.label is not None)

    def unlabelled(self) -> Iterator[SeedEntry]:
        return (e for e in self.entries if e.label is None)

    def section_counts(self) -> dict[str, Counter[str]]:
        counts: dict[str, Counter[str]] = {}
        for entry in self.entries:
            bucket = counts.setdefault(entry.section, Counter())
            bucket[entry.label.name if entry.label else "(unlabelled)"] += 1
        return counts


def _resolve_section(marker: str) -> str:
    upper = marker.upper()
    for keyword, name in _SECTION_BY_KEYWORD:
        if keyword in upper:
            return name
    # An unrecognised heading is not fatal: the data is still usable, it just
    # loses its provenance. Prefix it so it cannot collide with a known name.
    return f"other:{marker.strip()}"


def load_seed(path: str | Path) -> SeedCorpus:
    """Parse the seed file.

    Raises:
        SeedFormatError: if the path does not exist or contains no commands.
            An empty seed is never acceptable — a silently empty corpus would
            look exactly like "the model trained fine" in the pipeline output.
    """
    seed_path = Path(path)
    if not seed_path.exists():
        raise SeedFormatError(f"seed file not found: {seed_path}")

    entries: list[SeedEntry] = []
    unknown: list[tuple[int, str]] = []
    section = SECTION_UNLABELLED

    for lineno, raw in enumerate(seed_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue

        if line.startswith("#"):
            marker = _SECTION_MARKER_RE.match(line)
            if marker:
                section = _resolve_section(marker.group(1))
            continue

        label: Risk | None = None
        command = line
        if _LABEL_SPLIT in line:
            command, _, label_text = line.partition(_LABEL_SPLIT)
            command, label_text = command.strip(), label_text.strip().lower()
            if label_text not in _LABEL_NAMES:
                unknown.append((lineno, label_text))
                # Keep the command; a bad label is a reporting problem, not a
                # reason to throw away a real command.
                label = None
            else:
                label = _LABEL_NAMES[label_text]

        if command:
            entries.append(
                SeedEntry(command=command, label=label, section=section, line=lineno)
            )

    if not entries:
        raise SeedFormatError(f"no commands found in {seed_path}")

    return SeedCorpus(entries=tuple(entries), unknown_labels=tuple(unknown))