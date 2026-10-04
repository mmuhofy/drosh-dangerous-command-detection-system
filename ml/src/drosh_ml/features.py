"""Dense hand-crafted features, family B.

Why a second feature family at all
----------------------------------
The n-gram vectoriser can learn that ``rm`` near ``-rf`` near ``/`` is
dangerous, but it struggles with things that are dangerous *regardless of the
exact words*:

  * ``dd`` with any source writing to any block device — the n-grams are the
    same whether it says ``bs=1M`` or ``bs=4M``.
  * how many segments a command has (``a && b && c && ...`` compounds risk).
  * whether a recursive delete's target is a system path or a build artifact.
  * nesting depth of ``$(`` — a proxy for how much indirection is happening.

These are ~30 numbers, computed identically in Python and JavaScript, and they
give the model an explicit handle on exactly the distinctions the corpus is
built around. They also make the prototype's explanation panel meaningful: the
top contributors are then a mixture of recognisable n-grams and named signals
like ``delete_target_is_system_path``.

Every feature is a pure function of a :class:`Normalized`. Every value is a
small non-negative integer or 0/1 flag, so the exported coefficients need no
explanation beyond the name. The exported standardisation (mean/std) is
computed here and frozen into the model, so inference is identical.

The dense vector is appended to the sparse n-gram vector inside one
:class:`~sklearn.feature_extraction.FeatureUnion`-style matrix; the exported
model stores both the sparse vocabulary and the dense scaler.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .normalize import Normalized

__all__ = ["DENSE_FEATURE_NAMES", "dense_features", "DENSE_DIM"]

# Ordered: the export writes this order into the model, so appending a new
# feature at the end is safe but inserting one is not.
DENSE_FEATURE_NAMES: tuple[str, ...] = (
    "segment_count",
    "has_sudo",
    "has_su",
    "has_doas",
    "rm_recursive_force",
    "rm_any_recursive",
    "delete_target_system_path",
    "delete_target_home",
    "delete_target_wildcard",
    "delete_target_relative",
    "delete_target_disposable",
    "has_dd",
    "dd_to_block_device",
    "has_mkfs",
    "has_fs_table_tool",
    "has_fork_bomb",
    "has_chmod_777",
    "chmod_recursive",
    "has_chown_recursive_root",
    "has_curl_or_wget",
    "curl_pipe_to_shell",
    "has_base64_decode",
    "has_eval",
    "has_history_tamper",
    "touches_ssh_keys",
    "has_shutdown",
    "has_kill_broad",
    "has_git_destructive",
    "has_docker_prune",
    "has_kubectl_delete",
    "has_package_remove",
    "has_truncate_or_log_empty",
    "subshell_depth",
    "backtick_count",
    "quote_nesting",
    "flag_density",
    "path_depth",
    "mentions_only",
)

DENSE_DIM = len(DENSE_FEATURE_NAMES)
_INDEX = {name: i for i, name in enumerate(DENSE_FEATURE_NAMES)}

_SYSTEM_PATH_RE = re.compile(
    r"(?:^|[\s;&|(])/(?:etc|usr|var|bin|sbin|boot|lib|opt|srv|root|home|sys|proc|dev|sdcard"
    r"|storage|data)(?:/|\b)"
)
_HOME_PATH_RE = re.compile(r"(?:^|[\s;&|(])(~|\$\{?HOME\}?|\$\{?PWD\}?|\$\(pwd\))(?:/|\b|$)")
_WILDCARD_RE = re.compile(r"(?:^|[\s;&|(])\*(?:\.\*)?\*?(?:/|$)")
_RELATIVE_RE = re.compile(r"(?:^|[\s;&|(])\./|\s\./|\./")
_DISPOSABLE_RE = re.compile(
    r"node_modules|\bbuild/?\b|\bdist/?\b|\btarget/?\b|\.gradle|\.venv|\bvenv/?\b"
    r"|__pycache__|\.pytest_cache|\.mypy_cache|\.next|\.nuxt|\.cache|vendor/?\b"
    r"|pods/?\b|deriveddata|\.terraform|\bcoverage\b|\.tox|\.parcel-cache|\.turbo"
    r"|\*\.o\b|\*\.class\b|\*\.pyc\b|\.ds_store|\*\.log\b|\*\.tmp\b|\*\.bak\b"
)
_BLOCK_DEV_RE = re.compile(r"/dev/(?:sd[a-z]|nvme\d|mmcblk|hd[a-z])")
_FS_TABLE_RE = re.compile(r"\b(?:fdisk|parted|sgdisk|gparted|blkdiscard|wipefs)\b")


@dataclass(frozen=True, slots=True)
class DenseSpec:
    """Standardisation constants for the dense block.

    Attributes:
        mean: per-feature training mean.
        std: per-feature training std, floored away from zero so a constant
            feature cannot produce a division by zero at inference.
    """

    mean: np.ndarray
    std: np.ndarray


def _first_token(segment: str) -> str:
    """Leading word of a segment, ignoring leading env assignments."""
    tokens = segment.split()
    index = 0
    while index < len(tokens) and "=" in tokens[index] and not tokens[index].startswith("-"):
        # Skip VAR=value prefixes so `LC_ALL=C rm -rf /` sees `rm`.
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[index]):
            index += 1
        else:
            break
    if index < len(tokens):
        return tokens[index].strip("\"'")
    return ""


def _binary(text: str) -> bool:
    return 1.0 if text else 0.0


def dense_features(norm: Normalized) -> np.ndarray:
    """Compute the raw (un-standardised) dense vector.

    Every entry is a non-negative count or a 0/1 flag, so the exported model can
    be audited by reading the names in order.
    """
    text = norm.text
    joined = " ".join(norm.segments)
    has_rm_rf = bool(re.search(r"\brm\b[^;|&]*\s-{1,2}[a-z]*r[a-z]*f\b|\brm\s+-[a-z]*f[a-z]*r\b", text))
    rm_recursive = bool(re.search(r"\brm\b[^;|&]*\s-{1,2}(?:r\b|recursive)", text))

    # Which segment does the delete target? Take the segment that contains rm.
    rm_segment = next((s for s in norm.segments if re.search(r"\brm\b", s)), joined)
    target_is_system = _SYSTEM_PATH_RE.search(rm_segment) is not None
    target_is_home = _HOME_PATH_RE.search(rm_segment) is not None
    target_is_wild = _WILDCARD_RE.search(rm_segment) is not None
    target_is_rel = _RELATIVE_RE.search(rm_segment) is not None
    target_is_disposable = _DISPOSABLE_RE.search(rm_segment) is not None

    has_dd = bool(re.search(r"\bdd\b", text))
    has_mkfs = bool(re.search(r"\bmkfs\b|\bmke2fs\b|\bmkswap\b", text))

    curl_pipe = bool(
        re.search(r"(?:curl|wget)[^;|&]*\|\s*(?:sudo\s+)?(?:ba|z|k|da)?sh\b", text)
        or re.search(r"\|\s*(?:sudo\s+)?(?:python3?|perl|ruby|node|php)\b", text)
    )

    # How much indirection: $( ), backticks, quotes. A proxy for "how much is
    # this command hiding".
    subshell_depth = text.count("$(") + text.count("${")
    backticks = text.count("`")
    quotes = text.count('"') + text.count("'")

    flags = len(re.findall(r"(?:^|\s)-{1,2}[a-zA-Z][\w-]*", text))
    words = max(1, len(text.split()))
    # Flag density: a high ratio means flag-driven (often destructive) rather
    # than path- or text-driven.
    flag_density = flags / words

    # Path depth of the deepest slash-run, capped.
    paths = re.findall(r"(?:/|~|\$\{?HOME\}?)/[\w./-]*", text)
    path_depth = max((p.count("/") for p in paths), default=0)

    # A command that only *talks about* dangerous things: contains a dangerous
    # keyword but no dangerous verb. e.g. echo "rm -rf /", grep 'mkfs'.
    has_danger_word = bool(
        re.search(r"\brm\b|\bdd\b|mkfs|chmod 777|fork bomb|:\{", text)
    )
    has_danger_verb = bool(
        re.search(r"\brm\s+-|^\s*dd\b|\bmkfs\b|chmod|chown|shutdown|reboot|kill", text)
    )
    mentions_only = _binary(has_danger_word and not has_danger_verb)

    values = {
        "segment_count": float(len(norm.segments)),
        "has_sudo": _binary(re.search(r"(?:^|\s)sudo\b", text)),
        "has_su": _binary(re.search(r"(?:^|\s)su\s+-|\bsu\s+c\b", text)),
        "has_doas": _binary(re.search(r"(?:^|\s)doas\b", text)),
        "rm_recursive_force": _binary(has_rm_rf),
        "rm_any_recursive": _binary(rm_recursive),
        "delete_target_system_path": _binary(target_is_system),
        "delete_target_home": _binary(target_is_home),
        "delete_target_wildcard": _binary(target_is_wild),
        "delete_target_relative": _binary(target_is_rel),
        "delete_target_disposable": _binary(target_is_disposable),
        "has_dd": _binary(has_dd),
        "dd_to_block_device": _binary(has_dd and _BLOCK_DEV_RE.search(text)),
        "has_mkfs": _binary(has_mkfs),
        "has_fs_table_tool": _binary(_FS_TABLE_RE.search(text)),
        "has_fork_bomb": _binary(re.search(r":\s*\(\s*\)\s*\{", text)),
        "has_chmod_777": _binary(re.search(r"chmod[^;|&]*777", text)),
        "chmod_recursive": _binary(re.search(r"chmod\s+-R\b|chmod\s+-r\b", text)),
        "has_chown_recursive_root": _binary(
            re.search(r"chown\s+-R\b[^;|&]*(?:root|0:0)", text)
        ),
        "has_curl_or_wget": _binary(re.search(r"\b(?:curl|wget|fetch)\b", text)),
        "curl_pipe_to_shell": _binary(curl_pipe),
        "has_base64_decode": _binary(re.search(r"base64\s+(?:-d|--decode|-D)", text)),
        "has_eval": _binary(re.search(r"(?:^|\s)eval\b", text)),
        "has_history_tamper": _binary(
            re.search(r"history\s+-c|unset\s+HIST|>\s*~?/?\.bash_history", text)
        ),
        "touches_ssh_keys": _binary(re.search(r"\.ssh/|authorized_keys|id_rsa", text)),
        "has_shutdown": _binary(re.search(r"\b(?:shutdown|reboot|poweroff|halt|init\s+[06])\b", text)),
        "has_kill_broad": _binary(re.search(r"kill\s+-9\s+-1|killall|pkill", text)),
        "has_git_destructive": _binary(
            re.search(r"git\s+(reset\s+--hard|clean\s+-f|push\s+-[a-z]*f|--force)", text)
        ),
        "has_docker_prune": _binary(re.search(r"docker\s+(system|image|volume|container)\s+prune", text)),
        "has_kubectl_delete": _binary(re.search(r"kubectl\s+delete", text)),
        "has_package_remove": _binary(
            re.search(r"(?:apt|apt-get|yum|dnf)\s+(remove|purge|autoremove)|(?:pip|npm|gem)\s+uninstall", text)
        ),
        "has_truncate_or_log_empty": _binary(
            re.search(r"truncate\s+-s|>\s*[\w./-]*\.log", text)
        ),
        "subshell_depth": float(subshell_depth),
        "backtick_count": float(backticks),
        "quote_nesting": float(quotes // 2),
        "flag_density": flag_density,
        "path_depth": float(min(path_depth, 8)),
        "mentions_only": mentions_only,
    }

    assert set(values) == set(DENSE_FEATURE_NAMES), "dense feature name drift"
    return np.array([values[name] for name in DENSE_FEATURE_NAMES], dtype=np.float64)


def fit_standardiser(matrix: np.ndarray) -> DenseSpec:
    """Freeze mean/std from the training matrix.

    std is floored at 1e-6 so a feature that is constant across the corpus
    cannot blow up at inference.
    """
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.maximum(std, 1e-6)
    return DenseSpec(mean=mean, std=std)


def standardise(matrix: np.ndarray, spec: DenseSpec) -> np.ndarray:
    """Apply the frozen standardisation."""
    return (matrix - spec.mean) / spec.std