"""Label schema and the risk taxonomy behind it.

Three ordinal classes, not a binary flag
---------------------------------------
The feature this model serves is deliberately simple: the user types a command,
the model decides whether it is harmful, and a warning appears that the user can
swipe away. That decision is binary.

The *training* labels are three-class ordinal anyway, for two reasons.

1. Threshold headroom without retraining. ``RISKY`` is a genuinely different
   region of input space from ``SAFE``: it is where ``git clean -fdx`` and
   ``docker system prune`` and ``rm -rf node_modules`` live — commands that
   destroy a lot but are routine. Training on three classes forces the model to
   build a decision surface between "routine but heavy" and "unrecoverable",
   which is exactly the surface the exported thresholds slide along. Moving the
   warn/block line later is then an export-time constant, not a retrain.
2. Honest ground truth for evaluation. A binary model reports one number; a
   three-class model lets the evaluation report *which* boundary is failing. If
   recall on ``DESTRUCTIVE`` is 0.97, we need to know whether the 3% are
   mislabelled positives or genuinely ambiguous commands.

Cost asymmetry
--------------
Confusing a routine cleanup for a destructive command is a false positive. The
user sees a warning, judges it noise, and swipes it away — and then stops
reading warnings, which is how the feature kills itself. Missing a genuinely
destructive command is a false negative, and that is a data-loss event.

So the thresholds are tuned with an explicit cost ratio
(:data:`FALSE_NEGATIVE_COST`), never at 0.5. A false negative is treated as
roughly twenty times more expensive than a false positive. That ratio is a
product decision, not a statistical one, and it is recorded here so it can be
argued with rather than rediscovered.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

__all__ = [
    "Risk",
    "RISK_NAMES",
    "RISK_DESCRIPTIONS",
    "Category",
    "FALSE_NEGATIVE_COST",
    "label_for_category",
]


class Risk(IntEnum):
    """Ordinal risk classes. The numeric values are the regression targets."""

    SAFE = 0
    RISKY = 1
    DESTRUCTIVE = 2


RISK_NAMES: dict[Risk, str] = {
    Risk.SAFE: "safe",
    Risk.RISKY: "risky",
    Risk.DESTRUCTIVE: "destructive",
}

#: Turkish labels, used by the browser prototype. Kept beside the English names
#: so a new class cannot be added without deciding how it reads in the UI.
RISK_DESCRIPTIONS: dict[Risk, str] = {
    Risk.SAFE: "Güvenli",
    Risk.RISKY: "Şüpheli",
    Risk.DESTRUCTIVE: "Tehlikeli",
}

#: Used when picking decision thresholds. See module docstring for why this is
#: not 1.0. A false positive costs a swipe; a false negative can cost the user
#: their filesystem. Twenty is a starting point to be tuned against real
#: telemetry, not a measured value.
FALSE_NEGATIVE_COST: float = 20.0


@dataclass(frozen=True, slots=True)
class Category:
    """A family of commands that share a reason for being risky.

    Categories exist for two reasons beyond documentation. The generator in
    ``grammar.py`` expands each one combinatorially, so the label is correct by
    construction rather than predicted — an LLM guessing at 40k labels
    introduces label noise that is indistinguishable, in the loss curve, from a
    hard learning problem. And evaluation reports per-category recall, which
    turns "the model is at 94%" into "the model misses fork bombs and raw disk
    writes, and is fine on everything else" — an actionable statement.

    Attributes:
        name:        stable identifier, used as the evaluation grouping key.
        risk:        the class every generated member of this category gets.
        binaries:    the programs a user would actually type.
        rationale:   why this is dangerous, in the user's terms. Surfaced
                     verbatim in the prototype's explanation panel, which is why
                     it is written as a sentence rather than a note.
    """

    name: str
    risk: Risk
    binaries: tuple[str, ...]
    rationale: str


#: The taxonomy. Deliberately incomplete — it is a starting set that the
#: evaluation loop is expected to extend whenever a confusion-matrix cell turns
#: out to be underserved. Every category added here needs a matching set of hard
#: negatives in ``negatives.py``, or the model will learn the category's
#: vocabulary as a proxy for "risky" rather than learning the risk.
CATEGORIES: tuple[Category, ...] = (
    # --- unrecoverable data loss ---
    Category(
        name="recursive_delete_system",
        risk=Risk.DESTRUCTIVE,
        binaries=("rm",),
        rationale="Sistem dizinlerini veya ev dizinini kalıcı olarak siler.",
    ),
    Category(
        name="recursive_delete_wildcard",
        risk=Risk.DESTRUCTIVE,
        binaries=("rm",),
        rationale="Tüm dosyaları geri dönüşsüz olarak siler.",
    ),
    Category(
        name="raw_disk_write",
        risk=Risk.DESTRUCTIVE,
        binaries=("dd",),
        rationale="Doğrudan disk bloğu üzerine yazar; dosya sistemi ve bölüm tablosu bozulur.",
    ),
    Category(
        name="filesystem_format",
        risk=Risk.DESTRUCTIVE,
        binaries=("mkfs", "mke2fs", "mkswap", "fdisk", "parted", "sgdisk", "wipefs"),
        rationale="Dosya sistemini biçimlendirir veya bölüm tablosunu siler.",
    ),
    Category(
        name="fork_bomb",
        risk=Risk.DESTRUCTIVE,
        binaries=(":",),
        rationale="Sonsuz süreç çatallaması yaparak cihazı kilitler ve belleği tüketir.",
    ),
    # --- privilege and permission damage ---
    Category(
        name="permission_world_writable",
        risk=Risk.RISKY,
        binaries=("chmod",),
        rationale="Dosya izinlerini herkese açık hale getirir; güvenlik açığıdır.",
    ),
    Category(
        name="recursive_ownership_change",
        risk=Risk.RISKY,
        binaries=("chown",),
        rationale="Sahipliği kök olarak değiştirir; sonraki erişimleri bozabilir.",
    ),
    Category(
        name="sudo_privileged_destructive",
        risk=Risk.DESTRUCTIVE,
        binaries=("sudo", "doas", "su"),
        rationale="Yıkıcı işlemi kök yetkisiyle çalıştırır, geri alma şansı kalmaz.",
    ),
    # --- anti-forensics ---
    Category(
        name="history_tampering",
        risk=Risk.RISKY,
        binaries=("history", "shred", "srm"),
        rationale="Komut geçmişini veya kanıt dosyalarını yok eder.",
    ),
    Category(
        name="log_tampering",
        risk=Risk.RISKY,
        binaries=("journalctl", "truncate", "shred"),
        rationale="Sistem günlüklerini siler veya boşaltır.",
    ),
    # --- remote code execution ---
    Category(
        name="curl_pipe_shell",
        risk=Risk.DESTRUCTIVE,
        binaries=("curl", "wget"),
        rationale="İnternetten indirilen kodu doğrudan çalıştırır; kaynağı denetlenemez.",
    ),
    Category(
        name="encoded_payload_execution",
        risk=Risk.DESTRUCTIVE,
        binaries=("base64", "eval", "python", "perl", "node"),
        rationale="Kodlanmış içeriği çözüp çalıştırır; ne çalıştırdığı önceden okunamaz.",
    ),
    # --- remote access / exfiltration ---
    Category(
        name="ssh_key_injection",
        risk=Risk.DESTRUCTIVE,
        binaries=("ssh", "cat"),
        rationale="SSH yetkili anahtarını değiştirerek cihaza kalıcı erişim açar.",
    ),
    Category(
        name="reverse_shell",
        risk=Risk.DESTRUCTIVE,
        binaries=("nc", "ncat", "socat", "bash"),
        rationale="Cihazdan dışarıya etkileşimli kabuk açar; uzaktan tam kontrol sağlar.",
    ),
    # --- version control destruction ---
    Category(
        name="git_history_destructive",
        risk=Risk.RISKY,
        binaries=("git",),
        rationale="Commit geçmişini veya çalışma ağacını geri dönüşsüz olarak siler.",
    ),
    # --- package / dependency destruction ---
    Category(
        name="bulk_package_removal",
        risk=Risk.RISKY,
        binaries=("apt", "apt-get", "yum", "dnf", "pip", "npm"),
        rationale="Toplu olarak paket kaldırır veya bağımlılık zincirini bozar.",
    ),
    # --- container destruction ---
    Category(
        name="container_mass_prune",
        risk=Risk.RISKY,
        binaries=("docker", "podman", "kubectl"),
        rationale="İmaj, konteyner ve hacimleri toplu olarak siler.",
    ),
    # --- availability ---
    Category(
        name="shutdown_reboot",
        risk=Risk.RISKY,
        binaries=("shutdown", "reboot", "halt", "poweroff", "init"),
        rationale="Çalışan sistemi kapatır; bekleyen işi kaybettirir.",
    ),
    Category(
        name="kill_everything",
        risk=Risk.RISKY,
        binaries=("kill", "pkill", "killall"),
        rationale="Çok sayıda süreci toplu olarak sonlandırır.",
    ),
    Category(
        name="memory_exhaustion",
        risk=Risk.RISKY,
        binaries=("yes", "cat", "dd", "fallocate"),
        rationale="Belleği veya disk alanını tüketerek cihazı kullanılamaz hale getirir.",
    ),
)


def label_for_category(category_name: str) -> Risk:
    """Look up the risk class for a category name.

    Raises:
        KeyError: if the category is unknown. A typo in a generator must fail
            loudly at dataset-build time, not silently produce a bucket of
            mislabelled rows that only surfaces as an unexplained recall hole
            three days later.
    """
    for category in CATEGORIES:
        if category.name == category_name:
            return category.risk
    raise KeyError(f"unknown risk category: {category_name!r}")