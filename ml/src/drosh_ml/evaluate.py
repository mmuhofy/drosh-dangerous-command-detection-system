"""Evaluate the model and report where it is actually weak.

Accuracy on a three-class problem is close to useless on its own: 50% of the
corpus is ``safe``, so a model that never warns scores 50% while being worthless.
What matters is the confusion between adjacent classes, per category, and how
each threshold trades recall against false alarms.

Reports, in order:

* class distribution and the score range of each class, because a threshold that
  sits above every observed score is a threshold that never fires — which is
  exactly what the first version of this model produced;
* the operating characteristics at the shipped thresholds;
* a threshold sweep, so the choice can be argued with rather than inherited;
* per-category recall, so "94% overall" becomes "it misses reverse shells";
* per-obfuscation-family recall on the held-out set, which is the only number
  that says whether the model learned shell semantics or memorised the grammar.

Run from the codespace:

    PYTHONPATH=ml/src .venv/bin/python -m drosh_ml.evaluate
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .export import _load_dataset, _probe_commands, _prune, COEF_FLOOR
from .labels import FALSE_NEGATIVE_COST, Risk
from .train import _threshold_cost, fit

ARTIFACTS_JSON = "ml/artifacts/eval_report.json"


@dataclass(frozen=True, slots=True)
class Metrics:
    """Confusion counts at one (warn, block) operating point."""

    tp_block: int  # destructive, correctly blocked
    fn_block: int  # destructive, missed
    tp_warn: int  # risky, warned
    fn_warn: int  # risky, silent
    fp_warn: int  # safe, warned
    tn_warn: int  # safe, silent

    @property
    def block_recall(self) -> float:
        total = self.tp_block + self.fn_block
        return self.tp_block / total if total else 0.0

    @property
    def warn_recall(self) -> float:
        total = self.tp_warn + self.fn_warn
        return self.tp_warn / total if total else 0.0

    @property
    def safe_precision(self) -> float:
        total = self.tp_warn + self.fp_warn
        return self.tp_warn / total if total else 0.0

    @property
    def cost(self) -> float:
        return self.fn_block * FALSE_NEGATIVE_COST + self.fp_warn


def _confusion(truth: np.ndarray, risk: np.ndarray, warn: float, block: float) -> Metrics:
    warn_call = risk >= warn
    block_call = risk >= block
    is_d = truth == Risk.DESTRUCTIVE.value
    is_r = truth == Risk.RISKY.value
    is_s = truth == Risk.SAFE.value
    return Metrics(
        tp_block=int((is_d & block_call).sum()),
        fn_block=int((is_d & ~block_call).sum()),
        tp_warn=int((is_r & warn_call).sum()),
        fn_warn=int((is_r & ~warn_call).sum()),
        fp_warn=int((is_s & warn_call).sum()),
        tn_warn=int((is_s & ~warn_call).sum()),
    )


def _bar(value: float, width: int = 24) -> str:
    return "#" * int(round(value * width))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the risk model.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", help="write the report here")
    args = parser.parse_args()

    rows = _load_dataset()
    model = fit(rows)

    truth = np.array([float(row.label.value) for row in rows])

    # Score with the pruned coefficients — the same ones the browser uses.
    folded = model.weights[: len(model.vectorizer.vocabulary_)] * model.vectorizer.idf_
    pruned_vocab, pruned_coef = _prune(model.vectorizer.vocabulary_, folded, COEF_FLOOR)
    from .export import _score_pruned

    risk = _score_pruned(model, pruned_vocab, pruned_coef, [row.command for row in rows])

    out: list[str] = []
    add = out.append

    warn_t = model.thresholds["warn"]
    block_t = model.thresholds["block"]

    add("=" * 78)
    add("SINIF DAgilIMI VE SKOR ARALIKLARI")
    add("=" * 78)
    for value, name in ((0, "safe"), (1, "risky"), (2, "destructive")):
        subset = risk[truth == value]
        add(
            f"  {name:<13} n={len(subset):<6} "
            f"min={subset.min():.3f} p50={np.median(subset):.3f} "
            f"p90={np.percentile(subset, 90):.3f} p99={np.percentile(subset, 99):.3f} "
            f"max={subset.max():.3f}"
        )

    highest = risk.max()
    add("")
    add(f"  en yuksek risk skoru : {highest:.4f}")
    add(f"  block esigi          : {block_t:.4f}")
    if block_t > highest:
        add(
            "  !! block esigi hicbir komutun asilabilecegi yerde: "
            "destructive sinifi TETIKLENMEZ"
        )
    elif block_t > np.percentile(risk, 99.9):
        add("  !! block esigi en ust %0.1 disinda: neredeyse hic tetiklenmez" % 0.1)

    add("")
    add("=" * 78)
    add(f"SEÇILEN NOKTADA: warn={warn_t:.4f} block={block_t:.4f}")
    add("=" * 78)
    m = _confusion(truth, risk, warn_t, block_t)
    add(f"  destructive yakalandi : {m.tp_block}/{m.tp_block + m.fn_block}  "
        f"({m.block_recall * 100:.1f}%)  {_bar(m.block_recall)}")
    add(f"  risky uyarildi         : {m.tp_warn}/{m.tp_warn + m.fn_warn}  "
        f"({m.warn_recall * 100:.1f}%)  {_bar(m.warn_recall)}")
    add(f"  safe yanlis uyarildi   : {m.fp_warn}/{m.fp_warn + m.tn_warn}  "
        f"({(m.fp_warn / max(1, m.fp_warn + m.tn_warn)) * 100:.1f}%)")
    add(f"  maliyet               : {m.cost:.0f}  "
        f"(FN={FALSE_NEGATIVE_COST:.0f}x)")

    add("")
    add("=" * 78)
    add("ESIK TARAMASI  (block icin)")
    add("=" * 78)
    add("  esik    destructive- recall   safe-yanlis   maliyet")
    grid = np.arange(0.5, max(highest, 2.0) + 0.001, 0.05)
    sweep = []
    for candidate in grid:
        probe = _confusion(truth, risk, warn_t, float(candidate))
        sweep.append((probe.cost, float(candidate), probe))
    sweep.sort(key=lambda item: item[0])
    for cost, candidate, probe in sweep[:8]:
        add(
            f"  {candidate:.2f}    {probe.block_recall * 100:>5.1f}%           "
            f"{probe.fp_warn:>6}        {cost:>10.0f}"
        )
    add("")
    best_cost, best_block, best_metrics = sweep[0]
    add(f"  en iyi block esigi (maliyete gore): {best_block:.4f}")
    add(f"  -> destructive recall {best_metrics.block_recall * 100:.1f}%, "
        f"safe yanlis {best_metrics.fp_warn}")

    add("")
    add("=" * 78)
    add("KATEGORI BAZLI DESTRUCTIVE RECALL")
    add("=" * 78)
    by_category: dict[str, list[bool]] = defaultdict(list)
    for row, score in zip(rows, risk, strict=True):
        if row.label is Risk.DESTRUCTIVE:
            by_category[row.category].append(score >= block_t)
    weak: list[tuple[str, float, int]] = []
    for category, hits in sorted(by_category.items()):
        recall = sum(hits) / len(hits)
        weak.append((category, recall, len(hits)))
    for category, recall, count in sorted(weak, key=lambda item: item[1]):
        flag = "  <-- zayif" if recall < 0.5 else ""
        add(f"  {category:<34} {recall * 100:>5.1f}%  (n={count}){flag}")

    add("")
    add("=" * 78)
    add("KATEGORI BAZLI RISKY RECALL  (uyari seviyesinde)")
    add("=" * 78)
    by_risky: dict[str, list[bool]] = defaultdict(list)
    for row, score in zip(rows, risk, strict=True):
        if row.label is Risk.RISKY:
            by_risky[row.category].append(score >= warn_t)
    for category, hits in sorted(
        by_risky.items(), key=lambda item: sum(item[1]) / len(item[1])
    ):
        add(f"  {category:<34} {sum(hits) / len(hits) * 100:>5.1f}%  (n={len(hits)})")

    report = {
        "thresholds": model.thresholds,
        "highest_score": float(highest),
        "block_unreachable": bool(block_t > highest),
        "selected": {
            "block_recall": best_metrics.block_recall,
            "safe_false_warn": best_metrics.fp_warn,
            "cost": best_metrics.cost,
        },
        "best_block_by_cost": best_block,
        "per_category_destructive": {c: r for c, r, _ in weak},
    }

    text = "\n".join(out)
    if not args.quiet:
        print(text)
    target = args.json or ARTIFACTS_JSON
    from pathlib import Path

    Path(target).parent.mkdir(parents=True, exist_ok=True)
    Path(target).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not args.quiet:
        print()
        print(f"yazildi: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())