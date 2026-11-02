"""Evaluator: clean and robustness evaluation of all five detectors (report Section 3.3, steps 6-7).

Detectors: the four GPT-2 zero-shot scores (Log-Likelihood, Rank, Log-Rank, Entropy), each with its
own calibrator, and the fine-tuned DistilRoBERTa classifier.

* **Clean** - accuracy, per-class precision and recall, F1 (AI = positive) and AUROC on
  ``test.csv`` -> ``results/clean_metrics.csv`` / ``.md`` (percentages, 2 decimals) and the raw
  per-text probabilities in ``results/clean_predictions.csv``. When DistilRoBERTa and the app's
  zero-shot statistic are both evaluated, the web app's combined verdict (Algorithm 5) is scored
  too: coverage (share not "Uncertain"), accuracy on the decided texts and the share of human
  texts labelled "Likely AI-generated" -> ``results/verdict_metrics.csv`` and ``clean_metrics.md``.
* **Robustness** - the :class:`~src.perturb.Perturber` creates, for every test text, operation
  (swap, drop, key, add) and level (1 or 2 edited words), ``k`` = 10 deterministic variants.
  Following the survey (Section 7.1), a sample counts as correct only if **all** ``k`` variants
  are classified correctly; as in the standard definition of accuracy under attack (AUA), a text
  whose clean prediction is already wrong also counts as not robust, so AUA <= clean accuracy.
  The mean per-variant accuracy is reported as well. GPT-2 statistics of a variant are computed
  once (and cached) and reused by the four zero-shot detectors. Outputs: ``results/robustness.csv``
  (one row per detector and condition), ``results/robustness_table.md`` (survey Table 5 layout),
  ``results/robustness_per_sample.csv`` (one row per text, detector and condition),
  ``results/robustness_variants.csv.gz`` (one row per perturbed variant: its key, edited words and
  every detector's P(AI)) and ``results/robustness_chart.png``.

CLI: ``python -m src.evaluate --clean | --robust | --all [--k 10] [--limit N] [--detectors ...] [--results-dir DIR]``
(a reduced run - ``--limit``, another ``--k``, a subset of ``--detectors`` or other models - must name its
own ``--results-dir``, so it never replaces the full-size results in ``results/``).

The torch-based detectors are imported lazily, so ``metrics`` can be imported cheaply.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from src.base import configure_console, refuse_reduced_run, set_seed
from src.config import (
    CACHE_DIR,
    CLEAN_METRICS_CSV,
    CLEAN_METRICS_MD,
    CLEAN_PREDICTIONS_CSV,
    DATA_DIR,
    DETECTOR_DISPLAY,
    DETECTORS,
    FINETUNED_DIR,
    FINETUNED_KEY,
    K_VARIANTS,
    LABEL_AI,
    LABEL_HUMAN,
    MODELS_DIR,
    N_TEST,
    OP_DISPLAY,
    ROBUST_STATS_CACHE,
    ROBUSTNESS_CHART_PNG,
    ROBUSTNESS_CSV,
    ROBUSTNESS_MD,
    ROBUSTNESS_SAMPLES_CSV,
    ROBUSTNESS_VARIANTS_CSV,
    RESULTS_DIR,
    SEED,
    THRESHOLD,
    VERDICT_AI,
    VERDICT_HUMAN,
    VERDICT_METRICS_CSV,
    VERDICT_UNCERTAIN,
    ZERO_SHOT_JSON,
    ZS_BATCH_SIZE,
    ZS_METRICS,
)
from src.perturb import Perturber, edited_words, variant_key
from src.text_utils import word_count

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.finetune import FineTunedDetector
    from src.zero_shot import ZeroShotDetector

METRIC_COLUMNS = ("accuracy", "precision_human", "precision_ai", "recall_human", "recall_ai", "f1", "auroc")
METRIC_HEADERS = ("Accuracy", "Precision (human)", "Precision (AI)", "Recall (human)", "Recall (AI)", "F1", "AUROC")
VERDICT_COLUMNS = ("coverage", "accuracy_decided", "labelled_ai", "labelled_human", "uncertain")
VERDICT_HEADERS = ("Coverage", "Accuracy (decided)", "Labelled AI", "Labelled human", "Uncertain")


# ---------------------------------------------------------------------- pure metric helpers
def require_probabilities(y_prob: Sequence[float], what: str = "probabilities") -> np.ndarray:
    """Return ``y_prob`` as a float array; raise if any value is missing (NaN).

    A zero-shot score is missing only for a text with fewer than two GPT-2 tokens, which the
    20-word minimum rules out for every split. Comparing NaN with the threshold would silently
    count such a text as human, so a missing score stops the evaluation instead.
    """
    y_prob = np.asarray(y_prob, dtype=float)
    missing = int(np.isnan(y_prob).sum())
    if missing:
        raise ValueError(f"{missing} of {len(y_prob)} {what} are missing (NaN): texts too short to score")
    return y_prob


def metrics(y_true: Sequence[int], y_prob: Sequence[float], threshold: float = THRESHOLD) -> dict[str, float]:
    """Classification metrics for P(AI) scores; AI (label 1) is the positive class.

    Returns accuracy, precision/recall of each class, F1 of the AI class, AUROC (NaN when only
    one class is present) and ``n``. All values are fractions in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = require_probabilities(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[LABEL_HUMAN, LABEL_AI], zero_division=0
    )
    auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else float("nan"),
        "precision_human": float(precision[0]),
        "precision_ai": float(precision[1]),
        "recall_human": float(recall[0]),
        "recall_ai": float(recall[1]),
        "f1": float(f1[1]),
        "auroc": auroc,
        "n": int(len(y_true)),
    }


def robust_samples(clean_correct: np.ndarray, variant_correct: np.ndarray) -> np.ndarray:
    """Per sample: clean prediction correct **and** all ``k`` variants (columns) correct."""
    clean_correct = np.asarray(clean_correct, dtype=bool)
    variant_correct = np.asarray(variant_correct, dtype=bool)
    if variant_correct.ndim != 2 or variant_correct.shape[0] == 0:
        raise ValueError("expected a non-empty (n_samples, k) boolean matrix of variant results")
    if clean_correct.shape != (variant_correct.shape[0],):
        raise ValueError("expected one clean result per sample")
    return clean_correct & variant_correct.all(axis=1)


def verdict_metrics(y_true: Sequence[int], verdicts: Sequence[str]) -> pd.DataFrame:
    """Score the app's three-way verdict for all texts, human texts and AI texts (percentages).

    ``coverage`` = share with a verdict (not Uncertain); ``accuracy_decided`` = accuracy on those
    texts (NaN if none); ``labelled_ai`` / ``labelled_human`` / ``uncertain`` = share of the group
    with that verdict. For human texts ``labelled_ai`` is the false-accusation rate.
    """
    y_true = np.asarray(y_true, dtype=int)
    verdicts = np.asarray(verdicts, dtype=object)
    unknown = set(verdicts) - {VERDICT_AI, VERDICT_HUMAN, VERDICT_UNCERTAIN}
    if unknown:
        raise ValueError(f"unknown verdict labels: {sorted(unknown)}")
    predicted = np.where(verdicts == VERDICT_AI, LABEL_AI, LABEL_HUMAN)
    rows = []
    for group, mask in (("All texts", np.ones(len(y_true), dtype=bool)),
                        ("Human texts", y_true == LABEL_HUMAN), ("AI texts", y_true == LABEL_AI)):
        v, decided = verdicts[mask], verdicts[mask] != VERDICT_UNCERTAIN
        n = int(mask.sum())
        share = (lambda label: float(np.mean(v == label))) if n else (lambda label: float("nan"))
        accuracy = (float(np.mean(predicted[mask][decided] == y_true[mask][decided]))
                    if decided.any() else float("nan"))
        rows.append({"group": group, "n": n, "coverage": _pct(float(decided.mean()) if n else float("nan")),
                     "accuracy_decided": _pct(accuracy), "labelled_ai": _pct(share(VERDICT_AI)),
                     "labelled_human": _pct(share(VERDICT_HUMAN)), "uncertain": _pct(share(VERDICT_UNCERTAIN))})
    return pd.DataFrame(rows)


def accuracy_under_attack(clean_correct: np.ndarray, variant_correct: np.ndarray) -> float:
    """AUA: share of samples whose clean prediction and *all* their variants are correct.

    The original text is treated as the unedited variant 0 (standard accuracy-under-attack
    definition: an attack on an already misclassified text counts as successful), so AUA can
    never exceed the clean accuracy.
    """
    return float(robust_samples(clean_correct, variant_correct).mean())


def _pct(value: float) -> float:
    """Fraction -> percentage rounded to 2 decimals (NaN stays NaN)."""
    return round(100.0 * value, 2) if np.isfinite(value) else float("nan")


def _fmt(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.2f}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a GitHub-flavoured Markdown table (first column left-aligned, others right)."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join([":---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def condition_label(op: str, level: int) -> str:
    """Column label of a perturbation condition, e.g. ``Swap 1``."""
    return f"{OP_DISPLAY[op]} {level}"


def plot_robustness_chart(summary: pd.DataFrame, path: str | Path) -> Path:
    """Grouped bars per detector: clean accuracy vs mean 1-char AUA vs mean 2-char AUA."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    detectors = list(dict.fromkeys(summary["detector"]))
    groups = {"Clean accuracy": [], "Mean AUA, 1-char edits": [], "Mean AUA, 2-char edits": []}
    for det in detectors:
        rows = summary[summary["detector"] == det]
        groups["Clean accuracy"].append(float(rows.loc[rows["op"] == "none", "aua"].iloc[0]))
        for level, key in ((1, "Mean AUA, 1-char edits"), (2, "Mean AUA, 2-char edits")):
            groups[key].append(float(rows.loc[rows["level"] == level, "aua"].mean()))
    colors = ("#2b6cb0", "#dd6b20", "#c53030")
    x = np.arange(len(detectors))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    for offset, ((label, values), color) in enumerate(zip(groups.items(), colors)):
        bars = ax.bar(x + (offset - 1) * width, values, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    ax.set_xticks(x, [DETECTOR_DISPLAY.get(d, d) for d in detectors])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Clean accuracy vs accuracy under attack (AUA: clean and all k variants correct)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------- evaluator
class Evaluator:
    """Computes clean and robust metrics for the selected detectors on the test split.

    Args:
        test_df: test rows (columns ``id, source, label, text``).
        detectors: subset of :data:`src.config.DETECTORS`.
        k: perturbed variants per (text, operation, level).
        batch_size: texts per model pass.
        threshold: decision threshold on P(AI).
        results_dir / cache_dir: output folders.
        zero_shot_path / finetuned_dir: saved models.
        device: ``"cuda"``/``"cpu"`` (default: CUDA if available).
    """

    def __init__(
        self,
        test_df: pd.DataFrame,
        detectors: Sequence[str] = DETECTORS,
        k: int = K_VARIANTS,
        batch_size: int = ZS_BATCH_SIZE,
        threshold: float = THRESHOLD,
        results_dir: str | Path = RESULTS_DIR,
        cache_dir: str | Path = CACHE_DIR,
        zero_shot_path: str | Path = ZERO_SHOT_JSON,
        finetuned_dir: str | Path = FINETUNED_DIR,
        device: str | None = None,
    ) -> None:
        unknown = set(detectors) - set(DETECTORS)
        if unknown:
            raise ValueError(f"unknown detectors {sorted(unknown)}; choose from {DETECTORS}")
        if k < 1:
            raise ValueError("k must be >= 1")
        self.test = test_df.reset_index(drop=True)
        self.detectors = [d for d in DETECTORS if d in detectors]     # canonical order
        self.zs_metrics = [d for d in self.detectors if d in ZS_METRICS]
        self.k = k
        self.perturber = Perturber(k=k)
        self.conditions = self.perturber.conditions()                  # survey Table 5 order
        self.batch_size = batch_size
        self.threshold = threshold
        self.results_dir = Path(results_dir)
        self.cache_dir = Path(cache_dir)
        self.zero_shot_path = Path(zero_shot_path)
        self.finetuned_dir = Path(finetuned_dir)
        self.device = device
        self._zs: ZeroShotDetector | None = None
        self._ft: FineTunedDetector | None = None
        self._clean_probs: dict[str, np.ndarray] | None = None

    # ------------------------------------------------------------------ detectors
    def _zero_shot(self) -> "ZeroShotDetector":
        if self._zs is None:
            from src.zero_shot import ZeroShotDetector

            self._zs = ZeroShotDetector.load(self.zero_shot_path, device=self.device, batch_size=self.batch_size)
        return self._zs

    def _finetuned(self) -> "FineTunedDetector":
        if self._ft is None:
            from src.finetune import FineTunedDetector

            self._ft = FineTunedDetector.load(self.finetuned_dir, device=self.device, batch_size=self.batch_size)
        return self._ft

    def _release_models(self) -> None:
        for model in (self._zs, self._ft):
            if model is not None:
                model.release()

    def _probabilities(self, keys: Sequence[str], texts: Sequence[str], cache_path: Path, desc: str) -> dict[str, np.ndarray]:
        """P(AI) of every selected detector for ``texts`` (GPT-2 statistics computed once, cached)."""
        probs: dict[str, np.ndarray] = {}
        if self.zs_metrics:
            from src.zero_shot import cached_statistics

            zs = self._zero_shot()
            stats = cached_statistics(zs, keys, texts, cache_path, desc=f"GPT-2 {desc}")
            for metric in self.zs_metrics:
                probs[metric] = zs.proba_from_stats(stats, metric)
            zs.release()                     # free GPU memory before the next model
        if FINETUNED_KEY in self.detectors:
            ft = self._finetuned()
            probs[FINETUNED_KEY] = ft.predict_proba(texts, progress=True, desc=f"DistilRoBERTa {desc}")
        return probs

    @property
    def is_limited(self) -> bool:
        """True for smoke / limited runs whose numbers are not representative."""
        return len(self.test) < N_TEST or self.k < K_VARIANTS

    def _notes(self, robust: bool) -> list[str]:
        labels = self.test["label"]
        notes = [f"Test set: {len(self.test):,} texts ({int((labels == LABEL_HUMAN).sum()):,} human / "
                 f"{int((labels == LABEL_AI).sum()):,} AI), threshold P(AI) >= {self.threshold}."]
        if robust:
            notes.append(f"AUA = accuracy under attack: a test text counts as correct only if all k = {self.k} "
                         "perturbed variants are classified correctly; a text whose clean prediction is already "
                         "wrong counts as not robust (the original is treated as the unedited variant), so AUA "
                         "never exceeds Normal (= clean accuracy). The mean per-variant accuracy ignores the "
                         "clean prediction.")
        if self.is_limited:
            notes.append("**Limited run (smoke test / --limit / small k): these numbers are not representative.**")
        return notes

    # ------------------------------------------------------------------ clean evaluation
    def clean_probabilities(self) -> dict[str, np.ndarray]:
        """P(AI) of every selected detector on the clean test texts (memoised)."""
        if self._clean_probs is None:
            self._clean_probs = self._probabilities(
                self.test["id"].astype(str).tolist(), self.test["text"].tolist(),
                self.cache_dir / "gpt2_stats_test.csv", "clean test",
            )
        return self._clean_probs

    def run_clean(self) -> pd.DataFrame:
        """Clean metrics for every detector; writes CSV, Markdown and per-text predictions."""
        probs = self.clean_probabilities()
        labels = self.test["label"].to_numpy()
        rows = []
        for det in self.detectors:
            m = metrics(labels, probs[det], self.threshold)
            rows.append({"detector": det, "detector_name": DETECTOR_DISPLAY[det],
                         **{c: _pct(m[c]) for c in METRIC_COLUMNS}, "n": m["n"]})
        table = pd.DataFrame(rows)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(self.results_dir / CLEAN_METRICS_CSV.name, index=False, encoding="utf-8")

        predictions = self.test[["id", "source", "label"]].copy()
        for det in self.detectors:
            predictions[f"p_ai_{det}"] = np.round(probs[det], 6)
        selected = self._selected_zero_shot()
        verdict_table = None
        if FINETUNED_KEY in self.detectors and selected in self.zs_metrics:
            predictions["verdict"] = self.app_verdicts(probs[FINETUNED_KEY], probs[selected])
            verdict_table = verdict_metrics(labels, predictions["verdict"].to_numpy())
            verdict_table.to_csv(self.results_dir / VERDICT_METRICS_CSV.name, index=False, encoding="utf-8")
        predictions.to_csv(self.results_dir / CLEAN_PREDICTIONS_CSV.name, index=False, encoding="utf-8")

        md_rows = [[r["detector_name"], *(_fmt(r[c]) for c in METRIC_COLUMNS)] for r in rows]
        lines = ["# Clean test-set metrics (%)", "", markdown_table(("Detector", *METRIC_HEADERS), md_rows), ""]
        lines += [f"- {n}" for n in self._notes(robust=False)]
        lines.append("- F1 treats AI-generated text as the positive class; AUROC uses the raw probabilities.")
        if selected:
            lines.append(f"- Zero-shot score used by the web app: **{DETECTOR_DISPLAY[selected]}** (best validation F1).")
        if verdict_table is not None:
            lines += ["", "## Combined verdict of the web app (%)", "",
                      markdown_table(("Group", "Texts", *VERDICT_HEADERS),
                                     [[r["group"], r["n"], *(_fmt(r[c]) for c in VERDICT_COLUMNS)]
                                      for r in verdict_table.to_dict("records")]),
                      "",
                      f"- Decision rule of the app (Algorithm 5): DistilRoBERTa and the {DETECTOR_DISPLAY[selected]} "
                      "detector on the same side of 0.5 -> *Likely AI-generated* / *Likely human-written*, "
                      "otherwise *Uncertain*. Coverage = share of texts that get a verdict; Accuracy (decided) = "
                      "accuracy on those texts; Labelled AI = share of the group labelled *Likely AI-generated* "
                      "(for human texts: the false-accusation rate)."]
        (self.results_dir / CLEAN_METRICS_MD.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return table

    def app_verdicts(self, p_finetuned: np.ndarray, p_zeroshot: np.ndarray) -> list[str]:
        """The web app's verdict label (Algorithm 5, :func:`src.app.decide`) for every test text."""
        from src.app import decide          # imports Gradio, so only when needed

        return [decide(float(ft), float(zs), word_count(text), self.threshold).label
                for ft, zs, text in zip(p_finetuned, p_zeroshot, self.test["text"])]

    def _selected_zero_shot(self) -> str | None:
        """Statistic chosen for the app (from ``zero_shot.json``), if the file exists."""
        if not self.zero_shot_path.is_file():
            return None
        with open(self.zero_shot_path, encoding="utf-8") as handle:
            return json.load(handle).get("selected_metric")

    # ------------------------------------------------------------------ robustness evaluation
    def build_variants(self) -> pd.DataFrame:
        """All perturbed variants (made by the :class:`Perturber`), ordered condition -> test row -> ``j``.

        Columns: ``key`` (``"{id}|{op}|{level}|{j}"``; its CRC-32 is the variant's seed), ``row``
        (test row), ``op``, ``level``, ``j`` and ``text``.
        """
        records = []
        ids = self.test["id"].astype(str).tolist()
        texts = self.test["text"].tolist()
        for op, level in self.conditions:
            for row, (text_id, text) in enumerate(zip(ids, texts)):
                for j, variant in enumerate(self.perturber.variants(text, op, level, text_id=text_id)):
                    records.append((variant_key(text_id, op, level, j), row, op, level, j, variant))
        return pd.DataFrame(records, columns=["key", "row", "op", "level", "j", "text"])

    def gpt2_window(self, variants: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Which variants change the text GPT-2 actually reads (its first ``max_tokens`` tokens)?

        Returns a boolean flag per variant and the test rows longer than the window. A False flag
        means no edit lies inside the window (or the edit left the text unchanged), so the
        zero-shot detectors score the clean text. Only variants of long texts are tokenised.
        """
        zs = self._zero_shot()
        clean_texts = self.test["text"].tolist()
        lengths, clean_window = zs.window_token_ids(clean_texts)
        rows = variants["row"].to_numpy()
        flags = variants["text"].to_numpy() != np.asarray(clean_texts, dtype=object)[rows]
        long_rows = np.flatnonzero(np.asarray(lengths) > zs.max_tokens)
        idx = np.flatnonzero(np.isin(rows, long_rows))
        if len(idx):
            _, windows = zs.window_token_ids(variants["text"].to_numpy()[idx].tolist())
            flags[idx] = [w != clean_window[rows[i]] for i, w in zip(idx, windows)]
        return flags.astype(bool), long_rows

    def run_robustness(self) -> pd.DataFrame:
        """AUA and mean per-variant accuracy for every detector and condition; writes all outputs."""
        clean = self.clean_probabilities()
        variants = self.build_variants()
        n, k = len(self.test), self.k
        print(f"[evaluate] robustness: {n} test texts x {len(self.conditions)} conditions x k={k} "
              f"= {len(variants):,} variants per detector")
        started = time.perf_counter()
        probs = self._probabilities(variants["key"].tolist(), variants["text"].tolist(),
                                    self.cache_dir / ROBUST_STATS_CACHE.name, "variants")
        print(f"[evaluate] variants scored in {time.perf_counter() - started:.0f} s")

        labels = self.test["label"].to_numpy()
        variant_labels = labels[variants["row"].to_numpy()]
        summary_rows, sample_frames = [], []
        block = n * k
        for det in self.detectors:
            clean_correct = (require_probabilities(clean[det], f"{det} clean probabilities") >= self.threshold) == labels
            clean_acc = float(clean_correct.mean())
            summary_rows.append({"detector": det, "detector_name": DETECTOR_DISPLAY[det], "condition": "Normal",
                                 "op": "none", "level": 0, "aua": _pct(clean_acc),
                                 "mean_variant_accuracy": _pct(clean_acc), "n_texts": n, "k": k})
            correct_all = (require_probabilities(probs[det], f"{det} variant probabilities")
                           >= self.threshold) == variant_labels
            for b, (op, level) in enumerate(self.conditions):
                correct = correct_all[b * block:(b + 1) * block].reshape(n, k)
                summary_rows.append({"detector": det, "detector_name": DETECTOR_DISPLAY[det],
                                     "condition": condition_label(op, level), "op": op, "level": level,
                                     "aua": _pct(accuracy_under_attack(clean_correct, correct)),
                                     "mean_variant_accuracy": _pct(float(correct.mean())),
                                     "n_texts": n, "k": k})
                sample_frames.append(pd.DataFrame({
                    "id": self.test["id"], "source": self.test["source"], "label": labels,
                    "detector": det, "op": op, "level": level, "clean_correct": clean_correct,
                    "n_correct": correct.sum(axis=1), "k": k, "all_variants_correct": correct.all(axis=1),
                    "robust": robust_samples(clean_correct, correct),
                }))
        summary = pd.DataFrame(summary_rows)

        # One row per PERTURBED_SAMPLE with every detector's PREDICTION (report Figure 4).
        per_variant = pd.DataFrame({
            "key": variants["key"], "id": self.test["id"].to_numpy()[variants["row"].to_numpy()],
            "label": variant_labels, "op": variants["op"], "level": variants["level"], "j": variants["j"],
            "edits": [";".join(f"{a}>{b}" for a, b in edited_words(self.test["text"].iat[r], t))
                      for r, t in zip(variants["row"], variants["text"])],
        })
        window_note = None
        if self.zs_metrics:
            in_window, long_rows = self.gpt2_window(variants)
            per_variant["edit_in_gpt2_window"] = in_window
            window_note = self._window_note(variants, in_window, long_rows)
        for det in self.detectors:
            per_variant[f"p_ai_{det}"] = np.round(probs[det], 6)

        self.results_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(self.results_dir / ROBUSTNESS_CSV.name, index=False, encoding="utf-8")
        pd.concat(sample_frames, ignore_index=True).to_csv(
            self.results_dir / ROBUSTNESS_SAMPLES_CSV.name, index=False, encoding="utf-8")
        per_variant.to_csv(self.results_dir / ROBUSTNESS_VARIANTS_CSV.name, index=False, encoding="utf-8",
                           compression="gzip")
        (self.results_dir / ROBUSTNESS_MD.name).write_text(self.robustness_markdown(summary, window_note),
                                                           encoding="utf-8")
        plot_robustness_chart(summary, self.results_dir / ROBUSTNESS_CHART_PNG.name)
        return summary

    def _window_note(self, variants: pd.DataFrame, in_window: np.ndarray, long_rows: np.ndarray) -> str | None:
        """Table note on texts longer than GPT-2's window, whose late edits the zero-shot detectors miss."""
        if not len(long_rows):
            return None
        zs = self._zero_shot()
        labels = self.test["label"].to_numpy()[long_rows]
        of_long = np.isin(variants["row"].to_numpy(), long_rows)
        unseen = int((of_long & ~in_window).sum())
        return (f"GPT-2 reads only the first {zs.max_tokens} tokens. {len(long_rows):,} of {len(self.test):,} test "
                f"texts ({int((labels == LABEL_HUMAN).sum()):,} human / {int((labels == LABEL_AI).sum()):,} AI) are "
                f"longer; in {unseen:,} of their {int(of_long.sum()):,} variants no edit falls inside that window, "
                "so the four zero-shot detectors score the clean text there (DistilRoBERTa reads up to 8 x 256 "
                f"tokens). This can raise the zero-shot AUA by at most {100 * len(long_rows) / len(self.test):.2f} "
                "percentage points (column `edit_in_gpt2_window` of robustness_variants.csv.gz).")

    def robustness_markdown(self, summary: pd.DataFrame, window_note: str | None = None) -> str:
        """Two tables (AUA and mean per-variant accuracy) in survey Table 5 column order."""
        columns = ["Normal", *(condition_label(op, level) for op, level in self.conditions)]

        def table(value: str) -> str:
            rows = []
            for det in self.detectors:
                part = summary[summary["detector"] == det].set_index("condition")
                rows.append([DETECTOR_DISPLAY[det], *(_fmt(part.loc[c, value]) for c in columns)])
            return markdown_table(("Detector", *columns), rows)

        lines = [
            "# Robustness to character-level perturbations",
            "",
            f"## Accuracy under attack, AUA (%) - clean prediction and all k = {self.k} variants correct",
            "",
            table("aua"),
            "",
            "## Mean per-variant accuracy (%)",
            "",
            table("mean_variant_accuracy"),
            "",
            *(f"- {n}" for n in self._notes(robust=True)),
            *([f"- {window_note}"] if window_note else []),
            "- Operations (Pruthi et al., 2019): Swap = exchange two neighbouring letters, Drop = delete a letter, "
            "Add = insert a random letter, Key = replace a letter with a neighbouring QWERTY key; "
            "1 / 2 = number of edited words (one edit per word, first letter kept).",
        ]
        return "\n".join(lines) + "\n"

    def close(self) -> None:
        """Release GPU memory held by the detectors."""
        self._release_models()


# ---------------------------------------------------------------------- CLI
def _parse_detectors(value: str) -> list[str]:
    names = [v.strip() for v in value.split(",") if v.strip()]
    unknown = set(names) - set(DETECTORS)
    if unknown or not names:
        raise argparse.ArgumentTypeError(f"choose from {','.join(DETECTORS)}")
    return names


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluate",
        description="Clean and robustness evaluation of the five detectors on data/test.csv.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--clean", action="store_true", help="clean test-set metrics only")
    mode.add_argument("--robust", action="store_true", help="robustness (AUA) evaluation only")
    mode.add_argument("--all", action="store_true", help="clean + robustness (default)")
    parser.add_argument("--k", type=int, default=K_VARIANTS, help="variants per text, operation and level (default: %(default)s)")
    parser.add_argument("--limit", type=int, default=None, help="use at most N test texts (balanced)")
    parser.add_argument("--detectors", type=_parse_detectors, default=list(DETECTORS),
                        help=f"comma-separated subset of {','.join(DETECTORS)} (default: all)")
    parser.add_argument("--batch-size", type=int, default=ZS_BATCH_SIZE, help="texts per model pass (default: %(default)s)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="folder with test.csv")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR,
                        help="folder with zero_shot.json and distilroberta/ (default: %(default)s)")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help=f"output folder (default: {RESULTS_DIR}); a reduced run (--limit, another --k, "
                             "a subset of --detectors) must name its own")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR,
                        help="GPT-2 statistics cache (default: %(default)s)")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    configure_console()
    args = build_arg_parser().parse_args(argv)
    set_seed(SEED)
    from src.data import HC3DataModule

    run_clean = args.clean or args.all or not args.robust
    run_robust = args.robust or args.all or not args.clean
    try:
        test = HC3DataModule.load_split("test", args.data_dir, args.limit)
    except FileNotFoundError as error:
        print(f"[evaluate] {error}", file=sys.stderr)
        return 1
    reasons = []
    if len(test) < N_TEST:
        reasons.append(f"{len(test)} test texts instead of {N_TEST:,}")
    if run_robust and args.k != K_VARIANTS:
        reasons.append(f"k = {args.k} instead of {K_VARIANTS}")
    if set(args.detectors) != set(DETECTORS):
        reasons.append(f"{len(set(args.detectors))} of {len(DETECTORS)} detectors")
    if args.models_dir.resolve() != MODELS_DIR.resolve():
        reasons.append(f"models from {args.models_dir}")
    if refuse_reduced_run("evaluate", reasons, [] if args.results_dir else [("--results-dir", RESULTS_DIR)]):
        return 2
    results_dir = args.results_dir or RESULTS_DIR
    evaluator = Evaluator(test, detectors=args.detectors, k=args.k, batch_size=args.batch_size,
                          results_dir=results_dir, cache_dir=args.cache_dir,
                          zero_shot_path=args.models_dir / ZERO_SHOT_JSON.name,
                          finetuned_dir=args.models_dir / FINETUNED_DIR.name, device=args.device)
    try:
        if run_clean:
            table = evaluator.run_clean()
            print("\n" + (results_dir / CLEAN_METRICS_MD.name).read_text(encoding="utf-8"))
            extra = f", {VERDICT_METRICS_CSV.name}" if (results_dir / VERDICT_METRICS_CSV.name).is_file() else ""
            print(f"[evaluate] wrote {CLEAN_METRICS_CSV.name}, {CLEAN_METRICS_MD.name}, "
                  f"{CLEAN_PREDICTIONS_CSV.name}{extra} ({len(table)} detectors)")
        if run_robust:
            evaluator.run_robustness()
            print("\n" + (results_dir / ROBUSTNESS_MD.name).read_text(encoding="utf-8"))
            print(f"[evaluate] wrote {ROBUSTNESS_CSV.name}, {ROBUSTNESS_MD.name}, {ROBUSTNESS_SAMPLES_CSV.name}, "
                  f"{ROBUSTNESS_VARIANTS_CSV.name}, {ROBUSTNESS_CHART_PNG.name}")
    except FileNotFoundError as error:          # a model folder or JSON is missing
        print(f"[evaluate] {error}", file=sys.stderr)
        return 1
    finally:
        evaluator.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
