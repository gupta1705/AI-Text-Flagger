"""Tests for the v2 test-set report (before/after table) built from synthetic metric rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FD_KEY, FINETUNED_KEY, TEST_SETS, VERDICT_AI, VERDICT_HUMAN, VERDICT_UNCERTAIN
from src.evaluate import Evaluator, verdict_metrics


def _metric_rows() -> pd.DataFrame:
    rows = []
    for split in ("test", "test_ood"):
        for version, det, acc in (("v2", FD_KEY, 70.0), ("v2", FINETUNED_KEY, 80.0),
                                  ("v1", FINETUNED_KEY, 60.0), ("v1", "loglik", 55.0)):
            rows.append({"test_set": split, "test_set_name": TEST_SETS[split], "version": version, "detector": det,
                         "accuracy": acc + (5 if split == "test" else 0), "auroc": acc + 10, "n": 4})
    return pd.DataFrame(rows)


def _verdict_rows() -> pd.DataFrame:
    labels = np.array([0, 0, 1, 1])
    parts = []
    for split in ("test", "test_ood"):
        for version, verdicts in (("v1", [VERDICT_HUMAN, VERDICT_HUMAN, VERDICT_HUMAN, VERDICT_UNCERTAIN]),
                                  ("v2", [VERDICT_HUMAN, VERDICT_AI, VERDICT_AI, VERDICT_AI])):
            table = verdict_metrics(labels, np.asarray(verdicts, dtype=object))
            table.insert(0, "version", version)
            table.insert(0, "test_set_name", TEST_SETS[split])
            table.insert(0, "test_set", split)
            parts.append(table)
    return pd.concat(parts, ignore_index=True)


def test_markdown_has_one_column_per_test_set_and_before_after_rows() -> None:
    md = Evaluator.test_sets_markdown(_metric_rows(), _verdict_rows())
    assert "## Accuracy (%)" in md and "## AUROC (%)" in md and "## Web app verdict before vs after (%)" in md
    header = next(line for line in md.splitlines() if line.startswith("| Detector"))
    assert TEST_SETS["test"] in header and TEST_SETS["test_ood"] in header
    assert "| DistilRoBERTa v1 (HC3 only) | 65.00 | 60.00 |" in md
    assert "| Fast-DetectGPT | 75.00 | 70.00 |" in md
    # v1 caught no AI text (one uncertain), v2 caught both but wrongly flagged one human text
    assert f"| {TEST_SETS['test_ood']} | before (v1) | 75.00 | 66.67 | 0.00 | 0.00 |" in md
    assert f"| {TEST_SETS['test_ood']} | after (v2) | 100.00 | 75.00 | 100.00 | 50.00 |" in md


def test_markdown_without_verdicts_still_lists_detectors() -> None:
    md = Evaluator.test_sets_markdown(_metric_rows(), pd.DataFrame())
    assert "before vs after" not in md and "Fast-DetectGPT" in md
