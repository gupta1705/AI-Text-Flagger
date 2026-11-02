"""Tests for src.evaluate: metrics, AUA and the robustness aggregation (with stub detectors)."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from src.config import ROBUSTNESS_CONDITIONS, VERDICT_AI, VERDICT_HUMAN, VERDICT_UNCERTAIN
from src.evaluate import Evaluator, accuracy_under_attack, markdown_table, metrics, verdict_metrics
from src.perturb import Perturber, edited_words


def test_metrics_hand_computed_example() -> None:
    m = metrics([0, 0, 1, 1], [0.1, 0.6, 0.7, 0.4])
    assert m["accuracy"] == 0.5
    assert m["precision_human"] == 0.5 and m["precision_ai"] == 0.5
    assert m["recall_human"] == 0.5 and m["recall_ai"] == 0.5
    assert m["f1"] == 0.5
    assert m["auroc"] == 0.75
    assert m["n"] == 4


def test_metrics_threshold_is_inclusive_and_configurable() -> None:
    assert metrics([1], [0.5])["accuracy"] == 1.0
    assert metrics([1, 0], [0.6, 0.4], threshold=0.7)["recall_ai"] == 0.0


def test_metrics_single_class_has_nan_auroc_and_no_crash() -> None:
    m = metrics([1, 1, 1], [0.9, 0.2, 0.8])
    assert math.isnan(m["auroc"])
    assert m["precision_human"] == 0.0      # zero_division=0, no exception
    assert m["recall_ai"] == pytest.approx(2 / 3)


def test_metrics_refuse_missing_probabilities() -> None:
    with pytest.raises(ValueError, match="missing"):
        metrics([0, 1], [0.2, float("nan")])


def test_accuracy_under_attack_requires_all_variants_correct() -> None:
    correct = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 0]], dtype=bool)
    assert accuracy_under_attack(np.ones(4, dtype=bool), correct) == 0.5
    with pytest.raises(ValueError):
        accuracy_under_attack(np.ones(3, dtype=bool), np.ones(3, dtype=bool))
    with pytest.raises(ValueError):
        accuracy_under_attack(np.ones(2, dtype=bool), correct)


def test_accuracy_under_attack_never_exceeds_clean_accuracy() -> None:
    """A text the detector already gets wrong cannot become 'robust' because typos flip it."""
    clean_correct = np.array([True, False, True, False])
    variants_correct = np.ones((4, 3), dtype=bool)
    assert accuracy_under_attack(clean_correct, variants_correct) == 0.5 == clean_correct.mean()


def test_verdict_metrics() -> None:
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    verdicts = [VERDICT_HUMAN, VERDICT_HUMAN, VERDICT_AI, VERDICT_UNCERTAIN,
                VERDICT_AI, VERDICT_AI, VERDICT_UNCERTAIN, VERDICT_HUMAN]
    table = verdict_metrics(labels, verdicts).set_index("group")
    assert table.loc["All texts", "n"] == 8 and table.loc["All texts", "coverage"] == 75.0
    assert table.loc["All texts", "accuracy_decided"] == round(100 * 4 / 6, 2)
    assert table.loc["Human texts", "labelled_ai"] == 25.0                  # false accusations
    assert table.loc["AI texts", "labelled_ai"] == 50.0 and table.loc["AI texts", "uncertain"] == 25.0
    assert math.isnan(verdict_metrics([0], [VERDICT_UNCERTAIN]).loc[0, "accuracy_decided"])
    with pytest.raises(ValueError):
        verdict_metrics([0], ["maybe"])


def test_markdown_table() -> None:
    table = markdown_table(("A", "B"), [["x", 1.5]])
    assert table.splitlines() == ["| A | B |", "|:---|---:|", "| x | 1.5 |"]


def _test_frame(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame({
        "id": [f"{i}-{'h' if i % 2 == 0 else 'c'}0" for i in range(n)],
        "source": ["finance"] * n,
        "label": [i % 2 for i in range(n)],
        "text": [f"Sample answer number {i} with several longer words inside it" for i in range(n)],
    })


def test_build_variants_order_and_count() -> None:
    evaluator = Evaluator(_test_frame(), detectors=["distilroberta"], k=3)
    assert isinstance(evaluator.perturber, Perturber) and evaluator.conditions == list(ROBUSTNESS_CONDITIONS)
    variants = evaluator.build_variants()
    first_text = _test_frame()["text"][0]
    assert variants["text"].iloc[:3].tolist() == Perturber(k=3).variants(first_text, "swap", 1, text_id="0-h0")
    assert len(variants) == 6 * len(ROBUSTNESS_CONDITIONS) * 3
    first_block = variants.iloc[:18]
    assert (first_block["op"] == "swap").all() and (first_block["level"] == 1).all()
    assert first_block["row"].tolist() == [r for r in range(6) for _ in range(3)]
    assert variants["key"].iloc[0] == "0-h0|swap|1|0"


def test_run_robustness_with_stub_probabilities(tmp_path, monkeypatch) -> None:
    test = _test_frame()
    labels = test["label"].to_numpy()
    evaluator = Evaluator(test, detectors=["loglik", "distilroberta"], k=2,
                          results_dir=tmp_path, cache_dir=tmp_path / "cache")

    def fake_probabilities(keys, texts, cache_path, desc):
        keys = list(keys)
        if desc == "clean test":
            truth = labels.astype(float)
            return {"loglik": truth, "distilroberta": truth}
        # "variants": DistilRoBERTa stays perfect, log-likelihood fails variant j=1 of row 0 only.
        rows = np.array([int(k.split("-")[0]) for k in keys])
        perfect = labels[rows].astype(float)
        fragile = perfect.copy()
        broken = [i for i, k in enumerate(keys) if k.startswith("0-h0|") and k.endswith("|1")]
        fragile[broken] = 1.0                                   # row 0 is human -> wrong
        return {"loglik": fragile, "distilroberta": perfect}

    monkeypatch.setattr(evaluator, "_probabilities", fake_probabilities)
    monkeypatch.setattr(evaluator, "gpt2_window", lambda variants: (np.ones(len(variants), dtype=bool),
                                                                   np.array([], dtype=int)))
    summary = evaluator.run_robustness()

    ft = summary[summary["detector"] == "distilroberta"]
    assert (ft["aua"] == 100.0).all()
    ll = summary[(summary["detector"] == "loglik") & (summary["op"] != "none")]
    assert ll["aua"].tolist() == [round(100 * 5 / 6, 2)] * 8                  # one of six texts fails
    assert ll["mean_variant_accuracy"].tolist() == [round(100 * 11 / 12, 2)] * 8
    for name in ("robustness.csv", "robustness_table.md", "robustness_per_sample.csv", "robustness_chart.png"):
        assert (tmp_path / name).is_file()
    md = (tmp_path / "robustness_table.md").read_text(encoding="utf-8")
    assert "| Detector | Normal | Swap 1 | Swap 2 | Drop 1 | Drop 2 | Add 1 | Add 2 | Key 1 | Key 2 |" in md
    assert "not representative" in md                                         # tiny run is flagged
    per_sample = pd.read_csv(tmp_path / "robustness_per_sample.csv")
    assert len(per_sample) == 6 * 8 * 2
    assert {"clean_correct", "all_variants_correct", "robust"} <= set(per_sample.columns)
    per_variant = pd.read_csv(tmp_path / "robustness_variants.csv.gz")
    assert len(per_variant) == 6 * 8 * 2
    assert list(per_variant.columns) == ["key", "id", "label", "op", "level", "j", "edits",
                                         "edit_in_gpt2_window", "p_ai_loglik", "p_ai_distilroberta"]
    first = per_variant.iloc[0]
    assert first["key"] == "0-h0|swap|1|0" and first["id"] == "0-h0"
    variant_text = evaluator.build_variants()["text"].iloc[0]
    assert first["edits"] == ";".join(f"{a}>{b}" for a, b in edited_words(test["text"][0], variant_text))


def test_run_robustness_aua_requires_a_correct_clean_prediction(tmp_path, monkeypatch) -> None:
    test = _test_frame()
    labels = test["label"].to_numpy()
    evaluator = Evaluator(test, detectors=["distilroberta"], k=2, results_dir=tmp_path)

    def fake_probabilities(keys, texts, cache_path, desc):
        if desc == "clean test":
            clean = labels.astype(float)
            clean[0] = 1.0                                      # row 0 (human) wrong on clean text
            return {"distilroberta": clean}
        rows = [int(key.split("-")[0]) for key in keys]
        return {"distilroberta": labels[rows].astype(float)}   # every variant right

    monkeypatch.setattr(evaluator, "_probabilities", fake_probabilities)
    summary = evaluator.run_robustness()
    attacked = summary[summary["op"] != "none"]
    normal = float(summary.loc[summary["op"] == "none", "aua"].iloc[0])
    assert normal == round(100 * 5 / 6, 2)
    assert (attacked["aua"] == normal).all()                    # not 100: row 0 was never correct
    assert (attacked["mean_variant_accuracy"] == 100.0).all()


def test_run_clean_writes_tables(tmp_path, monkeypatch) -> None:
    test = _test_frame()
    evaluator = Evaluator(test, detectors=["rank", "distilroberta"], k=1, results_dir=tmp_path,
                          zero_shot_path=tmp_path / "missing.json")
    probs = {"rank": np.full(6, 0.9), "distilroberta": test["label"].to_numpy(dtype=float)}
    monkeypatch.setattr(evaluator, "_probabilities", lambda *args, **kwargs: probs)
    table = evaluator.run_clean()
    assert table["detector"].tolist() == ["rank", "distilroberta"]
    assert table.set_index("detector").loc["distilroberta", "accuracy"] == 100.0
    assert table.set_index("detector").loc["rank", "accuracy"] == 50.0
    md = (tmp_path / "clean_metrics.md").read_text(encoding="utf-8")
    assert "| DistilRoBERTa | 100.00 |" in md
    assert "Combined verdict" not in md                                       # no zero_shot.json
    predictions = pd.read_csv(tmp_path / "clean_predictions.csv")
    assert list(predictions.columns) == ["id", "source", "label", "p_ai_rank", "p_ai_distilroberta"]


def test_run_clean_scores_the_app_verdict(tmp_path, monkeypatch) -> None:
    test = _test_frame()                                         # labels 0, 1, 0, 1, 0, 1
    (tmp_path / "zero_shot.json").write_text(json.dumps({"selected_metric": "rank"}), encoding="utf-8")
    # v2: the app pairs DistilRoBERTa with Fast-DetectGPT, whatever GPT-2 score won on validation
    evaluator = Evaluator(test, detectors=["rank", "fastdetect", "distilroberta"], k=1, results_dir=tmp_path,
                          zero_shot_path=tmp_path / "zero_shot.json")
    probs = {"rank": np.full(6, 0.99),
             "fastdetect": np.array([0.2, 0.9, 0.7, 0.8, 0.1, 0.3]),
             "distilroberta": np.array([0.1, 0.8, 0.6, 0.9, 0.2, 0.9])}
    monkeypatch.setattr(evaluator, "_probabilities", lambda *args, **kwargs: probs)
    evaluator.run_clean()
    predictions = pd.read_csv(tmp_path / "clean_predictions.csv")
    assert predictions["verdict"].tolist() == [VERDICT_HUMAN, VERDICT_AI, VERDICT_AI, VERDICT_AI,
                                               VERDICT_HUMAN, VERDICT_UNCERTAIN]
    table = pd.read_csv(tmp_path / "verdict_metrics.csv").set_index("group")
    assert table.loc["All texts", "coverage"] == round(100 * 5 / 6, 2)
    assert table.loc["Human texts", "labelled_ai"] == round(100 / 3, 2)      # row 2 falsely flagged
    md = (tmp_path / "clean_metrics.md").read_text(encoding="utf-8")
    assert "## Combined verdict of the web app" in md and "the Fast-DetectGPT" in md


def test_evaluator_rejects_unknown_detectors() -> None:
    with pytest.raises(ValueError):
        Evaluator(_test_frame(), detectors=["gltr"])


class _WhitespaceWindow:
    """Stand-in for the GPT-2 tokenizer: one token per word, a 5-token window."""

    max_tokens = 5

    def window_token_ids(self, texts):
        tokens = [t.split() for t in texts]
        return [len(t) for t in tokens], [t[: self.max_tokens] for t in tokens]


def test_gpt2_window_flags_edits_beyond_the_zero_shot_window(monkeypatch) -> None:
    test = pd.DataFrame({"id": ["0-h0", "1-c0"], "source": ["finance"] * 2, "label": [0, 1],
                         "text": ["one two three four five six seven eight", "short text here"]})
    evaluator = Evaluator(test, detectors=["loglik"], k=1)
    monkeypatch.setattr(evaluator, "_zero_shot", lambda: _WhitespaceWindow())
    variants = pd.DataFrame({"row": [0, 0, 1, 1], "text": [
        "one two three four five six sevne eight",        # edit beyond the window -> unseen
        "one tow three four five six seven eight",        # edit inside the window
        "short txet here",
        "short text here",                                # edit was a no-op
    ]})
    flags, long_rows = evaluator.gpt2_window(variants)
    assert flags.tolist() == [False, True, True, False]
    assert long_rows.tolist() == [0]
    note = evaluator._window_note(variants, flags, long_rows)
    assert "1 of 2 test texts (1 human / 0 AI)" in note and "in 1 of their 2 variants" in note
    assert evaluator._window_note(variants, flags, np.array([], dtype=int)) is None
