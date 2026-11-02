"""Tests for the app's decision rule (Algorithm 5 / Listing 4) and UI construction without models."""

from __future__ import annotations

import dataclasses
import math

import gradio as gr
import numpy as np
import pandas as pd
import pytest

from src.app import (
    NO_ZERO_SHOT_NOTE,
    SHORT_TEXT_NOTE,
    TIMING_PLACEHOLDER,
    VERDICT_AI,
    VERDICT_HUMAN,
    VERDICT_UNCERTAIN,
    DetectorApp,
    Verdict,
    agreement_text,
    decide,
    dry_run,
    probability_bars_html,
    timing_sample,
    verdict_badge_html,
    word_count_text,
)


def test_verdict_fields_follow_listing_4() -> None:
    assert [f.name for f in dataclasses.fields(Verdict)] == ["label", "p_finetuned", "p_zeroshot", "agreement", "note"]


@pytest.mark.parametrize(
    ("p_ft", "p_zs", "label", "agreement"),
    [
        (0.9, 0.8, VERDICT_AI, True),
        (0.1, 0.3, VERDICT_HUMAN, True),
        (0.9, 0.2, VERDICT_UNCERTAIN, False),
        (0.4, 0.6, VERDICT_UNCERTAIN, False),
        (0.5, 0.5, VERDICT_AI, True),          # 0.5 counts as AI (p >= 0.5)
        (0.5, 0.49, VERDICT_UNCERTAIN, False),
    ],
)
def test_decision_rule(p_ft: float, p_zs: float, label: str, agreement: bool) -> None:
    verdict = decide(p_ft, p_zs, n_words=100)
    assert verdict.label == label
    assert verdict.agreement is agreement
    assert verdict.note == ""
    assert verdict.mean_probability == pytest.approx((p_ft + p_zs) / 2)


def test_short_text_note() -> None:
    assert decide(0.9, 0.9, n_words=19).note == SHORT_TEXT_NOTE
    assert decide(0.9, 0.9, n_words=20).note == ""


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan")])
def test_invalid_probabilities_are_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        decide(bad, 0.5, n_words=50)
    if not math.isnan(bad):
        with pytest.raises(ValueError):
            decide(0.5, bad, n_words=50)


@pytest.mark.parametrize("p_ft", [0.9, 0.1, 0.5])
def test_missing_zero_shot_score_is_never_a_vote(p_ft: float) -> None:
    """Fewer than two GPT-2 tokens -> no zero-shot score -> 'Uncertain', whatever DistilRoBERTa says."""
    verdict = decide(p_ft, float("nan"), n_words=1)
    assert verdict.label == VERDICT_UNCERTAIN and verdict.agreement is False
    assert verdict.note == f"{SHORT_TEXT_NOTE} {NO_ZERO_SHOT_NOTE}"
    assert not verdict.has_zeroshot and math.isnan(verdict.mean_probability)
    assert "No zero-shot score" in verdict_badge_html(verdict)
    assert "n/a (text too short)" in probability_bars_html(verdict, "Log-Rank")
    assert "cannot be checked" in agreement_text(verdict)


class _StubDetector:
    def __init__(self, p: float) -> None:
        self.p = p
        self.seen: list[str] = []

    def predict_proba(self, texts):
        self.seen += list(texts)
        return np.array([self.p] * len(texts))


def test_detect_normalises_and_reports_missing_zero_shot_score(tmp_path, monkeypatch) -> None:
    app = DetectorApp(models_dir=tmp_path, results_dir=tmp_path, data_dir=tmp_path)
    finetuned, zeroshot = _StubDetector(0.93), _StubDetector(float("nan"))
    monkeypatch.setattr(app, "_load_models", lambda: (finetuned, zeroshot))
    verdict = app.detect("  Hi\n")
    assert finetuned.seen == zeroshot.seen == ["Hi"]
    assert verdict.label == VERDICT_UNCERTAIN and not verdict.agreement


def test_html_rendering() -> None:
    verdict = decide(0.82, 0.64, n_words=120)
    badge = verdict_badge_html(verdict)
    assert VERDICT_AI in badge and "#c53030" in badge and "73%" in badge
    bars = probability_bars_html(verdict, "Log-Rank")
    assert "P(AI) = 82%" in bars and "P(AI) = 64%" in bars and "Log-Rank" in bars
    assert "#b7791f" in verdict_badge_html(decide(0.9, 0.1, n_words=50))


def test_word_count_text() -> None:
    assert word_count_text("") == "0 words"
    assert word_count_text("one").startswith("1 word -")
    assert word_count_text(" ".join(["w"] * 25)) == "25 words"


def test_app_builds_without_models(tmp_path) -> None:
    app = DetectorApp(models_dir=tmp_path / "models", results_dir=tmp_path / "results", data_dir=tmp_path / "data")
    assert not app.models_available()
    assert app.examples() == []
    assert "No results yet" in app.report_markdown()
    assert app.chart_path() is None
    demo = app.build()
    assert isinstance(demo, gr.Blocks)
    with pytest.raises(ValueError):
        app.detect("   \n ")                     # empty input is rejected before any model loads


@pytest.mark.parametrize("content", [None, "id,question_id,source,label,text,word_count\n"],
                         ids=["missing", "header-only"])
def test_dry_run_without_test_rows_uses_the_placeholder(tmp_path, monkeypatch, content) -> None:
    if content is not None:
        (tmp_path / "test.csv").write_text(content, encoding="utf-8")
    assert timing_sample(tmp_path) == TIMING_PLACEHOLDER
    app = DetectorApp(models_dir=tmp_path, results_dir=tmp_path, data_dir=tmp_path)
    scored = []
    monkeypatch.setattr(app, "models_available", lambda: True)
    monkeypatch.setattr(app, "detect", lambda text: scored.append(text) or decide(0.8, 0.7, n_words=300))
    assert dry_run(app) == 0
    assert scored == [TIMING_PLACEHOLDER] * 2


def test_app_examples_and_report_from_files(tmp_path) -> None:
    data_dir, results_dir = tmp_path / "data", tmp_path / "results"
    data_dir.mkdir()
    results_dir.mkdir()
    long_text = " ".join(["word"] * 80)
    pd.DataFrame({
        "id": ["1-h0", "1-c0"], "question_id": [1, 1], "source": ["finance"] * 2,
        "label": [0, 1], "text": [f"human {long_text}", f"ai {long_text}"], "word_count": [81, 81],
    }).to_csv(data_dir / "test.csv", index=False)
    (results_dir / "clean_metrics.md").write_text("# Clean table", encoding="utf-8")
    app = DetectorApp(models_dir=tmp_path / "models", results_dir=results_dir, data_dir=data_dir)
    examples = app.examples()
    assert [caption for caption, _ in examples] == ["Human answer (HC3 test set)", "ChatGPT answer (HC3 test set)"]
    assert examples[0][1].startswith("human ") and examples[1][1].startswith("ai ")
    assert app.report_markdown() == "# Clean table"
    assert isinstance(app.build(), gr.Blocks)
