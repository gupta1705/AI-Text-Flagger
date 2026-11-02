"""Tests for src.data.HC3DataModule on a synthetic HC3-style corpus (see conftest.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import DATA_DIR, SUMMARY_FILENAME
from src.data import HC3DataModule

SIZES = {"n_fit": 40, "n_val": 8, "n_test": 16}


def _module(hc3_file: Path, tmp_path: Path, **kwargs) -> HC3DataModule:
    params = {**SIZES, **kwargs}
    return HC3DataModule(hc3_path=hc3_file, data_dir=tmp_path / "data", cached_path=None, **params)


def test_load_flattens_and_normalises(hc3_file: Path, tmp_path: Path) -> None:
    frame = _module(hc3_file, tmp_path).load()
    assert list(frame.columns) == ["id", "question_id", "source", "label", "text", "word_count"]
    assert len(frame) == 163 + 80                      # human + ChatGPT answers
    assert set(frame["label"]) == {0, 1}
    assert frame["id"].is_unique
    first = frame.iloc[0]
    assert first["id"] == "0-h0" and first["question_id"] == 0 and first["source"] == "reddit_eli5"
    assert first["text"].startswith("Well, it isn't simple. Question 0 answer (first):")
    assert not frame["text"].str.contains("\n", regex=False).any()


def test_clean_removes_empty_short_and_duplicates(hc3_file: Path, tmp_path: Path) -> None:
    module = _module(hc3_file, tmp_path)
    cleaned = module.clean(module.load())
    assert module.removed == {"empty": 1, "too_short": 1, "duplicate": 1}
    assert len(cleaned) == 240
    assert (cleaned["word_count"] >= 20).all()
    assert cleaned["text"].is_unique
    duplicate_rows = cleaned[cleaned["text"].str.startswith("Shared duplicate")]
    assert duplicate_rows["id"].tolist() == ["1-h2"]   # first occurrence kept


def test_splits_are_balanced_disjoint_and_paired(hc3_file: Path, tmp_path: Path) -> None:
    module = _module(hc3_file, tmp_path)
    splits = module.make_splits(module.clean(module.load()))
    expected = {"train": 40, "val": 8, "test": 16}
    question_sets = {}
    for name, frame in splits.items():
        assert len(frame) == expected[name]
        assert (frame["label"] == 1).sum() == (frame["label"] == 0).sum()
        # one human and one AI answer per question when possible
        per_question = frame.groupby("question_id")["label"].agg(["count", "sum"])
        assert (per_question["count"] == 2).all() and (per_question["sum"] == 1).all()
        question_sets[name] = set(frame["question_id"])
    assert not question_sets["train"] & question_sets["val"]
    assert not question_sets["train"] & question_sets["test"]
    assert not question_sets["val"] & question_sets["test"]


def _splits(module: HC3DataModule) -> dict[str, pd.DataFrame]:
    return module.make_splits(module.clean(module.load()))


def test_splits_are_deterministic(hc3_file: Path, tmp_path: Path) -> None:
    a = _splits(_module(hc3_file, tmp_path))
    b = _splits(_module(hc3_file, tmp_path))
    for name in a:
        assert a[name]["id"].tolist() == b[name]["id"].tolist()
    c = _splits(_module(hc3_file, tmp_path, seed=7))
    assert a["test"]["id"].tolist() != c["test"]["id"].tolist()


def test_save_writes_csvs_and_summary(hc3_file: Path, tmp_path: Path) -> None:
    module = _module(hc3_file, tmp_path)
    summary = module.save(module.make_splits(module.clean(module.load())))
    data_dir = tmp_path / "data"
    for name, size in (("train", 40), ("val", 8), ("test", 16)):
        frame = HC3DataModule.load_split(name, data_dir)
        assert len(frame) == size
    on_disk = json.loads((data_dir / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(summary))
    assert summary["question_sets_disjoint"] is True
    assert summary["removed"] == {"empty": 1, "too_short": 1, "duplicate": 1}
    assert summary["splits"]["train"]["by_class"] == {"human": 20, "ai": 20}
    assert sum(v["human"] + v["ai"] for v in summary["splits"]["test"]["by_source"].values()) == 16
    # The raw corpus has the artefacts; after normalisation both classes are (near) free of them.
    raw_human = summary["artefact_rates_raw_all_answers"]["human"]
    assert raw_human["space_dot_space"] > 0.5 and raw_human["url_placeholder"] > 0.4
    assert raw_human["spaced_single_quote"] > 0.4
    for key in ("artefact_rates_after_normalisation", "artefact_rates_after_normalisation_all_clean_answers"):
        for cls in ("human", "ai"):
            assert summary[key][cls].keys() == raw_human.keys()
            for rate in summary[key][cls].values():
                assert rate < 0.02
    assert set(summary["markup_rates_after_normalisation"]["human"]) == {"asterisk", "quote_marker", "double_hyphen"}


def test_limit_caps_every_split(hc3_file: Path, tmp_path: Path) -> None:
    module = _module(hc3_file, tmp_path, limit=6)
    assert module.sizes == {"train": 6, "val": 6, "test": 6}
    splits = module.make_splits(module.clean(module.load()))
    assert all(len(frame) == 6 for frame in splits.values())


def test_load_split_limit_is_balanced(hc3_file: Path, tmp_path: Path) -> None:
    module = _module(hc3_file, tmp_path)
    module.save(module.make_splits(module.clean(module.load())))
    frame = HC3DataModule.load_split("train", tmp_path / "data", limit=10)
    assert len(frame) == 10 and frame["label"].sum() == 5
    with pytest.raises(FileNotFoundError):
        HC3DataModule.load_split("missing", tmp_path / "data")


def test_pool_too_small_raises(hc3_file: Path, tmp_path: Path) -> None:
    module = _module(hc3_file, tmp_path, n_val=40)
    with pytest.raises(ValueError, match="split 'val'"):
        module.make_splits(module.clean(module.load()))


def test_resolve_source_fallbacks(hc3_file: Path, tmp_path: Path) -> None:
    missing = tmp_path / "nowhere" / "all.jsonl"
    cached = HC3DataModule(hc3_path=missing, cached_path=hc3_file)
    assert cached.resolve_source() == hc3_file
    # Neither file exists -> "download" (a local file:// URL, so no network is needed).
    downloaded = HC3DataModule(hc3_path=missing, cached_path=None, url=hc3_file.as_uri())
    assert downloaded.resolve_source() == missing
    assert missing.read_bytes() == hc3_file.read_bytes()


@pytest.mark.skipif(not (DATA_DIR / SUMMARY_FILENAME).is_file(), reason="pipeline data not built yet")
def test_real_summary_has_no_class_specific_artefacts() -> None:
    summary = json.loads((DATA_DIR / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["question_sets_disjoint"] is True
    # Whole cleaned corpus (~74k answers): always a meaningful estimate.
    for cls in ("human", "ai"):
        for rate in summary["artefact_rates_after_normalisation_all_clean_answers"][cls].values():
            assert rate < 0.02
    # The split rows: only meaningful at full size (a --limit 64 smoke run has 96 texts per class).
    if summary["limit"] is None:
        for cls in ("human", "ai"):
            for rate in summary["artefact_rates_after_normalisation"][cls].values():
                assert rate < 0.02
    splits = {name: pd.read_csv(DATA_DIR / f"{name}.csv") for name in ("train", "val", "test")}
    assert not set(splits["train"]["question_id"]) & set(splits["test"]["question_id"])
