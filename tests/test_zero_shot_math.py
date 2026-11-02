"""Tests for the zero-shot maths (Algorithm 1 statistics, Algorithm 2 calibration) without GPT-2."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from src.zero_shot import (
    METRICS,
    Calibrator,
    ZeroShotDetector,
    cached_statistics,
    read_statistics_cache,
    sequence_statistics,
)


def test_sequence_statistics_on_hand_made_distributions() -> None:
    probs = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])
    target = torch.tensor([1, 2])
    loglik, rank, logrank, entropy = sequence_statistics(probs.log(), target)
    assert loglik == pytest.approx((math.log(0.3) + math.log(0.8)) / 2, abs=1e-6)
    assert rank == pytest.approx((2 + 1) / 2)                     # 0.3 has one better token
    assert logrank == pytest.approx((math.log(2) + math.log(1)) / 2, abs=1e-6)
    h1 = -sum(p * math.log(p) for p in (0.5, 0.3, 0.2))
    h2 = -sum(p * math.log(p) for p in (0.1, 0.1, 0.8))
    assert entropy == pytest.approx((h1 + h2) / 2, abs=1e-6)


def test_rank_counts_only_strictly_more_likely_tokens() -> None:
    logits = torch.tensor([[1.0, 1.0, 0.0]])           # tie between token 0 and 1
    assert sequence_statistics(logits, torch.tensor([1]))[1] == 1.0
    assert sequence_statistics(logits, torch.tensor([2]))[1] == 3.0


def test_statistics_are_shift_invariant_and_accept_fp16_logits() -> None:
    logits = torch.randn(5, 11, generator=torch.Generator().manual_seed(0))
    target = torch.tensor([0, 3, 5, 10, 2])
    assert sequence_statistics(logits, target) == pytest.approx(sequence_statistics(logits + 7.0, target), abs=1e-5)
    half = sequence_statistics(logits.half(), target)          # GPT-2 runs in fp16 on CUDA
    assert all(math.isfinite(v) for v in half)
    assert half[0] == pytest.approx(sequence_statistics(logits, target)[0], abs=1e-2)


def _separable(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    x = np.where(y == 1, rng.normal(-2.0, 0.5, n), rng.normal(-3.5, 0.5, n))   # AI: higher log-lik
    return x, y


def test_calibrator_matches_sklearn_on_standardised_scores() -> None:
    x, y = _separable()
    cal = Calibrator.fit("loglik", x, y)
    z = ((x - x.mean()) / x.std())[:, None]
    reference = LogisticRegression(random_state=42).fit(z, y).predict_proba(z)[:, 1]
    np.testing.assert_allclose(cal.predict_proba(x), reference, atol=1e-9)
    assert cal.coef > 0                                    # higher score -> more likely AI
    assert cal.predict_proba([cal.threshold_score])[0] == pytest.approx(0.5)


def test_calibrator_learns_negative_slope_for_rank_like_scores() -> None:
    x, y = _separable()
    cal = Calibrator.fit("logrank", -x, y)
    assert cal.coef < 0
    assert cal.predict_proba([-5.0])[0] > 0.5 > cal.predict_proba([5.0])[0]


def test_calibrator_nan_handling_and_json_round_trip() -> None:
    x, y = _separable()
    x_nan = np.append(x, np.nan)
    cal = Calibrator.fit("entropy", x_nan, np.append(y, 1))      # NaN row ignored when fitting
    assert np.isnan(cal.predict_proba([np.nan])[0])               # no evidence -> no probability
    assert cal.predict_proba([np.nan, 0.0])[1] == pytest.approx(cal.predict_proba([0.0])[0])
    restored = Calibrator.from_dict(json.loads(json.dumps(cal.to_dict())))
    assert restored == cal
    with pytest.raises(ValueError):
        Calibrator.fit("rank", [1.0, 2.0], [1, 1])


def _stats_frame(seed: int = 1) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], 50)
    stats = pd.DataFrame({
        "loglik": rng.normal(0, 1, 100),                             # uninformative
        "rank": rng.normal(0, 1, 100),                               # uninformative
        "logrank": np.where(y == 1, 1.0, 3.0) + rng.normal(0, 0.1, 100),   # separable, lower = AI
        "entropy": rng.normal(0, 1, 100),
    })
    return stats, y


def test_fit_select_save_load_round_trip(tmp_path) -> None:
    stats, y = _stats_frame()
    detector = ZeroShotDetector(device="cpu").fit_stats(stats, y)
    assert set(detector.calibrators) == set(METRICS)
    assert detector.select_metric(stats, y) == "logrank"
    assert detector.val_metrics["logrank"]["f1"] == 1.0
    path = detector.save(tmp_path / "zero_shot.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["selected_metric"] == "logrank" and payload["max_tokens"] == 512
    assert set(payload["calibrators"]) == set(METRICS)
    restored = ZeroShotDetector.load(path, device="cpu")
    assert restored.lm is None                                     # GPT-2 is loaded lazily
    np.testing.assert_allclose(restored.proba_from_stats(stats), detector.proba_from_stats(stats))
    np.testing.assert_allclose(restored.proba_from_stats(stats, "rank"), detector.proba_from_stats(stats, "rank"))


def test_unfitted_detector_raises() -> None:
    with pytest.raises(RuntimeError):
        ZeroShotDetector(device="cpu").proba_from_stats(pd.DataFrame({m: [0.0] for m in METRICS}))
    with pytest.raises(ValueError):
        ZeroShotDetector(metric="perplexity", device="cpu")


class _CountingDetector(ZeroShotDetector):
    """Stand-in for GPT-2: statistics derived from text length, calls are counted."""

    def __init__(self) -> None:
        super().__init__(device="cpu")
        self.scored: list[str] = []

    def token_statistics_batch(self, texts, batch_size=None, on_batch=None):
        texts = list(texts)
        self.scored += texts
        if on_batch:
            on_batch(len(texts))
        values = np.array([[len(t), len(t) + 1, len(t) + 2, len(t) + 3] for t in texts], dtype=float)
        return pd.DataFrame(values, columns=list(METRICS))


def test_cached_statistics_reuses_and_invalidates_rows(tmp_path) -> None:
    detector = _CountingDetector()
    cache = tmp_path / "gpt2_stats_test.csv"
    first = cached_statistics(detector, ["a", "b"], ["xx", "yyy"], cache, chunk_size=1)
    assert first.loc["b", "loglik"] == 3.0 and detector.scored == ["xx", "yyy"]
    again = cached_statistics(detector, ["b", "a"], ["yyy", "xx"], cache)
    assert detector.scored == ["xx", "yyy"]                     # nothing recomputed
    assert again.index.tolist() == ["b", "a"] and again.loc["a", "rank"] == 3.0
    changed = cached_statistics(detector, ["a", "c"], ["zzzz", "q"], cache)
    assert detector.scored[2:] == ["zzzz", "q"]                 # changed text and new key recomputed
    assert changed.loc["a", "loglik"] == 4.0


@pytest.mark.parametrize("cut", [b"c", b"c,12", b"c,2130142485,1.0,2"], ids=["key", "crc", "values"])
def test_cached_statistics_survives_a_cut_off_last_line(tmp_path, cut: bytes) -> None:
    detector = _CountingDetector()
    cache = tmp_path / "gpt2_stats_robust.csv"
    cached_statistics(detector, ["a", "b"], ["xx", "yyy"], cache)
    cache.write_bytes(cache.read_bytes() + cut)                # run killed while appending
    stats = cached_statistics(detector, ["a", "b", "c"], ["xx", "yyy", "z"], cache)
    assert detector.scored == ["xx", "yyy", "z"]               # a, b reused; c scored again
    assert stats.loc["c", "loglik"] == 1.0
    assert cache.read_bytes().endswith(b"\n")
    rows = cache.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "key,crc,loglik,rank,logrank,entropy" and len(rows) == 4
    cached_statistics(detector, ["a", "b", "c"], ["xx", "yyy", "z"], cache)
    assert len(detector.scored) == 3                           # fully cached now


def test_statistics_cache_ignores_corrupt_rows_and_keeps_nan_rows(tmp_path) -> None:
    cache = tmp_path / "cache.csv"
    cache.write_text(
        "key,crc,loglik,rank,logrank,entropy\n"
        "ok,7,1.0,2.0,3.0,4.0\n"
        "short,8,,,,\n"                     # < 2 tokens: all statistics missing -> valid
        "badcrc,x7,1.0,2.0,3.0,4.0\n"
        "partial,9,1.0,,3.0,4.0\n",
        encoding="utf-8",
    )
    known = read_statistics_cache(cache)
    assert set(known) == {"ok", "short"}
    assert known["ok"][0] == 7 and np.isnan(known["short"][1]).all()
    cache.write_text("something,else\n1,2\n", encoding="utf-8")
    assert read_statistics_cache(cache) == {} and not cache.exists()
