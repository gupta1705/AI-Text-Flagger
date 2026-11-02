"""Tests for the Fast-DetectGPT criterion, calibration, persistence and score cache (no language model)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src import fast_detect as fast_detect_module
from src.config import FD_KEY
from src.fast_detect import FastDetectGPTDetector, cached_scores, fast_detect_criterion


def _brute_force(logits: torch.Tensor, target: torch.Tensor) -> float:
    """The criterion written out token by token with plain Python floats."""
    ll_sum = mu_sum = var_sum = 0.0
    for row, tok in zip(logits.tolist(), target.tolist()):
        z = max(row)
        norm = math.log(sum(math.exp(v - z) for v in row)) + z
        logp = [v - norm for v in row]
        p = [math.exp(v) for v in logp]
        mu = sum(pi * li for pi, li in zip(p, logp))
        ll_sum += logp[tok]
        mu_sum += mu
        var_sum += sum(pi * li * li for pi, li in zip(p, logp)) - mu * mu
    return (ll_sum - mu_sum) / math.sqrt(var_sum)


def test_criterion_matches_the_formula() -> None:
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(7, 13, generator=gen) * 2
    target = torch.randint(0, 13, (7,), generator=gen)
    assert fast_detect_criterion(logits, target) == pytest.approx(_brute_force(logits, target), rel=1e-5)


def test_likely_tokens_score_higher_than_unlikely_ones() -> None:
    gen = torch.Generator().manual_seed(1)
    logits = torch.randn(20, 50, generator=gen) * 3
    top = logits.argmax(dim=-1)
    bottom = logits.argmin(dim=-1)
    assert fast_detect_criterion(logits, top) > 0 > fast_detect_criterion(logits, bottom)


def test_criterion_is_shift_invariant_and_accepts_fp16() -> None:
    logits = torch.randn(6, 9, generator=torch.Generator().manual_seed(2))
    target = torch.tensor([0, 1, 2, 3, 4, 5])
    base = fast_detect_criterion(logits, target)
    assert fast_detect_criterion(logits + 5.0, target) == pytest.approx(base, abs=1e-5)
    assert fast_detect_criterion(logits.half(), target) == pytest.approx(base, abs=2e-2)


def _calibrated(tmp_path=None) -> FastDetectGPTDetector:
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1], 100)
    scores = np.where(labels == 1, rng.normal(2.0, 0.7, 200), rng.normal(0.0, 0.7, 200))
    det = FastDetectGPTDetector(model_name="dummy/model", max_tokens=64, device="cpu")
    return det.fit_scores(scores, labels)


def test_calibration_is_monotonic_and_requires_fitting() -> None:
    det = _calibrated()
    probs = det.proba_from_scores(np.array([-1.0, 0.5, 1.0, 3.0]))
    assert np.all(np.diff(probs) > 0) and probs[0] < 0.5 < probs[-1]
    assert det.calibrator.metric == FD_KEY
    with pytest.raises(RuntimeError, match="not calibrated"):
        FastDetectGPTDetector(model_name="dummy/model", device="cpu").proba_from_scores([1.0])


def test_save_and_load_round_trip_without_loading_the_model(tmp_path) -> None:
    det = _calibrated()
    det.val_metrics = {"accuracy": 0.9}
    path = det.save(tmp_path / "fast_detect.json", {"hc3_test_metrics": {"auroc": 0.95}})
    loaded = FastDetectGPTDetector.load(path, device="cpu")
    assert loaded.lm is None                                         # the LM is loaded lazily
    assert (loaded.model_name, loaded.max_tokens, loaded.val_metrics) == ("dummy/model", 64, {"accuracy": 0.9})
    scores = np.array([-0.5, 1.0, 2.5])
    np.testing.assert_allclose(loaded.proba_from_scores(scores), det.proba_from_scores(scores))
    assert loaded.fingerprint == "fastdetect|dummy/model|64"


class _CountingDetector(FastDetectGPTDetector):
    def __init__(self, max_tokens: int = 64) -> None:
        super().__init__(model_name="dummy/model", max_tokens=max_tokens, device="cpu")
        self.scored: list[str] = []

    def scores(self, texts, on_batch=None):
        texts = list(texts)
        self.scored += texts
        if on_batch is not None:
            on_batch(len(texts))
        return np.array([float(len(t)) for t in texts])


def test_score_cache_reuses_and_invalidates(tmp_path, capsys) -> None:
    cache = tmp_path / "cache" / "fd_scores_test.csv"
    det = _CountingDetector()
    first = cached_scores(det, ["a", "b"], ["xx", "yyy"], cache)
    assert first.tolist() == [2.0, 3.0] and det.scored == ["xx", "yyy"]
    det.scored.clear()
    again = cached_scores(det, ["a", "b", "c"], ["xx", "changed", "z"], cache)
    assert again.tolist() == [2.0, 7.0, 1.0]
    assert det.scored == ["changed", "z"]                            # "a" came from the cache
    other = _CountingDetector(max_tokens=128)                       # other settings -> recomputed
    cached_scores(other, ["a"], ["xx"], cache.with_name("other.csv"))
    cache.with_name("other.csv").write_bytes(cache.read_bytes())
    cached_scores(other, ["b"], ["changed"], cache.with_name("other.csv"))
    assert other.scored == ["xx", "changed"]
    # A half-written last line (interrupted run) is dropped, not parsed.
    with open(cache, "ab") as handle:
        handle.write(b"d,123")
    det.scored.clear()
    assert cached_scores(det, ["a", "b"], ["xx", "changed"], cache).tolist() == [2.0, 7.0] and det.scored == []


def test_cli_refuses_a_reduced_run_into_the_default_model_file(tmp_path, monkeypatch, capsys) -> None:
    import pandas as pd

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for split in ("train", "val", "test"):
        pd.DataFrame({"id": [f"{split}-0", f"{split}-1"], "question_id": [1, 2], "source": ["x", "x"],
                      "label": [0, 1], "text": ["human text here.", "ai text here."],
                      "word_count": [3, 3]}).to_csv(data_dir / f"{split}.csv", index=False)
    default = tmp_path / "models" / "fast_detect.json"
    monkeypatch.setattr(fast_detect_module, "FAST_DETECT_JSON", default)
    monkeypatch.setattr(fast_detect_module, "FastDetectGPTDetector", None)     # no model may load
    assert fast_detect_module.main(["--data-dir", str(data_dir), "--cache-dir", str(tmp_path / "c")]) == 2
    assert "--output" in capsys.readouterr().err and not default.exists()
    assert fast_detect_module.main(["--data-dir", str(tmp_path / "missing")]) == 1
