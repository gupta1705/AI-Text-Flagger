"""Tests for the BaseDetector interface (report Figure 3): ``predict(texts, t=0.5)``."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.base import BaseDetector
from src.finetune import FineTunedDetector
from src.zero_shot import ZeroShotDetector


class _FixedDetector(BaseDetector):
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = np.array(probabilities)

    def fit(self, texts, labels, **kwargs):
        return self

    def predict_proba(self, texts):
        return self.probabilities[: len(texts)]


def test_predict_uses_threshold_t_inclusively() -> None:
    detector = _FixedDetector([0.2, 0.5, 0.7])
    assert detector.predict(["a", "b", "c"]).tolist() == [0, 1, 1]
    assert detector.predict(["a", "b", "c"], t=0.6).tolist() == [0, 0, 1]


def test_predict_refuses_texts_without_a_score() -> None:
    with pytest.raises(ValueError, match="no score"):
        _FixedDetector([0.9, float("nan")]).predict(["long text", "Hi"])


@pytest.mark.parametrize("cls", [ZeroShotDetector, FineTunedDetector])
def test_both_detectors_share_the_spec_signature(cls) -> None:
    parameters = inspect.signature(cls.predict).parameters
    assert list(parameters) == ["self", "texts", "t"] and parameters["t"].default == 0.5
