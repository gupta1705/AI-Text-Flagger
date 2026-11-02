"""Common detector interface (report Figure 3, ``BaseDetector``) and small runtime helpers.

``BaseDetector`` fixes the interface shared by the zero-shot and the fine-tuned detector:
``fit``, ``predict_proba`` (probability that a text is AI-generated) and ``predict``.
The helpers set random seeds, pick the compute device, make console output safe on Windows
code pages and keep a reduced run (``--limit`` etc.) from overwriting the full-size outputs.
``torch`` is imported lazily so that light modules stay light.
"""

from __future__ import annotations

import random
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from src.config import SEED, THRESHOLD

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


class BaseDetector(ABC):
    """Abstract AI-text detector: label 1 = AI-generated, label 0 = human-written."""

    #: Short machine-readable name used in result tables.
    name: str = "detector"

    @abstractmethod
    def fit(self, texts: Sequence[str], labels: Sequence[int], **kwargs) -> "BaseDetector":
        """Learn the detector's parameters from labelled texts and return ``self``."""

    @abstractmethod
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return P(AI-generated) for every text as a float array of shape ``(len(texts),)``."""

    def predict(self, texts: Sequence[str], t: float = THRESHOLD) -> np.ndarray:
        """Return hard labels: 1 (AI) when P(AI) >= ``t``, else 0 (human).

        Raises ``ValueError`` when a text has no probability (NaN - a zero-shot text with fewer
        than two tokens), because any hard label would be a guess.
        """
        proba = np.asarray(self.predict_proba(texts), dtype=float)
        if np.isnan(proba).any():
            raise ValueError(f"{int(np.isnan(proba).sum())} text(s) have no score (too short) - cannot label them")
        return (proba >= t).astype(int)


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and (if installed) PyTorch on CPU and all GPUs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard requirement of the detectors
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: "str | torch.device | None" = None) -> "torch.device":
    """Return ``device`` as a ``torch.device``; default to CUDA when available, else CPU."""
    import torch

    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def free_gpu_memory() -> None:
    """Release cached CUDA memory (no-op on CPU-only machines)."""
    try:
        import torch
    except ImportError:  # pragma: no cover
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def configure_console() -> None:
    """Never crash when printing non-ASCII text on a legacy Windows console code page."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


def refuse_reduced_run(prog: str, reasons: Sequence[str], defaulted: Sequence[tuple[str, object]]) -> bool:
    """Print why and return True when a reduced run would write into a default output folder.

    The default folders ``data/``, ``models/`` and ``results/`` hold the full-size run, which the
    README's results and the app's *Model report* come from. A run that differs from it (fewer
    rows with ``--limit``, other epochs, another ``k``, a subset of detectors) may write there only
    when the folder is named explicitly, so a quick check can never silently replace them.

    Args:
        prog: name used in the message, e.g. ``"evaluate"``.
        reasons: how the run differs from the full-size configuration (empty: it does not).
        defaulted: ``(option, default path)`` for every output the user left at its default.
    """
    if not reasons or not defaulted:
        return False
    where = ", ".join(f"{path} ({option})" for option, path in defaulted)
    options = " / ".join(option for option, _ in defaulted)
    print(f"[{prog}] not writing a reduced run ({'; '.join(reasons)}) into {where}: that folder holds the "
          f"full-size run. Choose another output with {options} (e.g. a folder under runs/), or name the "
          "default explicitly to overwrite it on purpose.", file=sys.stderr)
    return True
