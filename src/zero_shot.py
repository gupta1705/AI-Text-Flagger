"""ZeroShotDetector: GPT-2 token statistics + 1-D logistic calibration (report Algorithms 1-2, Listing 2).

Every text is truncated to 512 GPT-2 tokens and passed once through the model. For each token
after the first we record its log-probability, its rank in the predicted distribution
(1 = most likely) and the entropy of that distribution, and average them into four scores:

* ``loglik``  = mean log p(x_t | x_<t)      (AI text: higher)
* ``rank``    = mean (1 + #vocab entries with a higher probability)  (AI text: lower)
* ``logrank`` = mean log(rank)              (AI text: lower)
* ``entropy`` = mean -sum_v p_v log p_v     (AI text: lower)

One logistic regression per score, fitted on the standardised training scores, turns a score
into P(AI). The score with the best validation F1 is the one used by the web app. Calibrators
are stored as plain JSON (mean, std, coef, intercept) - no pickle.

GPT-2 statistics of the clean splits are cached in ``results/cache/gpt2_stats_<split>.csv``
(keyed by row id and a CRC of model + max tokens + text) so re-runs are fast.

CLI: ``python -m src.zero_shot [--batch-size B] [--limit N --output FILE]`` (a run on fewer rows than
the full splits must name its own ``--output``, so it never replaces ``models/zero_shot.json``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.base import BaseDetector, configure_console, free_gpu_memory, refuse_reduced_run, resolve_device, set_seed
from src.config import (
    CACHE_DIR,
    DATA_DIR,
    DETECTOR_DISPLAY,
    N_FIT,
    N_TEST,
    N_VAL,
    SEED,
    THRESHOLD,
    ZERO_SHOT_JSON,
    ZS_BATCH_SIZE,
    ZS_CACHE_CHUNK,
    ZS_MAX_TOKENS,
    ZS_METRICS,
    ZS_MODEL_NAME,
)
from src.data import HC3DataModule
from src.evaluate import metrics

METRICS = ZS_METRICS


def sequence_statistics(logits: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float, float]:
    """Mean log-likelihood, rank, log-rank and entropy of one sequence (Algorithm 1, lines 3-9).

    Args:
        logits: ``(T-1, V)`` next-token logits for positions 1..T-1 (any float dtype).
        target: ``(T-1,)`` ids of the tokens actually observed at positions 2..T.

    The log-softmax is computed in float32 for numerical safety.
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_lp = log_probs.gather(1, target[:, None]).squeeze(1)
    ranks = (log_probs > token_lp[:, None]).sum(dim=1) + 1        # 1 = most likely token
    entropy = -(log_probs.exp() * log_probs).sum(dim=1)
    ranks = ranks.float()
    return (
        token_lp.mean().item(),
        ranks.mean().item(),
        ranks.log().mean().item(),
        entropy.mean().item(),
    )


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return 0.5 * (1.0 + np.tanh(0.5 * z))


@dataclass(frozen=True)
class Calibrator:
    """1-D logistic regression ``P(AI | x) = sigmoid(coef * (x - mean) / std + intercept)``."""

    metric: str
    mean: float
    std: float
    coef: float
    intercept: float

    @classmethod
    def fit(cls, metric: str, scores: Sequence[float], labels: Sequence[int], seed: int = SEED) -> "Calibrator":
        """Standardise ``scores`` and fit a logistic regression on (score, label) pairs.

        Non-finite scores (texts with fewer than two tokens) are ignored.
        """
        x = np.asarray(scores, dtype=float)
        y = np.asarray(labels, dtype=int)
        finite = np.isfinite(x)
        x, y = x[finite], y[finite]
        if len(np.unique(y)) < 2:
            raise ValueError(f"calibrating '{metric}' needs finite scores of both classes")
        mean = float(x.mean())
        std = float(x.std())
        std = std if std > 1e-12 else 1.0
        model = LogisticRegression(random_state=seed).fit(((x - mean) / std)[:, None], y)
        return cls(metric, mean, std, float(model.coef_[0, 0]), float(model.intercept_[0]))

    def predict_proba(self, scores: Sequence[float]) -> np.ndarray:
        """P(AI) for every score.

        A non-finite score (a text with fewer than two GPT-2 tokens has no statistics) gives NaN:
        the detector has no evidence, and any number - even 0.5, which ``p >= 0.5`` would count
        as an AI vote - would pretend otherwise. The app reports such a text as *Uncertain*.
        """
        x = np.asarray(scores, dtype=float)
        with np.errstate(invalid="ignore"):
            p = _sigmoid(self.coef * (x - self.mean) / self.std + self.intercept)
        return np.where(np.isfinite(x), p, np.nan)

    @property
    def threshold_score(self) -> float | None:
        """Raw score at which P(AI) = 0.5 (``None`` if the slope is zero)."""
        if abs(self.coef) < 1e-12:
            return None
        return self.mean - self.intercept * self.std / self.coef

    def to_dict(self) -> dict:
        return {**asdict(self), "threshold_score": self.threshold_score}

    @classmethod
    def from_dict(cls, data: dict) -> "Calibrator":
        return cls(**{k: data[k] for k in ("metric", "mean", "std", "coef", "intercept")})


class ZeroShotDetector(BaseDetector):
    """GPT-2 zero-shot detector with one calibrator per token statistic.

    Args:
        metric: statistic used by ``predict_proba`` (set automatically by ``select_metric``).
        model_name: Hugging Face id of the surrogate language model.
        max_tokens: truncation length in tokens.
        batch_size: texts per forward pass.
        device: ``"cuda"``/``"cpu"``; default CUDA if available. The model runs in fp16 on CUDA.
    """

    name = "zero_shot"

    def __init__(
        self,
        metric: str | None = None,
        model_name: str = ZS_MODEL_NAME,
        max_tokens: int = ZS_MAX_TOKENS,
        batch_size: int = ZS_BATCH_SIZE,
        device: str | None = None,
    ) -> None:
        if metric is not None and metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        self.metric = metric
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.device = resolve_device(device)
        self.calibrators: dict[str, Calibrator] = {}
        self.val_metrics: dict[str, dict] = {}
        self.fit_info: dict = {}
        self.tokenizer = None
        self.lm = None

    # ------------------------------------------------------------------ model
    def load_tokenizer(self) -> None:
        """Load the GPT-2 tokenizer once (cheap; does not load the language model)."""
        if self.tokenizer is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token          # GPT-2 has no pad token
        self.tokenizer.padding_side = "right"

    def window_token_ids(self, texts: Sequence[str]) -> tuple[list[int], list[list[int]]]:
        """Full token count and the ids GPT-2 actually reads (first ``max_tokens``) per text."""
        self.load_tokenizer()
        ids = self.tokenizer(list(texts), verbose=False)["input_ids"]
        return [len(i) for i in ids], [i[:self.max_tokens] for i in ids]

    def load_model(self) -> None:
        """Load GPT-2 once (fp16 on CUDA, fp32 on CPU) in evaluation mode."""
        if self.lm is not None:
            return
        self.load_tokenizer()
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.lm = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype).to(self.device).eval()

    def release(self) -> None:
        """Free the language model (calibrators are kept)."""
        self.lm = None
        self.tokenizer = None
        free_gpu_memory()

    @property
    def fingerprint(self) -> str:
        """Identifies the statistics' configuration; part of every cache key."""
        return f"{self.model_name}|{self.max_tokens}"

    # ------------------------------------------------------------------ statistics
    @torch.inference_mode()
    def token_statistics(self, text: str) -> dict[str, float]:
        """Mean log-likelihood, rank, log-rank and entropy of ``text`` under GPT-2 (Algorithm 1)."""
        return self.token_statistics_batch([text]).iloc[0].to_dict()

    @torch.inference_mode()
    def token_statistics_batch(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
        on_batch: Callable[[int], None] | None = None,
    ) -> pd.DataFrame:
        """Statistics for many texts; returns a frame with columns :data:`METRICS` in input order.

        Texts are sorted by length (longest first) to minimise padding, padded on the right with
        the EOS token and truncated to ``max_tokens``. Per-sequence statistics use only real
        tokens (attention mask) and are computed one sequence at a time in float32 to bound
        memory (vocabulary 50,257 x up to 511 positions). Texts with fewer than two tokens get
        NaN. ``on_batch(n)`` is called after each batch of ``n`` texts (progress reporting).
        """
        self.load_model()
        texts = list(texts)
        size = batch_size or self.batch_size
        out = np.full((len(texts), len(METRICS)), np.nan)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        for start in range(0, len(order), size):
            idx = order[start:start + size]
            enc = self.tokenizer(
                [texts[i] for i in idx], return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_tokens,
            )
            input_ids = enc["input_ids"].to(self.device)
            mask = enc["attention_mask"].to(self.device)
            if input_ids.shape[1] >= 2:
                logits = self.lm(input_ids=input_ids, attention_mask=mask).logits
                lengths = enc["attention_mask"].sum(dim=1).tolist()       # real tokens per row
                for row, (i, length) in enumerate(zip(idx, lengths)):
                    if length >= 2:
                        out[i] = sequence_statistics(logits[row, :length - 1], input_ids[row, 1:length])
                del logits
            if on_batch is not None:
                on_batch(len(idx))
        return pd.DataFrame(out, columns=list(METRICS))

    # ------------------------------------------------------------------ calibration
    def fit(self, texts: Sequence[str], labels: Sequence[int], **kwargs) -> "ZeroShotDetector":
        """Compute statistics of ``texts`` and fit one calibrator per metric (Algorithm 2)."""
        return self.fit_stats(self.token_statistics_batch(texts), labels)

    def fit_stats(self, stats: pd.DataFrame, labels: Sequence[int]) -> "ZeroShotDetector":
        """Fit one calibrator per metric from precomputed statistics."""
        labels = np.asarray(labels, dtype=int)
        self.calibrators = {m: Calibrator.fit(m, stats[m].to_numpy(), labels) for m in METRICS}
        self.fit_info = {"n_fit": int(len(labels)), "n_fit_ai": int(labels.sum())}
        if self.metric is None:
            self.metric = METRICS[0]
        return self

    def select_metric(self, val_stats: pd.DataFrame, val_labels: Sequence[int]) -> str:
        """Choose the metric with the best validation F1 (AI positive); ties keep METRICS order."""
        self._require_fitted()
        self.val_metrics = {
            m: metrics(val_labels, self.calibrators[m].predict_proba(val_stats[m].to_numpy()))
            for m in METRICS
        }
        self.metric = max(METRICS, key=lambda m: (self.val_metrics[m]["f1"], -METRICS.index(m)))
        self.fit_info["n_val"] = int(len(val_labels))
        return self.metric

    def proba_from_stats(self, stats: pd.DataFrame, metric: str | None = None) -> np.ndarray:
        """P(AI) from precomputed statistics using ``metric`` (default: the selected metric)."""
        self._require_fitted()
        metric = metric or self.metric
        return self.calibrators[metric].predict_proba(stats[metric].to_numpy())

    def predict_proba(self, texts: Sequence[str], metric: str | None = None) -> np.ndarray:
        """Probability that each text is AI-generated."""
        return self.proba_from_stats(self.token_statistics_batch(texts), metric)

    def _require_fitted(self) -> None:
        if not self.calibrators:
            raise RuntimeError("ZeroShotDetector is not fitted - call fit()/fit_stats() or load()")

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path = ZERO_SHOT_JSON) -> Path:
        """Write calibrators, validation metrics and the chosen metric to JSON."""
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "selected_metric": self.metric,
            "selection_rule": "best validation F1 (AI = positive class)",
            "threshold": THRESHOLD,
            "calibrators": {m: c.to_dict() for m, c in self.calibrators.items()},
            "val_metrics": self.val_metrics,
            **self.fit_info,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path = ZERO_SHOT_JSON, device: str | None = None,
             batch_size: int = ZS_BATCH_SIZE) -> "ZeroShotDetector":
        """Restore a detector saved with :meth:`save` (GPT-2 itself is loaded lazily)."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found - run `python -m src.zero_shot` first")
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        detector = cls(metric=payload["selected_metric"], model_name=payload["model_name"],
                       max_tokens=payload["max_tokens"], batch_size=batch_size, device=device)
        detector.calibrators = {m: Calibrator.from_dict(c) for m, c in payload["calibrators"].items()}
        detector.val_metrics = payload.get("val_metrics", {})
        return detector


# ---------------------------------------------------------------------- statistics cache
CACHE_COLUMNS = ("key", "crc", *METRICS)


def _text_crc(fingerprint: str, text: str) -> int:
    return zlib.crc32(f"{fingerprint}|{text}".encode("utf-8"))


def read_statistics_cache(cache_path: str | Path) -> dict[str, tuple[int, np.ndarray]]:
    """Load a statistics cache as ``{key: (crc, statistics)}``, repairing what a crash can leave.

    * A run killed while appending can leave a cut-off last line. It is removed from the file
      (so the next append starts on a new line) and its text is scored again.
    * Rows whose crc is not a number, or whose statistics are only partly present, are ignored.
      Rows with *all* statistics missing are valid (texts with fewer than two tokens).
    * A file without the expected header is not a statistics cache of this version: it is
      deleted and rebuilt.
    """
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        return {}
    raw = cache_path.read_bytes()
    complete = raw[:raw.rfind(b"\n") + 1]
    if complete != raw:
        print(f"[zero-shot] {cache_path.name}: removed a cut-off last line left by an interrupted run")
        cache_path.write_bytes(complete)
    if not complete:                       # nothing complete was written (not even the header)
        cache_path.unlink()
        return {}
    header = complete.split(b"\n", 1)[0].decode("utf-8", errors="replace").strip()
    if header.split(",") != list(CACHE_COLUMNS):
        print(f"[zero-shot] {cache_path.name}: unexpected header {header[:60]!r}; rebuilding the cache")
        cache_path.unlink()
        return {}
    cached = pd.read_csv(cache_path, dtype={"key": str}, on_bad_lines="skip", encoding="utf-8")
    crc = pd.to_numeric(cached["crc"], errors="coerce")
    values = cached[list(METRICS)].apply(pd.to_numeric, errors="coerce")
    n_missing = values.isna().sum(axis=1)
    valid = cached["key"].notna() & crc.notna() & ((n_missing == 0) | (n_missing == len(METRICS)))
    keys = cached.loc[valid, "key"].tolist()
    crcs = crc[valid].astype("int64").tolist()
    rows = values[valid].to_numpy(dtype=float)
    return {k: (c, v) for k, c, v in zip(keys, crcs, rows)}          # later rows win


def cached_statistics(
    detector: ZeroShotDetector,
    keys: Sequence[str],
    texts: Sequence[str],
    cache_path: str | Path,
    chunk_size: int = ZS_CACHE_CHUNK,
    desc: str = "GPT-2 statistics",
) -> pd.DataFrame:
    """GPT-2 statistics for ``texts``, reusing and extending a CSV cache keyed by ``keys``.

    A cached row is reused only if its CRC of (model, max tokens, text) still matches. New rows
    are appended after every ``chunk_size`` texts, so an interrupted run resumes where it stopped
    (see :func:`read_statistics_cache` for how a half-written last line is handled).
    Returns a frame indexed by ``keys`` with columns :data:`METRICS`.
    """
    keys, texts = [str(k) for k in keys], list(texts)
    cache_path = Path(cache_path)
    crcs = [_text_crc(detector.fingerprint, t) for t in texts]
    known = read_statistics_cache(cache_path)
    missing = [i for i, (k, c) in enumerate(zip(keys, crcs)) if k not in known or known[k][0] != c]
    print(f"[zero-shot] {desc}: {len(texts) - len(missing)} cached, {len(missing)} to compute")
    if missing:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with tqdm(total=len(missing), desc=desc, unit="text") as bar:
            for start in range(0, len(missing), chunk_size):
                part = missing[start:start + chunk_size]
                stats = detector.token_statistics_batch([texts[i] for i in part], on_batch=bar.update)
                block = pd.DataFrame({"key": [keys[i] for i in part], "crc": [crcs[i] for i in part]})
                block = pd.concat([block, stats], axis=1)
                block.to_csv(cache_path, mode="a", header=not cache_path.is_file(), index=False, encoding="utf-8")
                for row, i in enumerate(part):
                    known[keys[i]] = (crcs[i], stats.iloc[row].to_numpy(dtype=float))
        rate = len(missing) / max(time.perf_counter() - started, 1e-9)
        print(f"[zero-shot] {desc}: {rate:.1f} texts/s")
    return pd.DataFrame([known[k][1] for k in keys], columns=list(METRICS), index=pd.Index(keys, name="key"))


# ---------------------------------------------------------------------- CLI
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.zero_shot",
        description="Compute GPT-2 token statistics, fit one calibrator per statistic, select the "
                    "best by validation F1 and save models/zero_shot.json.",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="use at most N rows per split (balanced); needs its own --output")
    parser.add_argument("--batch-size", type=int, default=ZS_BATCH_SIZE, help="texts per GPT-2 pass (default: %(default)s)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="folder with the split CSVs")
    parser.add_argument("--output", type=Path, default=None, help=f"calibrator JSON (default: {ZERO_SHOT_JSON})")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR,
                        help="GPT-2 statistics cache (default: %(default)s)")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    configure_console()
    args = build_arg_parser().parse_args(argv)
    set_seed(SEED)
    try:
        frames = {split: HC3DataModule.load_split(split, args.data_dir, args.limit) for split in ("train", "val", "test")}
    except FileNotFoundError as error:
        print(f"[zero-shot] {error}", file=sys.stderr)
        return 1
    full = {"train": N_FIT, "val": N_VAL, "test": N_TEST}
    reasons = [f"{len(frames[s])} {s} texts instead of {n:,}" for s, n in full.items() if len(frames[s]) < n]
    if refuse_reduced_run("zero-shot", reasons, [] if args.output else [("--output", ZERO_SHOT_JSON)]):
        return 2
    detector = ZeroShotDetector(batch_size=args.batch_size, device=args.device)
    print(f"[zero-shot] model {detector.model_name} on {detector.device}, max {detector.max_tokens} tokens")
    stats = {}
    for split, frame in frames.items():
        stats[split] = cached_statistics(
            detector, frame["id"], frame["text"],
            args.cache_dir / f"gpt2_stats_{split}.csv", desc=f"{split} ({len(frame)} texts)",
        )
    detector.fit_stats(stats["train"], frames["train"]["label"])
    chosen = detector.select_metric(stats["val"], frames["val"]["label"])
    path = detector.save(args.output or ZERO_SHOT_JSON)
    print(f"\n{'metric':<16}{'val acc':>9}{'val F1':>9}{'test acc':>10}{'test F1':>9}{'test AUROC':>12}")
    for m in METRICS:
        test = metrics(frames["test"]["label"], detector.proba_from_stats(stats["test"], m))
        val = detector.val_metrics[m]
        marker = "  <- app" if m == chosen else ""
        print(f"{DETECTOR_DISPLAY[m]:<16}{val['accuracy']:>9.4f}{val['f1']:>9.4f}"
              f"{test['accuracy']:>10.4f}{test['f1']:>9.4f}{test['auroc']:>12.4f}{marker}")
    print(f"\n[zero-shot] selected metric: {chosen}; saved {path}")
    detector.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
