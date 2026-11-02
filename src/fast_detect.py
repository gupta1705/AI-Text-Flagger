"""FastDetectGPTDetector: zero-shot detection by conditional probability curvature (Bao et al., ICLR 2024).

Why (v2): the GPT-2 scores of v1 (log-likelihood etc.) only ask "are these words likely?", so rich or
unusual wording from a modern model looks human to them. Fast-DetectGPT instead asks "is the text more
likely than what the model itself would typically write in the same places?" - machine text sits near
the model's own preferred choices, human text does not. It needs no training data, so it cannot
overfit to one dataset.

Score (analytic form, sampling model = scoring model q), over the T-1 predicted tokens of a text:

    ll_t   = log q(x_t | x_<t)
    mu_t   = sum_v q(v | x_<t) log q(v | x_<t)                         (expected log-prob)
    var_t  = sum_v q(v | x_<t) (log q(v | x_<t))^2 - mu_t^2
    score  = (sum_t ll_t - sum_t mu_t) / sqrt(sum_t var_t)             (AI text: higher)

The scoring model is a 2024 base language model (``Qwen/Qwen2.5-0.5B`` by default, fp16, chosen on the
validation split - see ``config.FD_MODEL_NAME``); texts are truncated to ``FD_MAX_TOKENS`` tokens. A 1-D
logistic regression (the same :class:`src.zero_shot.Calibrator`) fitted on the training split turns the
score into P(AI). Scores are cached in ``results/cache/fd_scores_<split>.csv``.

CLI: ``python -m src.fast_detect [--model M] [--limit N --output FILE]``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.base import BaseDetector, configure_console, free_gpu_memory, refuse_reduced_run, resolve_device, set_seed
from src.config import (
    CACHE_DIR,
    DATA_DIR,
    FAST_DETECT_JSON,
    FD_BATCH_SIZE,
    FD_KEY,
    FD_MAX_TOKENS,
    FD_MODEL_NAME,
    N_FIT,
    N_TEST,
    N_VAL,
    SEED,
)
from src.data import HC3DataModule
from src.evaluate import metrics
from src.zero_shot import Calibrator

CACHE_CHUNK = 512


def fast_detect_criterion(logits: torch.Tensor, target: torch.Tensor) -> float:
    """Analytic Fast-DetectGPT criterion of one sequence.

    Args:
        logits: ``(T-1, V)`` next-token logits for positions 1..T-1.
        target: ``(T-1,)`` observed token ids at positions 2..T.
    """
    log_q = torch.log_softmax(logits.float(), dim=-1)
    q = log_q.exp()
    ll = log_q.gather(1, target[:, None]).squeeze(1)
    mu = (q * log_q).sum(dim=-1)
    var = (q * log_q.square()).sum(dim=-1) - mu.square()
    denom = var.sum().clamp_min(1e-12).sqrt()
    return float(((ll - mu).sum() / denom).item())


class FastDetectGPTDetector(BaseDetector):
    """Fast-DetectGPT score of a modern small LM + logistic calibration."""

    name = FD_KEY

    def __init__(self, model_name: str = FD_MODEL_NAME, max_tokens: int = FD_MAX_TOKENS,
                 batch_size: int = FD_BATCH_SIZE, device: str | None = None) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.device = resolve_device(device)
        self.lm = None
        self.tokenizer = None
        self.calibrator: Calibrator | None = None
        self.val_metrics: dict = {}

    # ------------------------------------------------------------------ model
    def load_model(self) -> None:
        if self.lm is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.lm = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype).to(self.device).eval()

    def release(self) -> None:
        self.lm = None
        self.tokenizer = None
        free_gpu_memory()

    @property
    def fingerprint(self) -> str:
        return f"fastdetect|{self.model_name}|{self.max_tokens}"

    # ------------------------------------------------------------------ scores
    @torch.inference_mode()
    def scores(self, texts: Sequence[str], on_batch=None) -> np.ndarray:
        """Fast-DetectGPT criterion per text (NaN for texts with fewer than two tokens)."""
        self.load_model()
        texts = list(texts)
        out = np.full(len(texts), np.nan)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            enc = self.tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True,
                                 truncation=True, max_length=self.max_tokens)
            ids = enc["input_ids"].to(self.device)
            mask = enc["attention_mask"].to(self.device)
            if ids.shape[1] >= 2:
                logits = self.lm(input_ids=ids, attention_mask=mask).logits
                for row, (i, length) in enumerate(zip(idx, enc["attention_mask"].sum(dim=1).tolist())):
                    if length >= 2:
                        out[i] = fast_detect_criterion(logits[row, :length - 1], ids[row, 1:length])
                del logits
            if on_batch is not None:
                on_batch(len(idx))
        return out

    # ------------------------------------------------------------------ calibration / prediction
    def fit(self, texts: Sequence[str], labels: Sequence[int], **kwargs) -> "FastDetectGPTDetector":
        return self.fit_scores(self.scores(texts), labels)

    def fit_scores(self, scores: Sequence[float], labels: Sequence[int]) -> "FastDetectGPTDetector":
        self.calibrator = Calibrator.fit(FD_KEY, scores, labels)
        return self

    def proba_from_scores(self, scores: Sequence[float]) -> np.ndarray:
        if self.calibrator is None:
            raise RuntimeError("Fast-DetectGPT detector is not calibrated - run `python -m src.fast_detect`")
        return self.calibrator.predict_proba(scores)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return self.proba_from_scores(self.scores(texts))

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path = FAST_DETECT_JSON, extra: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "detector": "Fast-DetectGPT (analytic criterion, Bao et al. 2024)",
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "calibrator": self.calibrator.to_dict() if self.calibrator else None,
            "val_metrics": self.val_metrics,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **(extra or {}),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path = FAST_DETECT_JSON, device: str | None = None,
             batch_size: int = FD_BATCH_SIZE) -> "FastDetectGPTDetector":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        det = cls(model_name=data["model_name"], max_tokens=data["max_tokens"], batch_size=batch_size, device=device)
        det.calibrator = Calibrator.from_dict(data["calibrator"])
        det.val_metrics = data.get("val_metrics", {})
        return det


# ---------------------------------------------------------------------- cache
def cached_scores(detector: FastDetectGPTDetector, keys: Sequence[str], texts: Sequence[str],
                  cache_path: str | Path, desc: str = "Fast-DetectGPT") -> pd.Series:
    """Scores for ``texts``, reusing/extending a CSV cache keyed by id + CRC(model, max tokens, text)."""
    keys, texts = [str(k) for k in keys], list(texts)
    cache_path = Path(cache_path)
    crcs = [zlib.crc32(f"{detector.fingerprint}|{t}".encode("utf-8")) for t in texts]
    known: dict[str, tuple[int, float]] = {}
    if cache_path.is_file():
        raw = cache_path.read_bytes()
        complete = raw[:raw.rfind(b"\n") + 1]
        if complete != raw:
            cache_path.write_bytes(complete)
        if complete:
            frame = pd.read_csv(cache_path, dtype={"key": str}, on_bad_lines="skip", encoding="utf-8")
            frame["crc"] = pd.to_numeric(frame["crc"], errors="coerce")
            frame = frame.dropna(subset=["key", "crc"])
            known = {k: (int(c), float(s)) for k, c, s in zip(frame["key"], frame["crc"], frame["score"])}
    missing = [i for i, (k, c) in enumerate(zip(keys, crcs)) if k not in known or known[k][0] != c]
    print(f"[fast-detect] {desc}: {len(texts) - len(missing)} cached, {len(missing)} to compute")
    if missing:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with tqdm(total=len(missing), desc=desc, unit="text") as bar:
            for start in range(0, len(missing), CACHE_CHUNK):
                part = missing[start:start + CACHE_CHUNK]
                values = detector.scores([texts[i] for i in part], on_batch=bar.update)
                block = pd.DataFrame({"key": [keys[i] for i in part], "crc": [crcs[i] for i in part], "score": values})
                block.to_csv(cache_path, mode="a", header=not cache_path.is_file(), index=False, encoding="utf-8")
                for row, i in enumerate(part):
                    known[keys[i]] = (crcs[i], float(values[row]))
        rate = len(missing) / max(time.perf_counter() - started, 1e-9)
        print(f"[fast-detect] {desc}: {rate:.1f} texts/s")
    return pd.Series([known[k][1] for k in keys], index=pd.Index(keys, name="key"), name="score")


# ---------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    configure_console()
    parser = argparse.ArgumentParser(prog="python -m src.fast_detect",
                                     description="Score the splits with Fast-DetectGPT, calibrate on train, save JSON.")
    parser.add_argument("--model", default=FD_MODEL_NAME, help="scoring model (default: %(default)s)")
    parser.add_argument("--max-tokens", type=int, default=FD_MAX_TOKENS)
    parser.add_argument("--batch-size", type=int, default=FD_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None, help="at most N rows per split; needs --output")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=None, help=f"default: {FAST_DETECT_JSON}")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    set_seed(SEED)
    try:
        frames = {s: HC3DataModule.load_split(s, args.data_dir, args.limit) for s in ("train", "val", "test")}
    except FileNotFoundError as error:
        print(f"[fast-detect] {error}", file=sys.stderr)
        return 1
    full = {"train": N_FIT, "val": N_VAL, "test": N_TEST}
    reasons = [f"{len(frames[s])} {s} texts instead of {n:,}" for s, n in full.items() if len(frames[s]) < n]
    if args.model != FD_MODEL_NAME or args.max_tokens != FD_MAX_TOKENS:
        reasons.append(f"non-default model/max tokens ({args.model}, {args.max_tokens})")
    if refuse_reduced_run("fast-detect", reasons, [] if args.output else [("--output", FAST_DETECT_JSON)]):
        return 2
    det = FastDetectGPTDetector(args.model, args.max_tokens, args.batch_size, args.device)
    print(f"[fast-detect] model {det.model_name} on {det.device}, max {det.max_tokens} tokens")
    scores = {s: cached_scores(det, f["id"], f["text"], args.cache_dir / f"fd_scores_{s}.csv",
                               desc=f"{s} ({len(f)} texts)") for s, f in frames.items()}
    det.fit_scores(scores["train"].to_numpy(), frames["train"]["label"])
    det.val_metrics = metrics(frames["val"]["label"], det.proba_from_scores(scores["val"].to_numpy()))
    test = metrics(frames["test"]["label"], det.proba_from_scores(scores["test"].to_numpy()))
    path = det.save(args.output or FAST_DETECT_JSON, {"hc3_test_metrics": test})
    print(f"[fast-detect] val acc {det.val_metrics['accuracy']:.4f} F1 {det.val_metrics['f1']:.4f} "
          f"AUROC {det.val_metrics['auroc']:.4f} | HC3 test acc {test['accuracy']:.4f} AUROC {test['auroc']:.4f}")
    print(f"[fast-detect] saved {path}")
    det.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
