"""FineTunedDetector: DistilRoBERTa binary classifier (report Algorithm 3, Section 3.3 step 5).

Training (plain PyTorch loop, no ``Trainer``):

* ``distilroberta-base`` with a 2-class head (0 = human, 1 = AI);
* AdamW (lr 2e-5, weight decay 0.01), linear schedule with 10 % warm-up, 3 epochs, batch 8,
  optional gradient accumulation;
* addition to Algorithm 3: the gradient norm is clipped at ``FT_GRAD_CLIP_NORM`` = 1.0 before each
  optimiser step (Hugging Face Trainer's default; ``--grad-clip 0`` turns it off);
* sequences truncated to the first 256 tokens; fp16 autocast + GradScaler on CUDA; seed 42;
* after each epoch the validation F1 (AI = positive) is computed and the best checkpoint is saved
  to ``models/distilroberta/`` together with ``training_log.json``.

Inference (``predict_proba``): each text is split into consecutive 256-token chunks (at most 8),
the chunks are scored in length-sorted batches and P(AI) is averaged over the chunks of a text.
When a text needs more than one chunk, the last chunk is aligned to the end of the text (it
overlaps the previous one) so that every chunk is a full 256-token window like the training
inputs; otherwise a tail of a few tokens would get the same weight as a full chunk.

CLI: ``python -m src.finetune [--epochs 3] [--batch-size 8] [--limit N --output-dir DIR]`` (a run on fewer
rows or with other epochs than the full-size run must name its own ``--output-dir``, so it never
replaces ``models/distilroberta/``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.base import BaseDetector, configure_console, free_gpu_memory, refuse_reduced_run, resolve_device, set_seed
from src.config import (
    CLASS_NAMES,
    DATA_DIR,
    FINETUNED_DIR,
    FT_BATCH_SIZE,
    FT_EPOCHS,
    FT_GRAD_ACCUM,
    FT_GRAD_CLIP_NORM,
    FT_LEARNING_RATE,
    FT_MAX_CHUNKS,
    FT_MAX_LENGTH,
    FT_MODEL_NAME,
    FT_WARMUP_RATIO,
    FT_WEIGHT_DECAY,
    N_FIT,
    N_VAL,
    SEED,
    TRAINING_LOG_FILENAME,
)
from src.data import HC3DataModule
from src.evaluate import metrics


def chunk_token_ids(ids: Sequence[int], chunk_len: int, max_chunks: int) -> list[list[int]]:
    """Split content token ids into consecutive chunks of ``chunk_len`` (at most ``max_chunks``).

    Only the first ``chunk_len * max_chunks`` tokens are used. If more than one chunk is needed
    and the last one would be short, it is moved back to end at the last used token, so every
    chunk has exactly ``chunk_len`` tokens (the last one overlaps its predecessor). A text that
    fits in one chunk gives one (possibly short) chunk; an empty input gives one empty chunk so
    that every text receives a probability.
    """
    if chunk_len < 1 or max_chunks < 1:
        raise ValueError("chunk_len and max_chunks must be positive")
    if len(ids) == 0:
        return [[]]
    end = min(len(ids), chunk_len * max_chunks)
    starts = list(range(0, end, chunk_len))
    if len(starts) > 1 and end - starts[-1] < chunk_len:
        starts[-1] = end - chunk_len
    return [list(ids[i:i + chunk_len]) for i in starts]


class FineTunedDetector(BaseDetector):
    """DistilRoBERTa sequence classifier fine-tuned on HC3.

    Args:
        model_name_or_path: base model id or a folder written by :meth:`save`.
        max_length: tokens per training sequence and per inference chunk (incl. special tokens).
        max_chunks: maximum number of chunks scored per text at inference.
        batch_size: sequences per training step / inference pass.
        device: ``"cuda"``/``"cpu"``; default CUDA if available (mixed precision on CUDA).
    """

    name = "distilroberta"

    def __init__(
        self,
        model_name_or_path: str | Path = FT_MODEL_NAME,
        max_length: int = FT_MAX_LENGTH,
        max_chunks: int = FT_MAX_CHUNKS,
        batch_size: int = FT_BATCH_SIZE,
        device: str | None = None,
    ) -> None:
        if max_length < 3:
            raise ValueError("max_length must leave room for content tokens")
        self.model_name_or_path = str(model_name_or_path)
        self.max_length = max_length
        self.max_chunks = max_chunks
        self.batch_size = batch_size
        self.device = resolve_device(device)
        self.tokenizer = None
        self.model = None
        self.training_log: dict = {}

    @property
    def use_amp(self) -> bool:
        """Mixed precision is used on CUDA only."""
        return self.device.type == "cuda"

    # ------------------------------------------------------------------ model
    def load_model(self) -> None:
        """Load tokenizer and classifier once."""
        if self.model is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            num_labels=2,
            id2label=dict(CLASS_NAMES),
            label2id={v: k for k, v in CLASS_NAMES.items()},
        ).to(self.device)
        self.model.eval()

    def release(self) -> None:
        """Free the model and its GPU memory."""
        self.model = None
        self.tokenizer = None
        free_gpu_memory()

    def save(self, path: str | Path = FINETUNED_DIR) -> Path:
        """Save model weights and tokenizer with ``save_pretrained``."""
        if self.model is None:
            raise RuntimeError("nothing to save - the model is not loaded")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        return path

    @classmethod
    def load(cls, path: str | Path = FINETUNED_DIR, device: str | None = None,
             batch_size: int = FT_BATCH_SIZE) -> "FineTunedDetector":
        """Return a detector for a saved checkpoint (weights are loaded lazily)."""
        path = Path(path)
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"no fine-tuned model in {path} - run `python -m src.finetune` first")
        return cls(model_name_or_path=path, batch_size=batch_size, device=device)

    # ------------------------------------------------------------------ helpers
    def _pad(self, sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Right-pad token id lists into ``input_ids`` and ``attention_mask`` tensors."""
        width = max(len(s) for s in sequences)
        pad = self.tokenizer.pad_token_id
        ids = torch.full((len(sequences), width), pad, dtype=torch.long)
        mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for row, seq in enumerate(sequences):
            ids[row, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[row, :len(seq)] = 1
        return ids.to(self.device), mask.to(self.device)

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp)

    # ------------------------------------------------------------------ training
    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        val_texts: Sequence[str] | None = None,
        val_labels: Sequence[int] | None = None,
        epochs: int = FT_EPOCHS,
        lr: float = FT_LEARNING_RATE,
        weight_decay: float = FT_WEIGHT_DECAY,
        warmup_ratio: float = FT_WARMUP_RATIO,
        grad_accum: int = FT_GRAD_ACCUM,
        grad_clip_norm: float = FT_GRAD_CLIP_NORM,
        output_dir: str | Path = FINETUNED_DIR,
        seed: int = SEED,
        **kwargs,
    ) -> "FineTunedDetector":
        """Fine-tune, keep the best-validation-F1 epoch in ``output_dir`` and reload it.

        Without validation data the last epoch is kept. The training log is written to
        ``output_dir/training_log.json`` and stored in ``self.training_log``.
        """
        set_seed(seed)
        self.load_model()
        output_dir = Path(output_dir)
        labels = [int(y) for y in labels]
        encoded = self.tokenizer(list(texts), truncation=True, max_length=self.max_length)["input_ids"]
        n_batches = math.ceil(len(encoded) / self.batch_size)
        total_steps = math.ceil(n_batches / grad_accum) * epochs
        warmup_steps = int(round(warmup_ratio * total_steps))

        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        params = list(self.model.named_parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": [p for n, p in params if not n.endswith(no_decay)], "weight_decay": weight_decay},
                {"params": [p for n, p in params if n.endswith(no_decay)], "weight_decay": 0.0},
            ],
            lr=lr,
        )
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)
        generator = torch.Generator().manual_seed(seed)

        settings = {
            "base_model": self.model_name_or_path, "epochs": epochs, "batch_size": self.batch_size,
            "grad_accum": grad_accum, "effective_batch_size": self.batch_size * grad_accum,
            "learning_rate": lr, "weight_decay": weight_decay, "warmup_ratio": warmup_ratio,
            "warmup_steps": warmup_steps, "total_steps": total_steps, "max_length": self.max_length,
            "max_chunks_inference": self.max_chunks, "grad_clip_norm": grad_clip_norm or None,
            "fp16": self.use_amp, "device": str(self.device), "seed": seed,
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
        }
        history, best_f1, best_epoch = [], -1.0, 0
        started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            self.model.train()
            order = torch.randperm(len(encoded), generator=generator).tolist()
            batches = [order[i:i + self.batch_size] for i in range(0, len(order), self.batch_size)]
            loss_sum = 0.0
            optimizer.zero_grad(set_to_none=True)
            bar = tqdm(batches, desc=f"epoch {epoch}/{epochs}", unit="batch")
            for step, idx in enumerate(bar, start=1):
                input_ids, mask = self._pad([encoded[i] for i in idx])
                target = torch.tensor([labels[i] for i in idx], device=self.device)
                with self._autocast():
                    loss = self.model(input_ids=input_ids, attention_mask=mask, labels=target).loss
                scaler.scale(loss / grad_accum).backward()
                loss_sum += loss.item()
                if step % grad_accum == 0 or step == len(batches):
                    if grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                bar.set_postfix(loss=f"{loss_sum / step:.4f}")
            record = {"epoch": epoch, "train_loss": loss_sum / max(len(batches), 1)}
            if val_texts is not None and val_labels is not None:
                record["val"] = metrics(val_labels, self.predict_proba(val_texts))
                score = record["val"]["f1"]
            else:
                score = float(epoch)        # no validation data: keep the last epoch
            record["seconds"] = round(time.perf_counter() - epoch_start, 1)
            improved = score > best_f1
            if improved:
                best_f1, best_epoch = score, epoch
                self.save(output_dir)
            record["saved"] = improved
            history.append(record)
            val_text = f", val F1 {record['val']['f1']:.4f}" if "val" in record else ""
            print(f"[finetune] epoch {epoch}: train loss {record['train_loss']:.4f}{val_text}"
                  f"{' (best, saved)' if improved else ''}")

        self.training_log = {
            "model": "DistilRoBERTa binary classifier (0 = human, 1 = AI)",
            "settings": settings,
            "n_train": len(encoded),
            "n_val": len(val_texts) if val_texts is not None else 0,
            "epochs": history,
            "best_epoch": best_epoch,
            "best_val_f1": best_f1 if val_texts is not None else None,
            "total_seconds": round(time.perf_counter() - started, 1),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with open(output_dir / TRAINING_LOG_FILENAME, "w", encoding="utf-8") as handle:
            json.dump(self.training_log, handle, indent=2)
        # Reload the best checkpoint so the in-memory model matches what is on disk.
        self.release()
        self.model_name_or_path = str(output_dir)
        self.load_model()
        return self

    # ------------------------------------------------------------------ inference
    @torch.inference_mode()
    def predict_proba(self, texts: Sequence[str], batch_size: int | None = None,
                      progress: bool = False, desc: str = "DistilRoBERTa") -> np.ndarray:
        """P(AI) per text, averaged over its ``max_length``-token chunks (see :func:`chunk_token_ids`)."""
        self.load_model()
        self.model.eval()
        texts = list(texts)
        if not texts:
            return np.empty(0)
        content_len = self.max_length - 2               # room for <s> ... </s>
        cls_id, sep_id = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        token_ids = self.tokenizer(texts, add_special_tokens=False, verbose=False)["input_ids"]
        chunks, owner = [], []
        for i, ids in enumerate(token_ids):
            for chunk in chunk_token_ids(ids, content_len, self.max_chunks):
                chunks.append([cls_id, *chunk, sep_id])
                owner.append(i)
        order = sorted(range(len(chunks)), key=lambda j: len(chunks[j]), reverse=True)
        size = batch_size or self.batch_size
        probs = np.empty(len(chunks))
        for start in tqdm(range(0, len(order), size), desc=desc, unit="batch", disable=not progress):
            idx = order[start:start + size]
            input_ids, mask = self._pad([chunks[j] for j in idx])
            with self._autocast():
                logits = self.model(input_ids=input_ids, attention_mask=mask).logits
            probs[idx] = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        owner_arr = np.asarray(owner)
        sums = np.bincount(owner_arr, weights=probs, minlength=len(texts))
        counts = np.bincount(owner_arr, minlength=len(texts))
        return sums / counts


# ---------------------------------------------------------------------- CLI
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.finetune",
        description="Fine-tune DistilRoBERTa on the HC3 training split and keep the best "
                    "validation-F1 checkpoint in models/distilroberta/.",
    )
    parser.add_argument("--epochs", type=int, default=FT_EPOCHS, help="training epochs (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=FT_BATCH_SIZE, help="batch size (default: %(default)s)")
    parser.add_argument("--grad-accum", type=int, default=FT_GRAD_ACCUM,
                        help="gradient accumulation steps (default: %(default)s)")
    parser.add_argument("--lr", type=float, default=FT_LEARNING_RATE, help="AdamW learning rate (default: %(default)s)")
    parser.add_argument("--grad-clip", type=float, default=FT_GRAD_CLIP_NORM,
                        help="max gradient norm before each AdamW step, 0 = no clipping (default: %(default)s)")
    parser.add_argument("--limit", type=int, default=None,
                        help="use at most N rows of train and val (balanced); needs its own --output-dir")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="folder with the split CSVs")
    parser.add_argument("--output-dir", type=Path, default=None, help=f"checkpoint folder (default: {FINETUNED_DIR})")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    configure_console()
    args = build_arg_parser().parse_args(argv)
    try:
        train = HC3DataModule.load_split("train", args.data_dir, args.limit)
        val = HC3DataModule.load_split("val", args.data_dir, args.limit)
    except FileNotFoundError as error:
        print(f"[finetune] {error}", file=sys.stderr)
        return 1
    reasons = [f"{len(frame)} {name} texts instead of {n:,}"
               for name, frame, n in (("train", train, N_FIT), ("val", val, N_VAL)) if len(frame) < n]
    if args.epochs != FT_EPOCHS:
        reasons.append(f"{args.epochs} epoch(s) instead of {FT_EPOCHS}")
    if refuse_reduced_run("finetune", reasons, [] if args.output_dir else [("--output-dir", FINETUNED_DIR)]):
        return 2
    args.output_dir = args.output_dir or FINETUNED_DIR
    detector = FineTunedDetector(batch_size=args.batch_size, device=args.device)
    print(f"[finetune] {FT_MODEL_NAME} on {detector.device}: {len(train)} train / {len(val)} val texts, "
          f"{args.epochs} epoch(s), batch {args.batch_size} x accum {args.grad_accum}")
    detector.fit(
        train["text"].tolist(), train["label"].tolist(),
        val["text"].tolist(), val["label"].tolist(),
        epochs=args.epochs, lr=args.lr, grad_accum=args.grad_accum, grad_clip_norm=args.grad_clip,
        output_dir=args.output_dir,
    )
    log = detector.training_log
    best = next(r for r in log["epochs"] if r["epoch"] == log["best_epoch"])
    if "val" in best:
        val_m = best["val"]
        print(f"[finetune] best epoch {log['best_epoch']}: val accuracy {val_m['accuracy']:.4f}, "
              f"F1 {val_m['f1']:.4f}, AUROC {val_m['auroc']:.4f}")
    print(f"[finetune] saved model and {TRAINING_LOG_FILENAME} to {args.output_dir} "
          f"({log['total_seconds']:.0f} s)")
    detector.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
