"""HC3DataModule: download, clean and split the English HC3 corpus (report Section 3.3, steps 1-3).

Pipeline
--------
1. ``resolve_source`` finds ``all.jsonl`` (given path -> shared cache -> download with urllib).
2. ``load`` flattens every question into rows ``id, question_id, source, label, text, word_count``
   (label 0 = human, 1 = ChatGPT). The question id is the line index in ``all.jsonl``; every text
   is normalised with :func:`src.text_utils.normalize_text`.
3. ``clean`` drops empty texts, texts with fewer than 20 words and exact duplicates (keep first).
4. ``make_splits`` shuffles the question ids (seed 42), assigns ~70 % / 10 % / 20 % of them to
   train / val / test pools and draws balanced samples from each pool, preferring one human and
   one ChatGPT answer of the same question. No question appears in two splits.
5. ``save`` writes ``train.csv`` (4,500), ``val.csv`` (500), ``test.csv`` (1,500) and
   ``summary.json`` (counts, medians, disjointness check and, per class, the rate of every
   tokenisation artefact after normalisation - over the split rows and over the whole cleaned
   corpus - plus the rate of author-typed markup that is deliberately kept).

v2 (default): MAGE samples are added to ``train``/``val`` (``src/mage.py``) and the extra test sets
``test_mage.csv`` (MAGE in-distribution) and ``test_ood.csv`` (MAGE unseen domains + GPT-4) are written
next to the HC3 ``test.csv``. ``--no-mage`` reproduces the v1 HC3-only splits.

CLI: ``python -m src.data [--hc3-path P] [--no-mage] [--limit N --data-dir DIR]`` (a run with ``--limit``
must name its own ``--data-dir``, so it never replaces the full-size splits in ``data/``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.base import configure_console, refuse_reduced_run
from src.config import (
    CLASS_NAMES,
    DATA_DIR,
    HC3_CACHED_PATH,
    HC3_DEFAULT_PATH,
    HC3_URL,
    LABEL_AI,
    LABEL_HUMAN,
    MIN_WORDS,
    N_FIT,
    N_TEST,
    N_VAL,
    POOL_FRACTIONS,
    SEED,
    SPLIT_NAMES,
    SUMMARY_FILENAME,
)
from src.mage import build_mage_splits, ending_rates
from src.text_utils import MARKUP_PATTERNS, artefact_rates, normalize_text, word_count

COLUMNS = ["id", "question_id", "source", "label", "text", "word_count"]


class HC3DataModule:
    """Build balanced, question-disjoint HC3 splits for training, validation and testing.

    Args:
        hc3_path: location of ``all.jsonl``; defaults to ``data/raw/all.jsonl``.
        data_dir: folder that receives the split CSVs and ``summary.json``.
        seed: random seed for shuffling and sampling.
        n_fit / n_val / n_test: target split sizes (each balanced 50/50 between classes).
        limit: optional cap applied to every split size (for fast smoke runs).
        min_words: texts with fewer words are removed.
        cached_path: shared read-only copy tried when ``hc3_path`` is missing.
        url: download location used when neither file exists.
    """

    def __init__(
        self,
        hc3_path: str | Path | None = None,
        data_dir: str | Path = DATA_DIR,
        seed: int = SEED,
        n_fit: int = N_FIT,
        n_val: int = N_VAL,
        n_test: int = N_TEST,
        limit: int | None = None,
        min_words: int = MIN_WORDS,
        cached_path: str | Path | None = HC3_CACHED_PATH,
        url: str = HC3_URL,
    ) -> None:
        self.hc3_path = Path(hc3_path) if hc3_path else HC3_DEFAULT_PATH
        self.data_dir = Path(data_dir)
        self.seed = seed
        self.limit = limit
        cap = (lambda n: n) if limit is None else (lambda n: min(n, limit))
        self.sizes = {"train": cap(n_fit), "val": cap(n_val), "test": cap(n_test)}
        for name, size in self.sizes.items():
            if size < 2:
                raise ValueError(f"split '{name}' needs at least 2 rows, got {size}")
        self.min_words = min_words
        self.cached_path = Path(cached_path) if cached_path else None
        self.url = url
        self.source_path: Path | None = None
        self.raw_counts: dict[str, int] = {}
        self.raw_artefacts: dict[str, dict[str, float]] = {}
        self.corpus_artefacts: dict[str, dict[str, float]] = {}
        self.removed: dict[str, int] = {}
        self.pool_questions: dict[str, int] = {}

    # ------------------------------------------------------------------ acquisition
    def resolve_source(self) -> Path:
        """Return a readable ``all.jsonl``: given path, else the shared cache, else download it."""
        if self.hc3_path.is_file():
            return self.hc3_path
        if self.cached_path is not None and self.cached_path.is_file():
            print(f"[data] {self.hc3_path} not found; using cached copy {self.cached_path}")
            return self.cached_path
        self.hc3_path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.hc3_path.with_suffix(".part")
        print(f"[data] downloading HC3 from {self.url} (~74 MB) ...")
        with urllib.request.urlopen(self.url, timeout=60) as response, open(partial, "wb") as out:
            shutil.copyfileobj(response, out)
        partial.replace(self.hc3_path)
        return self.hc3_path

    # ------------------------------------------------------------------ loading
    def load(self) -> pd.DataFrame:
        """Flatten ``all.jsonl`` into one normalised row per answer (before cleaning)."""
        self.source_path = self.resolve_source()
        rows: list[dict] = []
        raw_texts: dict[int, list[str]] = {LABEL_HUMAN: [], LABEL_AI: []}
        n_questions = 0
        with open(self.source_path, encoding="utf-8") as handle:
            for question_id, line in enumerate(handle):
                if not line.strip():
                    continue
                record = json.loads(line)
                n_questions += 1
                answers = (
                    (LABEL_HUMAN, "h", record.get("human_answers") or []),
                    (LABEL_AI, "c", record.get("chatgpt_answers") or []),
                )
                for label, tag, texts in answers:
                    for j, raw in enumerate(texts):
                        raw = raw or ""
                        raw_texts[label].append(raw)
                        text = normalize_text(raw)
                        rows.append({
                            "id": f"{question_id}-{tag}{j}",
                            "question_id": question_id,
                            "source": record.get("source", "unknown"),
                            "label": label,
                            "text": text,
                            "word_count": word_count(text),
                        })
        self.raw_counts = {
            "questions": n_questions,
            "human_answers": len(raw_texts[LABEL_HUMAN]),
            "chatgpt_answers": len(raw_texts[LABEL_AI]),
        }
        self.raw_artefacts = {CLASS_NAMES[c]: artefact_rates(t) for c, t in raw_texts.items()}
        return pd.DataFrame(rows, columns=COLUMNS)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop empty texts, texts under ``min_words`` words and exact duplicates (keep first).

        Also records the post-normalisation artefact rates of the whole cleaned corpus, which
        stay meaningful when the splits themselves are tiny (``--limit``).
        """
        empty = df["text"].str.len() == 0
        short = ~empty & (df["word_count"] < self.min_words)
        kept = df[~empty & ~short]
        duplicate = kept["text"].duplicated(keep="first")
        self.removed = {
            "empty": int(empty.sum()),
            "too_short": int(short.sum()),
            "duplicate": int(duplicate.sum()),
        }
        kept = kept[~duplicate].reset_index(drop=True)
        self.corpus_artefacts = {
            CLASS_NAMES[c]: artefact_rates(kept.loc[kept["label"] == c, "text"].tolist())
            for c in (LABEL_HUMAN, LABEL_AI)
        }
        return kept

    # ------------------------------------------------------------------ splitting
    def make_splits(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Split by question into balanced ``train`` (fit), ``val`` and ``test`` frames."""
        rng = np.random.default_rng(self.seed)
        question_ids = np.sort(df["question_id"].unique())
        rng.shuffle(question_ids)
        n_train = int(round(POOL_FRACTIONS["train"] * len(question_ids)))
        n_val = int(round(POOL_FRACTIONS["val"] * len(question_ids)))
        pools = {
            "train": question_ids[:n_train],
            "val": question_ids[n_train:n_train + n_val],
            "test": question_ids[n_train + n_val:],
        }
        self.pool_questions = {name: int(len(ids)) for name, ids in pools.items()}
        splits = {}
        for name in SPLIT_NAMES:
            pool = df[df["question_id"].isin(pools[name])]
            sample = _sample_balanced(pool, self.sizes[name], rng, name)
            order = rng.permutation(len(sample))
            splits[name] = sample.iloc[order].reset_index(drop=True)
        _assert_disjoint(splits)
        return splits

    # ------------------------------------------------------------------ output
    def save(self, splits: dict[str, pd.DataFrame]) -> dict:
        """Write the split CSVs and ``summary.json``; return the summary dictionary."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in splits.items():
            frame[COLUMNS].to_csv(self.data_dir / f"{name}.csv", index=False, encoding="utf-8")
        summary = self.summary(splits)
        with open(self.data_dir / SUMMARY_FILENAME, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return summary

    def summary(self, splits: dict[str, pd.DataFrame]) -> dict:
        """Counts per split x class x source, medians and post-normalisation artefact rates."""
        everything = pd.concat(splits.values(), ignore_index=True)
        split_info = {}
        for name, frame in splits.items():
            by_source = (
                frame.groupby(["source", "label"]).size().unstack(fill_value=0)
                .reindex(columns=[LABEL_HUMAN, LABEL_AI], fill_value=0)
            )
            split_info[name] = {
                "rows": int(len(frame)),
                "questions": int(frame["question_id"].nunique()),
                "by_class": _class_counts(frame),
                "by_source": {
                    str(source): {CLASS_NAMES[c]: int(row[c]) for c in (LABEL_HUMAN, LABEL_AI)}
                    for source, row in by_source.iterrows()
                },
                "median_words": _median_words(frame),
            }
        return {
            "dataset": "HC3 English (all.jsonl), licence CC-BY-SA-4.0",
            "source_file": str(self.source_path) if self.source_path else None,
            "seed": self.seed,
            "limit": self.limit,
            "min_words": self.min_words,
            "raw": self.raw_counts,
            "removed": self.removed,
            "pool_fractions": POOL_FRACTIONS,
            "pool_questions": self.pool_questions,
            "splits": split_info,
            "median_words": _median_words(everything),
            "artefact_rates_raw_all_answers": self.raw_artefacts,
            "artefact_rates_after_normalisation": {
                CLASS_NAMES[c]: artefact_rates(everything.loc[everything["label"] == c, "text"].tolist())
                for c in (LABEL_HUMAN, LABEL_AI)
            },
            "artefact_rates_after_normalisation_all_clean_answers": self.corpus_artefacts,
            "markup_rates_after_normalisation": {
                CLASS_NAMES[c]: artefact_rates(everything.loc[everything["label"] == c, "text"].tolist(),
                                               MARKUP_PATTERNS)
                for c in (LABEL_HUMAN, LABEL_AI)
            },
            "markup_note": "author-typed markup (asterisks, '> ' quote markers, ' -- ') is kept on purpose; "
                           "these rates show the remaining class difference",
            "question_sets_disjoint": True,  # make_splits raises otherwise
        }

    def prepare(self) -> dict[str, pd.DataFrame]:
        """Run load -> clean -> make_splits -> save and return the splits."""
        splits = self.make_splits(self.clean(self.load()))
        self.save(splits)
        return splits

    # ------------------------------------------------------------------ reading
    @staticmethod
    def load_split(name: str, data_dir: str | Path = DATA_DIR, limit: int | None = None) -> pd.DataFrame:
        """Read ``<data_dir>/<name>.csv``; with ``limit`` keep a balanced subset of that size."""
        path = Path(data_dir) / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found - run `python -m src.data` first")
        frame = pd.read_csv(
            path,
            encoding="utf-8",
            dtype={"id": str, "source": str, "text": str},
            keep_default_na=False,
        )
        if limit is not None and limit < len(frame):
            frame = frame.groupby("label", sort=False).head(max(limit // 2, 1)).reset_index(drop=True)
        return frame


def _sample_balanced(pool: pd.DataFrame, n: int, rng: np.random.Generator, name: str) -> pd.DataFrame:
    """Draw ``n // 2`` human and ``n // 2`` AI rows from ``pool``, spreading over questions.

    Questions that have answers of both classes come first, so the sample is made of
    (human, AI) answer pairs of the same question whenever possible. Rows are taken in rounds:
    round ``r`` takes the ``r``-th (random) answer of each question, so a question contributes a
    second answer only when the pool has run out of fresh questions.
    """
    per_class = n // 2
    groups = {
        key: rng.permutation(idx)
        for key, idx in pool.groupby(["question_id", "label"], sort=True).indices.items()
    }
    questions = rng.permutation(np.sort(pool["question_id"].unique()))
    paired = {q for q in questions if (q, LABEL_HUMAN) in groups and (q, LABEL_AI) in groups}
    order = [q for q in questions if q in paired] + [q for q in questions if q not in paired]
    chosen: dict[int, list[int]] = {LABEL_HUMAN: [], LABEL_AI: []}
    depth, max_depth = 0, max((len(idx) for idx in groups.values()), default=0)
    while depth < max_depth and any(len(rows) < per_class for rows in chosen.values()):
        for question in order:
            for label, rows in chosen.items():
                idx = groups.get((question, label))
                if len(rows) < per_class and idx is not None and depth < len(idx):
                    rows.append(int(idx[depth]))
        depth += 1
    for label, rows in chosen.items():
        if len(rows) < per_class:
            raise ValueError(
                f"split '{name}': pool has only {len(rows)} {CLASS_NAMES[label]} texts, "
                f"{per_class} needed - reduce the split size or use more data"
            )
    return pool.iloc[chosen[LABEL_HUMAN] + chosen[LABEL_AI]]


def _assert_disjoint(splits: dict[str, pd.DataFrame]) -> None:
    """Raise if any question id occurs in more than one split."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = set(splits[a]["question_id"]) & set(splits[b]["question_id"])
            if overlap:
                raise AssertionError(f"questions shared by {a} and {b}: {sorted(overlap)[:5]}")


def _class_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {CLASS_NAMES[c]: int((frame["label"] == c).sum()) for c in (LABEL_HUMAN, LABEL_AI)}


def _median_words(frame: pd.DataFrame) -> dict[str, float]:
    return {
        CLASS_NAMES[c]: float(frame.loc[frame["label"] == c, "word_count"].median())
        if (frame["label"] == c).any() else 0.0
        for c in (LABEL_HUMAN, LABEL_AI)
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.data",
        description="Download/clean HC3 and write balanced, question-disjoint train/val/test CSVs.",
    )
    parser.add_argument("--hc3-path", type=Path, default=HC3_DEFAULT_PATH,
                        help="path to HC3 all.jsonl (default: %(default)s; falls back to the "
                             "shared cache, then downloads)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap every split at N rows for a fast run (default: full sizes "
                             f"{N_FIT}/{N_VAL}/{N_TEST}); needs its own --data-dir")
    parser.add_argument("--data-dir", type=Path, default=None, help=f"output folder (default: {DATA_DIR})")
    parser.add_argument("--seed", type=int, default=SEED, help="random seed (default: %(default)s)")
    parser.add_argument("--no-mage", action="store_true",
                        help="HC3 only (v1): do not add MAGE training data or write the MAGE test sets")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    configure_console()
    args = build_arg_parser().parse_args(argv)
    data_dir = args.data_dir or DATA_DIR
    module = HC3DataModule(hc3_path=args.hc3_path, data_dir=data_dir, seed=args.seed, limit=args.limit)
    full = {"train": N_FIT, "val": N_VAL, "test": N_TEST}
    reasons = [f"--limit {args.limit}"] if module.sizes != full else []
    if refuse_reduced_run("data", reasons, [] if args.data_dir else [("--data-dir", DATA_DIR)]):
        return 2
    splits = module.make_splits(module.clean(module.load()))
    mage_info = None
    if not args.no_mage:
        mage = build_mage_splits(seed=args.seed, limit=args.limit)
        mage_info = {
            "dataset": "MAGE (yaful/MAGE, Apache-2.0); labels flipped to 0 = human, 1 = AI",
            "rows": {name: _class_counts(frame) for name, frame in mage.items()},
            "domains": {name: frame.groupby("domain").size().to_dict() for name, frame in mage.items()},
            "generators_in_train": int(mage["train"].loc[mage["train"]["label"] == LABEL_AI, "source"]
                                       .str.split(":").str[2].nunique()),
            "ends_without_sentence_punctuation_after_trim": {n: ending_rates(f) for n, f in mage.items()},
            "artefact_rates_after_normalisation": {
                CLASS_NAMES[c]: artefact_rates(mage["train"].loc[mage["train"]["label"] == c, "text"].tolist())
                for c in (LABEL_HUMAN, LABEL_AI)
            },
        }
        splits["train"] = pd.concat([splits["train"], mage["train"][COLUMNS]], ignore_index=True)
        splits["val"] = pd.concat([splits["val"], mage["val"][COLUMNS]], ignore_index=True)
        order = np.random.default_rng(args.seed).permutation(len(splits["train"]))
        splits["train"] = splits["train"].iloc[order].reset_index(drop=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        mage["test"][COLUMNS].to_csv(data_dir / "test_mage.csv", index=False, encoding="utf-8")
        mage["ood"][COLUMNS].to_csv(data_dir / "test_ood.csv", index=False, encoding="utf-8")
    summary = module.save(splits)
    if mage_info is not None:
        summary["mage"] = mage_info
        with open(data_dir / SUMMARY_FILENAME, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"[data] MAGE rows: {mage_info['rows']}; wrote test_mage.csv and test_ood.csv")
    print(f"[data] source: {summary['source_file']}")
    print(f"[data] raw: {summary['raw']}  removed: {summary['removed']}")
    for name in SPLIT_NAMES:
        info = summary["splits"][name]
        print(f"[data] {name:5s}: {info['rows']:5d} rows, {info['questions']:5d} questions, "
              f"classes {info['by_class']}, median words {info['median_words']}")
    print(f"[data] artefact rates after normalisation: {summary['artefact_rates_after_normalisation']}")
    print(f"[data] wrote {', '.join(f'{n}.csv' for n in SPLIT_NAMES)} and {SUMMARY_FILENAME} to {data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
