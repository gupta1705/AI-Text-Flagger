"""Modern test set: AI texts from current models (2025-26) paired with human texts on the same topics.

Why: HC3 (ChatGPT 2022/23) and MAGE (27 generators up to GPT-3.5 / LLaMA-1) do not contain text from
current models. This module builds a small, out-of-distribution test set that the detectors never see
in training:

* **Human side** - human texts from the MAGE *test* split in six genres (stories, opinion essays,
  reviews, scientific abstracts, news, explanations), 80-400 words, not used in ``test_mage.csv``.
* **AI side** - for every human text, one text of the same genre, topic and length written by a current
  model on Groq (``qwen/qwen3.8-27b`` and ``openai/gpt-oss-20b``, alternating). The prompt gives only the
  first 60 words as a topic hint and asks the model not to copy them.

MAGE's human texts are pure ASCII (the dataset was ASCII-folded; dashes are written " - "), while
current models write em-dashes, "e"-acute, ellipses and similar characters. Left alone, any non-ASCII
character would reveal the class, so the AI texts are folded to ASCII the same way (:func:`ascii_fold`).

Every AI text records the generator, prompt and settings. Responses are cached in
``data/raw/modern_cache.jsonl`` so an interrupted run (e.g. a daily free-tier limit) resumes without
repeating calls. The API key is read from an env file (``GROQ_API_KEY``) and is never printed.

CLI: ``python -m src.modern_set --env-file PATH [--n-per-genre 25] [--rpm 20]``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import unicodedata
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.base import configure_console
from src.config import DATA_DIR, LABEL_AI, LABEL_HUMAN, MAGE_CACHED_DIR, RAW_DIR, SEED
from src.mage import mage_prepare_text
from src.text_utils import normalize_text, word_count

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GENERATORS = ("qwen/qwen3.8-27b", "openai/gpt-oss-20b")
GENRES = {  # MAGE domain -> genre named in the prompt
    "wp": "short story",
    "cmv": "opinion essay that argues for a point of view",
    "yelp": "customer review of a restaurant or business",
    "sci": "scientific paper abstract",
    "xsum": "news article",
    "eli5": "explanation that answers a curious reader's question",
}
HINT_WORDS = 60
MIN_HUMAN_WORDS, MAX_HUMAN_WORDS = 80, 400
TEMPERATURE = 0.7
CACHE_PATH = RAW_DIR / "modern_cache.jsonl"
OUTPUT_CSV = DATA_DIR / "test_modern.csv"


def build_prompt(genre: str, hint: str, n_words: int) -> str:
    """Instruction for one AI text of the same genre, topic and length as a human text."""
    return (
        f"Write a {genre} of about {n_words} words. Its topic is the same as the topic of the excerpt below, "
        "but do not copy or paraphrase the excerpt's sentences - write it entirely in your own words.\n\n"
        f"Excerpt (topic hint only): \"{hint}\"\n\n"
        f"Output only the {genre} itself as plain prose: no title, no headings, no lists, no markdown, "
        "no introduction or closing remarks."
    )


ASCII_MAP = str.maketrans({
    "\u2014": " - ", "\u2015": " - ", "\u2026": "...", "\u00a3": "GBP ", "\u20ac": "EUR ",
    "\u00b0": " degrees ", "\u00d7": "x", "\u2212": "-", "\u2113": "l", "\u00b5": "u", "\u03bc": "u",
})
NUMBER_RANGE = re.compile(r"(?<=\d)\u2013(?=\d)")


def ascii_fold(text: str) -> str:
    """Fold to ASCII like MAGE's human texts: dashes -> " - ", "cafe\u0301" -> "cafe", ellipsis -> "..."."""
    text = NUMBER_RANGE.sub("-", text).replace("\u2013", " - ").translate(ASCII_MAP)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text).strip()


def clean_generation(text: str) -> str:
    """Strip markdown/titles a model may add despite the instruction, then normalise like all data."""
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if lines and (lines[0].lstrip().startswith("#") or re.fullmatch(r"\**[^.!?]{1,80}\**:?", lines[0].strip())):
        lines = lines[1:]                              # a leading title line
    text = " ".join(lines)
    text = re.sub(r"\*\*|__|^#+\s*", "", text)         # bold / heading markers
    return mage_prepare_text(normalize_text(ascii_fold(normalize_text(text))))


def pick_human_texts(test_mage_ids: set[str], n_per_genre: int, seed: int = SEED) -> pd.DataFrame:
    """Human texts from the MAGE test split, per genre, not already in ``test_mage.csv``."""
    frame = pd.read_csv(MAGE_CACHED_DIR / "test.csv")
    frame = frame[frame["label"] == 1].reset_index()              # MAGE: 1 = human
    frame["domain"] = frame["src"].str.split("_").str[0]
    frame["id"] = "mage-test-" + frame["index"].astype(str)
    frame = frame[~frame["id"].isin(test_mage_ids) & frame["domain"].isin(GENRES)]
    frame["text"] = [mage_prepare_text(normalize_text(t)) for t in frame["text"].astype(str)]
    frame["word_count"] = [word_count(t) for t in frame["text"]]
    frame = frame[frame["word_count"].between(MIN_HUMAN_WORDS, MAX_HUMAN_WORDS)]
    parts = [g.sample(n=min(n_per_genre, len(g)), random_state=seed) for _, g in frame.groupby("domain", sort=True)]
    return pd.concat(parts).reset_index(drop=True)


class GroqWriter:
    """Minimal rate-limited, cached Groq client for generating the AI side."""

    def __init__(self, env_file: Path, rpm: int, cache_path: Path = CACHE_PATH):
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv(env_file, override=False)
        key = os.getenv("GROQ_API_KEY", "").strip()
        if not key:
            raise SystemExit(f"GROQ_API_KEY is not set in {env_file}")
        self.client = OpenAI(base_url=GROQ_BASE_URL, api_key=key, max_retries=0, timeout=90)
        self.interval = 60.0 / max(rpm, 1)
        self.last = 0.0
        self.cache_path = cache_path
        self.cache = {}
        if cache_path.is_file():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    entry = json.loads(line)
                    self.cache[entry["key"]] = entry
                except (json.JSONDecodeError, KeyError):
                    continue

    def generate(self, model: str, prompt: str, max_tokens: int) -> dict:
        key = hashlib.sha256(json.dumps([model, prompt, TEMPERATURE, max_tokens]).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        extra = {"reasoning_effort": "low"} if "gpt-oss" in model else {}
        for attempt in range(8):
            wait = self.interval - (time.monotonic() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()
            try:
                response = self.client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE, max_tokens=max_tokens, **extra)
                break
            except Exception as exc:  # rate limits / transient errors: back off, a daily limit stops
                status = getattr(exc, "status_code", None)
                headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
                retry_after = float(headers.get("retry-after", 0) or 0)
                if status == 429 and retry_after > 300:
                    raise SystemExit(f"{model}: daily limit reached (retry after {retry_after:.0f}s); "
                                     "completed texts are cached - rerun later") from exc
                if status not in (429, 500, 502, 503, 504, None) or attempt == 7:
                    raise
                time.sleep(max(retry_after, min(60, 2 ** attempt)))
        text = response.choices[0].message.content or ""
        entry = {"key": key, "model": model, "returned_model": response.model, "prompt": prompt,
                 "temperature": TEMPERATURE, "max_tokens": max_tokens, "text": text,
                 "finish_reason": response.choices[0].finish_reason,
                 "usage": getattr(response.usage, "total_tokens", None),
                 "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        self.cache[key] = entry
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry


def build(env_file: Path, n_per_genre: int, rpm: int, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    test_mage = data_dir / "test_mage.csv"
    used = set(pd.read_csv(test_mage, usecols=["id"])["id"].astype(str)) if test_mage.is_file() else set()
    humans = pick_human_texts(used, n_per_genre)
    writer = GroqWriter(env_file, rpm)
    rows = []
    for i, h in enumerate(humans.to_dict("records")):
        genre = GENRES[h["domain"]]
        hint = " ".join(h["text"].split()[:HINT_WORDS])
        n_words = int(round(h["word_count"], -1))
        model = GENERATORS[i % len(GENERATORS)]
        prompt = build_prompt(genre, hint, n_words)
        entry = writer.generate(model, prompt, max_tokens=int(n_words * 2.2) + 400)
        ai_text = clean_generation(entry["text"])
        pair = f"modern-{i:03d}"
        rows.append({"id": f"{pair}-h", "question_id": 5_000_000 + i, "source": f"modern:{h['domain']}:human",
                     "label": LABEL_HUMAN, "text": h["text"], "word_count": h["word_count"],
                     "generator": "human", "prompt": ""})
        rows.append({"id": f"{pair}-a", "question_id": 5_000_000 + i, "source": f"modern:{h['domain']}:{model}",
                     "label": LABEL_AI, "text": ai_text, "word_count": word_count(ai_text),
                     "generator": entry.get("returned_model") or model, "prompt": prompt})
        print(f"[modern] {i + 1}/{len(humans)} {h['domain']:5s} {model:20s} human {h['word_count']:3d} words, "
              f"AI {word_count(ai_text):3d} words")
    frame = pd.DataFrame(rows)
    frame = frame[frame["word_count"] >= 20].reset_index(drop=True)
    frame.to_csv(data_dir / OUTPUT_CSV.name, index=False, encoding="utf-8")
    return frame


def main(argv: Sequence[str] | None = None) -> int:
    configure_console()
    parser = argparse.ArgumentParser(prog="python -m src.modern_set", description=__doc__.splitlines()[0])
    parser.add_argument("--env-file", type=Path, required=True, help="file with GROQ_API_KEY=... (never printed)")
    parser.add_argument("--n-per-genre", type=int, default=25, help="human/AI pairs per genre (default: %(default)s)")
    parser.add_argument("--rpm", type=int, default=20, help="requests per minute (default: %(default)s)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    frame = build(args.env_file, args.n_per_genre, args.rpm, args.data_dir)
    counts = frame.groupby("label").size().to_dict()
    print(f"[modern] wrote {len(frame)} texts ({counts}) to {args.data_dir / OUTPUT_CSV.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
