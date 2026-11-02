"""Shared fixtures: a small synthetic HC3-style corpus (no network, no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SOURCES = ("reddit_eli5", "finance", "medicine", "open_qa", "wiki_csai")


def _words(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def write_synthetic_hc3(path: Path, n_questions: int = 80) -> Path:
    """Write an HC3-like ``all.jsonl``.

    Human answers are pre-tokenised (" . ", " , ", " n't", " ' so ' ") and carry an HC3 link
    placeholder ("[source] (URL_0)"); ChatGPT answers contain line breaks.
    Special cases: question 0 has an empty ChatGPT answer and a too-short human answer,
    question 1 repeats question 2's human answer (duplicate) and question 3 has no ChatGPT answer.
    """
    lines = []
    for q in range(n_questions):
        human = [
            f"Well , it is n't simple . Question {q} answer ( first ) : " + _words(f"h{q}a", 25),
            f"Another view for {q} , you do n't say ' so ' . [source] (URL_0) " + _words(f"h{q}b", 22),
        ]
        chatgpt = [f"Answer to question {q}.\nIt explains the topic clearly. " + _words(f"c{q}a", 25)]
        if q == 0:
            chatgpt.append("")
            human.append("too short answer .")
        if q == 1:
            human.append("Shared duplicate answer text , exactly the same . " + _words("dup", 20))
        if q == 2:
            human.append("Shared duplicate answer text , exactly the same . " + _words("dup", 20))
        if q == 3:
            chatgpt = []
        lines.append(json.dumps({
            "question": f"Question {q}?",
            "human_answers": human,
            "chatgpt_answers": chatgpt,
            "index": None,
            "source": SOURCES[q % len(SOURCES)],
        }))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def hc3_file(tmp_path: Path) -> Path:
    return write_synthetic_hc3(tmp_path / "raw" / "all.jsonl")
