"""Tests for the MAGE loader (v2 extra data): label flip, artefact removal, balanced domain sampling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import mage
from src.config import LABEL_AI, LABEL_HUMAN
from src.mage import END_OK, ending_rates, load_mage, mage_prepare_text, sample_by_domain

SENTENCE = "This is a complete sentence about the topic with enough words to count."
MACHINE = "A model wrote this other sentence about a different subject at some length."


def _text(n_sentences: int = 3, sentence: str = SENTENCE) -> str:
    return " ".join([sentence] * n_sentences)


def test_cmv_footer_is_removed() -> None:
    footer = (" Hello, users of CMV! This is a footnote from your moderators. "
              "We'd just like to remind you of a couple of things. Happy CMVing!")
    assert mage_prepare_text(_text(2) + footer) == _text(2)


def test_cut_off_text_is_trimmed_to_its_last_sentence() -> None:
    assert mage_prepare_text(_text(3) + " And then the model stopped in the") == _text(3)
    # A trim that would drop more than half of the text is not applied.
    short_head = "One sentence. " + "word " * 40
    assert mage_prepare_text(short_head) == short_head.strip()
    assert mage_prepare_text(_text(1)) == _text(1)                      # already complete
    assert mage_prepare_text('He said "stop."') == 'He said "stop."'


def _write_mage_csv(path: Path) -> Path:
    rows = [
        # MAGE: label 1 = human, 0 = machine
        (_text(3), "cmv_human", 1),
        (_text(3) + " Hello, users of CMV! moderators text. Happy CMVing!", "cmv_human", 1),
        (_text(3), "cmv_human", 1),                                     # duplicate of row 0
        (_text(3, MACHINE) + " cut off mid", "eli5_machine_continuation_gpt_j", 0),
        ("Way too short.", "xsum_machine_topical_flan_t5_xxl", 0),
        ("no sentence end anywhere " * 10, "yelp_machine_specified_opt_13b", 0),   # fragment -> dropped
        (_text(4), "eli5_eli5_machine_topical_gpt-3.5-trubo", 0),
    ]
    pd.DataFrame(rows, columns=["text", "src", "label"]).to_csv(path, index=False)
    return path


def test_load_mage_flips_labels_and_drops_artefacts(tmp_path: Path, monkeypatch) -> None:
    csv = _write_mage_csv(tmp_path / "train.csv")
    monkeypatch.setattr(mage, "resolve_file", lambda name: csv)
    frame = load_mage("train")
    assert list(frame.columns) == [*mage.COLUMNS, "domain"]
    # rows kept: human 0 (row 1 equals row 0 once the footer is gone, row 2 too), machine 3 and 6
    assert frame["label"].tolist() == [LABEL_HUMAN, LABEL_AI, LABEL_AI]
    assert frame["source"].tolist() == ["mage:cmv:human", "mage:eli5:gpt_j", "mage:eli5:gpt-3.5-trubo"]
    assert frame["question_id"].min() >= mage.QUESTION_ID_OFFSET["train"]
    assert all(END_OK.search(t) for t in frame["text"]) and "CMV" not in " ".join(frame["text"])
    assert frame["text"].is_unique and (frame["word_count"] >= 20).all()
    assert ending_rates(frame) == {"human": 0.0, "ai": 0.0}


def test_sample_by_domain_is_balanced_and_reproducible() -> None:
    rows = [{"label": label, "domain": dom, "text": f"{dom}-{label}-{i}"}
            for dom in ("a", "b", "c", "d") for label in (LABEL_HUMAN, LABEL_AI) for i in range(30)]
    frame = pd.DataFrame(rows)
    sample = sample_by_domain(frame, 40, seed=1)
    assert len(sample) == 40
    assert sample.groupby(["label", "domain"]).size().eq(5).all()
    assert sample.equals(sample_by_domain(frame, 40, seed=1))
    assert not sample["text"].equals(sample_by_domain(frame, 40, seed=2)["text"])


def test_ending_rates_counts_fragments_per_class() -> None:
    frame = pd.DataFrame({"label": [0, 0, 1, 1], "text": ["Done.", "not done", "Fine!", 'Quote."']})
    assert ending_rates(frame) == {"human": 0.5, "ai": 0.0}


@pytest.mark.parametrize("text", ["Ends well.", "Question?", "Wow!", 'Said "yes."', "(Aside.)"])
def test_end_ok_accepts_sentence_endings(text: str) -> None:
    assert END_OK.search(text)
