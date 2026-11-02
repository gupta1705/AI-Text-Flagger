"""Tests for the Modern-AI test-set helpers (prompt and clean-up); no API calls."""

from __future__ import annotations

import pytest

from src.mage import END_OK
from src.modern_set import GENRES, ascii_fold, build_prompt, clean_generation


def test_prompt_names_genre_length_and_hint() -> None:
    prompt = build_prompt("customer review of a restaurant", "the pasta was cold", 180)
    assert "customer review of a restaurant of about 180 words" in prompt
    assert '"the pasta was cold"' in prompt
    assert "no markdown" in prompt and "do not copy" in prompt


def test_every_genre_is_a_mage_domain() -> None:
    assert set(GENRES) <= {"cmv", "eli5", "hswag", "roct", "sci", "squad", "tldr", "wp", "xsum", "yelp"}


@pytest.mark.parametrize("raw", [
    "# The Quiet Forest\n\nThe forest was **quiet** that morning. Birds sang softly.",
    "**The Quiet Forest**\nThe forest was quiet that morning. Birds sang softly.",
    "The Quiet Forest:\n\nThe forest was quiet that morning.\nBirds sang softly.",
])
def test_titles_and_markdown_are_removed(raw: str) -> None:
    assert clean_generation(raw) == "The forest was quiet that morning. Birds sang softly."


def test_clean_generation_normalises_like_the_training_data() -> None:
    text = clean_generation("It is state‑of‑the‑art “work”. It ends here. And then")
    assert text == 'It is state-of-the-art "work". It ends here.'
    assert END_OK.search(text)
    assert clean_generation("") == ""


def test_ascii_fold_matches_mage_human_conventions() -> None:
    assert ascii_fold("A caf\u00e9 \u2014 na\u00efve\u2026 10\u201320 km at 5\u00b0") == \
        "A cafe - naive... 10-20 km at 5 degrees"
    assert ascii_fold("one \u2013 two") == "one - two"
    folded = clean_generation("The menu\u2014simple, honest\u2014was great. Prices in \u00a35 range.")
    # normalize_text then joins "word - word" exactly as it does for every human text
    assert folded.isascii() and folded == "The menu-simple, honest-was great. Prices in GBP 5 range."
