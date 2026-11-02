"""Tests for src.perturb (Algorithm 4): the four operations, levels and deterministic variants."""

from __future__ import annotations

import random
import re
import zlib

import pytest

from src.config import ROBUSTNESS_CONDITIONS
from src.perturb import (
    KEYBOARD,
    OPS,
    Perturber,
    edited_words,
    perturb,
    perturb_word,
    variant_key,
    variant_seed,
    variants,
)

WORD = "keyboard"
TEXT = "The quick brown fox jumps over lazy dogs, and then it rests."


@pytest.mark.parametrize("seed", range(20))
def test_swap_exchanges_neighbouring_letters(seed: int) -> None:
    out = perturb_word(WORD, "swap", random.Random(seed))
    assert len(out) == len(WORD) and out[0] == WORD[0] and sorted(out) == sorted(WORD)
    diff = [i for i, (a, b) in enumerate(zip(out, WORD)) if a != b]
    assert len(diff) in (0, 2) and (not diff or diff[1] == diff[0] + 1)


@pytest.mark.parametrize("seed", range(20))
def test_drop_deletes_one_inner_letter(seed: int) -> None:
    out = perturb_word(WORD, "drop", random.Random(seed))
    assert len(out) == len(WORD) - 1 and out[0] == WORD[0]
    assert any(WORD[:i] + WORD[i + 1:] == out for i in range(1, len(WORD) - 1))


@pytest.mark.parametrize("seed", range(20))
def test_key_uses_a_neighbouring_key(seed: int) -> None:
    out = perturb_word(WORD, "key", random.Random(seed))
    assert len(out) == len(WORD) and out[0] == WORD[0]
    diff = [i for i, (a, b) in enumerate(zip(out, WORD)) if a != b]
    assert len(diff) == 1
    i = diff[0]
    assert out[i] in KEYBOARD[WORD[i]]


@pytest.mark.parametrize("seed", range(20))
def test_add_inserts_one_letter(seed: int) -> None:
    out = perturb_word(WORD, "add", random.Random(seed))
    assert len(out) == len(WORD) + 1 and out[0] == WORD[0]
    assert any(out[:i] + out[i + 1:] == WORD for i in range(1, len(out) - 1))


def test_unknown_operation_raises() -> None:
    with pytest.raises(ValueError, match="unknown operation"):
        perturb_word(WORD, "flip", random.Random(0))
    with pytest.raises(ValueError):
        variants(TEXT, "flip", 1)


def _changed_words(a: str, b: str) -> int:
    tokens_a, tokens_b = re.split(r"(\W+)", a), re.split(r"(\W+)", b)
    assert len(tokens_a) == len(tokens_b)
    return sum(x != y for x, y in zip(tokens_a, tokens_b))


@pytest.mark.parametrize("op", ["drop", "add"])
@pytest.mark.parametrize("level", [1, 2])
def test_perturb_edits_level_words_and_keeps_punctuation(op: str, level: int) -> None:
    for seed in range(10):
        out = perturb(TEXT, op, level, seed=seed)
        assert _changed_words(TEXT, out) == level
        assert re.sub(r"\w", "", out) == re.sub(r"\w", "", TEXT)     # spaces/punctuation intact


def test_only_alphabetic_words_of_three_or_more_letters_are_eligible() -> None:
    text = "Hi, it is 2024 - ok?"
    for op in OPS:
        assert perturb(text, op, 2, seed=1) == text


def test_variants_are_deterministic_and_use_crc32_seeds() -> None:
    first = variants(TEXT, "swap", 2, k=10, text_id="17-h0")
    assert first == variants(TEXT, "swap", 2, k=10, text_id="17-h0")
    assert len(first) == 10 and len(set(first)) > 1
    assert first != variants(TEXT, "swap", 2, k=10, text_id="17-c0")
    assert variant_key("17-h0", "swap", 2, 3) == "17-h0|swap|2|3"
    assert variant_seed("17-h0", "swap", 2, 3) == zlib.crc32(b"17-h0|swap|2|3")
    assert first[3] == perturb(TEXT, "swap", 2, seed=zlib.crc32(b"17-h0|swap|2|3"))


def test_perturber_conditions_and_validation() -> None:
    perturber = Perturber(k=3)
    assert perturber.conditions() == list(ROBUSTNESS_CONDITIONS)          # one source of column order
    assert Perturber(ops=("key", "swap"), levels=(2,)).conditions() == [("swap", 2), ("key", 2)]
    assert perturber.variants(TEXT, "key", 1, text_id="x") == variants(TEXT, "key", 1, k=3, text_id="x")
    with pytest.raises(ValueError):
        Perturber(ops=("swap", "flip"))
    with pytest.raises(ValueError):
        Perturber(k=0)


@pytest.mark.parametrize("op", OPS)
def test_edited_words_lists_each_changed_word(op: str) -> None:
    variant = perturb(TEXT, op, 2, seed=5)
    edits = edited_words(TEXT, variant)
    assert 1 <= len(edits) <= 2                       # a swap of equal letters can be a no-op
    for original, edited in edits:
        assert original in TEXT.split(" ") or original in re.split(r"\W+", TEXT)
        assert edited in variant
    assert edited_words(TEXT, TEXT) == []
    with pytest.raises(ValueError):
        edited_words(TEXT, "something else entirely, with, more, tokens")
