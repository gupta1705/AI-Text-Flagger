"""Tests for the fine-tuned detector's chunking helper and loading checks (no model download)."""

from __future__ import annotations

import pytest

from src.finetune import FineTunedDetector, chunk_token_ids


def test_chunks_are_consecutive_and_capped() -> None:
    ids = list(range(10))
    assert chunk_token_ids(ids, 4, 8) == [[0, 1, 2, 3], [4, 5, 6, 7], [6, 7, 8, 9]]   # last chunk full
    assert chunk_token_ids(ids, 5, 8) == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    assert chunk_token_ids(ids, 4, 2) == [[0, 1, 2, 3], [4, 5, 6, 7]]       # at most max_chunks
    assert chunk_token_ids(ids, 10, 8) == [ids]
    assert chunk_token_ids(ids[:3], 4, 8) == [[0, 1, 2]]                    # short text: one chunk
    assert chunk_token_ids([], 4, 8) == [[]]                                 # empty text still scored


def test_no_tiny_tail_chunk_for_a_255_token_text() -> None:
    """254 content tokens fit in one 256-token chunk; token 255 must not form a 1-token chunk."""
    ids = list(range(255))
    chunks = chunk_token_ids(ids, 254, 8)
    assert [len(c) for c in chunks] == [254, 254]
    assert chunks[0] == ids[:254] and chunks[1] == ids[1:]
    assert all(len(c) == 254 for c in chunk_token_ids(list(range(1000)), 254, 8))


def test_chunk_arguments_are_validated() -> None:
    with pytest.raises(ValueError):
        chunk_token_ids([1, 2], 0, 1)


def test_load_requires_a_saved_checkpoint(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        FineTunedDetector.load(tmp_path / "distilroberta", device="cpu")


def test_construction_is_lazy() -> None:
    detector = FineTunedDetector(device="cpu")
    assert detector.model is None and detector.max_length == 256 and detector.max_chunks == 8
    assert not detector.use_amp
