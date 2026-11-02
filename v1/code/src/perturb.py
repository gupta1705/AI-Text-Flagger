"""Perturber: character-level typo perturbations (report Algorithm 4, Listing 3).

Implements the four adversarial-misspelling operations of Pruthi et al. (2019) used by the
robustness survey (Liu, Li and Li 2025, Section 7.1):

* ``swap`` - exchange two neighbouring letters,
* ``drop`` - delete one letter,
* ``key``  - replace a letter with a neighbouring QWERTY key,
* ``add``  - insert a random letter.

``perturb(text, op, n_chars)`` edits ``n_chars`` distinct, randomly chosen alphabetic words of at
least three letters, once each, keeping the first letter of every word. ``variants`` returns ``k``
reproducible variants whose seeds come from a stable CRC-32 hash of the variant key
``"{text_id}|{op}|{level}|{j}"`` (never Python's salted ``hash()``), so a stored key is enough to
regenerate its variant. ``Perturber`` is the class the Evaluator uses (report Figure 3).
"""

from __future__ import annotations

import random
import re
import string
import zlib

from src.config import K_VARIANTS, PERTURB_LEVELS, PERTURB_OPS, ROBUSTNESS_CONDITIONS

OPS = PERTURB_OPS
LEVELS = PERTURB_LEVELS
#: Operation order of the robustness table (survey Table 5): swap, drop, add, key.
TABLE_OP_ORDER = tuple(dict.fromkeys(op for op, _ in ROBUSTNESS_CONDITIONS))

KEYBOARD = {  # neighbouring keys on a QWERTY keyboard
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh", "u": "yij",
    "i": "uok", "o": "ipl", "p": "ol", "a": "qsz", "s": "awdz", "d": "sefx", "f": "drgc",
    "g": "fthv", "h": "gyjb", "j": "hukn", "k": "jilm", "l": "kop", "z": "asx", "x": "zsdc",
    "c": "xdfv", "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}


def perturb_word(word: str, op: str, rng: random.Random) -> str:
    """Apply one character-level edit to an alphabetic word of length >= 3."""
    i = rng.randrange(1, len(word) - 1)            # keep the first letter unchanged
    if op == "swap":                               # exchange two neighbouring letters
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "drop":                               # delete one letter
        return word[:i] + word[i + 1:]
    if op == "key":                                # replace with a neighbouring key
        c = word[i].lower()
        return word[:i] + rng.choice(KEYBOARD.get(c, c)) + word[i + 1:]
    if op == "add":                                # insert a random letter
        return word[:i] + rng.choice(string.ascii_lowercase) + word[i:]
    raise ValueError(f"unknown operation: {op}")


def perturb(text: str, op: str, n_chars: int = 1, seed: int = 0) -> str:
    """Return `text` with `n_chars` randomly chosen words edited once each."""
    rng = random.Random(seed)
    tokens = re.split(r"(\W+)", text)               # keep spaces and punctuation
    eligible = [k for k, t in enumerate(tokens) if t.isalpha() and len(t) >= 3]
    for k in rng.sample(eligible, min(n_chars, len(eligible))):
        tokens[k] = perturb_word(tokens[k], op, rng)
    return "".join(tokens)


def variant_key(text_id: str, op: str, level: int, j: int) -> str:
    """Identifier of variant ``j`` of ``text_id`` under (``op``, ``level``), e.g. ``"17-h0|swap|2|3"``."""
    return f"{text_id}|{op}|{level}|{j}"


def variant_seed(text_id: str, op: str, level: int, j: int) -> int:
    """Stable seed for variant ``j`` of ``text_id`` under (``op``, ``level``): CRC-32 of its key."""
    return zlib.crc32(variant_key(text_id, op, level, j).encode("utf-8"))


def edited_words(text: str, variant: str) -> list[tuple[str, str]]:
    """``(original, edited)`` word pairs that differ between ``text`` and its ``variant``.

    ``perturb`` keeps the token structure (words stay alphabetic), so the tokens line up.
    """
    original, edited = re.split(r"(\W+)", text), re.split(r"(\W+)", variant)
    if len(original) != len(edited):
        raise ValueError("variant does not come from perturb(text, ...)")
    return [(a, b) for a, b in zip(original, edited) if a != b]


def variants(text: str, op: str, n_chars: int, k: int = K_VARIANTS, text_id: str = "") -> list[str]:
    """Return ``k`` deterministic perturbed variants of ``text`` (``n_chars`` edited words each)."""
    if op not in OPS:
        raise ValueError(f"unknown operation: {op}")
    return [perturb(text, op, n_chars, seed=variant_seed(text_id, op, n_chars, j)) for j in range(k)]


class Perturber:
    """Creates the perturbed test variants used by the robustness evaluation.

    Args:
        k: number of variants per (text, operation, level).
        ops: operations to apply (subset of :data:`OPS`).
        levels: numbers of edited words per variant (subset of :data:`LEVELS`).
    """

    OPS = OPS
    LEVELS = LEVELS

    def __init__(self, k: int = K_VARIANTS, ops: tuple[str, ...] = OPS, levels: tuple[int, ...] = LEVELS) -> None:
        unknown = set(ops) - set(OPS)
        if unknown:
            raise ValueError(f"unknown operations: {sorted(unknown)}")
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.ops = tuple(ops)
        self.levels = tuple(levels)

    def variants(self, text: str, op: str, n_chars: int, text_id: str = "") -> list[str]:
        """``k`` deterministic variants of ``text`` for one operation and level."""
        return variants(text, op, n_chars, k=self.k, text_id=text_id)

    def conditions(self) -> list[tuple[str, int]]:
        """All (operation, level) pairs in the robustness-table order (Swap, Drop, Add, Key; 1, 2).

        With the default operations and levels this is exactly
        :data:`src.config.ROBUSTNESS_CONDITIONS`.
        """
        return [(op, level) for op in TABLE_OP_ORDER if op in self.ops for level in sorted(self.levels)]
