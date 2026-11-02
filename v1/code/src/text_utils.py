"""Text normalisation shared by every stage (training, evaluation and the web app).

HC3 human answers (mostly Reddit ELI5 / open_qa) are pre-tokenised ("word ." , "do n't",
"( text )", "can not", "* very *"), and the Reddit answers carry anonymised link placeholders
("[source] (URL_0)"), whereas ChatGPT answers are not tokenised but often contain line breaks.
A detector trained on raw HC3 would learn these formatting artefacts instead of writing style, so
*every* text of both classes passes through :func:`normalize_text` before it is used anywhere.

Rules, in the order they are applied (each one is unit-tested in ``tests/test_text_utils.py``).
Rules 1-8 are the required rules; 0 and 9-14 are additions found by measuring which artefacts
were still class-specific after rules 1-8 on the full HC3 corpus.

0. Map typographic quotes (U+2018/2019/201C/201D) to ASCII ``'`` and ``"`` (without it
   "it ’s" would escape rule 4).
1. Replace ``\\r``, ``\\n`` and ``\\t`` with a space, then collapse whitespace runs at once (rule 8
   applied early keeps every later regex linear in the input length).
9. Remove HC3's anonymised link placeholders: ``[text] (URL_0)`` -> ``text``, ``(URL_1)`` and
   ``[URL_2]`` -> nothing, a bare ``URL_3`` -> nothing. They never occur in real input.
2. Remove whitespace before closing punctuation ``. , ! ? ; : % ) ] }``.
3. Remove whitespace after opening brackets ``( [ {``.
4. Re-attach contractions `` n't 's 're 've 'll 'd 'm`` (case-insensitive) - but not a quoted
   letter such as ``the consonant 'm'``.
10. Re-join words the tokeniser split: ``can not`` -> ``cannot``, ``gon na`` -> ``gonna``,
    ``wan na`` -> ``wanna``, ``got ta`` -> ``gotta``.
5. Tighten spaced hyphens between word characters ("oscar - winning" -> "oscar-winning").
6. Tighten spaced double quotes around a phrase (``" Best Seller "`` -> ``"Best Seller"``). Only
   delimited pairs are tightened: the opening quote follows the start, a space or a bracket, both
   inner sides are spaced and the closing quote precedes a space, punctuation or the end. So an
   odd number of quotes (``a 12" pizza``) never glues words together.
7. ``"# 1"`` -> ``"#1"``.
11. ``"$ 5"`` -> ``"$5"``.
12. Tighten a spaced slash between word characters (``AC / DC`` -> ``AC/DC``).
13. Spaced single quotes: tighten a quoted phrase of up to six words (``' truth '`` ->
    ``'truth'``, delimited like rule 6), then attach a plural possessive (``friends ' car`` ->
    ``friends' car``).
14. Spaced markdown emphasis: ``* *`` -> ``**``, and ``* very *`` -> ``*very*`` /
    ``** in bold **`` -> ``**in bold**`` for spans of one to four words made of letters that do
    not follow a number, so arithmetic such as ``2 * 3 * 4`` is left alone.
8. Collapse runs of whitespace to one space and strip.

Rules 0 and 1 replace characters one-to-one (and do nothing on a second pass); every other rule
only deletes characters. The sequence is therefore repeated until the text stops changing
(usually one confirming pass) and always terminates. This makes the function idempotent even where rules interact,
e.g. a quoted phrase that only becomes short enough after its emphasis markers were tightened.

Not changed on purpose: markup the human authors typed themselves (bullet asterisks, ``>`` quote
markers, ``--`` dashes). It is reported separately in ``data/summary.json``
(:data:`MARKUP_PATTERNS`) and listed as a limitation in the README.
"""

from __future__ import annotations

import re

_MAX_PASSES = 5          # safety cap for the fixed-point loop (HC3 needs at most 2 passes)

_TYPOGRAPHIC_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
_CONTROL_WHITESPACE = re.compile(r"[\r\n\t]")
_WHITESPACE_RUN = re.compile(r"\s+")
_URL = r"_?URL_\d+_?"
_URL_LINK = re.compile(r"\[([^\[\]]*)\]\s*\(\s*" + _URL + r"\s*\)")
_URL_BRACKETED = re.compile(r"[(\[]\s*" + _URL + r"\s*[)\]]")
_URL_BARE = re.compile(r"(?<![A-Za-z0-9])" + _URL)
_SPACE_BEFORE_CLOSING = re.compile(r"\s+([.,!?;:%)\]}])")
_SPACE_AFTER_OPENING = re.compile(r"([(\[{])\s+")
_DETACHED_CONTRACTION = re.compile(r"\s+(n't|'s|'re|'ve|'ll|'d|'m)\b(?!')", re.IGNORECASE)
_SPLIT_WORD = re.compile(r"\b(can)\s+(not)\b|\b(gon|wan)\s+(na)\b|\b(got)\s+(ta)\b", re.IGNORECASE)
_SPACED_HYPHEN = re.compile(r"(?<=\w)\s+-\s+(?=\w)")
# A delimited pair opens at the start, after whitespace or after an opening bracket, and closes
# before whitespace, closing punctuation or the end of the text.
_OPENS = r"(?<![^\s(\[{])"
_CLOSES = r"(?=[\s.,!?;:)\]}]|$)"
_SPACED_DOUBLE_QUOTES = re.compile(_OPENS + r'"\s+([^"\s](?:[^"]*?[^"\s])?)\s+"' + _CLOSES)
_SPACED_HASH_NUMBER = re.compile(r"#\s+(?=\d)")
_SPACED_DOLLAR_NUMBER = re.compile(r"\$\s+(?=\d)")
_SPACED_SLASH = re.compile(r"(?<=\w)\s+/\s+(?=\w)")
_SPACED_SINGLE_QUOTES = re.compile(_OPENS + r"'\s+([^'\s]+(?:\s[^'\s]+){0,5})\s+'" + _CLOSES)
_PLURAL_POSSESSIVE = re.compile(r"(?<=[sS])\s+'(?=\s)")
_SPACED_DOUBLE_ASTERISK = re.compile(r"(?<=\*)\s+(?=\*)")
_EMPHASIS_WORD = r"[A-Za-z][A-Za-z'-]*[.,!?:;]?"
_SPACED_EMPHASIS = re.compile(
    _OPENS + r"(?<![\d)] )(\*{1,2})\s+(" + _EMPHASIS_WORD + r"(?:\s" + _EMPHASIS_WORD + r"){0,3})\s+\1" + _CLOSES
)


def _join_groups(match: re.Match) -> str:
    """Concatenate the matched groups of a split word ("can", "not" -> "cannot")."""
    return "".join(group for group in match.groups() if group)


def _normalise_once(text: str) -> str:
    """One pass of all rules (see the module docstring for the order)."""
    text = text.translate(_TYPOGRAPHIC_QUOTES)                 # rule 0
    text = _CONTROL_WHITESPACE.sub(" ", text)                  # rule 1
    text = _WHITESPACE_RUN.sub(" ", text)                      # (rule 8 early: linear time)
    text = _URL_LINK.sub(r"\1", text)                          # rule 9
    text = _URL_BRACKETED.sub("", text)
    text = _URL_BARE.sub("", text)
    text = _SPACE_BEFORE_CLOSING.sub(r"\1", text)              # rule 2
    text = _SPACE_AFTER_OPENING.sub(r"\1", text)               # rule 3
    text = _DETACHED_CONTRACTION.sub(r"\1", text)              # rule 4
    text = _SPLIT_WORD.sub(_join_groups, text)                 # rule 10
    text = _SPACED_HYPHEN.sub("-", text)                       # rule 5
    text = _SPACED_DOUBLE_QUOTES.sub(r'"\1"', text)            # rule 6
    text = _SPACED_HASH_NUMBER.sub("#", text)                  # rule 7
    text = _SPACED_DOLLAR_NUMBER.sub("$", text)                # rule 11
    text = _SPACED_SLASH.sub("/", text)                        # rule 12
    text = _SPACED_SINGLE_QUOTES.sub(r"'\1'", text)            # rule 13
    text = _PLURAL_POSSESSIVE.sub("'", text)
    text = _SPACED_DOUBLE_ASTERISK.sub("", text)               # rule 14
    text = _SPACED_EMPHASIS.sub(r"\1\2\1", text)
    return _WHITESPACE_RUN.sub(" ", text).strip()              # rule 8


def normalize_text(text: str | None) -> str:
    """Return ``text`` with HC3 tokenisation artefacts removed (see module docstring).

    The rules are applied until the text no longer changes, so the function is idempotent:
    ``normalize_text(normalize_text(s)) == normalize_text(s)``. ``None`` is treated as ``""``.
    Runtime is linear in the length of the input.
    """
    if not text:
        return ""
    for _ in range(_MAX_PASSES):
        normalised = _normalise_once(text)
        if normalised == text:
            break
        text = normalised
    return text


def word_count(text: str | None) -> int:
    """Number of whitespace-separated words in ``text`` (0 for empty or ``None``)."""
    return len(text.split()) if text else 0


#: Tokenisation / anonymisation artefacts that :func:`normalize_text` removes. Their rate per class
#: is reported in the data summary and must stay below 2 % after normalisation (pattern -> key).
ARTEFACT_PATTERNS: dict[str, str] = {
    " . ": "space_dot_space",
    " , ": "space_comma_space",
    " n't": "space_nt",
    "  ": "double_space",
    "URL_": "url_placeholder",
    " ' ": "spaced_single_quote",
    "* *": "spaced_double_asterisk",
    " / ": "spaced_slash",
    "$ ": "space_after_dollar",
    "can not": "can_not",
}

#: Markup typed by the human authors themselves; deliberately *not* removed, only reported so
#: that the remaining class differences are visible (pattern -> key).
MARKUP_PATTERNS: dict[str, str] = {
    "*": "asterisk",
    "> ": "quote_marker",
    " -- ": "double_hyphen",
}


def artefact_rates(texts: list[str], patterns: dict[str, str] | None = None) -> dict[str, float]:
    """Share of ``texts`` that contain each pattern (default: :data:`ARTEFACT_PATTERNS`)."""
    patterns = ARTEFACT_PATTERNS if patterns is None else patterns
    n = len(texts)
    if n == 0:
        return {key: 0.0 for key in patterns.values()}
    return {key: round(sum(pattern in t for t in texts) / n, 4) for pattern, key in patterns.items()}
