"""Tests for src.text_utils: one test per normalisation rule, idempotence and helpers."""

import time

import pytest

from src.text_utils import MARKUP_PATTERNS, artefact_rates, normalize_text, word_count


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # rule 0 (v2): non-breaking/figure hyphens and spaces -> ASCII (gpt-oss writes U+2011)
        ("state\u2011of\u2011the\u2011art and well\u2010known", "state-of-the-art and well-known"),
        ("100\u00a0km in 5\u202fmin,\u2009ok\u2007now", "100 km in 5 min, ok now"),
        # rule 0: typographic quotes -> ASCII (then rules 4 and 6 apply)
        ("it ’s “ quoted ”", 'it\'s "quoted"'),
        # rule 1: \r, \n, \t -> space
        ("line one\nline two\r\nthree\tfour", "line one line two three four"),
        # rule 2: no whitespace before closing punctuation
        ("Done . Yes , really ! Why ? So ; then : 50 % (a ) [b ] {c }",
         "Done. Yes, really! Why? So; then: 50% (a) [b] {c}"),
        # rule 3: no whitespace after opening brackets
        ("see ( this ) and [ that ] or { other }", "see (this) and [that] or {other}"),
        # rule 4: contractions re-attached
        ("I do n't know , it 's fine , they 're here , we 've gone , you 'll see , he 'd go , I 'm ok",
         "I don't know, it's fine, they're here, we've gone, you'll see, he'd go, I'm ok"),
        ("DO N'T SHOUT", "DON'T SHOUT"),
        # rule 5: spaced hyphen between word characters
        ("an oscar - winning book from 5 - 10", "an oscar-winning book from 5-10"),
        # rule 6: spaced double quotes around a phrase
        ('the " Best Seller " list', 'the "Best Seller" list'),
        # rule 7: "# 1" -> "#1"
        ("ranked # 1 today", "ranked #1 today"),
        # rule 8: collapse whitespace and strip
        ("  too   many    spaces  ", "too many spaces"),
        # rule 9: HC3 link placeholders are removed (markdown link -> its text)
        ("see [the paper] (URL_0) and ( URL_1 ) or [ URL_2 ] here: URL_3 . Done",
         "see the paper and or here:. Done"),
        # rule 10: words split by the tokeniser are re-joined
        ("You can not go, I 'm gon na try, wan na see, got ta run. CAN NOT",
         "You cannot go, I'm gonna try, wanna see, gotta run. CANNOT"),
        # rule 11: "$ 5" -> "$5"
        ("it costs $ 5 or $ 1,000", "it costs $5 or $1,000"),
        # rule 12: spaced slash between word characters
        ("AC / DC and /r / atheism, 10 / 2", "AC/DC and /r/atheism, 10/2"),
        # rule 13: spaced single quotes (short quoted phrase, plural possessive)
        ("they call it ' big words ' and my friends ' car", "they call it 'big words' and my friends' car"),
        # rule 14: spaced markdown emphasis
        ("It's a * very * strong drink, * * really * * good", "It's a *very* strong drink, **really** good"),
    ],
)
def test_normalisation_rules(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        'A 12" pizza is "big" today.',                  # odd number of quotes (inch mark)
        'a 5" screen and "quoted" text',
        'He said "hello and then "world" ok',          # unmatched opening quote
    ],
)
def test_odd_quote_counts_never_glue_words(text: str) -> None:
    assert normalize_text(text) == text


def test_spaced_quotes_with_odd_count_are_tightened_without_glue() -> None:
    assert normalize_text('A 12" pizza is " big " today.') == 'A 12" pizza is "big" today.'


def test_quoted_letters_are_not_taken_for_contractions() -> None:
    assert normalize_text("the consonant ' m ' and ' D ', it 's") == "the consonant 'm' and 'D', it's"


def test_arithmetic_asterisks_are_left_alone() -> None:
    assert normalize_text("2 * 3 * 4 = 24 and 5 * (x) * 2") == "2 * 3 * 4 = 24 and 5 * (x) * 2"


@pytest.mark.parametrize(
    "text",
    [
        'He said "' + " " * 20_000,                     # used to backtrack catastrophically
        'Title: "' + "\n" * 20_000 + "Body text follows here.",
        "Start of text" + " " * 50_000 + "end of text",
    ],
    ids=["open-quote-spaces", "open-quote-newlines", "space-run"],
)
def test_long_whitespace_runs_are_fast(text: str) -> None:
    started = time.perf_counter()
    normalize_text(text)
    assert time.perf_counter() - started < 1.0


def test_hc3_style_human_answer_matches_chatgpt_style() -> None:
    human = "It 's a \" big \" deal ( really ) . You do n't need # 1 , trust me .\n"
    chatgpt = 'It\'s a "big" deal (really). You don\'t need #1, trust me.'
    assert normalize_text(human) == normalize_text(chatgpt) == chatgpt


@pytest.mark.parametrize(
    "text",
    [
        'He said "hi" and "bye".',
        "A well-known fact: 3 - 1 = 2 - right?",
        "Plain ChatGPT text.\n\nWith paragraphs and (brackets).",
        "Odd \" quote count \" here \" and",
        "the name ' * * The IRA (Irish Republican Army) * * ' was used",   # needs a second pass
        "so ' A ' would be ' D ' . Sounds like ' d ' , ' t ' , ' m ' ok",
        "topic : * BBC * TIME Magazine * NPR and an associated chronology * Basic",
    ],
)
def test_idempotent(text: str) -> None:
    once = normalize_text(text)
    assert normalize_text(once) == once


def test_leaves_clean_text_unchanged() -> None:
    text = 'He said "hi" and "bye". It\'s 5% (roughly) [sic] {ok}; done: yes!'
    assert normalize_text(text) == text


def test_hyphen_only_between_word_characters() -> None:
    assert normalize_text("- item one") == "- item one"
    assert normalize_text("a list -") == "a list -"


@pytest.mark.parametrize("value", [None, "", "   \n\t "])
def test_empty_inputs(value) -> None:
    assert normalize_text(value) == ""


def test_word_count() -> None:
    assert word_count("one two  three\nfour") == 4
    assert word_count("") == 0
    assert word_count(None) == 0


def test_artefact_rates() -> None:
    raw = ["a . b", "c , d", "do n't", "x  y", "see URL_0", "my ' quote '", "* * b * *", "a / b", "$ 5", "can not"]
    rates = artefact_rates(raw)
    assert rates == {
        "space_dot_space": 0.1, "space_comma_space": 0.1, "space_nt": 0.1, "double_space": 0.1,
        "url_placeholder": 0.1, "spaced_single_quote": 0.1, "spaced_double_asterisk": 0.1,
        "spaced_slash": 0.1, "space_after_dollar": 0.1, "can_not": 0.1,
    }
    assert artefact_rates([])["space_dot_space"] == 0.0
    normalised = [normalize_text(t) for t in raw]
    assert all(v == 0.0 for v in artefact_rates(normalised).values())


def test_markup_rates_use_their_own_patterns() -> None:
    rates = artefact_rates(["* item one", "> quoted", "a -- b", "plain"], MARKUP_PATTERNS)
    assert rates == {"asterisk": 0.25, "quote_marker": 0.25, "double_hyphen": 0.25}
