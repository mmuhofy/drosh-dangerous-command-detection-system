"""Unit tests for the normalisation specification itself.

The cross-language guarantee lives in test_parity.py. This file covers the
behavioural contract of the rules — it is the place where a rule change is
*supposed* to fail, so each test names the rule it pins down rather than just
re-asserting an output.

The corpus in test_normalize.py is intentionally small and readable. If it ever
grows large it has become a second copy of test_parity.py's corpus, which would
be a maintenance trap.
"""

from __future__ import annotations

import pytest

from drosh_ml.normalize import (
    NGRAM_MAX,
    NGRAM_MIN,
    fold_ascii,
    normalize,
    presence_ngrams,
    strip_ansi,
)


class TestAnsiStripping:
    """Rule 1: escape sequences go before control characters are touched."""

    @pytest.mark.parametrize(
        "raw",
        [
            "\x1b[31mrm -rf /\x1b[0m",
            "\x1b[1;31;40mrm -rf /\x1b[0m",
            "\x1b[38;5;196mrm -rf /",
            "\x1b]0;title\x07rm -rf /",
            "\x1b]0;title\x1b\\rm -rf /",
            "\x1bPsome dcs\x1b\\rm -rf /",
            "\x1b(Brm -rf /",
            "\x1bMrm -rf /",
        ],
    )
    def test_escapes_leave_no_residue(self, raw: str) -> None:
        # The failure this guards against is subtle: deleting ESC but leaving
        # "31m" behind injects plausible-looking text into the n-gram space.
        assert normalize(raw).text == "rm -rf /"

    def test_strip_ansi_is_idempotent(self) -> None:
        once = strip_ansi("\x1b[32mgreen\x1b[0m")
        assert strip_ansi(once) == once


class TestSegmentation:
    """Rule 2: ; | & and newlines all separate commands."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("ls; rm -rf /", ("ls", "rm -rf /")),
            ("ls && rm -rf /", ("ls", "rm -rf /")),
            ("ls || rm -rf /", ("ls", "rm -rf /")),
            ("ls\nrm -rf /", ("ls", "rm -rf /")),
            ("ls\r\nrm -rf /", ("ls", "rm -rf /")),
            ("&&&", ()),
            (";;;", ()),
            ("  ", ()),
        ],
    )
    def test_split(self, raw: str, expected: tuple[str, ...]) -> None:
        assert normalize(raw).segments == expected

    def test_pipe_preserves_every_stage(self) -> None:
        assert normalize("ls | grep foo | wc -l").segments == ("ls", "grep foo", "wc -l")

    def test_empty_segments_are_dropped_not_kept(self) -> None:
        # "a;;b" must be two segments, not four with two blanks — a blank
        # segment contributes nothing but does shift the dense feature vector.
        assert normalize("a;;b").segments == ("a", "b")


class TestAsciiFolding:
    """Rule 3a: output is guaranteed printable ASCII, and Turkish must survive."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("RM -RF /", "rm -rf /"),
            ("café", "cafe"),
            ("CAFÉ", "cafe"),
            ("Łódź", "lodz"),
            ("Œuvre", "oeuvre"),
            ("ﬁle", "file"),
            ("naïve", "naive"),
        ],
    )
    def test_folds_to_expected_ascii(self, raw: str, expected: str) -> None:
        assert normalize(raw).text == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # The Turkish letters that have no NFKD decomposition. These are the
            # cases a naive normaliser silently drops, and a dropped "ş" fuses
            # the two tokens around it.
            ("şifre sil", "sifre sil"),
            ("ŞİĞÜÖÇ", "siguoc"),
            ("günlük.log", "gunluk.log"),
            ("ĞÜŞİÖÇ", "GUSIOC".lower()),
            ("ıİ", "ii"),
        ],
    )
    def test_turkish_folds_correctly(self, raw: str, expected: str) -> None:
        assert normalize(raw).text == expected

    def test_dotted_capital_i_survives_as_lowercase_i(self) -> None:
        # "İ" decomposes to I + combining dot. The dot is a combining mark and
        # must be stripped before lowercasing, or we get a stray codepoint.
        assert normalize("İndirilenler").text == "indirilenler"

    @pytest.mark.parametrize(
        "raw",
        ["İndirilenler", "şifre", "günlük", "😀 rm -rf /", "rm -rf /", "café"],
    )
    def test_output_is_always_printable_ascii(self, raw: str) -> None:
        text = normalize(raw).text
        assert all(0x20 <= ord(ch) <= 0x7E for ch in text), repr(text)

    def test_invisible_characters_are_removed_not_spaces(self) -> None:
        # A zero-width space that survived would turn "rm" into "r<ZWSP>m" and
        # the model would never see the n-gram "rm ". Same for a BOM.
        assert normalize("r\u200bm -rf /").text == "rm -rf /"
        assert normalize("\ufeffrm -rf /").text == "rm -rf /"

    def test_non_breaking_space_becomes_a_separator_not_a_deletion(self) -> None:
        # U+00A0 must map to a space, not be dropped: dropping it would fuse
        # "rm" and "-rf" into one token.
        assert normalize("rm -rf\u00a0/").text == "rm -rf /"

    def test_dropped_characters_are_counted(self) -> None:
        result = normalize("😀😀 rm")
        assert result.dropped_non_ascii == 2

    def test_fold_ascii_reports_drops_separately(self) -> None:
        text, dropped = fold_ascii("é€")
        assert text == "e"
        assert dropped == 1


class TestWhitespaceCollapsing:
    """Rule 3d: runs of whitespace become one space, ends are trimmed."""

    @pytest.mark.parametrize(
        "raw",
        [
            "rm\t\t-rf\t\t/",
            "rm     -rf     /",
            "   rm -rf /   ",
            "rm -rf /",
        ],
    )
    def test_collapses_to_single_spaces(self, raw: str) -> None:
        assert normalize(raw).text == "rm -rf /"

    def test_interior_single_spaces_are_preserved(self) -> None:
        assert normalize("rm -rf /").text == "rm -rf /"


class TestViews:
    """Rules 4-5: two views feed the n-gram extractor."""

    def test_text_keeps_quotes_and_unquoted_does_not(self) -> None:
        result = normalize('r"m" -rf /')
        assert result.text == 'r"m" -rf /'
        assert result.unquoted == "rm -rf /"

    def test_views_are_text_then_unquoted(self) -> None:
        result = normalize("echo 'a'")
        assert result.views == (result.text, result.unquoted)

    def test_quote_obfuscation_is_invisible_to_the_feature_set(self) -> None:
        # This is the property the second view exists for. Without it, the model
        # would need to have seen every quoted variant during training.
        plain = presence_ngrams(normalize("rm -rf /").views)
        for variant in ['r"m" -rf /', "r'm' -rf /", "r''m'' -rf /"]:
            assert plain <= presence_ngrams(normalize(variant).views), variant

    def test_segments_rejoin_into_text_with_the_declared_separator(self) -> None:
        result = normalize("ls; rm -rf /")
        assert result.text == "ls ; rm -rf /"
        assert result.text.split(" ; ") == list(result.segments)


class TestNgrams:
    """Rules 5-6: presence semantics over the configured window."""

    def test_window_bounds_are_the_documented_ones(self) -> None:
        assert (NGRAM_MIN, NGRAM_MAX) == (2, 5)

    def test_presence_is_a_set(self) -> None:
        grams = presence_ngrams(normalize("rm -rf /").views)
        assert all(isinstance(g, str) for g in grams)
        assert len(grams) == len(set(grams))

    def test_repetition_does_not_inflate_the_feature_set(self) -> None:
        # The reason counts are not used. Repeating the command must not keep
        # growing the feature set, because an attacker who cannot read the
        # model could otherwise inflate its score arbitrarily by repetition.
        once = presence_ngrams(normalize("rm -rf /").views)
        thrice = presence_ngrams(normalize("rm -rf / rm -rf / rm -rf /").views)
        ten_times = presence_ngrams(normalize("rm -rf / " * 10).views)

        # Repeating adds only the n-grams that straddle the joins, so the set
        # is essentially saturated after the second copy.
        assert once <= thrice
        assert len(thrice) - len(once) < len(once)
        assert len(ten_times) == len(thrice), "repetition kept growing the feature set"

    def test_all_window_lengths_are_present(self) -> None:
        grams = presence_ngrams(normalize("abcdefgh").views)
        assert "ab" in grams, "shortest window"
        assert "abcd" in grams, "longest window"
        assert "defgh" in grams, "longest window, at the end of the string"
        assert "a" not in grams, "1-grams are outside the window"
        assert "abcdefgh" not in grams, "8-grams exceed the maximum window"

    def test_short_strings_yield_nothing(self) -> None:
        # A single character cannot produce a 2-gram, so the feature set is
        # empty and the command is scored on the dense features alone.
        assert presence_ngrams(normalize("a").views) == set()
        assert presence_ngrams(normalize("").views) == set()
        assert presence_ngrams(normalize("ab").views) == {"ab"}

    def test_slicing_is_safe_on_the_ascii_boundary(self) -> None:
        # If the ASCII invariant in fold_ascii ever breaks, this is where it
        # shows up: Python would slice by code point and JS by code unit.
        text = normalize("\U0001f600 rm -rf /").text
        assert text.isascii()
        assert presence_ngrams([text])


class TestRawIsNeverMutated:
    def test_raw_is_preserved_verbatim(self) -> None:
        raw = "  \x1b[31mRM -RF /  "
        assert normalize(raw).raw == raw
