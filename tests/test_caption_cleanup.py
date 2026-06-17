"""Tests for caption post-processing (engine.base).

``clean_caption`` strips chat-template noise that VLMs prepend; the GGUF and
MLX engines both run every generated caption through it, so its behavior is
load-bearing for output quality.
"""

import pytest

from engine.base import apply_prefix_suffix, clean_caption


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  A red car  ", "A red car"),
        ("Caption: A red car", "A red car"),
        ("caption: a red car", "a red car"),
        ("Answer: 42", "42"),
        ("Description: a dog", "a dog"),
        ("Response: ok", "ok"),
        ("Here is a photo of a cat", "a photo of a cat"),
        ("Here's the scene", "the scene"),
        ("Sure, a sunset", "a sunset"),
        ("- a bullet caption", "a bullet caption"),
        (":: weird colons", "weird colons"),
        ("**bold lead", "bold lead"),
    ],
)
def test_clean_caption_strips_known_noise(raw, expected):
    assert clean_caption(raw) == expected


def test_clean_caption_strips_only_first_prefix():
    # The engine breaks after the first matching prefix — nested labels stay.
    assert clean_caption("Answer: Caption: x") == "Caption: x"


def test_clean_caption_empty_stays_empty():
    assert clean_caption("") == ""
    assert clean_caption("   ") == ""


def test_clean_caption_prefix_only_falls_back_to_original():
    # A caption that is *only* a prefix must not collapse to an empty string.
    assert clean_caption("Caption:") == "Caption:"


def test_clean_caption_is_idempotent():
    once = clean_caption("Caption: A red car")
    assert clean_caption(once) == once


def test_clean_caption_preserves_internal_punctuation():
    assert clean_caption("A cat: sitting") == "A cat: sitting"


def test_apply_prefix_only():
    assert apply_prefix_suffix("a cat", prefix="photo of") == "photo of a cat"


def test_apply_suffix_only():
    assert apply_prefix_suffix("a cat", suffix="indoors") == "a cat indoors"


def test_apply_prefix_and_suffix():
    assert apply_prefix_suffix("cat", prefix="a", suffix="b") == "a cat b"


def test_apply_prefix_suffix_noop_when_empty():
    assert apply_prefix_suffix("a cat") == "a cat"
    assert apply_prefix_suffix("a cat", "", "") == "a cat"


def test_apply_prefix_suffix_strips_affix_whitespace():
    assert (
        apply_prefix_suffix("cat", prefix="  photo  ", suffix="  now  ")
        == "photo cat now"
    )
