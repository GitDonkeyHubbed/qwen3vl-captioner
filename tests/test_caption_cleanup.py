"""Tests for caption post-processing (engine.base).

``clean_caption`` strips chat-template noise that VLMs prepend; the GGUF and
MLX engines both run every generated caption through it, so its behavior is
load-bearing for output quality.
"""

import pytest

from engine.base import (
    apply_prefix_suffix,
    clean_caption,
    frame_video_prompt,
    strip_reasoning,
)


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


# ── Reasoning traces must never reach a .txt sidecar ─────────────────────

def test_strip_reasoning_removes_a_think_block():
    """Qwen3.5 opens <think> before answering; only the answer is the caption."""
    raw = "<think>\nOkay, the user wants a caption. I see a woman...\n</think>\n\nA woman in a red coat."
    assert strip_reasoning(raw) == "A woman in a red coat."


def test_strip_reasoning_removes_a_gemma_thought_channel():
    raw = "<|channel|>think\nweighing the options\n<|channel|>\nA dog on a beach."
    assert strip_reasoning(raw) == "A dog on a beach."


def test_strip_reasoning_leaves_ordinary_captions_alone():
    assert strip_reasoning("A cat on a mat.") == "A cat on a mat."
    # An angle bracket that is not a reasoning tag must survive untouched.
    assert strip_reasoning("A sign reading <OPEN>.") == "A sign reading <OPEN>."


def test_strip_reasoning_keeps_an_unclosed_block():
    """Budget exhausted mid-thought: there is no caption to recover.

    Returning "" here would silently write an empty sidecar; returning the
    trace lets the caller see what happened.
    """
    raw = "<think>\nStill reasoning when the token budget ran out"
    assert strip_reasoning(raw) == raw


def test_clean_caption_strips_a_think_block_then_its_prefix():
    """The two passes compose: reasoning first, then the "Caption:" noise."""
    raw = "<think>\nplanning\n</think>\nCaption: A blue bicycle."
    assert clean_caption(raw) == "A blue bicycle."


def test_clean_caption_think_stripping_is_idempotent():
    once = clean_caption("<think>\nplanning\n</think>\nA blue bicycle.")
    assert clean_caption(once) == once


# ── Video prompts state the clip framing in text ─────────────────────────

def test_frame_video_prompt_states_the_count_and_keeps_the_prompt():
    assert frame_video_prompt("Describe the video.", 4) == (
        "The 4 images above are frames sampled in order from a single video "
        "clip. Describe the video."
    )


def test_frame_video_prompt_is_shared_by_both_backends():
    """Both engines import this helper, so the framing cannot drift apart."""
    from engine import inference, mlx_engine

    assert inference.frame_video_prompt is frame_video_prompt
    assert mlx_engine.frame_video_prompt is frame_video_prompt
