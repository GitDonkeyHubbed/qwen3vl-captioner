"""Guards on the GGUF caption path (``Qwen3VLEngine.caption_image``).

The model is a stand-in. These cover the sampling gate, malformed
chat-completion payloads, and cancel — the ways a caption run corrupts or
drops its text without a real llama.cpp load.
"""

import pytest
from PIL import Image

from engine.inference import Qwen3VLEngine


def _image(tmp_path):
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (4, 4), (1, 2, 3)).save(path, format="JPEG")
    return path


class _Model:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(self.response)
        return self.response


def _engine(tmp_path, model):
    engine = Qwen3VLEngine()
    engine.model = model
    engine._is_loaded = True
    return engine, _image(tmp_path)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("temperature", "top_p", "expect_temp", "expect_top_p"),
    [(0, 0.1, 0, 1.0), (0.6, 0.4, 0.6, 0.4)],
)
def test_sampling_kwargs_follow_the_temperature_gate(
    tmp_path, stream, temperature, top_p, expect_temp, expect_top_p,
):
    """temperature 0 is greedy: top_p must be 1.0 or nucleus sampling
    collapses the greedy decode. A positive temperature must keep the
    caller's top_p. Both the streaming and the blocking branches."""
    if stream:
        response = [{"choices": [{"delta": {"content": "ok"}}]}]
    else:
        response = {"choices": [{"message": {"content": "ok"}}]}
    engine, path = _engine(tmp_path, _Model(response))
    seen = []
    out = engine.caption_image(
        path, "describe",
        temperature=temperature, top_p=top_p,
        stream_callback=seen.append if stream else None,
    )

    assert out == "ok"
    call = engine.model.calls[0]
    assert call["stream"] is stream
    assert call["temperature"] == expect_temp
    assert call["top_p"] == expect_top_p
    if stream:
        assert seen == ["ok"]


def test_malformed_stream_chunks_do_not_drop_the_caption(tmp_path):
    """An empty or null ``choices`` list used to raise mid-stream and
    throw away the tokens already produced."""
    chunks = [
        {"choices": [{"delta": {"content": "red "}}]},
        {},
        {"choices": []},
        {"choices": None},
        {"choices": [None]},
        {"choices": [{}]},
        {"choices": [{"delta": None}]},
        {"choices": [{"delta": {}}]},
        {"choices": [{"delta": {"content": ""}}]},
        {"choices": [{"delta": {"content": "car"}}]},
    ]
    engine, path = _engine(tmp_path, _Model(chunks))
    seen = []

    out = engine.caption_image(path, "describe", stream_callback=seen.append)

    assert out == "red car"
    assert seen == ["red ", "car"]


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"choices": []},
        {"choices": None},
        {"choices": [None]},
        {"choices": [{}]},
        {"choices": [{"message": None}]},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": None}}]},
    ],
)
def test_nonstream_without_text_returns_empty(tmp_path, response):
    engine, path = _engine(tmp_path, _Model(response))
    assert engine.caption_image(path, "describe") == ""


def test_nonstream_result_is_cleaned_and_affixed(tmp_path):
    engine, path = _engine(
        tmp_path,
        _Model({"choices": [{"message": {"content": "Answer: a cat"}}]}),
    )
    assert engine.caption_image(
        path, "describe", prefix="pre", suffix="suf",
    ) == "pre a cat suf"


def test_cancel_stops_before_later_tokens(tmp_path):
    chunks = [
        {"choices": [{"delta": {"content": "a cat"}}]},
        {"choices": [{"delta": {"content": " in a hat"}}]},
    ]
    engine, path = _engine(tmp_path, _Model(chunks))
    seen = []

    out = engine.caption_image(
        path, "describe",
        prefix="pre", suffix="suf",
        stream_callback=seen.append,
        cancel_check=lambda: len(seen) >= 1,
    )

    assert seen == ["a cat"]
    assert out == "pre a cat suf"


def test_caption_sends_the_image_and_the_prompt(tmp_path):
    engine, path = _engine(
        tmp_path,
        _Model({"choices": [{"message": {"content": "ok"}}]}),
    )

    out = engine.caption_image(
        path, "describe it", system_prompt="sys", temperature=0.2, top_p=0.5,
    )

    assert out == "ok"
    messages = engine.model.calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": "sys"}
    user = messages[1]["content"]
    assert user[1] == {"type": "text", "text": "describe it"}
    uri = user[0]["image_url"]["url"]
    assert uri.startswith("data:image/jpeg;base64,")
    assert len(uri) > len("data:image/jpeg;base64,")
