"""Keep Gemma4 media input and E4B generation prompts consistent with MTMD."""

from copy import deepcopy
import gc
from types import SimpleNamespace
import weakref

import pytest

from engine import inference


MODEL_TURN = "<|turn>model\n"
THOUGHT_PREFILL = "<|channel>thought\n<channel|>"


@pytest.fixture
def load_engine(monkeypatch, tmp_path):
    def load(metadata=None, filename="Gemma-4-E4B.gguf", family="gemma4", template=None):
        class Template:
            text = "image-and-history" + MODEL_TURN + THOUGHT_PREFILL

            def render(self, **context):
                self.context = context
                return self.text

        template = template or Template()

        class Handler:
            def __init__(self, clip_model_path, verbose=False):
                self.chat_template = template

        monkeypatch.setattr(inference, "LLAMA_CPP_AVAILABLE", True)
        monkeypatch.setattr(
            inference, "llama_chat_format",
            SimpleNamespace(**{inference.CHAT_FAMILY_HANDLERS[family]: Handler}),
            raising=False,
        )
        monkeypatch.setattr(
            inference, "Llama",
            lambda **kwargs: SimpleNamespace(metadata=metadata or {}, n_ctx=lambda: 4096),
            raising=False,
        )
        model = tmp_path / filename
        mmproj = tmp_path / "mmproj.gguf"
        model.touch()
        mmproj.touch()
        engine = inference.Qwen3VLEngine()
        engine.load_model(model, mmproj, chat_family=family)
        return engine, template

    return load


def test_e4b_load_removes_only_generation_prefill_and_forwards_context(load_engine):
    engine, original = load_engine(
        metadata={"general.name": "Gemma-4-E4B-Custom", "general.architecture": "gemma4"},
        filename="renamed.gguf",
    )
    history = "<|turn>model\n" + THOUGHT_PREFILL + "An earlier answer.<turn|>"
    original.text = history + "<|image|>data:image/jpeg;base64,abc" + MODEL_TURN + THOUGHT_PREFILL
    messages = [{"role": "user", "content": "Describe the image."}]
    context = dict(messages=messages, add_generation_prompt=True, enable_thinking=False)

    rendered = engine.chat_handler.chat_template.render(**context)

    assert rendered == history + "<|image|>data:image/jpeg;base64,abc" + MODEL_TURN
    assert original.context == context
    assert original.context["messages"] is messages


@pytest.mark.parametrize("metadata,filename,family,adapt", [
    ({}, "Gemma-4-E4B-Q4_K_M.gguf", "gemma4", True),
    ({"general.name": "Gemma 4 E4B"}, "renamed.gguf", "gemma4", True),
    ({"general.name": "Gemma-4-31B"}, "Gemma-4-E4B.gguf", "gemma4", False),
    ({"general.name": "Gemma-4-E2B"}, "renamed.gguf", "gemma4", False),
    ({"general.name": "Gemma-4-E4B", "general.architecture": "gemma3"},
     "Gemma-4-E4B.gguf", "gemma4", False),
    ({"general.name": "Gemma-4-E4B"}, "renamed.gguf", "gemma3", False),
    ({}, "Gemma-4-E4B.gguf", "qwen3vl", False),
])
def test_prompt_adaptation_is_limited_to_gemma4_e4b(
    load_engine, metadata, filename, family, adapt,
):
    engine, original = load_engine(metadata, filename, family)
    rendered = engine.chat_handler.chat_template.render(add_generation_prompt=True)
    assert rendered == ("image-and-history" + MODEL_TURN if adapt else original.text)
    if family != "gemma4" or metadata.get("general.architecture") == "gemma3":
        assert engine.chat_handler.chat_template is original


@pytest.mark.parametrize("text,generation", [
    ("prompt" + MODEL_TURN + THOUGHT_PREFILL, False),
    ("prompt" + MODEL_TURN, True),  # Already-correct future wheel.
    ("prompt<|turn>user\n" + THOUGHT_PREFILL, True),
    ("prompt" + MODEL_TURN + THOUGHT_PREFILL + "existing answer", True),
])
def test_prompt_adapter_preserves_other_template_output(load_engine, text, generation):
    engine, original = load_engine()
    original.text = text
    assert engine.chat_handler.chat_template.render(add_generation_prompt=generation) == text


class ImageTemplate:
    """Render the pinned handler's image branch without loading native llama.

    Gemma4ChatHandler emits ``'<|image|>' + url_val`` in its content loop.
    A fixed future wheel can emit just ``url_val``; MTMD adds the boundaries.
    """

    def __init__(self, image_prefix="<|image|>"):
        self.image_prefix = image_prefix

    def render(self, **context):
        text = ""
        for message in context["messages"]:
            role = "model" if message["role"] == "assistant" else message["role"]
            text += "<|turn>" + role + "\n"
            content = message["content"]
            if isinstance(content, str):
                text += content.strip()
            else:
                for item in content:
                    if item["type"] == "text":
                        text += item["text"].strip()
                    elif item["type"] == "image_url":
                        image = item["image_url"]
                        url = image if isinstance(image, str) else image["url"]
                        text += self.image_prefix + url
                    elif item["type"] == "audio_url":
                        text += "<|audio|>" + item["audio_url"]
            text += "<turn|>\n"
        if context.get("add_generation_prompt"):
            text += MODEL_TURN + THOUGHT_PREFILL
        return text


@pytest.mark.parametrize("name,strip_thought", [
    ("Gemma-4-E4B", True),
    ("Gemma-4-E2B", False),
    ("Gemma-4-31B", False),
    ("Gemma-4-26B-A4B", False),
])
@pytest.mark.parametrize("mapping_url", [False, True])
def test_all_gemma4_images_omit_hard_placeholder(
    load_engine, name, strip_thought, mapping_url,
):
    uri = "data:image/jpeg;base64,abc"
    image_url = {"url": uri, "detail": "high"} if mapping_url else uri
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": image_url},
    ]}]
    original_messages = deepcopy(messages)
    engine, _ = load_engine(
        metadata={"general.name": name}, template=ImageTemplate(),
    )

    rendered = engine.chat_handler.chat_template.render(
        messages=messages, add_generation_prompt=True,
    )

    suffix = MODEL_TURN + ("" if strip_thought else THOUGHT_PREFILL)
    assert rendered == "<|turn>user\nDescribe this image." + uri + "<turn|>\n" + suffix
    assert messages == original_messages
    # This is the pinned MTMD handler's next step. The native tokenizer will
    # insert BOI + embeddings + EOI at this marker, so no hard image token
    # should accompany it.
    mtmd_text = rendered.replace(uri, "<__media__>")
    assert "<|image|>" not in mtmd_text
    assert mtmd_text.count("<__media__>") == 1


@pytest.mark.parametrize("image_prefix", ["<|image|>", ""])
def test_video_frames_and_history_preserve_literal_image_tokens(load_engine, image_prefix):
    first_uri = "data:image/jpeg;base64,first"
    second_uri = "data:image/jpeg;base64,second"
    messages = [
        {"role": "user", "content": "Keep the literal <|image|> in the history."},
        {"role": "assistant", "content": "The token is <|image|>."},
        {"role": "user", "content": [
            {"type": "text", "text": "Frame 1, literal token <|image|>"},
            {"type": "image_url", "image_url": {"url": first_uri}},
            {"type": "text", "text": "Frame 2 "},
            {"type": "image_url", "image_url": second_uri},
            {"type": "text", "text": "Repeated frame "},
            {"type": "image_url", "image_url": first_uri},
            {"type": "audio_url", "audio_url": "data:audio/wav;base64,audio"},
        ]},
    ]
    original_messages = deepcopy(messages)
    engine, _ = load_engine(template=ImageTemplate(image_prefix=image_prefix))

    rendered = engine.chat_handler.chat_template.render(
        messages=messages, add_generation_prompt=True,
    )

    assert rendered == (
        "<|turn>user\nKeep the literal <|image|> in the history.<turn|>\n"
        "<|turn>model\nThe token is <|image|>.<turn|>\n"
        "<|turn>user\nFrame 1, literal token <|image|>" + first_uri
        + "Frame 2" + second_uri + "Repeated frame" + first_uri
        + "<|audio|>data:audio/wav;base64,audio<turn|>\n" + MODEL_TURN
    )
    assert messages == original_messages


def test_media_urls_in_text_do_not_lose_literal_prefixes(load_engine):
    uri = "data:image/jpeg;base64,abc"
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Literal: <|image|>" + uri + "\n"},
        {"type": "image_url", "image_url": uri},
    ]}]
    engine, _ = load_engine(template=ImageTemplate())

    rendered = engine.chat_handler.chat_template.render(
        messages=messages, add_generation_prompt=True,
    )

    assert rendered == (
        "<|turn>user\nLiteral: <|image|>" + uri + uri + "<turn|>\n" + MODEL_TURN
    )


@pytest.mark.parametrize("name", ["Gemma-4-E4B", "Gemma-4-12B"])
def test_bos_survives_repeated_media_requests_without_duplicating_fresh_bos(load_engine, name):
    class BosTemplate(ImageTemplate):
        def render(self, **context):
            return context["bos_token"] + super().render(**context)

    engine, _ = load_engine(metadata={"general.name": name}, template=BosTemplate())
    engine.model.token_bos = lambda: 2
    detokenizations = []

    def detokenize(tokens, special=False):
        detokenizations.append((tokens, special))
        return b"<bos>" if special else b""

    engine.model.detokenize = detokenize
    for n_tokens, uris in [(0, ["first"]), (100, ["second"]), (200, ["frame1", "frame2"]),
                           (0, ["after-reset"]), (100, ["last"])]:
        engine.model.n_tokens = n_tokens
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": uri}} for uri in uris
        ]}]
        context = dict(bos_token="", messages=messages)

        rendered = engine.chat_handler.chat_template.render(**context)

        # MTMD supplies automatic BOS only on the fresh/reset ledger. Every
        # reused ledger needs the literal special token in its template.
        literal_bos = "<bos>" if n_tokens else ""
        assert rendered == literal_bos + "<|turn>user\n" + "".join(uris) + "<turn|>\n"
        assert context["bos_token"] == ""
    assert detokenizations == [([2], True)] * 3


def test_correct_future_bos_is_forwarded_unchanged(load_engine):
    engine, original = load_engine(metadata={"general.name": "Gemma-4-12B"})
    engine.model.n_tokens = 100
    context = dict(bos_token="<bos>", add_generation_prompt=True)

    engine.chat_handler.chat_template.render(**context)

    assert original.context == context


def test_template_does_not_keep_engine_or_unloaded_model_alive(load_engine):
    engine, _ = load_engine()
    adapter = engine.chat_handler.chat_template
    engine.unload()
    engine_ref = weakref.ref(engine)
    del engine
    gc.collect()

    assert engine_ref() is None
    assert adapter.model_provider() is None
