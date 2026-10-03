"""Keep E4B generation prompts consistent with its embedded GGUF template."""

from types import SimpleNamespace

import pytest

from engine import inference


MODEL_TURN = "<|turn>model\n"
THOUGHT_PREFILL = "<|channel>thought\n<channel|>"


@pytest.fixture
def load_engine(monkeypatch, tmp_path):
    def load(metadata=None, filename="Gemma-4-E4B.gguf", family="gemma4"):
        class Template:
            text = "image-and-history" + MODEL_TURN + THOUGHT_PREFILL

            def render(self, **context):
                self.context = context
                return self.text

        template = Template()

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
    if not adapt:
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
