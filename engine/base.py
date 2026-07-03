"""
Shared engine interface and caption post-processing.

The app supports two inference backends behind one duck-typed interface:

  - Qwen3VLEngine (engine/inference.py): GGUF models via llama-cpp-python.
    Works on Windows (CUDA), macOS (Metal), and Linux. Needs an mmproj
    vision encoder file next to the model.
  - MlxVlmEngine (engine/mlx_engine.py): MLX models via Apple's mlx-vlm.
    Apple Silicon only. Models are folders of safetensors — no mmproj.

Every engine implements:
  load_model(model_path, mmproj_path, *, progress_callback=None)
    (mmproj_path is REQUIRED by the GGUF engine — pairing a model with a
    missing/mismatched vision encoder crashes llama.cpp natively — and is
    accepted-but-ignored by the MLX engine, whose models embed the tower)
  caption_image(image_path, prompt, ..., stream_callback, cancel_check) -> str
  unload()
  get_model_info() -> dict
  is_loaded -> bool          (property)
  last_inference_time -> float (property)
"""

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that describes images accurately and in detail."
)

# VLMs often prepend formatting noise like ":", "Answer:", "Caption:", etc.
_STRIP_PREFIXES = [
    "answer:", "caption:", "description:", "response:",
    "here is", "here's", "sure,", "sure.",
]


def clean_caption(caption: str) -> str:
    """Strip chat-template artifacts from a generated caption."""
    cleaned = caption.strip()
    for pfx in _STRIP_PREFIXES:
        if cleaned.lower().startswith(pfx):
            cleaned = cleaned[len(pfx):]
            break
    # Strip any remaining leading colons, dashes, dots, asterisks, whitespace
    cleaned = cleaned.lstrip(":;-–—.*• \t\n")
    return cleaned if cleaned else caption.strip()


def apply_prefix_suffix(caption: str, prefix: str = "", suffix: str = "") -> str:
    """Apply the user's fixed prefix/suffix to a cleaned caption."""
    if prefix:
        caption = prefix.strip() + " " + caption
    if suffix:
        caption = caption + " " + suffix.strip()
    return caption
