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

from PIL import Image, ImageOps

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that describes images accurately and in detail."
)

# Longest side, in pixels, an image is scaled to before it reaches a vision
# encoder. Every backend must apply it: a native-resolution 16.7 MP photo costs
# seconds of encode time and gives no better caption than the clamped one.
MAX_IMAGE_DIM = 1280


def load_image_for_inference(image_path, max_dim: int = MAX_IMAGE_DIM) -> Image.Image:
    """Open an image, apply EXIF orientation, and clamp its longest side.

    JPEG sources are decoded through `draft()`, which lets libjpeg downscale
    by 1/2, 1/4 or 1/8 *while decoding*. Without it a 40 MP photo was fully
    decoded and then LANCZOS-resized from full resolution on the caption
    worker — roughly half a second of pure CPU per image, straight onto batch
    wall time.

    exif_transpose applies the EXIF Orientation tag (3/6/8 — ubiquitous in
    phone/camera JPEGs). Without it the model receives sideways pixels and
    captions a rotated scene, invisibly, because the Qt preview applies
    orientation on its own.
    """
    # Open inside a context manager so the source file handle is released
    # deterministically — exif_transpose + convert() force the pixel load, so
    # the detached RGB copy needs no further access to the file. (Prevents a
    # descriptor leak / Windows file lock during batch runs.)
    with Image.open(image_path) as src:
        w, h = src.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            src.draft("RGB", (max(1, int(w * scale)), max(1, int(h * scale))))
        img = ImageOps.exif_transpose(src).convert("RGB")

    # draft() only lands on a power-of-two fraction, so a final resize is still
    # needed — but from a much smaller image. Clamp to >=1 px so an extreme
    # aspect ratio (e.g. 10000x1) can't scale a side to zero and crash resize.
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS
        )
    return img

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
