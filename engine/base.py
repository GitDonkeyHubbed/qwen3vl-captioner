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
    accepted-but-ignored by the MLX engine, whose models embed the tower.
    The GGUF engine additionally accepts chat_family= to pick the chat
    template handler; the MLX engine does NOT take it, so callers must
    only pass it to GGUF loads — see ModelLoadWorker in gui/main_window.)
  caption_image(image_path, prompt, ..., stream_callback, cancel_check) -> str
  caption_video(video_path, prompt, ..., num_frames=DEFAULT_VIDEO_FRAMES) -> str
    (samples evenly-spaced frames and captions the clip in one multi-image
    turn; frame limits below are shared so both backends behave identically)
  unload()
  get_model_info() -> dict
  is_loaded -> bool          (property)
  last_inference_time -> float (property)
"""

import re

from PIL import Image, ImageOps

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that describes images accurately and in detail."
)

# Video captioning limits shared by both engines so they sample and downscale
# identically. Frames are encoded smaller than single images (640 vs 1280)
# because N frames share one context window / unified-memory budget and
# vision tokens grow with pixel area.
VIDEO_FRAME_MAX_DIM = 640
DEFAULT_VIDEO_FRAMES = 8
MAX_VIDEO_FRAMES = 16

# Longest side, in pixels, an image is scaled to before it reaches a vision
# encoder. Every backend must apply it: a native-resolution 16.7 MP photo costs
# seconds of encode time and gives no better caption than the clamped one.
MAX_IMAGE_DIM = 1280


def clamp_image_dim(img: Image.Image, max_dim: int) -> Image.Image:
    """Downscale a decoded image so neither side exceeds max_dim.

    Shared by the file path (load_image_for_inference) and the video paths,
    which clamp already-decoded frames. Sides are clamped to >=1 px so an
    extreme aspect ratio (e.g. 10000x1) can't scale one to zero and crash
    resize.
    """
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    scale = max_dim / max(w, h)
    return img.resize(
        (max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS
    )


def frame_video_prompt(prompt: str, n_frames: int) -> str:
    """Prefix a caption prompt with the fact that the images are one clip.

    Without it the model sees N unrelated pictures and describes them one by
    one. The chat handlers' own ``add_vision_id`` labelling ("Picture N:")
    would say something similar, but it is switched off so the single-image
    prompt every existing user captions with is unchanged — and Gemma-4's
    handler has no such flag at all. Stating it in the text works for every
    family and for both backends, which is why it lives here rather than in
    either engine.
    """
    return (
        f"The {n_frames} images above are frames sampled in order from a "
        f"single video clip. {prompt}"
    )


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
    # needed — but from a much smaller image.
    return clamp_image_dim(img, max_dim)


# VLMs often prepend formatting noise like ":", "Answer:", "Caption:", etc.
_STRIP_PREFIXES = [
    "answer:", "caption:", "description:", "response:",
    "here is", "here's", "sure,", "sure.",
]


# Reasoning models (Qwen3.5, Gemma-4) emit a thinking block before the answer.
# The handlers are constructed with thinking disabled, but a model can still
# open one on its own, and a stray trace must never reach a .txt sidecar.
_THINK_BLOCK = re.compile(
    r"\A\s*(?:<think>|<\|channel\|>\s*think).*?(?:</think>|<\|/?channel\|>)\s*",
    re.DOTALL | re.IGNORECASE,
)


def strip_reasoning(caption: str) -> str:
    """Remove a leading <think>...</think> (or Gemma thought-channel) block.

    Returns the text after the block. An unclosed block means the model spent
    its whole budget reasoning, so there is no caption to recover: the text is
    returned unchanged for the caller to surface rather than silently saving a
    monologue.
    """
    stripped = _THINK_BLOCK.sub("", caption, count=1)
    if stripped != caption:
        return stripped
    return caption


def clean_caption(caption: str) -> str:
    """Strip chat-template artifacts from a generated caption."""
    cleaned = strip_reasoning(caption).strip()
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
