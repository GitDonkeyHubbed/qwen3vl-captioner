"""
Qwen3-VL GGUF Inference Engine

Provides GPU-accelerated vision-language model inference using llama-cpp-python.
Supports single image captioning and video captioning (a batch of temporally
sampled frames sent as one multi-image turn), with streaming token output and
configurable generation parameters. Thread-safe for Qt signal integration.
"""

import base64
import io
import math
import time
from pathlib import Path
from typing import Callable, Optional

from PIL import Image, ImageOps


from engine.base import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_VIDEO_FRAMES,
    MAX_VIDEO_FRAMES,
    VIDEO_FRAME_MAX_DIM,
    apply_prefix_suffix,
    clean_caption,
)
from engine.cuda_setup import setup_cuda_dll_path, startup_failure_advice

# Setup CUDA DLL path before importing llama_cpp
setup_cuda_dll_path()

# A failed DLL load raises RuntimeError (not ImportError), so catch broadly
# and keep the error text — the GUI uses it to show actionable advice.
LLAMA_CPP_IMPORT_ERROR: Optional[str] = None
try:
    from llama_cpp import Llama
    from llama_cpp import llama_chat_format
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
    LLAMA_CPP_AVAILABLE = True
except Exception as _e:
    LLAMA_CPP_AVAILABLE = False
    LLAMA_CPP_IMPORT_ERROR = str(_e)


# Supported image file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"}


# Chat handler class name (in llama_cpp.llama_chat_format) per model family.
CHAT_FAMILY_HANDLERS = {
    "qwen3vl": "Qwen3VLChatHandler",
    "qwen35": "Qwen35ChatHandler",
    "gemma4": "Gemma4ChatHandler",
    "gemma3": "Gemma3ChatHandler",
}


def is_image_file(path: Path) -> bool:
    """Check if a file path has a supported image extension."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def infer_chat_family(model_path: str | Path) -> str:
    """
    Infer the chat-template family from a GGUF filename.

    Returns 'qwen3vl', 'qwen35', 'gemma4', or 'gemma3'. Anything not
    recognized falls back to 'qwen3vl' (the app's primary model line).
    """
    name = Path(model_path).name.lower()
    if "gemma-4" in name or "gemma4" in name:
        return "gemma4"
    if "gemma-3" in name or "gemma3" in name:
        return "gemma3"
    # The Qwen35 handler covers both the 3.5 and 3.6 model lines.
    if any(tag in name for tag in ("qwen3.5", "qwen3_5", "qwen35", "qwen3.6", "qwen3_6")):
        return "qwen35"
    return "qwen3vl"


def _resolve_chat_handler_cls(family: str):
    """
    Resolve the vision chat handler class for a model family.

    Tries the family's own handler first, then Qwen3VLChatHandler, then
    Qwen25VLChatHandler. The pinned JamePeng wheel ships all of them; the
    fallbacks guard other llama-cpp-python builds.
    """
    for name in (CHAT_FAMILY_HANDLERS.get(family), "Qwen3VLChatHandler", "Qwen25VLChatHandler"):
        if name:
            handler_cls = getattr(llama_chat_format, name, None)
            if handler_cls is not None:
                return handler_cls
    # Unreachable on the pinned wheel — a last resort for exotic builds (and
    # it keeps the direct legacy-handler import alive for external callers).
    return Qwen25VLChatHandler


def estimate_vision_tokens(width: int, height: int) -> int:
    """Estimate vision tokens for one frame (Qwen3-VL: one per 32x32 patch)."""
    return math.ceil(width / 32) * math.ceil(height / 32)


def _clamp_to_max_dim(img: Image.Image, max_dim: int) -> Image.Image:
    """Downscale so neither dimension exceeds max_dim, keeping aspect ratio."""
    # Clamp to >=1 px so an extreme aspect ratio (e.g. 10000x1) can't scale a
    # side to zero and crash resize.
    w, h = img.size
    if w > max_dim or h > max_dim:
        scale = max_dim / max(w, h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


def pil_image_to_data_uri(img: Image.Image, max_dim: int = 1280) -> str:
    """
    Resize a decoded image if needed (keeping aspect ratio) and convert to
    a base64 data URI suitable for llama-cpp-python vision input.

    Args:
        img: The PIL image (callers handle EXIF orientation before this).
        max_dim: Maximum dimension (width or height) to resize to.

    Returns:
        A data URI string like 'data:image/png;base64,...'
    """
    img = _clamp_to_max_dim(img, max_dim)

    # Convert to PNG bytes then base64
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def image_to_data_uri(image_path: Path, max_dim: int = 1280) -> str:
    """
    Load an image, resize if needed (keeping aspect ratio), and convert to
    a base64 data URI suitable for llama-cpp-python vision input.

    Args:
        image_path: Path to the image file.
        max_dim: Maximum dimension (width or height) to resize to.

    Returns:
        A data URI string like 'data:image/png;base64,...'
    """
    # Open inside a context manager so the source file handle is released
    # deterministically — exif_transpose + convert() force the pixel load, so
    # the detached RGB copy needs no further access to the file. (Prevents a
    # descriptor leak / Windows file lock during batch runs.)
    #
    # exif_transpose applies the EXIF Orientation tag (3/6/8 — ubiquitous in
    # phone/camera JPEGs). Without it the model receives sideways pixels and
    # captions a rotated scene — invisibly, because the Qt preview applies
    # orientation on its own.
    with Image.open(image_path) as src:
        img = ImageOps.exif_transpose(src).convert("RGB")

    return pil_image_to_data_uri(img, max_dim=max_dim)


class Qwen3VLEngine:
    """
    Inference engine for Qwen3-VL GGUF models via llama-cpp-python.

    Usage:
        engine = Qwen3VLEngine()
        engine.load_model(model_path, mmproj_path)
        caption = engine.caption_image(image_path, prompt)
        engine.unload()
    """

    def __init__(self):
        self.model: Optional[Llama] = None
        self.chat_handler = None
        self.model_path: Optional[Path] = None
        self.mmproj_path: Optional[Path] = None
        self.chat_family: Optional[str] = None
        self._n_ctx: int = 0
        self._is_loaded = False
        self._last_inference_time: float = 0.0

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and self.model is not None

    @property
    def last_inference_time(self) -> float:
        """Time in seconds for the last inference call."""
        return self._last_inference_time

    def load_model(
        self,
        model_path: str | Path,
        mmproj_path: str | Path,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        verbose: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None,
        chat_family: Optional[str] = None,
    ) -> None:
        """
        Load the GGUF model and vision encoder.

        Args:
            model_path: Path to the main .gguf model file.
            mmproj_path: Path to the mmproj vision encoder .gguf file.
            n_ctx: Context window size (tokens).
            n_gpu_layers: Number of layers to offload to GPU (-1 = all).
            verbose: Enable llama.cpp verbose logging.
            progress_callback: Optional callback for status messages.
            chat_family: Chat template family ('qwen3vl', 'qwen35', 'gemma4',
                'gemma3'). None infers it from the model filename.
        """
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError(
                startup_failure_advice(LLAMA_CPP_IMPORT_ERROR or "llama-cpp-python is not installed")
            )

        model_path = Path(model_path)
        mmproj_path = Path(mmproj_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not mmproj_path.exists():
            raise FileNotFoundError(f"Vision encoder (mmproj) not found: {mmproj_path}")

        # Unload any existing model first
        if self._is_loaded:
            self.unload()

        family = chat_family or infer_chat_family(model_path)

        if progress_callback:
            progress_callback("Loading vision encoder (mmproj)...")

        # Create the vision chat handler matching the model's chat template.
        # Qwen3-VL models get Qwen3VLChatHandler — their proper template,
        # verified multi-image — instead of the former Qwen2.5-VL one.
        handler_cls = _resolve_chat_handler_cls(family)
        self.chat_handler = handler_cls(
            clip_model_path=str(mmproj_path),
            verbose=verbose,
        )

        if progress_callback:
            progress_callback("Loading language model (this may take a minute)...")

        # Load the main model with GPU acceleration
        self.model = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,  # Use GPU acceleration
            chat_handler=self.chat_handler,
            verbose=verbose,
        )

        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.chat_family = family
        self._n_ctx = n_ctx
        self._is_loaded = True

        if progress_callback:
            progress_callback("Model loaded successfully.")

    def caption_image(
        self,
        image_path: str | Path,
        prompt: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        prefix: str = "",
        suffix: str = "",
        stream_callback: Optional[Callable[[str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> str:
        """
        Generate a caption for a single image.

        Args:
            image_path: Path to the image file.
            prompt: The captioning prompt/instruction.
            system_prompt: System message for the conversation.
            temperature: Sampling temperature (0 = greedy, higher = more creative).
            top_p: Nucleus sampling threshold.
            max_tokens: Maximum tokens to generate.
            prefix: Fixed text to prepend to the caption.
            suffix: Fixed text to append to the caption.
            stream_callback: Called with each generated token for streaming display.
            cancel_check: Function that returns True if generation should be cancelled.

        Returns:
            The complete generated caption string (with prefix/suffix if provided).
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Convert image to data URI
        image_uri = image_to_data_uri(image_path)

        # Build the chat messages with image
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        return self._generate(
            messages, temperature, top_p, max_tokens,
            stream_callback, cancel_check, prefix, suffix,
        )

    def caption_video(
        self,
        video_path: str | Path,
        prompt: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        prefix: str = "",
        suffix: str = "",
        stream_callback: Optional[Callable[[str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        num_frames: int = DEFAULT_VIDEO_FRAMES,
    ) -> str:
        """
        Generate a caption for a video from evenly sampled frames.

        The frames are sent as one multi-image user turn in temporal order,
        so the model sees the whole clip in a single generation.

        Args:
            video_path: Path to the video file.
            prompt: The captioning prompt/instruction.
            system_prompt: System message for the conversation.
            temperature: Sampling temperature (0 = greedy, higher = more creative).
            top_p: Nucleus sampling threshold.
            max_tokens: Maximum tokens to generate.
            prefix: Fixed text to prepend to the caption.
            suffix: Fixed text to append to the caption.
            stream_callback: Called with each generated token for streaming display.
            cancel_check: Function that returns True if generation should be cancelled.
            num_frames: Frames to sample (clamped to 2..MAX_VIDEO_FRAMES).

        Returns:
            The complete generated caption string (with prefix/suffix if provided).
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        num_frames = max(2, min(num_frames, MAX_VIDEO_FRAMES))

        # Imported lazily so cv2 (pulled in by engine.video) stays an
        # optional dependency for image-only use.
        from engine import video as video_module

        frames = video_module.sample_frames(video_path, num_frames=num_frames)

        image_parts = []
        vision_tokens = 0
        for frame in frames:
            frame = _clamp_to_max_dim(frame, VIDEO_FRAME_MAX_DIM)
            uri = pil_image_to_data_uri(frame, max_dim=VIDEO_FRAME_MAX_DIM)
            w, h = frame.size
            vision_tokens += estimate_vision_tokens(w, h)
            image_parts.append({"type": "image_url", "image_url": {"url": uri}})

        # Preflight the context budget: Qwen3-VL's M-RoPE cannot context-shift,
        # so overflowing n_ctx would be a hard crash mid-generation — refuse up
        # front instead. 1.15x covers vision-encoder/template overhead; 256
        # covers the text prompt and chat scaffolding.
        needed = math.ceil(1.15 * vision_tokens) + max_tokens + 256
        if needed > self._n_ctx:
            raise RuntimeError(
                f"{num_frames} video frames need ~{needed} context tokens "
                f"(vision + {max_tokens} generation), but the context window "
                f"is only {self._n_ctx}. Lower \"Frames per video\" and try again."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                # Frames in temporal order, then the instruction.
                "content": image_parts + [{"type": "text", "text": prompt}],
            },
        ]

        return self._generate(
            messages, temperature, top_p, max_tokens,
            stream_callback, cancel_check, prefix, suffix,
        )

    def _generate(
        self,
        messages: list[dict],
        temperature: float,
        top_p: float,
        max_tokens: int,
        stream_callback: Optional[Callable[[str], None]],
        cancel_check: Optional[Callable[[], bool]],
        prefix: str,
        suffix: str,
    ) -> str:
        """Run chat completion on prepared messages and post-process the caption."""
        start_time = time.perf_counter()

        if stream_callback:
            # Streaming mode
            caption_parts = []

            response = self.model.create_chat_completion(
                messages=messages,
                temperature=temperature if temperature > 0 else 0,
                top_p=top_p if temperature > 0 else 1.0,
                max_tokens=max_tokens,
                stream=True,
            )

            for chunk in response:
                if cancel_check and cancel_check():
                    break

                choices = chunk.get("choices") or [{}]
                delta = (choices[0] or {}).get("delta") or {}
                token_text = delta.get("content", "")
                if token_text:
                    caption_parts.append(token_text)
                    stream_callback(token_text)

            caption = "".join(caption_parts).strip()
        else:
            # Non-streaming mode
            response = self.model.create_chat_completion(
                messages=messages,
                temperature=temperature if temperature > 0 else 0,
                top_p=top_p if temperature > 0 else 1.0,
                max_tokens=max_tokens,
                stream=False,
            )

            choices = response.get("choices") or [{}]
            message = (choices[0] or {}).get("message") or {}
            caption = (message.get("content") or "").strip()

        self._last_inference_time = time.perf_counter() - start_time

        caption = clean_caption(caption)
        return apply_prefix_suffix(caption, prefix, suffix)

    def unload(self) -> None:
        """Unload the model and free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.chat_handler is not None:
            del self.chat_handler
            self.chat_handler = None
        self._is_loaded = False
        self.model_path = None
        self.mmproj_path = None
        self.chat_family = None
        self._n_ctx = 0

        # Force garbage collection to free VRAM
        import gc
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def get_model_info(self) -> dict:
        """Return a dictionary with model metadata for the status panel."""
        if not self.is_loaded:
            return {"status": "Not loaded"}

        return {
            "status": "Loaded",
            "model_file": self.model_path.name if self.model_path else "unknown",
            "mmproj_file": self.mmproj_path.name if self.mmproj_path else "unknown",
            "chat_family": self.chat_family or "unknown",
            "last_inference_s": round(self._last_inference_time, 2),
        }
