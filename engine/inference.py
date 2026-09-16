"""
Qwen3-VL GGUF Inference Engine

Provides GPU-accelerated vision-language model inference using llama-cpp-python.
Supports single image captioning and video captioning (a batch of temporally
sampled frames sent as one multi-image turn), with streaming token output and
configurable generation parameters. Thread-safe for Qt signal integration.
"""

import base64
import inspect
import io
import math
import time
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from engine.base import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_VIDEO_FRAMES,
    MAX_IMAGE_DIM,
    MAX_VIDEO_FRAMES,
    VIDEO_FRAME_MAX_DIM,
    apply_prefix_suffix,
    clamp_image_dim,
    VideoCancelled,
    clean_caption,
    frame_video_prompt,
    load_image_for_inference,
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
    "qwen25vl": "Qwen25VLChatHandler",
    "gemma4": "Gemma4ChatHandler",
    "gemma3": "Gemma3ChatHandler",
}


def is_image_file(path: Path) -> bool:
    """Check if a file path has a supported image extension."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def infer_chat_family(model_path: str | Path) -> str:
    """
    Infer the chat-template family from a GGUF filename.

    Returns 'qwen3vl', 'qwen35', 'qwen25vl', 'gemma4', or 'gemma3'. Anything
    not recognized falls back to 'qwen3vl' (the app's primary model line).
    """
    name = Path(model_path).name.lower()
    if "gemma-4" in name or "gemma4" in name:
        return "gemma4"
    if "gemma-3" in name or "gemma3" in name:
        return "gemma3"
    # The Qwen35 handler covers both the 3.5 and 3.6 model lines.
    if any(tag in name for tag in ("qwen3.5", "qwen3_5", "qwen35", "qwen3.6", "qwen3_6")):
        return "qwen35"
    # Browsed legacy Qwen2-VL / Qwen2.5-VL files keep the Qwen2.5-VL handler —
    # without the plain-Qwen2 tags they would fall through to the Qwen3-VL
    # template. The Qwen2-VL tags spell out the "-vl" so they cannot swallow
    # "qwen2.5"/"qwen2_5" names, and no Qwen3 name contains them.
    if any(tag in name for tag in ("qwen2.5", "qwen2_5", "qwen25",
                                   "qwen2-vl", "qwen2vl", "qwen2_vl")):
        return "qwen25vl"
    return "qwen3vl"


# Families whose handlers share the ChatML lineage (<|im_start|> turns), so
# one can stand in for another on a build missing the newest handler. Gemma is
# deliberately absent: its template uses <start_of_turn>/<end_of_turn> and a
# different image-token protocol, so a Qwen handler would not degrade a Gemma
# model's output, it would garble it.
_CHATML_FALLBACKS = ("Qwen3VLChatHandler", "Qwen25VLChatHandler")


def _resolve_chat_handler_cls(family: str):
    """
    Resolve the vision chat handler class for a model family.

    Tries the family's own handler first. If this build does not ship it, a
    Qwen family falls back along _CHATML_FALLBACKS; a non-ChatML family (Gemma)
    raises instead.

    Substituting across templates is not a degraded mode, it is a wrong one:
    the handler supplies the chat template and image-token protocol, so a
    Gemma model driven by a Qwen handler emits the wrong control tokens and
    either produces nonsense or fails inside media evaluation, with nothing
    pointing at the cause. The pinned JamePeng wheel ships every handler, so
    this only bites on other builds — which is exactly where a clear error
    beats silent garbage.
    """
    own = CHAT_FAMILY_HANDLERS.get(family)
    if own:
        handler_cls = getattr(llama_chat_format, own, None)
        if handler_cls is not None:
            return handler_cls

    if own and own not in _CHATML_FALLBACKS and family.startswith("gemma"):
        raise RuntimeError(
            f"This llama-cpp-python build has no {own}, which the {family} "
            f"chat template requires. Falling back to a Qwen handler would "
            f"send the wrong control tokens. Install the pinned wheel (see "
            f"setup.bat / setup.sh), or choose a Qwen3-VL model instead."
        )

    for name in _CHATML_FALLBACKS:
        handler_cls = getattr(llama_chat_format, name, None)
        if handler_cls is not None:
            return handler_cls
    # Unreachable on the pinned wheel — a last resort for exotic builds (and
    # it keeps the direct legacy-handler import alive for external callers).
    return Qwen25VLChatHandler


def _construct_chat_handler(handler_cls, mmproj_path, verbose: bool):
    """Build a vision chat handler with caption-safe defaults.

    Two of the pinned wheel's defaults are wrong for captioning:

    * ``enable_thinking=True`` (Qwen35ChatHandler, Gemma4ChatHandler) ends the
      generation prompt inside an open ``<think>`` block, so the model's
      reasoning trace IS the caption and usually exhausts the token budget
      before any description appears.
    * ``add_vision_id=True`` (Qwen3VLChatHandler, Qwen35ChatHandler) prefixes
      every image with ``Picture N:``. That labelling is aimed at multi-image
      prompts, and on the single-image path it is redundant noise.
      ``caption_video`` states the frame ordering in its text prompt instead,
      which also works for Gemma-4, whose handler has no such flag.

    Switching Qwen3-VL to its own handler DOES change the single-image prompt
    (it is a deliberate correctness fix, not a no-op). Rendered from the
    pinned wheel's jinja templates on the same message list, the old
    Qwen25VLChatHandler emitted
    ``<|im_start|>user\\nPicture 1: <|vision_start|> <img> <|vision_end|>PROMPT``
    and Qwen3VLChatHandler with ``add_vision_id=False`` emits
    ``<|im_start|>user\\n<|vision_start|><img><|vision_end|>PROMPT`` — the
    hardcoded prefix and the spaces around the placeholder are gone, and the
    trailing newline after ``<|im_start|>assistant`` that Qwen3-VL's template
    specifies is added. An A/B over 7 images (including OCR and chart reading)
    found no factual regression from the change.

    Unknown keywords are dropped by inspecting the constructor signature so
    other llama-cpp-python builds, whose handlers may not accept them, still
    load. Filtering up front (rather than retrying on TypeError) means a
    TypeError raised from *inside* a compatible constructor propagates on the
    first attempt instead of triggering silent retries that could repeat the
    constructor's side effects before surfacing the real error.

    A constructor whose signature cannot be read at all — an extension type
    with no ``__text_signature__`` — falls back to trying progressively
    fewer flags, because there is nothing left to inspect. That path does
    risk repeated calls, which is exactly what inspection avoids elsewhere;
    it is confined to the case with no alternative, since refusing to load a
    model is the worse outcome. Every handler in the pinned wheel is a plain
    Python class, so this is the compatibility path, not the normal one.
    """
    kwargs = {"clip_model_path": str(mmproj_path), "verbose": verbose}
    extra = {"enable_thinking": False, "add_vision_id": False}

    try:
        params = inspect.signature(handler_cls).parameters
    except (TypeError, ValueError):
        params = None

    if params is not None:
        if not any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        ):
            # No **kwargs catch-all, so only pass flags the constructor names.
            extra = {k: v for k, v in extra.items() if k in params}
        return handler_cls(**kwargs, **extra)

    # Opaque constructor (an extension type with no readable signature).
    # Trying is the only way left to learn which flags it takes, so narrow
    # the set on each rejection. This reintroduces the repeated-call risk
    # that inspection exists to avoid, which is why it is confined to the
    # path that has no alternative: failing to load a model at all is the
    # worse outcome.
    #
    # Only a TypeError that names an unexpected keyword is treated as "this
    # flag is unsupported". Retrying on ANY TypeError would let a handler
    # that rejects the flag combination in its own body be retried until
    # some narrower set happened to get past the raise — silently accepting
    # a construction the handler meant to refuse. A body error stops the
    # chain and propagates, so the opaque path keeps the same guarantee the
    # inspection path gets for free.
    attempts = (
        extra,
        {"enable_thinking": False},
        {"add_vision_id": False},
        {},
    )
    for i, attempt in enumerate(attempts):
        try:
            return handler_cls(**kwargs, **attempt)
        except TypeError as exc:
            unsupported_kw = "unexpected keyword argument" in str(exc)
            if not unsupported_kw or i == len(attempts) - 1:
                raise


def estimate_vision_tokens(width: int, height: int) -> int:
    """Estimate vision tokens for one frame (Qwen3-VL: one per 32x32 patch)."""
    return math.ceil(width / 32) * math.ceil(height / 32)


def _encode_data_uri(img: Image.Image) -> str:
    """Encode a clamped image as a JPEG q95 base64 data URI.

    JPEG rather than PNG: the chat handler decodes whatever it is given and
    re-encodes to JPEG q95 itself before handing bytes to llama.cpp, so a PNG
    here bought nothing and cost a slow, entropy-coded compression pass (plus
    several times the base64 payload) on every caption.
    """
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def image_to_data_uri(image_path: Path, max_dim: int = MAX_IMAGE_DIM) -> str:
    """
    Load an image, resize if needed (keeping aspect ratio), and convert to
    a base64 data URI suitable for llama-cpp-python vision input.

    Args:
        image_path: Path to the image file.
        max_dim: Maximum dimension (width or height) to resize to.

    Returns:
        A data URI string like 'data:image/jpeg;base64,...'
    """
    img = load_image_for_inference(image_path, max_dim)

    return _encode_data_uri(img)


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
        self.chat_handler = _construct_chat_handler(
            handler_cls, mmproj_path, verbose
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
        # Record the context llama.cpp actually gave us, not the argument.
        # n_ctx=0 means "use the model's native context size", so storing the
        # raw 0 would make caption_video's preflight compare a positive budget
        # against a zero-token window and refuse every clip on a model that
        # has plenty of room. Llama.n_ctx() reports the effective value.
        try:
            self._n_ctx = int(self.model.n_ctx())
        except Exception:
            # A build without the accessor: fall back to the argument, and
            # treat 0 as "unknown" rather than "no context at all" so the
            # preflight degrades to not-checking instead of always-failing.
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

        # Extraction runs before any token loop exists to notice a cancel —
        # on a long clip that is seconds of scanning. A cancel here returns
        # "" like one during generation, so the caller sees a single
        # cancellation contract.
        try:
            frames = video_module.sample_frames(
                video_path, num_frames=num_frames, cancel_check=cancel_check
            )
        except VideoCancelled:
            return ""

        # Clamp and measure first, encode second. Both passes are needed
        # anyway, and splitting them means an over-budget clip is refused
        # before paying for a JPEG encode per frame — otherwise the preflight
        # below, whose whole point is to fail cheaply, still charged the full
        # CPU cost of encoding every frame it was about to reject.
        clamped = []
        vision_tokens = 0
        for frame in frames:
            if cancel_check and cancel_check():
                return ""
            frame = clamp_image_dim(frame, VIDEO_FRAME_MAX_DIM)
            w, h = frame.size
            vision_tokens += estimate_vision_tokens(w, h)
            clamped.append(frame)

        # Tell the model the images are one clip rather than unrelated
        # pictures (shared with the MLX backend so both frame it identically).
        framed_prompt = frame_video_prompt(prompt, len(clamped))

        # Preflight the context budget: Qwen3-VL's M-RoPE cannot context-shift,
        # so overflowing n_ctx loses the caption — llama.cpp logs "decode:
        # failed to find a memory slot for batch" and the wheel raises after
        # seconds of wasted GPU work (measured: 2.6 s; the process survives,
        # so this is a failed caption rather than a crash). Refusing up front
        # costs ~0.05 s and says why. 1.15x covers vision-encoder/template
        # overhead; the text prompts are measured (they are user-editable and
        # unbounded, so a flat allowance would let a long prompt slip past the
        # check); 128 covers chat scaffolding.
        text_tokens = self._count_text_tokens(system_prompt + "\n" + framed_prompt)
        needed = math.ceil(1.15 * vision_tokens) + max_tokens + text_tokens + 128
        if self._n_ctx and needed > self._n_ctx:
            raise RuntimeError(
                f"{len(clamped)} video frames need ~{needed} context "
                f"tokens (vision + {text_tokens} prompt + {max_tokens} "
                f"generation), but the context window is only {self._n_ctx}. "
                f"Caption fewer frames (num_frames) or load the model with a "
                f"larger context window."
            )

        # Budget cleared — now pay for the encode.
        image_parts = []
        for frame in clamped:
            if cancel_check and cancel_check():
                return ""
            image_parts.append(
                {"type": "image_url", "image_url": {"url": _encode_data_uri(frame)}}
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                # Frames in temporal order, then the instruction.
                "content": image_parts + [{"type": "text", "text": framed_prompt}],
            },
        ]

        return self._generate(
            messages, temperature, top_p, max_tokens,
            stream_callback, cancel_check, prefix, suffix,
        )

    def _count_text_tokens(self, text: str) -> int:
        """Count tokens for prompt text via the loaded model's tokenizer,
        falling back to a conservative character-based estimate."""
        try:
            return len(self.model.tokenize(text.encode("utf-8"), special=True))
        except Exception:
            # ~3 chars/token is conservative for English prose.
            return len(text) // 3 + 16

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

        # Stream whenever EITHER a token callback or a cancel predicate is
        # supplied. A blocking completion has no point at which to poll, so a
        # caller that passed cancel_check without stream_callback used to get
        # no cancellation at all once generation started — the call ran to
        # completion and returned the caption it had asked to abandon. The
        # MLX backend cancels regardless of streaming, so this also brings the
        # two engines to the same contract.
        if stream_callback or cancel_check:
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
                    # Only when the caller actually wants the tokens — this
                    # branch now also runs for cancel-only callers.
                    if stream_callback:
                        stream_callback(token_text)

            caption = "".join(caption_parts).strip()
        else:
            # No callback and no cancel predicate: nothing to poll for.
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
