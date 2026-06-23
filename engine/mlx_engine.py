"""
MLX inference engine for Apple Silicon Macs.

Runs Qwen3-VL models in MLX format (folders of safetensors) via Apple's
mlx-vlm package. MLX models embed the vision tower — no separate mmproj
file is needed — and typically outperform llama.cpp Metal on M-series
chips, especially for image encoding.

Only available on macOS arm64 with mlx-vlm installed (setup.sh installs
it automatically on Apple Silicon).
"""

import platform
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from engine.base import DEFAULT_SYSTEM_PROMPT, apply_prefix_suffix, clean_caption

MLX_SUPPORTED = sys.platform == "darwin" and platform.machine() == "arm64"

MLX_VLM_IMPORT_ERROR: Optional[str] = None
if MLX_SUPPORTED:
    try:
        import importlib
        importlib.import_module("mlx_vlm")
        MLX_VLM_AVAILABLE = True
    except Exception as _e:
        MLX_VLM_AVAILABLE = False
        MLX_VLM_IMPORT_ERROR = str(_e)
else:
    MLX_VLM_AVAILABLE = False
    MLX_VLM_IMPORT_ERROR = "MLX requires an Apple Silicon Mac"


def is_mlx_model_dir(path: Path) -> bool:
    """An MLX model is a directory containing config.json + safetensors."""
    path = Path(path)
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and any(path.glob("*.safetensors"))
    )


def _load_make_sampler():
    """Return mlx's make_sampler factory, or None on older/missing mlx.

    Newer mlx-lm exposes sampling parameters through a sampler callable; if it
    isn't importable we fall back to the legacy temperature/top_p kwargs.
    """
    try:
        from mlx_lm.sample_utils import make_sampler
        return make_sampler
    except ImportError:
        # mlx_lm missing or too old to expose make_sampler — use legacy kwargs.
        # (By the time this runs mlx is already imported, so a non-import error
        # here would be a genuine problem and should not be masked.)
        return None


def _stream_with_sampling(
    stream_generate, args, kwargs, temperature, top_p, make_sampler="auto"
):
    """Invoke mlx-vlm's stream_generate across a breaking sampling-API change.

    mlx-vlm/mlx-lm dropped the ``temperature``/``top_p`` keyword arguments on
    ``stream_generate`` in favour of a ``sampler`` callable, and the project
    pins only ``mlx-vlm>=0.1.0`` (no upper bound) — so a fresh setup.sh install
    pulls the newer API where the old kwargs raise ``TypeError`` on every
    caption. Prefer the sampler path; if this build doesn't accept ``sampler``
    (older 0.1.x), fall back to the legacy kwargs so captioning works on either
    version.
    """
    if make_sampler == "auto":
        make_sampler = _load_make_sampler()
    if make_sampler is not None:
        # Build the sampler OUTSIDE the try: a failure here is a real
        # make_sampler problem and must not be mistaken for "this build has no
        # sampler kwarg" (which would wrongly retry the legacy kwargs and crash
        # on a newer mlx-vlm that removed them).
        sampler = make_sampler(temp=temperature, top_p=top_p)
        try:
            return stream_generate(*args, sampler=sampler, **kwargs)
        except TypeError as exc:
            # Only fall back when stream_generate specifically rejects the
            # `sampler` keyword (older 0.1.x). Any other TypeError is a genuine
            # error and must propagate.
            if "sampler" not in str(exc):
                raise
    return stream_generate(*args, temperature=temperature, top_p=top_p, **kwargs)


class MlxVlmEngine:
    """
    Inference engine for MLX-format Qwen3-VL models via mlx-vlm.

    Mirrors the Qwen3VLEngine interface (see engine/base.py) so the GUI
    can drive either backend through the same calls.
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.config = None
        self.model_path: Optional[Path] = None
        self._is_loaded = False
        self._last_inference_time: float = 0.0

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and self.model is not None

    @property
    def last_inference_time(self) -> float:
        return self._last_inference_time

    def load_model(
        self,
        model_path,
        mmproj_path=None,  # unused — MLX models embed the vision tower
        n_ctx: int = 8192,  # unused — kept for interface compatibility
        n_gpu_layers: int = -1,  # unused — MLX always runs on the GPU
        verbose: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Load an MLX model folder."""
        if not MLX_VLM_AVAILABLE:
            raise RuntimeError(
                "mlx-vlm is not available.\n\n"
                + (MLX_VLM_IMPORT_ERROR or "")
                + "\n\nOn Apple Silicon, run setup.sh (or: pip install mlx-vlm)."
            )

        model_path = Path(model_path)
        if not is_mlx_model_dir(model_path):
            raise FileNotFoundError(
                f"Not an MLX model folder (needs config.json + safetensors): "
                f"{model_path}"
            )

        if self._is_loaded:
            self.unload()

        if progress_callback:
            progress_callback("Loading MLX model (this may take a minute)...")

        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        self.model, self.processor = load(str(model_path))
        self.config = load_config(str(model_path))

        self.model_path = model_path
        self._is_loaded = True

        if progress_callback:
            progress_callback("MLX model loaded successfully.")

    def caption_image(
        self,
        image_path,
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
        """Generate a caption for a single image (streams tokens if asked)."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        from mlx_vlm import stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        # Include the system prompt so the MLX backend conditions the model
        # the same way the GGUF backend does (engine parity). Skip an empty
        # system turn if the caller passes "".
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        formatted_prompt = apply_chat_template(
            self.processor, self.config, messages, num_images=1
        )

        start_time = time.perf_counter()
        caption_parts = []

        temp = temperature if temperature > 0 else 0.0
        nucleus = top_p if temperature > 0 else 1.0
        token_stream = _stream_with_sampling(
            stream_generate,
            (self.model, self.processor, formatted_prompt),
            {"image": [str(image_path)], "max_tokens": max_tokens},
            temp,
            nucleus,
        )
        for chunk in token_stream:
            if cancel_check and cancel_check():
                break
            text = getattr(chunk, "text", None)
            if text is None:
                text = str(chunk)
            if text:
                caption_parts.append(text)
                if stream_callback:
                    stream_callback(text)

        self._last_inference_time = time.perf_counter() - start_time

        caption = clean_caption("".join(caption_parts))
        return apply_prefix_suffix(caption, prefix, suffix)

    def unload(self) -> None:
        """Unload the model and free GPU memory."""
        self.model = None
        self.processor = None
        self.config = None
        self._is_loaded = False
        self.model_path = None

        import gc
        gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass

    def get_model_info(self) -> dict:
        """Return metadata for the status panel."""
        if not self.is_loaded:
            return {"status": "Not loaded"}
        return {
            "status": "Loaded",
            "backend": "MLX (Apple Silicon)",
            "model_file": self.model_path.name if self.model_path else "unknown",
            "last_inference_s": round(self._last_inference_time, 2),
        }
