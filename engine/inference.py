"""
Qwen3-VL GGUF Inference Engine

Provides GPU-accelerated vision-language model inference using llama-cpp-python.
Supports single image captioning with streaming token output and configurable
generation parameters. Thread-safe for Qt signal integration.
"""

import base64
import io
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

from PIL import Image


def _setup_cuda_dll_path():
    """
    Manually preload CUDA DLLs using ctypes before llama.cpp initialization.

    On Windows, llama-cpp-python with CUDA support requires CUDA runtime DLLs
    (cudart64_12.dll, cublas64_12.dll, etc.) to be loaded. When launching
    the GUI via double-click, the CUDA bin directory may not be in PATH, causing
    "access violation reading 0x0000000000000000" errors during llama_backend_init().

    We manually preload the DLLs using ctypes.CDLL() to ensure they're loaded
    into the process before llama.cpp tries to use them.
    """
    if sys.platform != "win32":
        return  # Only needed on Windows

    # Common CUDA installation paths to check
    cuda_paths = [
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin"),
    ]

    # Also check CUDA_PATH environment variable
    cuda_env = os.environ.get("CUDA_PATH")
    if cuda_env:
        cuda_paths.insert(0, Path(cuda_env) / "bin")

    # Find first existing CUDA path
    cuda_bin = None
    for path in cuda_paths:
        if path.exists():
            cuda_bin = path
            break

    if not cuda_bin:
        return  # No CUDA found, will fall back to CPU

    # Critical CUDA DLLs that must be preloaded
    critical_dlls = [
        "cudart64_12.dll",
        "cublas64_12.dll",
        "cublasLt64_12.dll",
    ]

    # Preload DLLs using ctypes
    import ctypes
    for dll_name in critical_dlls:
        dll_path = cuda_bin / dll_name
        if dll_path.exists():
            try:
                ctypes.CDLL(str(dll_path))
            except Exception:
                pass  # Continue even if one DLL fails


# Setup CUDA DLL path before importing llama_cpp
_setup_cuda_dll_path()

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False


# Supported image file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"}


def is_image_file(path: Path) -> bool:
    """Check if a file path has a supported image extension."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


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
    img = Image.open(image_path).convert("RGB")
    
    # Resize if any dimension exceeds max_dim
    w, h = img.size
    if w > max_dim or h > max_dim:
        scale = max_dim / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    
    # Convert to PNG bytes then base64
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


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
        """
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError(
                "llama-cpp-python is not installed. Run setup.bat or:\n"
                "  pip install llama-cpp-python"
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
        
        if progress_callback:
            progress_callback("Loading vision encoder (mmproj)...")
        
        # Create the vision chat handler for Qwen VL models
        self.chat_handler = Qwen25VLChatHandler(
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
        self._is_loaded = True
        
        if progress_callback:
            progress_callback("Model loaded successfully.")
    
    def caption_image(
        self,
        image_path: str | Path,
        prompt: str,
        system_prompt: str = "You are a helpful assistant that describes images accurately and in detail.",
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
                
                delta = chunk.get("choices", [{}])[0].get("delta", {})
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
            
            caption = response["choices"][0]["message"]["content"].strip()
        
        self._last_inference_time = time.perf_counter() - start_time
        
        # --- Clean up chat-template artifacts ---
        # VLMs often prepend formatting noise like ":", "Answer:", "Caption:", etc.
        # Strip known prefixes first, then any leftover leading punctuation.
        _strip_prefixes = [
            "answer:", "caption:", "description:", "response:",
            "here is", "here's", "sure,", "sure.",
        ]
        cleaned = caption
        for pfx in _strip_prefixes:
            if cleaned.lower().startswith(pfx):
                cleaned = cleaned[len(pfx):]
                break
        # Strip any remaining leading colons, dashes, dots, asterisks, whitespace
        cleaned = cleaned.lstrip(":;-–—.*• \t\n")
        if cleaned:
            caption = cleaned
        
        # Apply prefix and suffix
        if prefix:
            caption = prefix.strip() + " " + caption
        if suffix:
            caption = caption + " " + suffix.strip()
        
        return caption
    
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
            "last_inference_s": round(self._last_inference_time, 2),
        }
