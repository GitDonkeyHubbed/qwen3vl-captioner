"""
Qwen3-VL Captioner — Entry Point

Initializes the Qt application with the dark theme, auto-detects the
GGUF model in the parent directory, and launches the main window.
"""

import sys
from pathlib import Path

from engine.cuda_setup import setup_cuda_dll_path, startup_failure_advice
from gui.version import APP_VERSION

# GPU backend llama.cpp uses on this platform — Metal on Apple Silicon, CUDA on
# Windows. Used only to label the startup log messages below accurately.
_GPU_BACKEND = (
    "Metal" if sys.platform == "darwin"
    else "CUDA" if sys.platform == "win32"
    else "GPU"
)

# CRITICAL: Load GPU runtime libraries and initialize llama.cpp BEFORE importing
# PyQt6! On Windows, PyQt6 init can interfere with CUDA context creation;
# setup_cuda_dll_path() is a safe no-op on macOS/Linux.
setup_cuda_dll_path()

_engine_startup_error = None
try:
    import llama_cpp
    llama_cpp.llama_backend_init()
    print(f"[OK] {_GPU_BACKEND} backend initialized successfully (before PyQt6 import)")
except Exception as e:
    _engine_startup_error = startup_failure_advice(str(e))
    print(f"[WARNING] Failed to initialize {_GPU_BACKEND} backend early: {e}")
    print(_engine_startup_error)
    print("[INFO] Will attempt regular initialization later")

# Now safe to import PyQt6
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtGui import QFont, QImageReader

from gui.main_window import MainWindow
from gui.theme import get_stylesheet


def main():
    """Application entry point."""
    app = QApplication(sys.argv)

    # Remove Qt's default 256MB image allocation limit.
    # High-res images (e.g. 8000x6000 camera photos) exceed 256MB when decoded
    # to 32-bit RGBA, causing "Rejecting image" errors. 0 = unlimited.
    QImageReader.setAllocationLimit(0)

    # Set application metadata
    app.setApplicationName("VL-CAPTIONER")
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("Qwen3VL-Captioner")

    # Apply the SAVED theme (the stylesheet default is dark, so without this
    # a user's persisted light-mode choice was ignored on every launch)
    from gui.config import get_theme
    from gui.theme import set_theme, apply_placeholder_palette
    theme_mode = get_theme()
    set_theme(theme_mode)
    app.setStyleSheet(get_stylesheet(theme_mode))
    # Placeholder color must go through the palette — Qt Style Sheets have no
    # ::placeholder sub-control (the old QSS rule was silently ignored).
    apply_placeholder_palette(app)

    # Set default font
    font = QFont("Segoe UI", 10)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    app.setFont(font)

    # Auto-detect model directory (parent of this script's directory)
    app_dir = Path(__file__).resolve().parent
    model_dir = app_dir.parent  # The parent folder should contain the .gguf

    # Create and show main window
    window = MainWindow(model_dir=model_dir)
    window.show()

    # If the inference engine failed to initialize, tell the user in the GUI
    # with specific remediation steps — not just in the console they may
    # never see when launching via double-click.
    from engine.inference import LLAMA_CPP_AVAILABLE
    if not LLAMA_CPP_AVAILABLE or _engine_startup_error:
        QMessageBox.warning(
            window, "Engine Problem Detected",
            (_engine_startup_error or "The inference engine failed to load.")
            + "\n\nThe app will stay open, but captioning will not work "
            "until this is fixed.",
        )

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
