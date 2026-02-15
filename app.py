"""
Qwen3-VL Captioner — Entry Point

Initializes the Qt application with the dark theme, auto-detects the
GGUF model in the parent directory, and launches the main window.
"""

import sys
from pathlib import Path

# CRITICAL: Initialize CUDA BEFORE importing PyQt6!
# PyQt6 initialization interferes with CUDA context creation
try:
    import llama_cpp
    llama_cpp.llama_backend_init()
    print("[OK] CUDA backend initialized successfully (before PyQt6 import)")
except Exception as e:
    print(f"[WARNING] Failed to initialize CUDA backend early: {e}")
    print("[INFO] Will attempt regular initialization later")

# Now safe to import PyQt6
from PyQt6.QtWidgets import QApplication
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
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Qwen3VL-Captioner")
    
    # Apply dark theme
    app.setStyleSheet(get_stylesheet())
    
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
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
