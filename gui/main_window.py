"""
Main Application Window

Orchestrates the three-panel layout:
  Left:   FileBrowserPanel (project files / thumbnails)
  Center: ImageViewer + CaptionPanel
  Right:  SettingsPanel (model config / parameters)

Wires up signals between all components and the inference engine.
"""

import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QFont, QAction, QPainter, QColor, QPen, QBrush, QScreen
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSplitter, QStatusBar, QProgressBar, QApplication, QFileDialog,
    QMessageBox, QSizePolicy, QStackedWidget,
)

from gui.file_browser import FileBrowserPanel
from gui.image_viewer import ImageViewer
from gui.caption_panel import CaptionPanel
from gui.settings_panel import SettingsPanel
from gui.dataset_panel import DatasetPanel
from gui.notification_panel import NotificationStore, NotificationPanel
from gui.theme import COLORS
from engine.inference import Qwen3VLEngine, is_image_file
from engine.model_downloader import ensure_mmproj, find_mmproj_file


# --- Worker for background model loading ---
class ModelLoadWorker(QObject):
    """Loads the GGUF model in a background thread."""
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, engine: Qwen3VLEngine, model_path: Path, mmproj_path: Path):
        super().__init__()
        self.engine = engine
        self.model_path = model_path
        self.mmproj_path = mmproj_path

    def run(self):
        try:
            self.engine.load_model(
                self.model_path,
                self.mmproj_path,
                progress_callback=lambda msg: self.progress.emit(msg),
            )
            self.finished.emit()
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")


# --- Worker for background caption generation ---
class CaptionWorker(QObject):
    """Generates a caption in a background thread."""
    new_token = pyqtSignal(str)
    finished = pyqtSignal(str)  # full caption
    error = pyqtSignal(str)

    def __init__(
        self, engine: Qwen3VLEngine, image_path: Path,
        prompt: str, temperature: float, top_p: float,
        max_tokens: int, prefix: str, suffix: str,
    ):
        super().__init__()
        self.engine = engine
        self.image_path = image_path
        self.prompt = prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.prefix = prefix
        self.suffix = suffix
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            caption = self.engine.caption_image(
                image_path=self.image_path,
                prompt=self.prompt,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                prefix=self.prefix,
                suffix=self.suffix,
                stream_callback=lambda t: self.new_token.emit(t),
                cancel_check=lambda: self._cancelled,
            )
            self.finished.emit(caption)
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class _ShieldIcon(QWidget):
    """Small blue rounded-square with a shield glyph, matching the Figma brand icon."""
    def __init__(self, size=28, parent=None):
        super().__init__(parent)
        self._size = size
        self.setFixedSize(size, size)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Rounded blue background
        p.setBrush(QBrush(QColor(COLORS["accent"])))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(0, 0, self._size, self._size, 6, 6)
        # Shield outline (simplified)
        p.setPen(QPen(QColor(255, 255, 255, 180), 1.5))
        p.setBrush(QBrush(QColor(255, 255, 255, 40)))
        cx = self._size / 2
        cy = self._size / 2
        s = self._size * 0.32  # half-width
        from PyQt6.QtGui import QPolygonF
        from PyQt6.QtCore import QPointF
        shield = QPolygonF([
            QPointF(cx, cy - s * 1.1),      # top center
            QPointF(cx + s, cy - s * 0.6),   # top right
            QPointF(cx + s, cy + s * 0.1),   # mid right
            QPointF(cx, cy + s * 1.1),       # bottom center
            QPointF(cx - s, cy + s * 0.1),   # mid left
            QPointF(cx - s, cy - s * 0.6),   # top left
        ])
        p.drawPolygon(shield)
        p.end()


class MainWindow(QMainWindow):
    """
    Main application window with three-panel layout matching the Figma mockup.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        super().__init__()
        self.setWindowTitle("VL-CAPTIONER Studio Pro v1.4.2 — GGUF Engine")
        self.setMinimumSize(1000, 650)

        # Screen-aware sizing: use 85% of available screen, clamped to minimums
        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            w = max(1000, int(avail.width() * 0.85))
            h = max(650, int(avail.height() * 0.85))
            x = avail.x() + (avail.width() - w) // 2
            y = avail.y() + (avail.height() - h) // 2
            self.setGeometry(x, y, w, h)
        else:
            self.setGeometry(50, 50, 1400, 850)

        # State
        self._engine = Qwen3VLEngine()
        self._model_dir = model_dir
        self._current_image: Optional[Path] = None
        self._captions: Dict[str, str] = {}  # str(path) -> caption

        # Thread references — MUST be stored as instance attrs to prevent GC
        self._model_load_thread: Optional[QThread] = None
        self._model_load_worker: Optional[ModelLoadWorker] = None
        self._generation_thread: Optional[QThread] = None
        self._caption_worker: Optional[CaptionWorker] = None
        self._is_generating = False
        self._batch_queue: List[Path] = []
        self._batch_index = 0
        self._download_thread: Optional[QThread] = None
        self._download_worker = None  # ModelDownloadWorker (lazy import)

        # NVML (GPU monitoring)
        self._nvml_handle = None
        self._init_nvml()

        # Periodic GPU refresh timer (5 seconds)
        self._gpu_timer = QTimer(self)
        self._gpu_timer.setInterval(5000)
        self._gpu_timer.timeout.connect(self._update_gpu_info)

        # Notification system
        self._notification_store = NotificationStore(self)
        self._notification_panel: Optional[NotificationPanel] = None  # created lazily after bell btn exists

        # Build UI
        self._build_nav_bar()
        self._build_main_layout()
        self._build_status_bar()
        self._connect_signals()

        # Populate model pick-list from available GGUF files on disk
        self._refresh_model_combo()

        # Kick off GPU monitoring immediately (don't wait for model load)
        self._update_gpu_info()
        self._gpu_timer.start()

    def _build_nav_bar(self):
        """Build the top navigation bar matching the Figma Header component."""
        nav_bar = QFrame()
        nav_bar.setProperty("class", "nav-bar")
        nav_bar.setFixedHeight(52)
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(16, 0, 16, 0)
        nav_layout.setSpacing(0)

        # --- Left side: Brand + Nav tabs ---
        left_group = QHBoxLayout()
        left_group.setSpacing(6)

        # Shield icon
        shield = _ShieldIcon(28)
        left_group.addWidget(shield)

        # Brand text block
        brand_block = QVBoxLayout()
        brand_block.setContentsMargins(0, 0, 0, 0)
        brand_block.setSpacing(0)

        brand_title = QLabel("VL-CAPTIONER")
        brand_title.setProperty("class", "brand-title")
        brand_title.setStyleSheet(
            f"color: {COLORS['text_primary']}; font-size: 13px; font-weight: 700; "
            f"letter-spacing: 0.5px; padding: 0; margin: 0; background: transparent;"
        )
        brand_block.addWidget(brand_title)

        brand_sub = QLabel("GGUF Studio Pro v1.4.2")
        brand_sub.setProperty("class", "brand-subtitle")
        brand_sub.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: 9px; font-family: 'Consolas', 'Courier New', monospace; "
            f"letter-spacing: 0.3px; padding: 0; margin: 0; background: transparent; text-transform: uppercase;"
        )
        brand_block.addWidget(brand_sub)

        brand_container = QWidget()
        brand_container.setLayout(brand_block)
        brand_container.setStyleSheet("background: transparent;")
        left_group.addWidget(brand_container)

        left_group.addSpacing(24)

        # Nav tabs — Project (active), Dataset
        self._tab_buttons: Dict[str, QPushButton] = {}
        for tab_name, is_active in [("Project", True), ("Dataset", False)]:
            btn = QPushButton(tab_name)
            btn.setProperty("class", "nav-tab-active" if is_active else "nav-tab")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked, t=tab_name: self._switch_tab(t))
            left_group.addWidget(btn)
            self._tab_buttons[tab_name] = btn

        left_widget = QWidget()
        left_widget.setLayout(left_group)
        left_widget.setStyleSheet("background: transparent;")
        nav_layout.addWidget(left_widget)

        nav_layout.addStretch()

        # --- Right side: GPU pill + icon buttons + user ---
        right_group = QHBoxLayout()
        right_group.setSpacing(4)

        # GPU / VRAM pill
        gpu_pill = QFrame()
        gpu_pill.setProperty("class", "gpu-pill")
        gpu_pill_layout = QHBoxLayout(gpu_pill)
        gpu_pill_layout.setContentsMargins(10, 4, 10, 4)
        gpu_pill_layout.setSpacing(8)

        # Emerald pulse dot + GPU %
        self._gpu_dot = QLabel("\u2022")
        self._gpu_dot.setStyleSheet(f"color: {COLORS['success']}; font-size: 14px; background: transparent;")
        gpu_pill_layout.addWidget(self._gpu_dot)

        self._gpu_label = QLabel("GPU: --")
        self._gpu_label.setStyleSheet(
            f"color: {COLORS['success']}; font-size: 10px; font-weight: 600; "
            f"letter-spacing: 0.5px; text-transform: uppercase; background: transparent;"
        )
        gpu_pill_layout.addWidget(self._gpu_label)

        # Vertical separator inside pill
        pill_sep = QFrame()
        pill_sep.setFixedSize(1, 14)
        pill_sep.setStyleSheet(f"background: {COLORS['border']};")
        gpu_pill_layout.addWidget(pill_sep)

        # VRAM info
        self._vram_label = QLabel("-- VRAM")
        self._vram_label.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: 10px; font-weight: 500; "
            f"letter-spacing: 0.5px; text-transform: uppercase; background: transparent;"
        )
        gpu_pill_layout.addWidget(self._vram_label)

        right_group.addWidget(gpu_pill)
        right_group.addSpacing(8)

        # Icon buttons: Terminal, Bell, Settings
        icon_chars = [
            ("\u2318", "Terminal"),    # ⌘
            ("\U0001F514", "Alerts"),  # 🔔
            ("\u2699", "Settings"),    # ⚙
        ]
        self._terminal_btn = None
        self._bell_btn = None
        self._settings_gear_btn = None
        for icon_char, tooltip in icon_chars:
            btn = QPushButton(icon_char)
            btn.setProperty("class", "icon-button")
            btn.setFixedSize(32, 32)
            btn.setToolTip(tooltip)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            right_group.addWidget(btn)
            if tooltip == "Terminal":
                self._terminal_btn = btn
            elif tooltip == "Alerts":
                self._bell_btn = btn
            elif tooltip == "Settings":
                self._settings_gear_btn = btn

        # Bell badge (red dot with unread count, overlaid on bell button)
        self._bell_badge = QLabel("0", self._bell_btn)
        self._bell_badge.setFixedSize(16, 16)
        self._bell_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._bell_badge.setStyleSheet(
            f"background-color: {COLORS['error']}; color: #ffffff; "
            f"font-size: 9px; font-weight: 700; border-radius: 8px; "
            f"border: none; padding: 0px;"
        )
        self._bell_badge.move(self._bell_btn.width() - 14, -2)
        self._bell_badge.setVisible(False)

        # Create the notification panel now that the bell button exists
        self._notification_panel = NotificationPanel(self._notification_store, self)

        # Vertical separator
        sep = QFrame()
        sep.setFixedSize(1, 24)
        sep.setStyleSheet(f"background: {COLORS['border']};")
        right_group.addWidget(sep)
        right_group.addSpacing(4)

        # Admin + user avatar
        admin_label = QLabel("Admin")
        admin_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px; font-weight: 500; background: transparent;"
        )
        right_group.addWidget(admin_label)

        avatar = QLabel("\U0001F464")  # 👤
        avatar.setFixedSize(28, 28)
        avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        avatar.setStyleSheet(
            f"background: {COLORS['bg_surface']}; border: 1px solid {COLORS['border_light']}; "
            f"border-radius: 14px; font-size: 13px;"
        )
        right_group.addWidget(avatar)

        right_widget = QWidget()
        right_widget.setLayout(right_group)
        right_widget.setStyleSheet("background: transparent;")
        nav_layout.addWidget(right_widget)

        # Set as menu bar area (above central widget)
        nav_container = QWidget()
        nav_container_layout = QVBoxLayout(nav_container)
        nav_container_layout.setContentsMargins(0, 0, 0, 0)
        nav_container_layout.setSpacing(0)
        nav_container_layout.addWidget(nav_bar)

        self._nav_widget = nav_container

    def _build_main_layout(self):
        """Build the three-panel main layout."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Nav bar at top
        main_layout.addWidget(self._nav_widget)

        # Three-panel splitter
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setHandleWidth(1)

        # Left: File Browser
        self._file_browser = FileBrowserPanel()
        self._splitter.addWidget(self._file_browser)

        # Center: Image Viewer + Caption Panel
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        self._image_viewer = ImageViewer()
        center_layout.addWidget(self._image_viewer, 3)

        self._caption_panel = CaptionPanel()
        center_layout.addWidget(self._caption_panel, 1)

        self._splitter.addWidget(center_widget)

        # Right: Settings Panel
        self._settings_panel = SettingsPanel()
        self._splitter.addWidget(self._settings_panel)

        # Proportional splitter sizes: ~17% left, ~58% center, ~25% right
        total_w = self.width()
        left_w = max(180, int(total_w * 0.17))
        right_w = max(300, int(total_w * 0.25))
        center_w = total_w - left_w - right_w
        self._splitter.setSizes([left_w, center_w, right_w])
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setStretchFactor(2, 0)

        # Dataset panel (shown when Dataset tab is active)
        self._dataset_panel = DatasetPanel()
        self._dataset_panel.set_refresh_callback(self._refresh_dataset)

        # Stack: index 0 = Project view (splitter), index 1 = Dataset view
        self._main_stack = QStackedWidget()
        self._main_stack.addWidget(self._splitter)
        self._main_stack.addWidget(self._dataset_panel)

        main_layout.addWidget(self._main_stack, 1)

    def _build_status_bar(self):
        """Build the bottom status bar matching the Figma footer."""
        self._status_bar = QStatusBar()
        self._status_bar.setFixedHeight(24)
        self._status_bar.setStyleSheet(
            f"QStatusBar {{ "
            f"  background: {COLORS['bg_darkest']}; "
            f"  border-top: 1px solid {COLORS['border']}; "
            f"  color: {COLORS['text_dim']}; "
            f"  font-size: 10px; "
            f"  font-weight: 500; "
            f"  padding: 0 4px; "
            f"}} "
            f"QStatusBar::item {{ border: none; }}"
        )
        self.setStatusBar(self._status_bar)

        # Left side: connection indicator + queue
        left_container = QWidget()
        left_container.setStyleSheet("background: transparent;")
        left_layout = QHBoxLayout(left_container)
        left_layout.setContentsMargins(8, 0, 0, 0)
        left_layout.setSpacing(0)

        # Emerald dot
        conn_dot = QLabel("\u2022")
        conn_dot.setStyleSheet(f"color: {COLORS['success']}; font-size: 12px; padding-right: 4px; background: transparent;")
        left_layout.addWidget(conn_dot)
        self._conn_dot = conn_dot

        # Connection text
        self._conn_label = QLabel("Connected to Local Node: 127.0.0.1:8188")
        self._conn_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px; background: transparent;")
        left_layout.addWidget(self._conn_label)

        # Separator
        sep1 = QFrame()
        sep1.setFixedSize(1, 12)
        sep1.setStyleSheet(f"background: {COLORS['border']}; margin: 0 8px;")
        left_layout.addWidget(sep1)

        # Queue info
        self._queue_label = QLabel("")
        self._queue_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px; background: transparent;")
        left_layout.addWidget(self._queue_label)

        self._status_bar.addWidget(left_container)

        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        spacer.setStyleSheet("background: transparent;")
        self._status_bar.addWidget(spacer)

        # Progress bar (hidden by default)
        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedWidth(200)
        self._progress_bar.setFixedHeight(4)
        self._progress_bar.setVisible(False)
        self._progress_bar.setStyleSheet(
            f"QProgressBar {{ "
            f"  background: {COLORS['bg_surface']}; "
            f"  border: none; border-radius: 2px; "
            f"}} "
            f"QProgressBar::chunk {{ "
            f"  background: {COLORS['accent']}; "
            f"  border-radius: 2px; "
            f"}}"
        )
        self._status_bar.addPermanentWidget(self._progress_bar)

        # Right side: inference time + RAM + UTF-8
        right_container = QWidget()
        right_container.setStyleSheet("background: transparent;")
        right_layout = QHBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 8, 0)
        right_layout.setSpacing(12)

        self._inference_label = QLabel("")
        self._inference_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px; background: transparent;")
        right_layout.addWidget(self._inference_label)

        self._ram_label = QLabel("")
        self._ram_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px; background: transparent;")
        right_layout.addWidget(self._ram_label)

        utf8_label = QLabel("UTF-8")
        utf8_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 10px; background: transparent;")
        right_layout.addWidget(utf8_label)

        self._status_bar.addPermanentWidget(right_container)

        # Update RAM info on start
        self._update_ram_info()

    def _connect_signals(self):
        """Wire up all component signals."""
        # File browser -> display
        self._file_browser.image_selected.connect(self._on_image_selected)
        self._file_browser.clear_requested.connect(self._on_clear_all)

        # Caption panel
        self._caption_panel.regenerate_requested.connect(self._generate_caption)
        self._caption_panel.save_requested.connect(self._save_current_caption)

        # Settings panel
        self._settings_panel.load_model_requested.connect(self._load_model)
        self._settings_panel.unload_model_requested.connect(self._unload_model)
        self._settings_panel.batch_caption_requested.connect(self._batch_caption_all)
        self._settings_panel.export_requested.connect(self._export_all_captions)
        self._settings_panel.settings_changed.connect(self._on_settings_changed)
        self._settings_panel.download_model_requested.connect(self._download_model)

        # Header icon buttons
        if self._terminal_btn:
            self._terminal_btn.clicked.connect(self._show_engine_status)
        if self._bell_btn:
            self._bell_btn.clicked.connect(self._toggle_notifications)
        if self._settings_gear_btn:
            self._settings_gear_btn.clicked.connect(self._open_app_settings)

        # Notification badge updates
        self._notification_store.notification_added.connect(self._update_bell_badge)

    # --- Image Selection ---

    def _on_image_selected(self, path: Path):
        """Handle image selection from the file browser."""
        self._current_image = path
        self._image_viewer.set_image(path)

        # Show existing caption if available
        key = str(path)
        if key in self._captions:
            self._caption_panel.set_caption(self._captions[key])
        else:
            # Check for existing .txt sidecar
            txt_path = path.with_suffix(".txt")
            if txt_path.exists():
                try:
                    caption = txt_path.read_text(encoding="utf-8").strip()
                    self._captions[key] = caption
                    self._caption_panel.set_caption(caption)
                except Exception:
                    self._caption_panel.clear_caption()
            else:
                self._caption_panel.clear_caption()

    def _on_clear_all(self):
        """Reset the workspace — clear all images, captions, and viewer state."""
        # Cancel any in-progress batch
        self._batch_queue.clear()
        self._batch_index = 0

        # Clear captions cache
        self._captions.clear()
        self._current_image = None

        # Reset viewer and caption panel
        self._image_viewer.clear()
        self._caption_panel.clear_caption()

        # Reset status bar
        self._progress_bar.setVisible(False)
        self._queue_label.setText("")

        self._notify("Workspace cleared", "info")

    # --- Model Loading ---

    def _load_model(self):
        """Load the GGUF model in a background thread."""
        if self._engine.is_loaded:
            return

        # Find model file
        model_path = self._find_model_file()
        if not model_path:
            QMessageBox.warning(
                self, "Model Not Found",
                "Could not find a .gguf model file.\n\n"
                "Expected location: parent directory of this app.\n"
                "Please place your Qwen3-VL GGUF file there."
            )
            return

        # Find or download mmproj
        model_dir = model_path.parent
        self._settings_panel.set_model_status("Checking for vision encoder...")

        try:
            mmproj_path = ensure_mmproj(
                model_dir,
                progress_callback=lambda msg, _: self._settings_panel.set_model_status(msg),
            )
        except Exception as e:
            QMessageBox.critical(
                self, "mmproj Error",
                f"Failed to find/download vision encoder:\n{e}"
            )
            self._settings_panel.set_model_status("Error: mmproj not found")
            return

        # Load in background thread
        self._settings_panel.set_model_status("Loading model...")
        self._settings_panel.load_model_btn.setEnabled(False)
        self._set_connection_status("loading", "Loading model...")

        # Store as instance attrs to prevent garbage collection (QThread crash fix)
        self._model_load_thread = QThread()
        self._model_load_worker = ModelLoadWorker(self._engine, model_path, mmproj_path)
        self._model_load_worker.moveToThread(self._model_load_thread)

        self._model_load_thread.started.connect(self._model_load_worker.run)
        self._model_load_worker.progress.connect(
            lambda msg: self._settings_panel.set_model_status(msg)
        )
        self._model_load_worker.finished.connect(self._on_model_loaded)
        self._model_load_worker.error.connect(self._on_model_load_error)
        self._model_load_worker.finished.connect(self._model_load_thread.quit)
        self._model_load_worker.error.connect(self._model_load_thread.quit)

        self._model_load_thread.start()

    def _on_model_loaded(self):
        """Handle successful model load."""
        info = self._engine.get_model_info()
        model_name = info.get('model_file', 'Model')
        self._settings_panel.set_model_status(
            f"{model_name} loaded and ready.",
            detail=f"Vision encoder: {info['mmproj_file']}",
            is_loaded=True,
        )
        self._set_connection_status("ready", "Model ready")
        self._settings_panel.model_combo.setEnabled(False)  # Must unload before switching
        self._update_gpu_info()
        if not self._gpu_timer.isActive():
            self._gpu_timer.start()
        self._notify(f"{model_name} loaded successfully", "success")

    def _on_model_load_error(self, error: str):
        """Handle model load failure."""
        self._settings_panel.set_model_status(f"Error loading model", detail=error[:100])
        self._settings_panel.load_model_btn.setEnabled(True)
        self._set_connection_status("error", "Error")
        self._notify(f"Model load failed: {error[:80]}", "error")
        QMessageBox.critical(self, "Model Load Error", f"Failed to load model:\n\n{error}")

    def _unload_model(self):
        """Unload the current model and reset UI state."""
        if not self._engine.is_loaded:
            return

        # Don't unload while generating
        if self._is_generating:
            QMessageBox.warning(
                self, "Cannot Unload",
                "Please wait for the current generation to finish before unloading."
            )
            return

        self._engine.unload()

        # Reset UI state
        self._settings_panel.set_model_status("Model unloaded", is_loaded=False)
        self._set_connection_status("ready", "Model unloaded")
        self._settings_panel.model_combo.setEnabled(True)

        # Refresh GPU display (timer keeps running to show VRAM)
        self._update_gpu_info()
        self._notify("Model unloaded", "info")

    def _show_engine_status(self):
        """Show engine status info in a dialog (terminal icon action)."""
        info = self._engine.get_model_info()
        lines = [f"{k}: {v}" for k, v in info.items()]
        msg = "\n".join(lines) if lines else "No model loaded."
        QMessageBox.information(self, "Engine Status", msg)

    def _open_app_settings(self):
        """Open the application settings dialog (gear icon action)."""
        from gui.app_settings_dialog import AppSettingsDialog
        dlg = AppSettingsDialog(self)
        dlg.theme_changed.connect(self._on_theme_changed)
        dlg.paths_changed.connect(self._refresh_model_combo)
        dlg.exec()

    def _on_theme_changed(self, mode: str):
        """Handle theme switch from settings dialog."""
        from gui.theme import set_theme, get_stylesheet
        set_theme(mode)
        from PyQt6.QtWidgets import QApplication
        app_instance = QApplication.instance()
        if app_instance:
            app_instance.setStyleSheet(get_stylesheet(mode))

    # --- Model Downloading ---

    def _download_model(self, model_name: str):
        """Handle a download request from the settings panel."""
        from gui.model_download_manager import (
            get_model_info, model_file_exists, ModelDownloadWorker,
        )
        from gui.config import get_hf_token

        info = get_model_info(model_name)
        if info is None:
            QMessageBox.information(
                self, "Download Unavailable",
                f"No auto-download entry for '{model_name}'.\n\n"
                "Place a compatible .gguf file in the model directory\n"
                "and restart the application.",
            )
            return

        # Determine target directory (same as model search logic)
        target_dir = self._model_dir or Path(__file__).resolve().parent.parent

        if model_file_exists(target_dir, info["filename"]):
            QMessageBox.information(
                self, "Already Downloaded",
                f"{info['filename']} already exists in:\n{target_dir}"
            )
            return

        # Confirm with user (file size warning)
        answer = QMessageBox.question(
            self, "Download Model",
            f"Download {info['filename']}?\n\n"
            f"Size: ~{info['size_gb']:.1f} GB\n"
            f"From: {info['repo_id']}\n"
            f"To:   {target_dir}\n\n"
            "This may take several minutes.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        # Show indeterminate progress bar
        self._progress_bar.setRange(0, 0)  # indeterminate
        self._progress_bar.setVisible(True)
        self._queue_label.setText(f"Downloading {info['filename']}...")
        self._notify(f"Downloading {info['filename']}...", "download")

        # Start download in background thread
        token = get_hf_token()
        self._download_thread = QThread()
        self._download_worker = ModelDownloadWorker(
            repo_id=info["repo_id"],
            filename=info["filename"],
            target_dir=target_dir,
            hf_token=token,
        )
        self._download_worker.moveToThread(self._download_thread)

        self._download_thread.started.connect(self._download_worker.run)
        self._download_worker.progress.connect(self._on_download_progress)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_worker.error.connect(self._on_download_error)
        self._download_worker.finished.connect(self._download_thread.quit)
        self._download_worker.error.connect(self._download_thread.quit)

        self._download_thread.start()

    def _on_download_progress(self, message: str, fraction: float):
        """Handle download progress updates."""
        self._queue_label.setText(message)

    def _on_download_finished(self, local_path: str):
        """Handle successful download."""
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setVisible(False)
        filename = Path(local_path).name
        self._queue_label.setText(f"Downloaded: {filename}")
        self._notify(f"Download complete: {filename}", "success")
        # Refresh the model pick-list so the new file appears
        self._refresh_model_combo()

    def _on_download_error(self, error: str):
        """Handle download failure."""
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setVisible(False)
        self._queue_label.setText("Download failed")
        self._notify(f"Download failed: {error[:80]}", "error")
        QMessageBox.critical(
            self, "Download Error",
            f"Failed to download model:\n\n{error}"
        )

    # --- Notification helpers ---

    def _notify(self, message: str, category: str = "info"):
        """Add a notification to the store (and badge updates automatically)."""
        self._notification_store.add(message, category)

    def _toggle_notifications(self):
        """Show/hide the notification dropdown below the bell button."""
        if self._notification_panel and self._notification_panel.isVisible():
            self._notification_panel.hide()
        else:
            self._notification_panel.show_below(self._bell_btn)
            self._update_bell_badge()

    def _update_bell_badge(self):
        """Show/hide the red unread-count badge on the bell icon."""
        count = self._notification_store.unread_count()
        if count > 0:
            self._bell_badge.setText(str(min(count, 9)))
            self._bell_badge.setVisible(True)
        else:
            self._bell_badge.setVisible(False)

    # --- Model file scanning helpers ---

    def _get_search_dirs(self) -> List[Path]:
        """Return the list of directories to scan for GGUF files."""
        from gui.config import get_model_search_paths
        dirs: List[Path] = []
        if self._model_dir:
            dirs.append(self._model_dir)
        app_dir = Path(__file__).resolve().parent
        dirs.append(app_dir.parent)
        dirs.append(app_dir)
        # Append user-configured extra search paths
        for p in get_model_search_paths():
            d = Path(p)
            if d not in dirs:
                dirs.append(d)
        return dirs

    def _scan_model_files(self) -> List[Path]:
        """Return all non-mmproj GGUF files found in the search directories."""
        seen: set = set()
        results: List[Path] = []
        for dir_path in self._get_search_dirs():
            if not dir_path.is_dir():
                continue
            for f in sorted(dir_path.iterdir()):
                if (
                    f.is_file()
                    and f.suffix.lower() == ".gguf"
                    and "mmproj" not in f.name.lower()
                    and f.name not in seen
                ):
                    seen.add(f.name)
                    results.append(f)
        return results

    def _refresh_model_combo(self):
        """Populate the settings panel model combo with discovered GGUF files."""
        models = self._scan_model_files()
        combo = self._settings_panel.model_combo
        prev = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        if models:
            for m in models:
                # Display filename, store full path as item data
                combo.addItem(m.stem, str(m))
            # Try to restore the previous selection
            idx = combo.findText(prev)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        else:
            combo.addItem("(no models found)")
        combo.blockSignals(False)

    def _find_model_file(self) -> Optional[Path]:
        """Return the model Path currently selected in the combo box."""
        combo = self._settings_panel.model_combo
        path_str = combo.currentData()
        if path_str:
            p = Path(path_str)
            if p.is_file():
                return p
        # Fallback: first available GGUF
        models = self._scan_model_files()
        return models[0] if models else None

    # --- Caption Generation ---

    def _generate_caption(self):
        """Generate caption for the current image."""
        if not self._engine.is_loaded:
            QMessageBox.warning(self, "Model Required", "Please load the model first.")
            return

        if not self._current_image:
            QMessageBox.warning(self, "No Image", "Please select an image first.")
            return

        if self._is_generating:
            return

        self._is_generating = True
        self._caption_panel.clear_caption()
        self._caption_panel.set_generating(True)
        self._settings_panel.set_generating(True)
        self._image_viewer.set_processing(True)
        self._set_connection_status("generating", "Generating...")

        # Start generation in background thread — store as instance attrs
        self._generation_thread = QThread()
        self._caption_worker = CaptionWorker(
            engine=self._engine,
            image_path=self._current_image,
            prompt=self._settings_panel.get_prompt(),
            temperature=self._settings_panel.get_temperature(),
            top_p=self._settings_panel.get_top_p(),
            max_tokens=self._settings_panel.get_max_tokens(),
            prefix=self._settings_panel.get_prefix(),
            suffix=self._settings_panel.get_suffix(),
        )
        self._caption_worker.moveToThread(self._generation_thread)

        self._generation_thread.started.connect(self._caption_worker.run)
        self._caption_worker.new_token.connect(self._caption_panel.append_token)
        self._caption_worker.finished.connect(self._on_caption_finished)
        self._caption_worker.error.connect(self._on_caption_error)
        self._caption_worker.finished.connect(self._generation_thread.quit)
        self._caption_worker.error.connect(self._generation_thread.quit)

        self._generation_thread.start()

    def _on_caption_finished(self, caption: str):
        """Handle completed caption generation."""
        self._is_generating = False
        self._caption_panel.set_generating(False)
        self._settings_panel.set_generating(False)
        self._image_viewer.set_processing(False)

        # Cache the caption
        if self._current_image:
            self._captions[str(self._current_image)] = caption
            self._file_browser.set_item_caption(self._current_image, caption)
            self._file_browser.set_item_status(self._current_image, "done")

            # Auto-save .txt sidecar during batch (or always if checkbox is on)
            if self._settings_panel.auto_save_cb.isChecked():
                txt_path = self._current_image.with_suffix(".txt")
                try:
                    txt_path.write_text(caption, encoding="utf-8")
                except Exception as e:
                    self._notify(f"Auto-save failed for {txt_path.name}: {e}", "error")

        # Update inference time
        inf_time = self._engine.last_inference_time
        self._settings_panel.set_inference_time(inf_time)
        self._inference_label.setText(f"Inference: {inf_time:.1f}s")

        self._set_connection_status("ready", "Ready")

        # Continue batch if active
        if self._batch_queue:
            self._process_next_batch_item()

    def _on_caption_error(self, error: str):
        """Handle caption generation error."""
        self._is_generating = False
        self._caption_panel.set_generating(False)
        self._settings_panel.set_generating(False)
        self._image_viewer.set_processing(False)
        self._caption_panel.show_feedback(f"Error: {error[:80]}", is_success=False)
        self._set_connection_status("error", "Error")
        self._notify(f"Caption error: {error[:80]}", "error")

        # Cancel batch on error
        self._batch_queue.clear()

    # --- Status helpers ---

    def _set_connection_status(self, state: str, text: str):
        """Update status bar connection indicator. state: ready|loading|generating|error"""
        color_map = {
            "ready": COLORS["success"],
            "loading": COLORS["warning"],
            "generating": COLORS["warning"],
            "error": COLORS["error"],
        }
        icon_map = {
            "ready": "\u2022",       # ●
            "loading": "\u23F3",     # ⏳
            "generating": "\u26A1",  # ⚡
            "error": "\u2022",       # ●
        }
        color = color_map.get(state, COLORS["text_dim"])
        icon = icon_map.get(state, "\u2022")

        self._conn_dot.setStyleSheet(f"color: {color}; font-size: 12px; padding-right: 4px; background: transparent;")
        self._conn_label.setText(text)
        self._conn_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px; background: transparent;")

    # --- Settings Change Handler ---

    def _on_settings_changed(self):
        """Handle settings panel changes — update caption panel format badge."""
        preset_id = self._settings_panel.get_active_preset()
        if preset_id:
            from gui.settings_panel import TARGET_PRESETS
            for preset in TARGET_PRESETS:
                if preset["id"] == preset_id:
                    self._caption_panel.set_format_label(preset["name"])
                    break

    # --- Batch Captioning ---

    def _batch_caption_all(self):
        """Start batch captioning for all imported images."""
        if not self._engine.is_loaded:
            QMessageBox.warning(self, "Model Required", "Please load the model first.")
            return

        all_paths = self._file_browser.get_all_paths()
        if not all_paths:
            QMessageBox.warning(self, "No Images", "Please import images first.")
            return

        # Optionally skip images that already have a .txt sidecar
        skip_existing = self._settings_panel.skip_existing_cb.isChecked()
        if skip_existing:
            filtered = [p for p in all_paths if not p.with_suffix(".txt").exists()]
            skipped = len(all_paths) - len(filtered)
            if not filtered:
                QMessageBox.information(
                    self, "Nothing to Do",
                    f"All {len(all_paths)} images already have .txt captions."
                )
                return
            if skipped:
                self._notify(f"Skipping {skipped} images with existing captions", "info")
            all_paths = filtered

        self._batch_queue = list(all_paths)
        self._batch_index = 0

        # Mark all as queued
        for p in self._batch_queue:
            self._file_browser.set_item_status(p, "queued")

        self._progress_bar.setRange(0, len(self._batch_queue))
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._queue_label.setText(f"Queue: {len(self._batch_queue)} remaining")

        self._process_next_batch_item()

    def _process_next_batch_item(self):
        """Process the next image in the batch queue."""
        if not self._batch_queue:
            self._on_batch_complete()
            return

        path = self._batch_queue.pop(0)
        self._batch_index += 1

        # Update UI
        self._file_browser.set_item_status(path, "processing")
        self._file_browser.select_item(path)
        self._settings_panel.set_batch_progress(
            self._batch_index, self._batch_index + len(self._batch_queue)
        )
        self._progress_bar.setValue(self._batch_index)
        self._queue_label.setText(f"Queue: {len(self._batch_queue)} remaining")

        # Generate (will call _process_next_batch_item on finish via _on_caption_finished)
        QTimer.singleShot(100, self._generate_caption)

    def _on_batch_complete(self):
        """Handle batch completion."""
        total = self._batch_index
        self._batch_index = 0
        self._progress_bar.setVisible(False)
        self._queue_label.setText(f"Batch complete: {total} images captioned")
        self._settings_panel.set_batch_progress(total, total)
        self._caption_panel.show_feedback(f"Batch complete! {total} images captioned.")
        self._notify(f"Batch complete: {total} images captioned", "success")

    # --- Save / Export ---

    def _save_current_caption(self):
        """Save the current caption as a .txt sidecar file."""
        if not self._current_image:
            return

        caption = self._caption_panel.get_caption()
        if not caption:
            self._caption_panel.show_feedback("Nothing to save", is_success=False)
            return

        txt_path = self._current_image.with_suffix(".txt")
        try:
            txt_path.write_text(caption, encoding="utf-8")
            self._captions[str(self._current_image)] = caption
            self._caption_panel.show_feedback(f"Saved: {txt_path.name}")
            self._file_browser.set_item_status(self._current_image, "done")
        except Exception as e:
            self._caption_panel.show_feedback(f"Save error: {e}", is_success=False)

    def _export_all_captions(self):
        """Export all cached captions as .txt sidecar files."""
        if not self._captions:
            QMessageBox.information(
                self, "Nothing to Export",
                "No captions to export. Generate captions first."
            )
            return

        saved = 0
        errors = 0
        for path_str, caption in self._captions.items():
            try:
                img_path = Path(path_str)
                txt_path = img_path.with_suffix(".txt")
                txt_path.write_text(caption, encoding="utf-8")
                saved += 1
            except Exception:
                errors += 1

        msg = f"Exported {saved} caption files."
        if errors:
            msg += f"\n{errors} error(s) occurred."

        QMessageBox.information(self, "Export Complete", msg)

    # --- Tab Switching ---

    def _switch_tab(self, tab_name: str):
        """Switch between Project and Dataset views."""
        for name, btn in self._tab_buttons.items():
            if name == tab_name:
                btn.setProperty("class", "nav-tab-active")
            else:
                btn.setProperty("class", "nav-tab")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        if tab_name == "Dataset":
            self._main_stack.setCurrentIndex(1)
            self._refresh_dataset()
        else:
            self._main_stack.setCurrentIndex(0)

    def _refresh_dataset(self):
        """Populate the dataset panel from the file browser's loaded images."""
        paths = self._file_browser.get_all_paths()
        self._dataset_panel.populate(paths)

    # --- GPU / RAM Info ---

    def _init_nvml(self):
        """Initialize NVIDIA Management Library for real VRAM monitoring."""
        self._pynvml = None
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, module="pynvml")
                import pynvml
            self._pynvml = pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._nvml_handle = None

    def _update_gpu_info(self):
        """Update GPU/VRAM display in the nav bar pill using pynvml."""
        if self._nvml_handle is not None and self._pynvml is not None:
            try:
                mem_info = self._pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                mem_used_gb = mem_info.used / (1024 ** 3)
                mem_total_gb = mem_info.total / (1024 ** 3)
                pct = int(mem_info.used / mem_info.total * 100) if mem_info.total > 0 else 0

                self._gpu_label.setText(f"GPU: {pct}%")
                self._vram_label.setText(f"{mem_used_gb:.1f}/{mem_total_gb:.0f}GB")

                # Color-code: green <70%, yellow 70-90%, red >90%
                if pct >= 90:
                    color = COLORS["error"]
                elif pct >= 70:
                    color = COLORS["warning"]
                else:
                    color = COLORS["success"]

                self._gpu_dot.setStyleSheet(f"color: {color}; font-size: 14px; background: transparent;")
                self._gpu_label.setStyleSheet(
                    f"color: {color}; font-size: 10px; font-weight: 600; "
                    f"letter-spacing: 0.5px; text-transform: uppercase; background: transparent;"
                )
                return
            except Exception:
                pass

        self._gpu_label.setText("GPU: Active")
        self._vram_label.setText("VRAM: N/A")

    def _update_ram_info(self):
        """Update RAM display in the status bar."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            used_gb = mem.used / (1024 ** 3)
            total_gb = mem.total / (1024 ** 3)
            self._ram_label.setText(f"RAM: {used_gb:.1f} / {total_gb:.0f} GB")
        except ImportError:
            self._ram_label.setText("")

    # --- Cleanup ---

    def closeEvent(self, event):
        """Clean up all threads on close."""
        # Cancel any active caption generation
        if self._caption_worker:
            self._caption_worker.cancel()
        if self._generation_thread and self._generation_thread.isRunning():
            self._generation_thread.quit()
            self._generation_thread.wait(3000)

        # Wait for model load thread if running
        if self._model_load_thread and self._model_load_thread.isRunning():
            self._model_load_thread.quit()
            self._model_load_thread.wait(5000)

        # Wait for download thread if running
        if self._download_thread and self._download_thread.isRunning():
            self._download_thread.quit()
            self._download_thread.wait(5000)

        self._engine.unload()

        # Shutdown pynvml
        self._gpu_timer.stop()
        if self._nvml_handle is not None and self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_handle = None

        event.accept()
