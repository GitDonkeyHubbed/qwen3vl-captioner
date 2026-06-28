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

from PyQt6.QtCore import Qt, QThread, QEventLoop, pyqtSignal, QObject, QTimer
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSplitter, QStatusBar, QProgressBar, QApplication, QFileDialog,
    QMessageBox, QSizePolicy, QStackedWidget, QProgressDialog,
)

from gui.file_browser import FileBrowserPanel
from gui.image_viewer import ImageViewer
from gui.caption_panel import CaptionPanel
from gui.settings_panel import SettingsPanel
from gui.dataset_panel import DatasetPanel
from gui.notification_panel import NotificationStore, NotificationPanel
from gui.theme import COLORS
from engine.inference import Qwen3VLEngine
from engine.model_downloader import ensure_mmproj, find_mmproj_file, download_named_mmproj


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


# --- Worker for a blocking download run on a background thread ---
class _BlockingDownloadWorker(QObject):
    """Runs a blocking download callable on a QThread.

    The callable receives a ``progress_callback(message, fraction)`` and
    returns its result (e.g. a Path). Used to move the vision-encoder
    (mmproj) download off the Qt UI thread so the window stays responsive
    while it runs (see MainWindow._download_mmproj_blocking).
    """
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)   # result of the callable
    error = pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        try:
            result = self._fn(
                lambda msg, _f=None: self.progress.emit(str(msg))
            )
        except Exception as e:
            self.error.emit(str(e))
            return
        self.finished.emit(result)


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
            # A cancelled generation returns the partial text via a normal
            # return (the engine breaks out of the stream loop), so route it
            # to error('cancelled') rather than finished() — otherwise the
            # truncated caption would be cached/auto-saved as a real result.
            if self._cancelled:
                self.error.emit("cancelled")
            else:
                self.finished.emit(caption)
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")





class MainWindow(QMainWindow):
    """
    Main application window with three-panel layout matching the Figma mockup.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        super().__init__()
        from gui.version import APP_VERSION
        self.setWindowTitle(f"QWEN 3 VL ABL Captioner V{APP_VERSION} — GGUF Engine")
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
        self._is_loading = False  # guards re-entrant model loads (during mmproj dialogs)
        self._batch_queue: List[Path] = []
        self._batch_index = 0
        self._batch_active = False  # True from batch start until completion/abort
        self._download_thread: Optional[QThread] = None
        self._download_worker = None  # ModelDownloadWorker (lazy import)
        self._finished_threads: List[QThread] = []  # keep refs until done
        self._pending_mmproj = None  # (repo_id, filename, target_dir) to chain

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

        # Shield icon -> Qwen Logo (from file)
        logo_path = Path(__file__).parent / "qwen-icon-logo-png_seeklogo-611724.png"
        logo_label = QLabel()
        logo_label.setFixedSize(28, 28)
        if logo_path.exists():
            from PyQt6.QtGui import QPixmap
            pixmap = QPixmap(str(logo_path))
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    28, 28,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                logo_label.setPixmap(scaled)
        left_group.addWidget(logo_label)

        # Brand text block
        brand_block = QVBoxLayout()
        brand_block.setContentsMargins(0, 0, 0, 0)
        brand_block.setSpacing(0)

        brand_title = QLabel("QWEN 3 VL ABL Captioner")
        brand_title.setProperty("class", "brand-title")
        brand_title.setStyleSheet(
            f"color: {COLORS['text_primary']}; font-size: 13px; font-weight: 700; "
            f"letter-spacing: 0.5px; padding: 0; margin: 0; background: transparent;"
        )
        brand_block.addWidget(brand_title)

        from gui.version import APP_VERSION
        brand_sub = QLabel(f"V{APP_VERSION}")
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

        # Engine status text
        self._conn_label = QLabel("Local llama.cpp engine — no model loaded")
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

        # Stop button — shown right next to the progress bar only while a model
        # download is running, so cancelling is discoverable where you watch it.
        self._dl_stop_btn = QPushButton("✕ Stop")
        self._dl_stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._dl_stop_btn.setToolTip(
            "Stop the current download and clear the partial file"
        )
        self._dl_stop_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {COLORS['error']}; "
            f"border: 1px solid {COLORS['error']}; border-radius: 4px; "
            f"font-size: 10px; font-weight: 600; padding: 1px 8px; margin-left: 8px; }} "
            f"QPushButton:hover {{ background: {COLORS['error']}; color: #ffffff; }}"
        )
        self._dl_stop_btn.setVisible(False)
        self._dl_stop_btn.clicked.connect(self._cancel_download)
        self._status_bar.addPermanentWidget(self._dl_stop_btn)

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
        self._settings_panel.browse_model_requested.connect(self._browse_for_model)
        self._settings_panel.cancel_requested.connect(self._cancel_generation)

        # Populate the model dropdown with what's actually on disk
        self._refresh_model_list()

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
        self._batch_active = False

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

    def _selected_backend(self) -> str:
        """Return "mlx" or "gguf" for the current dropdown selection."""
        from gui.model_download_manager import get_model_info

        kind, value = self._settings_panel.get_selected_model()
        if kind == "registry":
            info = get_model_info(value)
            if info and info.get("backend") == "mlx":
                return "mlx"
        return "gguf"

    def _ensure_engine(self, backend: str):
        """Swap the engine instance to match the requested backend."""
        from engine.mlx_engine import MlxVlmEngine

        if backend == "mlx":
            if not isinstance(self._engine, MlxVlmEngine):
                self._engine = MlxVlmEngine()
        else:
            if not isinstance(self._engine, Qwen3VLEngine):
                self._engine = Qwen3VLEngine()

    def _load_model(self):
        """Load the selected model, guarding against re-entrant loads.

        The mmproj dialogs and the modal encoder-download dialog in the
        implementation spin the Qt event loop while is_loaded is still False
        and the Load button is enabled, so a second trigger could otherwise
        start a second concurrent load on a single engine (racing self.model
        assignment). The _is_loading flag prevents that.
        """
        if self._is_loading or (self._model_load_thread and self._model_load_thread.isRunning()):
            return
        if self._engine.is_loaded:
            return
        self._is_loading = True
        try:
            self._load_model_impl()
        finally:
            # Keep the flag set only if a background load actually started;
            # otherwise clear it so early-return paths (model not found,
            # load cancelled, mmproj download failed) let the user retry.
            if not (self._model_load_thread and self._model_load_thread.isRunning()):
                self._is_loading = False

    def _load_model_impl(self):
        """Resolve the model + vision encoder and start the load thread."""
        if self._engine.is_loaded:
            return

        # Find the model file (GGUF) or folder (MLX)
        model_path = self._find_model_file()
        if not model_path:
            QMessageBox.warning(
                self, "Model Not Found",
                "Could not find the selected model on disk.\n\n"
                "Download it with the ⬇ button, or use 📁 Browse to pick a "
                "local GGUF file."
            )
            return

        backend = self._selected_backend()
        self._ensure_engine(backend)

        if backend == "mlx":
            # MLX models embed the vision tower — no mmproj needed
            self._start_model_load(model_path, None)
            return

        # GGUF: resolve the vision encoder (mmproj). Prefer the encoder that
        # MATCHES the selected model — pairing a model with a different model's
        # mmproj crashes llama.cpp natively (this was the load-crash bug). Only
        # fall back to "any mmproj in the folder" for local/unknown models that
        # have no registry entry.
        model_dir = model_path.parent
        self._settings_panel.set_model_status("Checking for vision encoder...")

        info = self._selected_registry_info()
        expected_mmproj = info.get("mmproj_filename") if info else None

        if expected_mmproj:
            candidate = model_dir / expected_mmproj
            mmproj_path = candidate if candidate.is_file() else None
        else:
            mmproj_path = find_mmproj_file(model_dir)

        if mmproj_path is None:
            mmproj_path = self._resolve_missing_mmproj(model_dir, info, expected_mmproj)
            if mmproj_path is None:
                return  # cancelled or download failed (status already set)

        self._start_model_load(model_path, mmproj_path)

    def _selected_registry_info(self):
        """Registry info dict for the currently selected model, or None for a
        local/unknown selection (browsed file with no registry entry)."""
        from gui.model_download_manager import get_model_info
        kind, value = self._settings_panel.get_selected_model()
        if kind != "registry":
            return None
        return get_model_info(value)

    def _download_mmproj_blocking(self, title, fn):
        """Run a blocking mmproj download off the UI thread.

        ``fn`` takes a ``progress_callback(message, fraction)`` and returns the
        downloaded Path. The work runs on a QThread while a modal, indeterminate
        progress dialog spins a local event loop, so the main window keeps
        repainting (no 'Not Responding') instead of freezing for the whole
        multi-hundred-MB encoder transfer. Returns the Path, or re-raises the
        worker's exception so the caller's existing error handling applies.
        """
        thread = QThread()
        worker = _BlockingDownloadWorker(fn)
        worker.moveToThread(thread)

        dialog = QProgressDialog(
            f"{title}\nThis can take a while for large encoders…",
            None, 0, 0, self,   # no Cancel button; 0/0 = indeterminate
        )
        dialog.setWindowTitle("Downloading Vision Encoder")
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)

        loop = QEventLoop()
        state = {"result": None, "error": None}

        def on_progress(msg):
            dialog.setLabelText(msg)
            self._settings_panel.set_model_status(msg)

        def on_finished(result):
            state["result"] = result
            loop.quit()

        def on_error(msg):
            state["error"] = msg
            loop.quit()

        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.started.connect(worker.run)

        thread.start()
        dialog.show()
        loop.exec()            # keeps the UI alive until the worker signals done
        dialog.close()

        thread.quit()
        # Wait without a timeout: loop.exec() only returns after the worker
        # emitted finished/error (the last thing run() does), so the thread is
        # already finishing. A bounded wait that timed out would let thread/
        # worker fall out of scope while still running -> "QThread: Destroyed
        # while thread is still running" and a possible crash. Once wait()
        # returns the thread has fully stopped, so dropping the references on
        # return is safe.
        thread.wait()

        if state["error"] is not None:
            raise RuntimeError(state["error"])
        return state["result"]

    def _resolve_missing_mmproj(self, model_dir, info, expected_mmproj):
        """Obtain a vision encoder when the one matching the model isn't on disk.

        For a known registry model, download/browse for ITS specific encoder —
        never a generic default, which would mismatch and crash. For local
        models, fall back to the legacy default-download/browse flow. Returns a
        Path, or None if the user cancels or the download fails.
        """
        if info and expected_mmproj:
            mismatch_note = (
                "\n\nA different model's vision encoder is present, but pairing "
                "mismatched encoders crashes the engine — this model needs its own."
                if find_mmproj_file(model_dir) is not None else ""
            )
            answer = QMessageBox.question(
                self, "Vision Encoder Needed",
                f"The vision encoder for this model isn't downloaded:\n"
                f"  {expected_mmproj}\n\n"
                f"Download it now from {info['repo_id']}?{mismatch_note}\n\n"
                "Choose No to browse for it manually.",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes,
            )
            if answer == QMessageBox.StandardButton.Yes:
                try:
                    self._settings_panel.set_model_status(
                        "Downloading matching vision encoder..."
                    )
                    return self._download_mmproj_blocking(
                        f"Downloading matching vision encoder ({expected_mmproj})…",
                        lambda cb: download_named_mmproj(
                            info["repo_id"], expected_mmproj, model_dir,
                            progress_callback=cb,
                        ),
                    )
                except Exception as e:
                    QMessageBox.critical(
                        self, "mmproj Error",
                        f"Failed to download vision encoder:\n{e}"
                    )
                    self._settings_panel.set_model_status("Error: mmproj not found")
                    return None
            if answer == QMessageBox.StandardButton.No:
                file_path, _ = QFileDialog.getOpenFileName(
                    self, "Select matching mmproj (Vision Encoder)",
                    str(model_dir), "GGUF models (*.gguf)"
                )
                if not file_path:
                    self._settings_panel.set_model_status("Load cancelled")
                    return None
                return Path(file_path)
            self._settings_panel.set_model_status("Load cancelled")
            return None

        # Local/unknown model: legacy default-download or browse flow.
        answer = QMessageBox.question(
            self, "Vision Encoder Needed",
            "No mmproj (vision encoder) .gguf found next to this model.\n\n"
            "Download the default Qwen3-VL 8B mmproj into that folder?\n\n"
            "Choose No to browse for an mmproj file manually\n"
            "(use the mmproj published alongside your model).",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if answer == QMessageBox.StandardButton.Yes:
            try:
                return self._download_mmproj_blocking(
                    "Downloading default vision encoder…",
                    lambda cb: ensure_mmproj(model_dir, progress_callback=cb),
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "mmproj Error",
                    f"Failed to download vision encoder:\n{e}"
                )
                self._settings_panel.set_model_status("Error: mmproj not found")
                return None
        if answer == QMessageBox.StandardButton.No:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select mmproj (Vision Encoder)",
                str(model_dir), "GGUF models (*.gguf)"
            )
            if not file_path:
                self._settings_panel.set_model_status("Load cancelled")
                return None
            return Path(file_path)
        self._settings_panel.set_model_status("Load cancelled")
        return None

    def _start_model_load(self, model_path: Path, mmproj_path: Optional[Path]):
        """Kick off the background model-load thread (both backends)."""
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
        self._is_loading = False
        info = self._engine.get_model_info()
        model_name = info.get('model_file', 'Model')
        vision = info.get('mmproj_file') or "built into model (MLX)"
        self._settings_panel.set_model_status(
            f"{model_name} loaded and ready.",
            detail=f"Vision encoder: {vision}",
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
        self._is_loading = False
        self._settings_panel.set_model_status("Error loading model", detail=error[:100])
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
            get_model_info, model_file_exists, mlx_model_exists,
        )

        info = get_model_info(model_name)
        if info is None:
            QMessageBox.information(
                self, "Download Unavailable",
                f"No auto-download entry for '{model_name}'.\n\n"
                "Place a compatible .gguf file in the model directory\n"
                "and restart the application.",
            )
            return

        is_mlx = info.get("backend") == "mlx"
        display_name = info["folder"] if is_mlx else info["filename"]

        # Destination for a genuinely-new download
        target_dir = self._model_dir or Path(__file__).resolve().parent.parent

        # "Already downloaded" must use the SAME multi-dir scan as discovery
        # (_find_model_file / _refresh_model_list), otherwise a model already
        # present in another search dir is treated as missing and re-downloaded
        # as a duplicate.
        existing_dir = None
        for d in self._model_search_dirs():
            found = (
                mlx_model_exists(d, info["folder"]) if is_mlx
                else model_file_exists(d, info["filename"])
            )
            if found:
                existing_dir = d
                break
        if existing_dir is not None:
            QMessageBox.information(
                self, "Already Downloaded",
                f"{display_name} already exists in:\n{existing_dir}"
            )
            return

        # Confirm with user (file size warning)
        answer = QMessageBox.question(
            self, "Download Model",
            f"Download {display_name}?\n\n"
            f"Size: ~{info['size_gb']:.1f} GB\n"
            f"From: {info['repo_id']}\n"
            f"To:   {target_dir}\n\n"
            "This may take several minutes.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        # Queue the matching mmproj to auto-download right after the model
        # (skipped if any mmproj already exists in the target dir)
        self._pending_mmproj = None
        if not is_mlx and info.get("mmproj_filename"):
            if find_mmproj_file(target_dir) is None:
                self._pending_mmproj = (
                    info["repo_id"], info["mmproj_filename"], target_dir
                )

        self._start_file_download(
            repo_id=info["repo_id"],
            filename=info.get("filename", ""),
            target_dir=target_dir,
            display_name=display_name,
            snapshot_folder=info["folder"] if is_mlx else None,
        )

    def _start_file_download(
        self, repo_id: str, filename: str, target_dir: Path,
        display_name: str, snapshot_folder: Optional[str] = None,
    ):
        """Start a background download with progress UI (model or mmproj)."""
        from gui.model_download_manager import ModelDownloadWorker
        from gui.config import get_hf_token

        self._progress_bar.setRange(0, 0)  # indeterminate until fractions arrive
        self._progress_bar.setVisible(True)
        self._queue_label.setText(f"Downloading {display_name}...")
        self._notify(f"Downloading {display_name}...", "download")
        self._settings_panel.set_download_in_progress(True)
        self._dl_stop_btn.setEnabled(True)
        self._dl_stop_btn.setText("✕ Stop")
        self._dl_stop_btn.setVisible(True)

        # Keep references to finishing threads so chained downloads don't
        # garbage-collect a QThread that is still shutting down
        self._finished_threads = [t for t in self._finished_threads if t.isRunning()]
        if self._download_thread is not None:
            self._finished_threads.append(self._download_thread)

        self._download_thread = QThread()
        self._download_worker = ModelDownloadWorker(
            repo_id=repo_id,
            filename=filename,
            target_dir=target_dir,
            hf_token=get_hf_token(),
            snapshot_folder=snapshot_folder,
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
        if 0.0 < fraction <= 1.0:
            # Switch from indeterminate to a real percentage bar
            if self._progress_bar.maximum() == 0:
                self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(int(fraction * 100))

    def _on_download_finished(self, local_path: str):
        """Handle successful download."""
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setVisible(False)
        self._settings_panel.set_download_in_progress(False)
        self._hide_download_stop_btn()
        filename = Path(local_path).name
        self._queue_label.setText(f"Downloaded: {filename}")
        self._notify(f"Download complete: {filename}", "success")
        # Refresh the dropdown so the new model shows its ✓ marker
        self._refresh_model_list()

        # Chain the matching vision encoder download if one was queued
        if self._pending_mmproj and "mmproj" not in filename.lower():
            repo_id, mmproj_name, target_dir = self._pending_mmproj
            self._pending_mmproj = None
            if find_mmproj_file(target_dir) is None:
                QTimer.singleShot(
                    150,
                    lambda: self._start_file_download(
                        repo_id, mmproj_name, target_dir,
                        f"vision encoder ({mmproj_name})",
                    ),
                )

    def _on_download_error(self, error: str):
        """Handle download failure."""
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setVisible(False)
        self._settings_panel.set_download_in_progress(False)
        self._hide_download_stop_btn()
        self._pending_mmproj = None  # don't leave a stale mmproj queued after a failure/cancel
        self._queue_label.setText("Download failed")
        if "cancelled" not in error.lower():
            self._notify(f"Download failed: {error[:80]}", "error")
            QMessageBox.critical(
                self, "Download Error",
                f"Failed to download model:\n\n{error}"
            )
        else:
            self._queue_label.setText("Download cancelled")
            self._notify("Download cancelled by user", "info")

    def _cancel_download(self):
        """Stop an in-progress model download and clear its partial file.

        Wired to the status-bar Stop button so a wrong/slow download can be
        aborted and a different model selected.
        """
        if not (
            self._download_worker
            and self._download_thread
            and self._download_thread.isRunning()
        ):
            self._hide_download_stop_btn()
            return

        answer = QMessageBox.question(
            self, "Stop Download",
            "Stop the current download and delete the partial file?\n\n"
            "You can then choose a different model.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        # Don't auto-start the matching mmproj after the model is cancelled
        self._pending_mmproj = None
        self._download_worker.cancel()
        self._dl_stop_btn.setEnabled(False)
        self._dl_stop_btn.setText("Stopping…")
        # An MLX folder (snapshot) download can only stop between files — a
        # large shard already in flight has to finish first — so be honest
        # about the delay instead of looking hung.
        if getattr(self._download_worker, "snapshot_folder", None):
            msg = "Stopping after the current file finishes…"
            self._queue_label.setText(msg)
            self._notify(msg, "info")
        else:
            self._notify("Stopping download…", "info")

    def _hide_download_stop_btn(self):
        """Reset and hide the status-bar download Stop button."""
        self._dl_stop_btn.setVisible(False)
        self._dl_stop_btn.setEnabled(True)
        self._dl_stop_btn.setText("✕ Stop")

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

    def _model_search_dirs(self) -> List[Path]:
        """Directories scanned for GGUF model files."""
        search_dirs = []
        if self._model_dir:
            search_dirs.append(self._model_dir)
        app_dir = Path(__file__).resolve().parent.parent
        search_dirs.append(app_dir.parent)
        search_dirs.append(app_dir)
        # Dedupe while preserving order
        seen = set()
        unique = []
        for d in search_dirs:
            r = d.resolve()
            if r not in seen:
                seen.add(r)
                unique.append(r)
        return unique

    def _refresh_model_list(self):
        """Rebuild the model dropdown from the registry, files on disk, and
        user-added custom models (issue #7)."""
        from gui.model_download_manager import (
            MODEL_REGISTRY, MLX_MODEL_REGISTRY, mlx_model_exists,
            mlx_backend_supported,
        )
        from gui.config import get_custom_models

        search_dirs = self._model_search_dirs()

        # Which registry models are already downloaded?
        downloaded = set()
        registry_filenames = set()
        for name, info in MODEL_REGISTRY.items():
            registry_filenames.add(info["filename"])
            for d in search_dirs:
                if (d / info["filename"]).is_file():
                    downloaded.add(name)
                    break
        if mlx_backend_supported():
            for name, info in MLX_MODEL_REGISTRY.items():
                for d in search_dirs:
                    if mlx_model_exists(d, info["folder"]):
                        downloaded.add(name)
                        break

        # Local models: user-added paths (anywhere on disk) plus unknown
        # GGUF files found in the search dirs
        local_paths: List[Path] = []
        seen_local = set()
        for p_str in get_custom_models():
            p = Path(p_str)
            if p.is_file() and p.resolve() not in seen_local:
                seen_local.add(p.resolve())
                local_paths.append(p)
        for d in search_dirs:
            if not d.is_dir():
                continue
            try:
                entries = sorted(d.iterdir())
            except OSError:
                continue
            for f in entries:
                if (
                    f.is_file()
                    and f.suffix == ".gguf"
                    and "mmproj" not in f.name.lower()
                    and f.name not in registry_filenames
                    and f.resolve() not in seen_local
                ):
                    seen_local.add(f.resolve())
                    local_paths.append(f)

        # Memory budget (GB) so the dropdown can flag models that won't fit.
        # On NVIDIA: total VRAM via NVML. On Apple Silicon (no NVML, where the
        # MLX models live): currently-available unified memory via psutil, so
        # the fit hints reflect what's actually usable rather than the full
        # machine RAM.
        vram_gb = None
        if self._nvml_handle is not None and self._pynvml is not None:
            try:
                mem_info = self._pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                vram_gb = mem_info.total / (1024 ** 3)
            except Exception:
                pass
        if vram_gb is None and sys.platform == "darwin":
            try:
                import psutil
                vram_gb = psutil.virtual_memory().available / (1024 ** 3)
            except Exception:
                pass

        self._settings_panel.populate_models(local_paths, downloaded, vram_gb)

    def _browse_for_model(self):
        """Let the user pick any GGUF model file from disk (issue #7)."""
        start_dir = str(self._model_dir or Path.home())
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF Model", start_dir, "GGUF models (*.gguf)"
        )
        if not file_path:
            return

        path = Path(file_path)
        if "mmproj" in path.name.lower():
            QMessageBox.warning(
                self, "Vision Encoder Selected",
                "That file looks like an mmproj (vision encoder), not a main "
                "model.\n\nSelect the main model GGUF instead — the mmproj in "
                "the same folder is picked up automatically.",
            )
            return

        from gui.config import add_custom_model
        add_custom_model(str(path))
        self._refresh_model_list()
        self._settings_panel.select_local_model(path)
        self._notify(f"Added local model: {path.name}", "success")

    def _find_model_file(self) -> Optional[Path]:
        """Resolve the model file (GGUF) or folder (MLX) for the selection."""
        from gui.model_download_manager import get_model_info, mlx_model_exists

        kind, value = self._settings_panel.get_selected_model()

        if kind == "local":
            path = Path(value)
            return path if path.is_file() else None

        model_info = get_model_info(value)

        # MLX models live in folders, not single files
        if model_info and model_info.get("backend") == "mlx":
            for dir_path in self._model_search_dirs():
                if mlx_model_exists(dir_path, model_info["folder"]):
                    return dir_path / model_info["folder"]
            return None

        target_filename = model_info["filename"] if model_info else None
        search_dirs = self._model_search_dirs()

        # First pass: look for the specific selected model file
        if target_filename:
            for dir_path in search_dirs:
                candidate = dir_path / target_filename
                if candidate.is_file():
                    return candidate

        # Fallback: any non-mmproj GGUF file
        for dir_path in search_dirs:
            if not dir_path.is_dir():
                continue
            for f in dir_path.iterdir():
                if f.is_file() and f.suffix == ".gguf" and "mmproj" not in f.name.lower():
                    return f

        return None

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

        # Keep references to finishing generation threads (mirrors the download
        # path) so a still-shutting-down QThread from the previous batch item
        # isn't garbage-collected while its C++ side is alive ("QThread
        # destroyed while still running").
        self._finished_threads = [t for t in self._finished_threads if t.isRunning()]
        if self._generation_thread is not None:
            self._finished_threads.append(self._generation_thread)

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

    def _cancel_generation(self):
        """Cancel the current caption generation or batch process."""
        cancelled_something = False

        # Cancel active caption worker
        if self._caption_worker and self._is_generating:
            self._caption_worker.cancel()
            cancelled_something = True

        # Cancel batch queue / active batch (the queue may already be empty if
        # the LAST item is mid-generation, so also check the batch flag)
        if self._batch_queue or self._batch_active:
            remaining = len(self._batch_queue)
            self._batch_queue.clear()
            self._batch_index = 0
            self._batch_active = False
            self._progress_bar.setVisible(False)
            self._queue_label.setText(f"Batch cancelled ({remaining} remaining skipped)")
            self._notify(f"Batch cancelled — {remaining} images skipped", "info")
            cancelled_something = True

        # Cancel active download
        if self._download_worker and self._download_thread and self._download_thread.isRunning():
            self._download_worker.cancel()
            self._notify("Cancelling download...", "info")
            cancelled_something = True

        if cancelled_something:
            self._caption_panel.show_feedback("Cancelled", is_success=False)
            self._set_connection_status("ready", "Cancelled")
        else:
            self._caption_panel.show_feedback("Nothing to cancel", is_success=False)

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

        # Update inference time
        inf_time = self._engine.last_inference_time
        self._settings_panel.set_inference_time(inf_time)
        self._inference_label.setText(f"Inference: {inf_time:.1f}s")

        self._set_connection_status("ready", "Ready")

        # ── Auto-Save ──
        # Branch on the batch FLAG, not the queue: the queue is popped before
        # the final image generates, so it is already empty while the last
        # caption is in flight. Using the flag ensures the last item is saved
        # silently like the rest and that _on_batch_complete still runs.
        if self._batch_active:
            self._auto_save_caption(self._current_image, caption)
            self._process_next_batch_item()
        elif self._settings_panel.get_auto_save():
            self._auto_save_caption(self._current_image, caption)
        else:
            # Single image: ask the user
            self._prompt_auto_save(caption)

    def _prompt_auto_save(self, caption: str):
        """Show a Yes/No dialog asking whether to auto-save the caption file."""
        if not self._current_image or not caption:
            return

        txt_path = self._current_image.with_suffix(".txt")
        answer = QMessageBox.question(
            self, "Auto Save Caption",
            f"Caption generated!\n\n"
            f"Save as: {txt_path.name}\n"
            f"Location: {txt_path.parent}\n\n"
            "Would you like to save this caption now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,  # default to Yes
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._auto_save_caption(self._current_image, caption)

    def _auto_save_caption(self, image_path: Path, caption: str):
        """Silently save a caption as a .txt sidecar file."""
        if not image_path or not caption:
            return
        txt_path = image_path.with_suffix(".txt")
        try:
            txt_path.write_text(caption, encoding="utf-8")
            self._captions[str(image_path)] = caption
            self._file_browser.set_item_status(image_path, "done")
            self._caption_panel.show_feedback(f"Saved: {txt_path.name}")
        except Exception as e:
            self._caption_panel.show_feedback(f"Save error: {e}", is_success=False)

    def _on_caption_error(self, error: str):
        """Handle a caption generation error (or a user cancellation)."""
        self._is_generating = False
        self._caption_panel.set_generating(False)
        self._settings_panel.set_generating(False)
        self._image_viewer.set_processing(False)

        was_cancel = "cancel" in error.lower()
        if was_cancel:
            self._caption_panel.show_feedback("Generation cancelled", is_success=False)
            self._set_connection_status("ready", "Cancelled")
            self._notify("Caption generation cancelled", "info")
        else:
            self._caption_panel.show_feedback(f"Error: {error[:80]}", is_success=False)
            self._set_connection_status("error", "Error")
            self._notify(f"Caption error: {error[:80]}", "error")

        # End any active batch cleanly. The success path resets the batch UI via
        # _on_batch_complete; the error/cancel path must do it here so the
        # progress bar and queue label don't linger in a misleading state.
        if self._batch_active or self._batch_queue:
            self._batch_queue.clear()
            self._batch_index = 0
            self._batch_active = False
            self._progress_bar.setVisible(False)
            if not was_cancel:
                self._queue_label.setText("Batch stopped (error)")
            self._settings_panel.set_batch_progress(0, 0)

    # --- Status helpers ---

    def _set_connection_status(self, state: str, text: str):
        """Update status bar connection indicator. state: ready|loading|generating|error"""
        color_map = {
            "ready": COLORS["success"],
            "loading": COLORS["warning"],
            "generating": COLORS["warning"],
            "error": COLORS["error"],
        }
        color = color_map.get(state, COLORS["text_dim"])

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

        self._batch_queue = list(all_paths)
        self._batch_index = 0
        self._batch_active = True

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
        QTimer.singleShot(100, self._start_deferred_batch_caption)

    def _start_deferred_batch_caption(self):
        """Fire the next batch caption, unless the batch was cancelled.

        There's a ~100 ms gap between batch items where _is_generating is
        already False; if the user cancels in that window, _batch_active is
        cleared. Guard here so the queued timer doesn't start an orphan
        generation on the last-selected image after cancellation.
        """
        if self._batch_active:
            self._generate_caption()

    def _on_batch_complete(self):
        """Handle batch completion."""
        self._batch_active = False
        total = self._batch_index
        self._batch_index = 0
        self._progress_bar.setVisible(False)
        self._queue_label.setText(f"Batch complete: {total} images captioned")
        self._settings_panel.set_batch_progress(total, total)
        self._caption_panel.show_feedback(f"Batch complete! {total} images captioned.")
        self._notify(f"Batch complete: {total} images captioned", "success")

        # Prompt to auto-save all batch captions
        answer = QMessageBox.question(
            self, "Auto Save All Captions",
            f"Batch captioning complete! ({total} images)\n\n"
            "All captions were saved during batch processing.\n"
            "Would you also like to export all captions as .txt files?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._export_all_captions()

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

        # macOS: Apple Silicon shares unified memory between CPU and GPU,
        # so show system memory pressure instead of a CUDA VRAM readout.
        if sys.platform == "darwin":
            try:
                import psutil
                mem = psutil.virtual_memory()
                used_gb = (mem.total - mem.available) / (1024 ** 3)
                total_gb = mem.total / (1024 ** 3)
                pct = int(mem.percent)

                self._gpu_label.setText(f"MEM: {pct}%")
                self._vram_label.setText(f"{used_gb:.1f}/{total_gb:.0f}GB UNIFIED")

                if pct >= 90:
                    color = COLORS["error"]
                elif pct >= 70:
                    color = COLORS["warning"]
                else:
                    color = COLORS["success"]
                self._gpu_dot.setStyleSheet(
                    f"color: {color}; font-size: 14px; background: transparent;"
                )
                self._gpu_label.setStyleSheet(
                    f"color: {color}; font-size: 10px; font-weight: 600; "
                    f"letter-spacing: 0.5px; text-transform: uppercase; background: transparent;"
                )
                return
            except ImportError:
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
        threads_clean = True

        # Caption generation: cancel only if actually generating (idempotent,
        # avoids touching an already-finished worker).
        if self._caption_worker and self._is_generating:
            self._caption_worker.cancel()
        if self._generation_thread and self._generation_thread.isRunning():
            self._generation_thread.quit()
            if not self._generation_thread.wait(10000):
                threads_clean = False

        # The model-load thread cannot be interrupted mid-construction of the
        # native Llama handler — give it a long blocking wait rather than
        # racing unload() against a still-running load.
        if self._model_load_thread and self._model_load_thread.isRunning():
            self._model_load_thread.quit()
            if not self._model_load_thread.wait(60000):
                threads_clean = False

        # Download: quit() alone can't stop the blocking run()/thread pool —
        # signal cancel first so the worker threads actually exit. Capture the
        # wait() result like the other threads: a download stuck mid-shard must
        # block unload() rather than be destroyed while still running.
        if self._download_thread and self._download_thread.isRunning():
            if self._download_worker:
                self._download_worker.cancel()
            self._download_thread.quit()
            if not self._download_thread.wait(10000):
                threads_clean = False

        # Any earlier download/generation threads still shutting down (kept alive
        # in _finished_threads so they aren't GC'd mid-stop) must also be joined
        # before unload(), so none is destroyed while still running.
        for t in getattr(self, "_finished_threads", []):
            if t.isRunning():
                t.quit()
                if not t.wait(10000):
                    threads_clean = False

        # Only free the native engine objects once no worker can still be using
        # them. On a dirty exit (a wait timed out), skip unload() — the process
        # is terminating and will reclaim the memory anyway; deleting the model
        # under a live worker can crash llama.cpp.
        if threads_clean:
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
