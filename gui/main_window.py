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
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSplitter, QStatusBar, QProgressBar, QApplication, QFileDialog,
    QMessageBox, QSizePolicy, QStackedWidget, QProgressDialog,
)

from gui.caption_io import caption_path, read_caption, write_caption
from gui.file_browser import FileBrowserPanel
from gui.image_viewer import ImageViewer
from gui.caption_panel import CaptionPanel
from gui.settings_panel import SettingsPanel
from gui.dataset_panel import DatasetPanel
from gui.notification_panel import NotificationStore, NotificationPanel
from gui.theme import COLORS
from engine.inference import Qwen3VLEngine
from engine.model_downloader import (
    default_mmproj_fits, download_named_mmproj, ensure_mmproj, find_mmproj_file,
)


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


class _UnclosableProgressDialog(QProgressDialog):
    """A QProgressDialog that ignores Esc and the title-bar close button.

    QProgressDialog hides itself on Esc/close even with no Cancel button,
    which would silently drop application modality while the nested event
    loop of _download_mmproj_blocking is still running (leaving an invisible,
    unstoppable download and a fully interactive main window mid-load).
    Closing is re-enabled via allow_close() once the worker has finished.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._allow_close = False

    def allow_close(self):
        self._allow_close = True

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and not self._allow_close:
            event.ignore()
            return
        super().keyPressEvent(event)

    def reject(self):
        # Esc reaches the dialog as reject(); swallow it while the worker runs.
        if self._allow_close:
            super().reject()

    def closeEvent(self, event):
        if self._allow_close:
            super().closeEvent(event)
        else:
            event.ignore()


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
        self._app_title = f"QWEN 3 VL ABL Captioner V{APP_VERSION}"
        self.setWindowTitle(self._app_title)
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
        # Caption cache. `_captions` holds the text; `_caption_mtimes` records
        # the sidecar mtime each cached entry was read from or written to, so a
        # file edited outside the app is re-read instead of served stale; and
        # `_unsaved` marks generated captions the user has NOT accepted, which
        # must never be pushed to disk by Export.
        self._captions: Dict[str, str] = {}  # str(path) -> caption
        self._caption_mtimes: Dict[str, Optional[float]] = {}
        self._unsaved: set[str] = set()
        # Partial text of the in-flight generation, so switching away from and
        # back to the generating image restores the stream instead of leaving
        # a stale caption that later tokens append to.
        self._stream_buffer = ""

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
        self._batch_total = 0
        self._batch_saved = 0
        self._batch_failed = 0
        self._batch_current_path: Optional[Path] = None  # item pinned for the deferred timer
        self._download_thread: Optional[QThread] = None
        self._download_worker = None  # ModelDownloadWorker (lazy import)
        self._finished_threads: List[QThread] = []  # keep refs until done
        self._pending_mmproj = None  # (repo_id, filename, target_dir) to chain

        # NVML (GPU monitoring)
        self._nvml_handle = None
        self._init_nvml()

        # Periodic GPU/RAM refresh timer (5 seconds)
        self._gpu_timer = QTimer(self)
        self._gpu_timer.setInterval(5000)
        self._gpu_timer.timeout.connect(self._update_gpu_info)
        self._gpu_timer.timeout.connect(self._update_ram_info)

        # Notification system
        self._notification_store = NotificationStore(self)
        self._notification_panel: Optional[NotificationPanel] = None  # created lazily after bell btn exists

        # Build UI
        self._build_nav_bar()
        self._build_main_layout()
        self._build_status_bar()
        self._connect_signals()
        self._setup_shortcuts()

        # Folders/images can be dropped anywhere on the window, not just on
        # the narrow file-browser strip (events are delegated to it).
        self.setAcceptDrops(True)

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
        # No inline `color:` — the brand-title rule supplies it, so the title
        # follows a runtime theme switch instead of staying zinc-100 on a
        # light nav bar (1.15:1).
        brand_title.setStyleSheet(
            "letter-spacing: 0.5px; padding: 0; margin: 0; background: transparent;"
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
        self._file_browser.stem_collision_detected.connect(
            lambda msg: self._notify(msg, "warning")
        )
        self._file_browser.caption_decode_warning.connect(
            lambda msg: self._notify(msg, "warning")
        )
        self._file_browser.import_failed.connect(
            lambda msg: self._notify(msg, "error")
        )
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

    def _setup_shortcuts(self):
        """Keyboard shortcuts for the core loop: navigate, generate, save.

        Navigation deliberately uses Ctrl+arrows / PageUp+PageDown rather than
        bare arrows, which must keep working inside the caption text editor.
        """
        QShortcut(QKeySequence.StandardKey.Save, self, self._save_current_caption)
        QShortcut(QKeySequence("Ctrl+G"), self, self._generate_caption)
        QShortcut(QKeySequence("Ctrl+Right"), self, lambda: self._select_adjacent(1))
        QShortcut(QKeySequence("Ctrl+Left"), self, lambda: self._select_adjacent(-1))
        QShortcut(QKeySequence(Qt.Key.Key_PageDown), self, lambda: self._select_adjacent(1))
        QShortcut(QKeySequence(Qt.Key.Key_PageUp), self, lambda: self._select_adjacent(-1))

    def _select_adjacent(self, delta: int):
        """Select the next/previous image in the file browser."""
        paths = self._file_browser.get_all_paths()
        if not paths:
            return
        try:
            idx = paths.index(self._current_image) + delta
        except ValueError:
            idx = 0
        idx = max(0, min(idx, len(paths) - 1))
        self._file_browser.select_item(paths[idx])

    # --- Window-level drag & drop (delegated to the file browser) ---

    def dragEnterEvent(self, event):
        self._file_browser.dragEnterEvent(event)

    def dragLeaveEvent(self, event):
        # Without this the "Drop images here" overlay stayed visible after a
        # drag left the window without dropping.
        self._file_browser.dragLeaveEvent(event)

    def dropEvent(self, event):
        self._file_browser.dropEvent(event)

    # --- Image Selection ---

    # --- Caption cache ---

    def _cache_caption(self, path: Path, caption: str, *, saved: bool,
                       mtime: Optional[float] = None):
        """Record a caption in the cache.

        `saved=False` marks it as generated-but-not-accepted, which keeps
        Export from writing it over a good file on disk.
        """
        key = str(path)
        self._captions[key] = caption
        if saved:
            self._unsaved.discard(key)
            self._caption_mtimes[key] = (
                mtime if mtime is not None else read_caption(path).mtime
            )
        else:
            self._unsaved.add(key)
            self._caption_mtimes[key] = None

    def _load_caption(self, path: Path) -> str:
        """Return the caption to show for *path*, re-reading a changed sidecar.

        The cache used to be write-once: once an entry existed the `.txt` was
        never read again, so an edit made outside the app was invisible and
        the panel could show text that no longer matched the file.
        """
        key = str(path)
        if key in self._unsaved:
            # Not on disk yet — the cache is the only copy.
            return self._captions.get(key, "")

        info = read_caption(path)
        if key in self._captions and info.mtime == self._caption_mtimes.get(key):
            return self._captions[key]

        if info.read_error:
            self._notify(
                f"Could not read {caption_path(path).name}: {info.read_error}",
                "error",
            )
            self._captions.pop(key, None)
            self._caption_mtimes.pop(key, None)
            return ""

        if not info.exists:
            self._captions.pop(key, None)
            self._caption_mtimes.pop(key, None)
            return ""

        if info.decode_error:
            self._notify(
                f"{caption_path(path).name} is not valid UTF-8 — shown with "
                "replacement characters; saving will rewrite it as UTF-8.",
                "warning",
            )
        self._captions[key] = info.text
        self._caption_mtimes[key] = info.mtime
        return info.text

    def _confirm_discard_caption_edit(self) -> bool:
        """Offer to save an unsaved hand-edit. False means "abandon the action"."""
        if not self._caption_panel.is_dirty():
            return True
        if not self._current_image:
            self._caption_panel.mark_clean()
            return True

        answer = QMessageBox.question(
            self, "Unsaved Caption",
            f"The caption for {self._current_image.name} has unsaved edits.\n\n"
            "Save it before continuing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            # A failed write must not be treated as "saved" — keep the edit
            # and let the user deal with the error.
            if not self._save_current_caption():
                return False
            return True
        self._caption_panel.mark_clean()
        return True

    # --- Image Selection ---

    def _on_image_selected(self, path: Path):
        """Handle image selection from the file browser."""
        if (
            self._current_image is not None
            and path != self._current_image
            and not self._batch_active
            and not self._confirm_discard_caption_edit()
        ):
            # Put the highlight back without re-entering this handler.
            self._file_browser.select_item(self._current_image, emit=False)
            return

        self._current_image = path
        self._image_viewer.set_image(path)

        # A generation in flight on this very image: restore the partial
        # stream, not the (older) cached caption the tokens would append to.
        if (
            self._is_generating
            and self._caption_worker is not None
            and self._caption_worker.image_path == path
        ):
            self._caption_panel.set_caption(self._stream_buffer)
            return

        caption = self._load_caption(path)
        if caption:
            self._caption_panel.set_caption(caption)
        else:
            self._caption_panel.clear_caption()

    def _on_clear_all(self):
        """Reset the workspace — clear all images, captions, and viewer state."""
        if not self._confirm_discard_caption_edit():
            return

        # Cancel an in-flight generation first, or an orphan worker keeps
        # streaming tokens into the freshly cleared panel.
        if self._caption_worker and self._is_generating:
            self._caption_worker.cancel()

        # Cancel any in-progress batch, including its progress UI — leaving
        # these set stranded the Batch button disabled and labelled
        # "Processing N/M...".
        self._reset_batch_state()

        # The panel does not clear itself on the Clear All click so this
        # handler can still abandon the action above.
        self._file_browser.clear_all()

        # Clear captions cache
        self._captions.clear()
        self._caption_mtimes.clear()
        self._unsaved.clear()
        self._stream_buffer = ""
        self._current_image = None

        # Reset viewer and caption panel
        self._image_viewer.clear()
        self._caption_panel.clear_caption()

        # Reset status bar
        self._progress_bar.setVisible(False)
        self._queue_label.setText("")

        self._notify("Workspace cleared", "info")

    def _reset_batch_state(self):
        """Clear the batch queue and every piece of UI that reflects it."""
        self._batch_queue.clear()
        self._batch_index = 0
        self._batch_active = False
        self._batch_current_path = None
        self._progress_bar.setVisible(False)
        self._settings_panel.set_batch_progress(0, 0)

    def _reset_batch_statuses(self):
        """Return queued/processing thumbnails to a truthful state.

        An aborted batch used to leave the in-flight thumbnail stuck at
        "Captioning..." and every remaining one at "Queued" indefinitely.
        """
        for path in self._file_browser.get_all_paths():
            status = self._file_browser.get_item_status(path)
            if status in ("queued", "processing"):
                key = str(path)
                if key in self._unsaved:
                    self._file_browser.set_item_status(path, "generated")
                elif read_caption(path).has_caption:
                    self._file_browser.set_item_status(path, "done")
                else:
                    self._file_browser.set_item_status(path, "idle")

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
        # The title used to hard-code "GGUF Engine" even while an MLX model
        # was loaded. Name the engine actually in use, or nothing at all.
        self.setWindowTitle(
            f"{self._app_title} — {'MLX' if backend == 'mlx' else 'GGUF'} Engine"
        )

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
            # For a registry model we know exactly what to fetch — offer to
            # download it right here instead of pointing at the ⬇ button.
            kind, value = self._settings_panel.get_selected_model()
            info = self._selected_registry_info()
            if kind == "registry" and info:
                answer = QMessageBox.question(
                    self, "Model Not Downloaded",
                    f"'{value}' isn't downloaded yet "
                    f"(~{info.get('size_gb', 0):.1f} GB).\n\n"
                    "Download it now?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if answer == QMessageBox.StandardButton.Yes:
                    self._download_model(value)
                return
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
            mmproj_path = find_mmproj_file(model_dir, model_path)

        if mmproj_path is None:
            mmproj_path = self._resolve_missing_mmproj(
                model_dir, info, expected_mmproj, model_path
            )
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

        dialog = _UnclosableProgressDialog(
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
        dialog.allow_close()
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

    def _resolve_missing_mmproj(self, model_dir, info, expected_mmproj,
                                model_path=None):
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
                if find_mmproj_file(model_dir) is not None else ""  # any encoder at all
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

        # Local/unknown model: legacy default-download or browse flow. The
        # built-in download only ships the Qwen3-VL 8B encoder, so it is
        # offered only when it can actually pair with this model — handing it
        # to, say, a 4B model crashes llama.cpp natively on the first caption.
        if not default_mmproj_fits(model_path):
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select the mmproj (Vision Encoder) for this model",
                str(model_dir), "GGUF models (*.gguf)"
            )
            if not file_path:
                QMessageBox.warning(
                    self, "Vision Encoder Needed",
                    f"No vision encoder matching {Path(model_path).name} was "
                    "found next to it.\n\n"
                    "The built-in download only provides the Qwen3-VL 8B "
                    "encoder, which this model cannot use — download the "
                    "mmproj published alongside your model and put it in the "
                    "same folder.",
                )
                self._settings_panel.set_model_status("Error: mmproj not found")
                return None
            return Path(file_path)

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
                    lambda cb: ensure_mmproj(
                        model_dir, progress_callback=cb, model_path=model_path
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
        # Show the human-readable error up front; tuck the raw traceback into
        # the expandable details instead of dumping it as the message body.
        msg, _, tb = error.partition("\nTraceback")
        box = QMessageBox(
            QMessageBox.Icon.Critical, "Model Load Error",
            f"Failed to load model:\n\n{msg.strip()[:600]}", parent=self,
        )
        if tb:
            box.setDetailedText("Traceback" + tb)
        box.exec()

    def _unload_model(self):
        """Unload the current model and reset UI state."""
        if not self._engine.is_loaded:
            return

        # Don't unload while generating — including the ~100 ms gap between
        # batch items, where _is_generating is briefly False but the batch
        # timer is about to fire on the unloaded engine (zombie batch).
        if self._is_generating or self._batch_active or self._batch_queue:
            QMessageBox.warning(
                self, "Cannot Unload",
                "Please wait for the current generation/batch to finish "
                "(or cancel it) before unloading."
            )
            return

        self._engine.unload()

        # Reset UI state
        self.setWindowTitle(self._app_title)
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
        dlg.deleteLater()  # don't accumulate a dialog per gear click

    def _on_theme_changed(self, mode: str):
        """Handle theme switch from settings dialog."""
        from gui.theme import set_theme, get_stylesheet, apply_placeholder_palette
        set_theme(mode)
        from PyQt6.QtWidgets import QApplication
        app_instance = QApplication.instance()
        if app_instance:
            app_instance.setStyleSheet(get_stylesheet(mode))
            apply_placeholder_palette(app_instance)

        # Widgets that paint or set colours themselves cannot be reached by the
        # app stylesheet — tell them to re-resolve against the new palette.
        for widget in (
            self._settings_panel, self._dataset_panel, self._file_browser,
        ):
            refresh = getattr(widget, "refresh_theme", None)
            if refresh is not None:
                refresh()
        if self._notification_panel is not None:
            self._notification_panel.refresh_theme()

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

        # Re-entrancy guard: starting a second download while one is running
        # would leave the first worker orphaned and both writing the same
        # .part file.
        if self._download_thread is not None and self._download_thread.isRunning():
            self._notify(
                f"A download is already running — {display_name} was not started.",
                "warning",
            )
            return

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

        # Capture the worker BEFORE the modal dialog: while it is open the
        # event loop keeps running, so the download can finish and a chained
        # one (e.g. the auto-mmproj) can start — cancelling the new worker
        # because of a dialog answered about the old one would be wrong.
        worker_at_prompt = self._download_worker

        answer = QMessageBox.question(
            self, "Stop Download",
            "Stop the current download and delete the partial file?\n\n"
            "You can then choose a different model.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        if (
            worker_at_prompt is not self._download_worker
            or not (self._download_thread and self._download_thread.isRunning())
        ):
            return  # the download the user answered about is already gone

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
        memory_is_shared = False
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
                memory_is_shared = True
            except Exception:
                pass

        self._settings_panel.populate_models(
            local_paths, downloaded, vram_gb, memory_is_shared
        )

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

        # Fallback: any non-mmproj GGUF file — ONLY for a selection with no
        # registry entry. For a registry model this fallback silently loaded a
        # different GGUF off disk, then downloaded that entry's vision encoder
        # and paired it with the foreign model: exactly the mismatched-mmproj
        # crash the surrounding code exists to prevent. Returning None instead
        # lets the "isn't downloaded yet — download it now?" prompt fire.
        if model_info is not None:
            return None

        for dir_path in search_dirs:
            if not dir_path.is_dir():
                continue
            for f in dir_path.iterdir():
                if f.is_file() and f.suffix == ".gguf" and "mmproj" not in f.name.lower():
                    return f

        return None

    # --- Caption Generation ---

    def _generate_caption(self, image_path: Optional[Path] = None):
        """Generate a caption for *image_path* (default: the current image).

        The target is pinned HERE, at start — everything downstream
        (_on_caption_finished) attributes the result to the worker's own
        image_path, never to whatever happens to be selected when the
        caption completes.
        """
        target = image_path or self._current_image
        if not self._engine.is_loaded:
            QMessageBox.warning(self, "Model Required", "Please load the model first.")
            return

        if not target:
            QMessageBox.warning(self, "No Image", "Please select an image first.")
            return

        if self._is_generating:
            return

        # A regenerate wipes the caption box — offer to keep a hand-edit first.
        if not self._batch_active and not self._confirm_discard_caption_edit():
            return

        self._is_generating = True
        self._stream_buffer = ""
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
            image_path=target,
            prompt=self._settings_panel.get_prompt(),
            temperature=self._settings_panel.get_temperature(),
            top_p=self._settings_panel.get_top_p(),
            max_tokens=self._settings_panel.get_max_tokens(),
            prefix=self._settings_panel.get_prefix(),
            suffix=self._settings_panel.get_suffix(),
        )
        self._caption_worker.moveToThread(self._generation_thread)

        self._generation_thread.started.connect(self._caption_worker.run)
        self._caption_worker.new_token.connect(self._on_new_token)
        self._caption_worker.finished.connect(self._on_caption_finished)
        self._caption_worker.error.connect(self._on_caption_error)
        self._caption_worker.finished.connect(self._generation_thread.quit)
        self._caption_worker.error.connect(self._generation_thread.quit)

        self._generation_thread.start()

    def _on_new_token(self, token: str):
        """Append a streamed token, but only to its own image's caption box.

        Tokens used to be wired straight to the panel, so selecting a
        different image mid-generation kept appending the running caption on
        top of the newly selected image's text — and Save/Ctrl+S then wrote
        that foreign partial text into the wrong .txt and cache.
        """
        self._stream_buffer += token
        worker = self._caption_worker
        if worker is None or worker.image_path != self._current_image:
            return
        self._caption_panel.append_token(token)

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
            self._reset_batch_state()
            self._queue_label.setText(f"Batch cancelled ({remaining} remaining skipped)")
            self._notify(f"Batch cancelled — {remaining} images skipped", "info")
            cancelled_something = True

        if cancelled_something:
            self._reset_batch_statuses()
            self._caption_panel.show_feedback("Cancelled", is_success=False)
            self._set_connection_status("ready", "Cancelled")
        else:
            # A model download is NOT cancelled here: it has its own Cancel
            # button, which confirms first. Silently deleting a multi-GB
            # partial because the user meant to stop a batch is not a
            # recoverable mistake.
            self._caption_panel.show_feedback("Nothing to cancel", is_success=False)

    def _on_caption_finished(self, caption: str):
        """Handle completed caption generation.

        CRITICAL: the result is attributed to the image the WORKER was started
        for, not to self._current_image — the user may have clicked another
        thumbnail while the caption streamed, and saving under the selection
        would silently overwrite that other image's caption file.
        """
        worker_path = (
            self._caption_worker.image_path if self._caption_worker
            else self._current_image
        )

        self._is_generating = False
        self._caption_panel.set_generating(False)
        self._settings_panel.set_generating(False)
        self._image_viewer.set_processing(False)

        # Cache the caption under the image it belongs to. It is NOT saved
        # yet — the status stays "generated" (no green check) until a write
        # actually succeeds, and Export must not push it over a good file.
        if worker_path:
            self._cache_caption(worker_path, caption, saved=False)
            self._file_browser.set_item_caption(worker_path, caption)
            self._file_browser.set_item_status(worker_path, "generated")

        if worker_path != self._current_image and self._current_image:
            # The selection moved mid-generation: the panel holds the streamed
            # text of ANOTHER image — restore the selected image's own caption.
            own = self._load_caption(self._current_image)
            if own:
                self._caption_panel.set_caption(own)
            else:
                self._caption_panel.clear_caption()
        elif worker_path:
            # Still on the generated image: replace the raw streamed text with
            # the processed caption (clean_caption plus prefix/suffix) that was
            # cached and will be saved, so a later Save can't silently strip
            # the preset's prefix/suffix back off.
            self._caption_panel.set_caption(caption)

        self._stream_buffer = ""

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
            if self._auto_save_caption(worker_path, caption):
                self._batch_saved += 1
            else:
                self._batch_failed += 1
            self._process_next_batch_item()
        elif self._settings_panel.get_auto_save():
            self._auto_save_caption(worker_path, caption)
        else:
            # Single image: ask the user
            self._prompt_auto_save(worker_path, caption)

    def _prompt_auto_save(self, image_path: Optional[Path], caption: str):
        """Show a Yes/No dialog asking whether to auto-save the caption file."""
        if not image_path or not caption:
            return

        txt_path = image_path.with_suffix(".txt")
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
            self._auto_save_caption(image_path, caption)

    def _auto_save_caption(self, image_path: Path, caption: str) -> bool:
        """Silently save a caption as a .txt sidecar file.

        Returns True only when the bytes actually reached disk. A failed write
        used to be swallowed: the item was still marked "done", nothing
        reached the notification store, and the batch dialog reported every
        caption as saved.
        """
        if not image_path or not caption:
            return False
        txt_path = caption_path(image_path)
        try:
            mtime = write_caption(image_path, caption)
        except Exception as e:
            self._caption_panel.show_feedback(f"Save error: {e}", is_success=False)
            self._notify(f"Save failed for {image_path.name}: {e}", "error")
            return False
        self._cache_caption(image_path, caption, saved=True, mtime=mtime)
        self._file_browser.set_item_status(image_path, "done")
        self._caption_panel.show_feedback(f"Saved: {txt_path.name}")
        return True

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
            self._reset_batch_state()
            if not was_cancel:
                self._queue_label.setText("Batch stopped (error)")
            self._reset_batch_statuses()
        elif not self._is_generating:
            # A single generation that failed leaves its own thumbnail at
            # "Captioning..." otherwise.
            self._reset_batch_statuses()

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
        if not preset_id:
            # No preset is active — say so instead of leaving the last
            # preset's name (or the startup default "SDXL") on the badge while
            # a completely different prompt is being sent.
            self._caption_panel.set_format_label("Custom")
            return
        from gui.settings_panel import TARGET_PRESETS
        for preset in TARGET_PRESETS:
            if preset["id"] == preset_id:
                self._caption_panel.set_format_label(preset["name"])
                break

    # --- Batch Captioning ---

    def _batch_caption_all(self):
        """Start batch captioning for all imported images."""
        if self._batch_active or self._is_generating:
            return  # re-entrancy guard: a batch or single caption is running

        if not self._engine.is_loaded:
            QMessageBox.warning(self, "Model Required", "Please load the model first.")
            return

        all_paths = self._file_browser.get_all_paths()
        if not all_paths:
            QMessageBox.warning(self, "No Images", "Please import images first.")
            return

        # Batch re-captions everything and overwrites every .txt sidecar,
        # hand-edited ones included. Ask before destroying existing work.
        already = [p for p in all_paths if read_caption(p).has_caption]
        if already:
            box = QMessageBox(self)
            box.setWindowTitle("Existing Captions Found")
            box.setIcon(QMessageBox.Icon.Question)
            box.setText(
                f"{len(already)} of {len(all_paths)} images already have a "
                ".txt caption."
            )
            box.setInformativeText(
                "Re-captioning them overwrites those files, including any "
                "captions you edited by hand."
            )
            skip_btn = box.addButton(
                f"Skip {len(already)} already-captioned",
                QMessageBox.ButtonRole.AcceptRole,
            )
            overwrite_btn = box.addButton(
                "Overwrite all", QMessageBox.ButtonRole.DestructiveRole
            )
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.setDefaultButton(skip_btn)
            box.exec()

            clicked = box.clickedButton()
            if clicked is overwrite_btn:
                queue = list(all_paths)
            elif clicked is skip_btn:
                done = {str(p) for p in already}
                queue = [p for p in all_paths if str(p) not in done]
                if not queue:
                    QMessageBox.information(
                        self, "Nothing to Caption",
                        "Every imported image already has a caption.",
                    )
                    return
            else:
                return
        else:
            queue = list(all_paths)

        self._batch_current_path: Optional[Path] = None
        self._batch_queue = queue
        self._batch_index = 0
        self._batch_active = True
        self._batch_saved = 0
        self._batch_failed = 0

        # Mark all as queued
        for p in self._batch_queue:
            self._file_browser.set_item_status(p, "queued")

        self._batch_total = len(self._batch_queue)
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
        # Pin the batch target: generation must run on the POPPED item, not on
        # whatever the selection happens to be when the deferred timer fires —
        # a user click in the 100 ms gap would otherwise redirect the batch.
        self._batch_current_path = path

        # Update UI
        self._file_browser.set_item_status(path, "processing")
        self._file_browser.select_item(path)
        self._settings_panel.set_batch_progress(
            self._batch_index, self._batch_index + len(self._batch_queue)
        )
        self._progress_bar.setValue(self._batch_index)
        eta_txt = ""
        inf_t = getattr(self._engine, "last_inference_time", 0) or 0
        if inf_t > 0:
            eta_s = (len(self._batch_queue) + 1) * inf_t
            eta_txt = (
                f" (~{eta_s:.0f}s left)" if eta_s < 90
                else f" (~{eta_s / 60:.0f} min left)"
            )
        self._queue_label.setText(
            f"Queue: {len(self._batch_queue)} remaining{eta_txt}"
        )

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
            self._generate_caption(self._batch_current_path)

    def _on_batch_complete(self):
        """Handle batch completion."""
        total = self._batch_index
        saved = self._batch_saved
        failed = self._batch_failed
        self._reset_batch_state()

        summary = f"{saved} saved"
        if failed:
            summary += f", {failed} failed"
        self._queue_label.setText(f"Batch complete: {total} captioned ({summary})")
        self._caption_panel.show_feedback(
            f"Batch complete! {summary}.", is_success=not failed
        )
        self._notify(
            f"Batch complete: {total} captioned — {summary}",
            "error" if failed else "success",
        )

        # Every caption was already written as a .txt sidecar during the run —
        # a "would you also like to export .txt files?" question here was a
        # confusing no-op, so just confirm what happened. Report the failures
        # too: claiming all captions were saved when a write failed sent users
        # away believing work was on disk that never made it.
        if failed:
            QMessageBox.warning(
                self, "Batch Finished With Errors",
                f"Batch finished — {saved} caption(s) saved as .txt files, "
                f"but {failed} could not be written.\n\n"
                "See the notification bell for the individual errors; those "
                "captions are still in the app and can be saved from the "
                "caption box.",
            )
        else:
            QMessageBox.information(
                self, "Batch Complete",
                f"Batch complete — {saved} captions saved as .txt files "
                "next to the images.",
            )

    # --- Save / Export ---

    def _save_current_caption(self) -> bool:
        """Save the current caption as a .txt sidecar file. True on success."""
        if not self._current_image:
            return False

        # Refuse while a caption is streaming: the box is mid-generation and
        # may not even hold this image's text yet.
        if self._is_generating:
            self._caption_panel.show_feedback(
                "Still generating — save when it finishes", is_success=False
            )
            return False

        caption = self._caption_panel.get_caption()
        if not caption:
            self._caption_panel.show_feedback("Nothing to save", is_success=False)
            return False

        txt_path = caption_path(self._current_image)
        try:
            mtime = write_caption(self._current_image, caption)
        except Exception as e:
            self._caption_panel.show_feedback(f"Save error: {e}", is_success=False)
            self._notify(
                f"Save failed for {self._current_image.name}: {e}", "error"
            )
            return False

        self._cache_caption(self._current_image, caption, saved=True, mtime=mtime)
        self._caption_panel.mark_clean()
        self._caption_panel.show_feedback(f"Saved: {txt_path.name}")
        self._file_browser.set_item_caption(self._current_image, caption)
        self._file_browser.set_item_status(self._current_image, "done")
        return True

    def _export_all_captions(self):
        """Export cached captions as .txt sidecar files.

        The cache is compared against disk first. Blindly rewriting every
        sidecar from memory destroyed good on-disk captions — including
        replacing one with a generated caption the user had explicitly
        declined to save.
        """
        if not self._captions:
            QMessageBox.information(
                self, "Nothing to Export",
                "No captions to export. Generate captions first."
            )
            return

        new_files: List[tuple] = []
        conflicts: List[tuple] = []
        unchanged = 0
        for path_str, caption in self._captions.items():
            img_path = Path(path_str)
            info = read_caption(img_path)
            if not info.has_caption:
                new_files.append((img_path, caption))
            elif info.text == caption.strip():
                unchanged += 1
            else:
                conflicts.append((img_path, caption))

        targets = list(new_files)
        if conflicts:
            box = QMessageBox(self)
            box.setWindowTitle("Existing Caption Files Differ")
            box.setIcon(QMessageBox.Icon.Warning)
            box.setText(
                f"{len(conflicts)} image(s) already have a .txt caption whose "
                "text differs from the one held in the app."
            )
            box.setInformativeText(
                f"{len(new_files)} caption(s) have no file yet and can be "
                "written safely."
            )
            write_new_btn = box.addButton(
                f"Write {len(new_files)} new only",
                QMessageBox.ButtonRole.AcceptRole,
            )
            overwrite_btn = box.addButton(
                f"Overwrite {len(conflicts)} too",
                QMessageBox.ButtonRole.DestructiveRole,
            )
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.setDefaultButton(write_new_btn)
            box.exec()

            clicked = box.clickedButton()
            if clicked is overwrite_btn:
                targets += conflicts
            elif clicked is not write_new_btn:
                return

        if not targets:
            QMessageBox.information(
                self, "Nothing to Export",
                f"Every caption is already on disk ({unchanged} up to date).",
            )
            return

        saved = 0
        errors = 0
        for img_path, caption in targets:
            try:
                mtime = write_caption(img_path, caption)
            except Exception as e:
                errors += 1
                self._notify(f"Export failed for {img_path.name}: {e}", "error")
                continue
            self._cache_caption(img_path, caption, saved=True, mtime=mtime)
            self._file_browser.set_item_status(img_path, "done")
            saved += 1

        msg = f"Exported {saved} caption files."
        if unchanged:
            msg += f"\n{unchanged} already matched the file on disk."
        if errors:
            msg += f"\n{errors} error(s) occurred — see the notification bell."

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

    def _unsaved_summary(self) -> List[str]:
        """Names of images holding a caption that is not on disk."""
        names = [Path(k).name for k in sorted(self._unsaved)]
        if self._caption_panel.is_dirty() and self._current_image:
            name = self._current_image.name
            if name not in names:
                names.insert(0, name)
        return names

    def closeEvent(self, event):
        """Clean up all threads on close."""
        # Unsaved captions used to be discarded without a word: the close was
        # accepted unconditionally, and a caption whose save prompt was
        # declined still showed the green "done" check.
        pending = self._unsaved_summary()
        if pending:
            shown = ", ".join(pending[:5])
            if len(pending) > 5:
                shown += f", and {len(pending) - 5} more"
            answer = QMessageBox.question(
                self, "Unsaved Captions",
                f"{len(pending)} caption(s) have not been saved to disk "
                f"({shown}).\n\nClose anyway and lose them?",
                QMessageBox.StandardButton.Cancel
                | QMessageBox.StandardButton.Discard,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Discard:
                event.ignore()
                return

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
