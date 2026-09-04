"""
File Browser Panel (Left Sidebar)

Displays imported images as a searchable thumbnail list with caption preview
snippets. Matches the Figma "VL-CAPTIONER Studio Pro" sidebar design:
  - Blue-500 FolderOpen icon + "Project Files" header + pill count badge
  - Search bar with icon
  - Thumbnail list with blue left-border selection highlight
  - Emerald CheckCircle overlay on completed items
  - Status text: blue for processing, zinc-500 for pending
  - Import button at bottom
  - Drag & Drop support for images and folders
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import (
    QPixmap, QPainter, QColor, QPen, QDragEnterEvent, QDropEvent, QImageReader,
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QScrollArea, QFrame, QFileDialog,
)

from engine.inference import IMAGE_EXTENSIONS
from gui.caption_io import read_caption
from gui.theme import COLORS


THUMB_SIZE = 48


class _CheckCircleOverlay(QWidget):
    """Small emerald check-circle painted in the top-right corner of a thumbnail."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(16, 16)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Filled circle
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(COLORS["success"]))
        painter.drawEllipse(1, 1, 13, 13)
        # White checkmark
        painter.setPen(QPen(QColor("#ffffff"), 1.8))
        painter.drawLine(4, 7, 6, 10)
        painter.drawLine(6, 10, 11, 4)
        painter.end()


def is_importable_image(path: Path) -> bool:
    """True for a real image file the app should import.

    AppleDouble sidecars (`._IMG_0001.jpg`) carry an image extension, sort
    first in a folder, and are not decodable — importing one aborted the whole
    batch on item 1. Dot-prefixed files are metadata, never user images.
    """
    if path.name.startswith("."):
        return False
    return path.suffix.lower() in IMAGE_EXTENSIONS


def scan_directory(dir_path: Path) -> List[Path]:
    """Return the importable images in a directory, sorted. May raise OSError."""
    return sorted(f for f in dir_path.iterdir() if f.is_file() and is_importable_image(f))


def _stem_key(path: Path) -> str:
    """Collision key for the `.txt` sidecar a file would claim.

    Case-insensitive on Windows and macOS, where `Photo.jpg` and `photo.png`
    still map to a single sidecar.
    """
    stem = str(path.with_suffix(""))
    stem = os.path.normcase(stem)
    if sys.platform == "darwin":
        stem = stem.lower()
    return stem


class ThumbnailItem(QFrame):
    """A single thumbnail entry in the file list."""

    clicked = pyqtSignal(Path)

    def __init__(self, image_path: Path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self._is_selected = False
        self._caption_preview = ""
        self._status = "idle"  # idle, queued, processing, generated, done

        self.setProperty("class", "thumbnail-item")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(64)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(10)

        # Thumbnail image container (relative positioning for overlay)
        thumb_container = QWidget()
        thumb_container.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        thumb_container.setStyleSheet("background: transparent;")

        self.thumb_label = QLabel(thumb_container)
        self.thumb_label.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        self.thumb_label.setProperty("class", "thumb-image")
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._load_thumbnail()

        # Check overlay (hidden by default)
        self._check_overlay = _CheckCircleOverlay(thumb_container)
        self._check_overlay.move(THUMB_SIZE - 14, -2)
        self._check_overlay.setVisible(False)

        layout.addWidget(thumb_container)

        # Text info
        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 2, 0, 2)
        text_layout.setSpacing(3)

        self.name_label = QLabel(image_path.name)
        self.name_label.setProperty("class", "thumb-name")
        self.name_label.setWordWrap(False)
        text_layout.addWidget(self.name_label)

        self.preview_label = QLabel("")
        self.preview_label.setProperty("class", "thumb-preview")
        self.preview_label.setWordWrap(False)
        text_layout.addWidget(self.preview_label)

        layout.addLayout(text_layout, 1)

    def _load_thumbnail(self):
        """Load and scale the thumbnail image.

        Decode via QImageReader with a target size so the JPEG decoder
        downscales DURING decode — QPixmap(path) decoded every image at full
        resolution (a 40MP photo → ~160 MB of pixels) just to draw a 56px
        thumbnail, freezing the UI thread on large imports.
        """
        reader = QImageReader(str(self.image_path))
        reader.setAutoTransform(True)  # honor EXIF orientation like the viewer
        size = reader.size()
        if size.isValid():
            scaled = size.scaled(
                THUMB_SIZE, THUMB_SIZE, Qt.AspectRatioMode.KeepAspectRatio
            )
            reader.setScaledSize(scaled)
        image = reader.read()
        if not image.isNull():
            self.thumb_label.setPixmap(QPixmap.fromImage(image))
        else:
            self.thumb_label.setText("?")

    def set_selected(self, selected: bool):
        """Update visual selection state."""
        self._is_selected = selected
        self.setProperty("class", "thumbnail-selected" if selected else "thumbnail-item")
        self.style().unpolish(self)
        self.style().polish(self)

    def set_caption_preview(self, text: str):
        """Show a preview snippet of the caption."""
        self._caption_preview = text
        self._set_preview(text)

    def _set_preview(self, text: str, active: bool = False):
        """Set the preview line, styled by class so it follows the theme."""
        self.preview_label.setText(
            text[:40] + "..." if len(text) > 40 else text
        )
        self.preview_label.setProperty(
            "class", "thumb-preview-active" if active else "thumb-preview"
        )
        self.preview_label.style().unpolish(self.preview_label)
        self.preview_label.style().polish(self.preview_label)

    def set_status(self, status: str):
        """Set the status badge (idle, queued, processing, generated, done).

        "generated" and "done" look the same except for the check overlay:
        the check means "written to disk", so a caption the user has not
        saved yet must not wear it.
        """
        self._status = status

        # Show/hide check overlay
        self._check_overlay.setVisible(status == "done")

        if status == "idle" and self._caption_preview:
            self._set_preview(self._caption_preview)
        elif status == "queued":
            self._set_preview("Queued")
        elif status == "processing":
            self._set_preview("Captioning...", active=True)
        elif status in ("done", "generated"):
            if self._caption_preview:
                self._set_preview(self._caption_preview)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.image_path)
        super().mousePressEvent(event)


class _DropOverlay(QFrame):
    """Semi-transparent overlay shown when dragging files over the panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # The overlay must not eat clicks: a drag that entered the window and
        # left without dropping used to strand it visible, swallowing every
        # click on thumbnails and buttons underneath.
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setVisible(False)
        self._label = QLabel("📂  Drop images here", self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.refresh_theme()

    def refresh_theme(self):
        self.setStyleSheet(
            f"background: rgba(59, 130, 246, 0.12); "
            f"border: 2px dashed {COLORS['accent']}; "
            f"border-radius: 8px;"
        )
        self._label.setStyleSheet(
            f"color: {COLORS['accent_text']}; font-size: 14px; font-weight: 600; "
            f"background: transparent; border: none;"
        )

    def resizeEvent(self, event):
        self._label.setGeometry(0, 0, self.width(), self.height())
        super().resizeEvent(event)


class FileBrowserPanel(QFrame):
    """
    Left sidebar panel with project file browser.
    Shows imported images as searchable thumbnail list.
    Supports drag & drop of image files and folders.
    """

    image_selected = pyqtSignal(Path)
    images_imported = pyqtSignal(list)  # list[Path]
    stem_collision_detected = pyqtSignal(str)  # warning text for the user
    import_failed = pyqtSignal(str)            # a path could not be scanned
    caption_decode_warning = pyqtSignal(str)   # non-UTF-8 sidecars were read lossily
    clear_requested = pyqtSignal()      # emitted when user clicks Clear All

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "sidebar-panel")
        self.setMinimumWidth(220)
        self.setMaximumWidth(320)

        # Enable drag & drop
        self.setAcceptDrops(True)

        self._items: Dict[str, ThumbnailItem] = {}  # str(path) -> ThumbnailItem
        self._current_selection: Optional[Path] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Header ──
        header = QFrame()
        header.setStyleSheet(
            f"background: {COLORS['bg_darkest']}; "
            f"border-bottom: 1px solid {COLORS['border']};"
        )
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(16, 12, 16, 10)
        header_layout.setSpacing(8)

        title_row = QHBoxLayout()
        title_row.setSpacing(8)

        # Folder icon (blue)
        folder_icon = QLabel("\U0001F4C2")  # open folder emoji
        folder_icon.setStyleSheet(
            f"font-size: 16px; color: {COLORS['accent_text']};"
        )
        title_row.addWidget(folder_icon)

        title_label = QLabel("Project Files")
        title_label.setStyleSheet(
            f"font-size: 14px; font-weight: 600; color: {COLORS['text_primary']};"
        )
        title_row.addWidget(title_label)

        title_row.addStretch()

        # File count pill badge
        self.count_label = QLabel("0")
        self.count_label.setStyleSheet(
            f"background: {COLORS['bg_hover']}; color: {COLORS['text_secondary']}; "
            f"border-radius: 10px; padding: 2px 8px; "
            f"font-size: 11px; font-weight: 600;"
        )
        title_row.addWidget(self.count_label)

        header_layout.addLayout(title_row)

        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search images...")
        self.search_input.textChanged.connect(self._filter_items)
        header_layout.addWidget(self.search_input)

        layout.addWidget(header)

        # ── Scrollable thumbnail list ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._list_widget = QWidget()
        self._list_widget.setProperty("class", "thumbnail-list")
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setContentsMargins(0, 4, 0, 4)
        self._list_layout.setSpacing(0)
        self._list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        scroll.setWidget(self._list_widget)
        layout.addWidget(scroll, 1)

        # ── Action buttons ──
        btn_frame = QFrame()
        # Scoped to QFrame: an unselectored `background:` here cascades onto
        # every child, overriding the accent-button rule and rendering the
        # primary "Open Folder" button white-on-white in light mode.
        btn_frame.setObjectName("browserActions")
        self._btn_frame = btn_frame
        btn_frame.setStyleSheet(
            f"QFrame#browserActions {{ background: {COLORS['surface_translucent']}; "
            f"border-top: 1px solid {COLORS['border']}; }}"
        )
        btn_layout = QVBoxLayout(btn_frame)
        btn_layout.setContentsMargins(12, 10, 12, 10)
        btn_layout.setSpacing(6)

        # Import Folder (primary action)
        self.import_folder_btn = QPushButton("📂  Open Folder")
        self.import_folder_btn.setProperty("class", "accent-button")
        self.import_folder_btn.setFixedHeight(32)
        self.import_folder_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.import_folder_btn.setToolTip("Import all images from a folder (dataset directory)")
        self.import_folder_btn.clicked.connect(self._on_import_folder_clicked)
        btn_layout.addWidget(self.import_folder_btn)

        # Add individual files (secondary)
        add_row = QHBoxLayout()
        add_row.setSpacing(6)

        self.import_btn = QPushButton("📄 Add Files")
        self.import_btn.setProperty("class", "secondary-button")
        self.import_btn.setFixedHeight(28)
        self.import_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.import_btn.setToolTip("Add individual image files")
        self.import_btn.clicked.connect(self._on_import_clicked)
        add_row.addWidget(self.import_btn)

        # Clear All / Reset
        self.clear_btn = QPushButton("✕ Clear All")
        self.clear_btn.setProperty("class", "secondary-button")
        self.clear_btn.setFixedHeight(28)
        self.clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_btn.setToolTip("Remove all images and reset for a new dataset")
        self.clear_btn.setStyleSheet(
            f"QPushButton {{ color: {COLORS['error']}; border: 1px solid {COLORS['border']}; "
            f"border-radius: 4px; font-size: 11px; font-weight: 600; background: transparent; }}"
            f"QPushButton:hover {{ background: {COLORS['error']}; color: #ffffff; "
            f"border-color: {COLORS['error']}; }}"
        )
        self.clear_btn.clicked.connect(self._on_clear_clicked)
        add_row.addWidget(self.clear_btn)

        btn_layout.addLayout(add_row)

        # Drag & drop hint label
        drop_hint = QLabel("💡 Tip: drag & drop images or folders here")
        drop_hint.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: 9px; font-style: italic; padding-top: 2px;"
        )
        drop_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btn_layout.addWidget(drop_hint)

        layout.addWidget(btn_frame)

        # ── Drop overlay (shown during drag) ──
        self._drop_overlay = _DropOverlay(self)

    # ─── Drag & Drop ──────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Accept drag if it contains file URLs."""
        if event.mimeData() and event.mimeData().hasUrls():
            # Check if any URL is an image or directory
            has_valid = False
            for url in event.mimeData().urls():
                if not url.isLocalFile():
                    continue
                path = Path(url.toLocalFile())
                try:
                    if path.is_dir() or (path.is_file() and is_importable_image(path)):
                        has_valid = True
                        break
                except OSError:
                    continue
            if has_valid:
                event.acceptProposedAction()
                self._drop_overlay.setGeometry(0, 0, self.width(), self.height())
                self._drop_overlay.setVisible(True)
                return
        event.ignore()

    def dragLeaveEvent(self, event):
        """Hide drop overlay when drag leaves."""
        self._drop_overlay.setVisible(False)

    def dropEvent(self, event: QDropEvent):
        """Handle dropped files and folders."""
        self._drop_overlay.setVisible(False)
        if not event.mimeData() or not event.mimeData().hasUrls():
            event.ignore()
            return

        image_paths: List[Path] = []
        failed: List[str] = []
        for url in event.mimeData().urls():
            # Skip non-file URLs. Path(url.toLocalFile()) for, say, an image
            # dragged out of a browser tab is Path(""), whose is_dir() is True
            # for the process working directory — so the drop imported every
            # image in it.
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            try:
                if path.is_dir():
                    image_paths.extend(scan_directory(path))
                elif path.is_file() and is_importable_image(path):
                    image_paths.append(path)
            except OSError as e:
                # A permission-denied or dropped network share raised out of a
                # Qt slot, which aborts the whole PyQt6 process.
                failed.append(f"{path.name}: {e}")

        if failed:
            self.import_failed.emit(
                f"{len(failed)} item(s) could not be read: " + "; ".join(failed[:3])
            )

        if image_paths:
            self.add_images(image_paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    def resizeEvent(self, event):
        """Keep the drop overlay sized to the panel."""
        super().resizeEvent(event)
        if hasattr(self, "_drop_overlay"):
            self._drop_overlay.setGeometry(0, 0, self.width(), self.height())

    # ─── Public API ───────────────────────────────────────────

    def add_images(self, paths: List[Path]):
        """Add images to the file browser."""
        new_paths = []
        decode_warnings: List[str] = []
        for p in paths:
            key = str(p)
            if key not in self._items:
                item = ThumbnailItem(p)
                item.clicked.connect(self._on_item_clicked)
                self._list_layout.addWidget(item)
                self._items[key] = item
                new_paths.append(p)

                # Check for existing caption .txt file
                info = read_caption(p)
                if info.has_caption:
                    item.set_caption_preview(info.text)
                    item.set_status("done")
                if info.decode_error:
                    decode_warnings.append(p.name)

        self.count_label.setText(str(len(self._items)))

        if decode_warnings:
            example = ", ".join(sorted(decode_warnings)[:3])
            self.caption_decode_warning.emit(
                f"{len(decode_warnings)} caption file(s) are not valid UTF-8 "
                f"and were read with replacement characters (e.g. {example}). "
                "Saving over one will rewrite it as UTF-8."
            )

        # Warn about stem collisions: photo.jpg and photo.png share ONE
        # photo.txt sidecar, so captioning both silently overwrites one
        # caption with the other. Only groups a NEWLY added file participates
        # in are reported — recomputing over everything would re-fire the same
        # warning on every later, unrelated import.
        stems: dict = {}
        for item in self._items.values():
            p = item.image_path
            stems.setdefault(_stem_key(p), []).append(p.name)
        new_stems = {_stem_key(p) for p in new_paths}
        relevant = [
            names for stem, names in stems.items()
            if len(names) > 1 and stem in new_stems
        ]
        if relevant:
            example = " / ".join(sorted(relevant[0]))
            self.stem_collision_detected.emit(
                f"{len(relevant)} image(s) share a name with a different "
                f"extension (e.g. {example}) — they will share ONE .txt "
                "caption file, overwriting each other."
            )

        if new_paths:
            self.images_imported.emit(new_paths)

    def refresh_theme(self):
        """Re-resolve colours set inline, after a runtime theme switch."""
        self._btn_frame.setStyleSheet(
            f"QFrame#browserActions {{ background: {COLORS['surface_translucent']}; "
            f"border-top: 1px solid {COLORS['border']}; }}"
        )
        self._drop_overlay.refresh_theme()

    def clear_all(self):
        """Remove all items from the file browser."""
        for item in self._items.values():
            item.setParent(None)
            item.deleteLater()
        self._items.clear()
        self._current_selection = None
        self.count_label.setText("0")

    def get_all_paths(self) -> List[Path]:
        """Return all image paths in the browser."""
        return [item.image_path for item in self._items.values()]

    def set_item_status(self, path: Path, status: str):
        """Update the status badge for a specific item."""
        key = str(path)
        if key in self._items:
            self._items[key].set_status(status)

    def get_item_status(self, path: Path) -> Optional[str]:
        """Return the status badge of an item, or None if it isn't loaded."""
        item = self._items.get(str(path))
        return item._status if item else None

    def set_item_caption(self, path: Path, caption: str):
        """Set the caption preview for a specific item."""
        key = str(path)
        if key in self._items:
            self._items[key].set_caption_preview(caption)

    def select_item(self, path: Path, emit: bool = True):
        """Programmatically select an item.

        `emit=False` moves the highlight without emitting image_selected —
        used to restore the previous selection when the user cancels out of
        an unsaved-caption prompt, which must not re-enter that prompt.
        """
        self._on_item_clicked(path, emit=emit)

    def _on_item_clicked(self, path: Path, emit: bool = True):
        """Handle thumbnail click — update selection and emit signal."""
        # Deselect previous
        if self._current_selection:
            key = str(self._current_selection)
            if key in self._items:
                self._items[key].set_selected(False)

        # Select new
        key = str(path)
        if key in self._items:
            self._items[key].set_selected(True)

        self._current_selection = path
        if emit:
            self.image_selected.emit(path)

    def _on_import_clicked(self):
        """Open file dialog to import images (remembers the last-used folder)."""
        from gui.config import get_last_import_dir, set_last_import_dir
        ext_filter = "Images (" + " ".join(f"*{ext}" for ext in IMAGE_EXTENSIONS) + ")"
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import Images", get_last_import_dir(), ext_filter,
        )
        if paths:
            set_last_import_dir(str(Path(paths[0]).parent))
            self.add_images([Path(p) for p in paths])

    def _on_import_folder_clicked(self):
        """Open folder dialog to import all images from a directory
        (remembers the last-used folder)."""
        from gui.config import get_last_import_dir, set_last_import_dir
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Image Folder", get_last_import_dir(),
            QFileDialog.Option.ShowDirsOnly,
        )
        if dir_path:
            set_last_import_dir(dir_path)
            self.import_directory(Path(dir_path))

    def _on_clear_clicked(self):
        """Ask the window to clear the workspace.

        The panel deliberately does NOT clear itself here: MainWindow may
        still prompt about an unsaved caption edit and needs the option to
        abandon the clear, so it calls back into clear_all() once it has
        decided.
        """
        if not self._items:
            return
        self.clear_requested.emit()

    def import_directory(self, dir_path: Path):
        """Import all images from a directory."""
        try:
            if not dir_path.is_dir():
                return
            image_paths = scan_directory(dir_path)
        except OSError as e:
            # Unhandled inside a Qt slot, an OSError here (permission denied, a
            # network share dropping mid-scan) aborts the whole PyQt6 process.
            self.import_failed.emit(f"Could not read {dir_path}: {e}")
            return
        self.add_images(image_paths)

    def _filter_items(self, text: str):
        """Filter visible thumbnails based on search text."""
        text_lower = text.lower()
        for item in self._items.values():
            visible = text_lower in item.image_path.name.lower() if text_lower else True
            item.setVisible(visible)
