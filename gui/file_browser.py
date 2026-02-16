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

from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal, QSize, QTimer, QMimeData
from PyQt6.QtGui import QPixmap, QIcon, QFont, QPainter, QColor, QPen, QBrush, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QScrollArea, QFrame, QFileDialog, QSizePolicy,
)

from engine.inference import IMAGE_EXTENSIONS
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


class ThumbnailItem(QFrame):
    """A single thumbnail entry in the file list."""

    clicked = pyqtSignal(Path)

    def __init__(self, image_path: Path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self._is_selected = False
        self._caption_preview = ""
        self._status = "idle"  # idle, queued, processing, done

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
        self.thumb_label.setStyleSheet(
            f"border-radius: 4px; background: {COLORS['bg_hover']}; "
            f"border: 1px solid {COLORS['border_light']};"
        )
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
        self.name_label.setStyleSheet(
            f"font-weight: 500; font-size: 12px; color: {COLORS['text_primary']};"
        )
        self.name_label.setWordWrap(False)
        text_layout.addWidget(self.name_label)

        self.preview_label = QLabel("")
        self.preview_label.setStyleSheet(
            f"font-size: 10px; color: {COLORS['text_dim']};"
        )
        self.preview_label.setWordWrap(False)
        text_layout.addWidget(self.preview_label)

        layout.addLayout(text_layout, 1)

    def _load_thumbnail(self):
        """Load and scale the thumbnail image."""
        pixmap = QPixmap(str(self.image_path))
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                THUMB_SIZE, THUMB_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.thumb_label.setPixmap(scaled)
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
        preview = text[:40] + "..." if len(text) > 40 else text
        self.preview_label.setText(preview)
        self.preview_label.setStyleSheet(
            f"font-size: 10px; color: {COLORS['text_dim']};"
        )

    def set_status(self, status: str):
        """Set the status badge (idle, queued, processing, done)."""
        self._status = status

        # Show/hide check overlay
        self._check_overlay.setVisible(status == "done")

        if status == "idle" and self._caption_preview:
            self.preview_label.setText(
                self._caption_preview[:40] + ("..." if len(self._caption_preview) > 40 else "")
            )
            self.preview_label.setStyleSheet(
                f"font-size: 10px; color: {COLORS['text_dim']};"
            )
        elif status == "queued":
            self.preview_label.setText("Queued")
            self.preview_label.setStyleSheet(
                f"font-size: 10px; color: {COLORS['text_dim']};"
            )
        elif status == "processing":
            self.preview_label.setText("Captioning...")
            self.preview_label.setStyleSheet(
                f"font-size: 10px; color: {COLORS['accent_text']};"
            )
        elif status == "done":
            if self._caption_preview:
                self.preview_label.setText(
                    self._caption_preview[:40]
                    + ("..." if len(self._caption_preview) > 40 else "")
                )
                self.preview_label.setStyleSheet(
                    f"font-size: 10px; color: {COLORS['text_dim']};"
                )

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.image_path)
        super().mousePressEvent(event)


class _DropOverlay(QFrame):
    """Semi-transparent overlay shown when dragging files over the panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"background: rgba(59, 130, 246, 0.12); "
            f"border: 2px dashed {COLORS['accent']}; "
            f"border-radius: 8px;"
        )
        self.setVisible(False)
        self._label = QLabel("📂  Drop images here", self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
        self._list_widget.setStyleSheet(f"background: {COLORS['bg_darkest']};")
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setContentsMargins(0, 4, 0, 4)
        self._list_layout.setSpacing(0)
        self._list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        scroll.setWidget(self._list_widget)
        layout.addWidget(scroll, 1)

        # ── Action buttons ──
        btn_frame = QFrame()
        btn_frame.setStyleSheet(
            f"background: rgba(24, 24, 27, 0.5); "
            f"border-top: 1px solid {COLORS['border']};"
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
                path = Path(url.toLocalFile())
                if path.is_dir() or (path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
                    has_valid = True
                    break
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
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_dir():
                # Import all images from the dropped directory
                for f in sorted(path.iterdir()):
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                        image_paths.append(f)
            elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(path)

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
        for p in paths:
            key = str(p)
            if key not in self._items:
                item = ThumbnailItem(p)
                item.clicked.connect(self._on_item_clicked)
                self._list_layout.addWidget(item)
                self._items[key] = item
                new_paths.append(p)

                # Check for existing caption .txt file
                txt_path = p.with_suffix(".txt")
                if txt_path.exists():
                    try:
                        caption = txt_path.read_text(encoding="utf-8").strip()
                        item.set_caption_preview(caption)
                        item.set_status("done")
                    except Exception:
                        pass

        self.count_label.setText(str(len(self._items)))

        if new_paths:
            self.images_imported.emit(new_paths)

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

    def set_item_caption(self, path: Path, caption: str):
        """Set the caption preview for a specific item."""
        key = str(path)
        if key in self._items:
            self._items[key].set_caption_preview(caption)

    def select_item(self, path: Path):
        """Programmatically select an item."""
        self._on_item_clicked(path)

    def _on_item_clicked(self, path: Path):
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
        self.image_selected.emit(path)

    def _on_import_clicked(self):
        """Open file dialog to import images."""
        ext_filter = "Images (" + " ".join(f"*{ext}" for ext in IMAGE_EXTENSIONS) + ")"
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import Images", "", ext_filter,
        )
        if paths:
            self.add_images([Path(p) for p in paths])

    def _on_import_folder_clicked(self):
        """Open folder dialog to import all images from a directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Image Folder", "",
            QFileDialog.Option.ShowDirsOnly,
        )
        if dir_path:
            self.import_directory(Path(dir_path))

    def _on_clear_clicked(self):
        """Clear all loaded images and reset for a new dataset."""
        if not self._items:
            return
        self.clear_all()
        self.clear_requested.emit()

    def import_directory(self, dir_path: Path):
        """Import all images from a directory."""
        if not dir_path.is_dir():
            return

        image_paths = sorted([
            f for f in dir_path.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ])
        self.add_images(image_paths)

    def _filter_items(self, text: str):
        """Filter visible thumbnails based on search text."""
        text_lower = text.lower()
        for key, item in self._items.items():
            visible = text_lower in item.image_path.name.lower() if text_lower else True
            item.setVisible(visible)
