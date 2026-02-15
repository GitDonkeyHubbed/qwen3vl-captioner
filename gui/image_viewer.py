"""
Image Viewer Panel (Center Top)

Large image preview with canvas toolbar (filename, zoom controls, reset,
maximize) and a processing overlay during caption generation. Matches the
Figma "VL-CAPTIONER Studio Pro" CaptionWorkspace design.
"""

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QSize, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QWheelEvent, QPainter, QColor, QPen, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSizePolicy, QScrollArea,
)

from gui.theme import COLORS


class _SpinnerWidget(QWidget):
    """Animated spinner for the processing overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(48, 48)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._rotate)
        self._timer.start(30)

    def _rotate(self):
        self._angle = (self._angle + 6) % 360
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.translate(24, 24)
        painter.rotate(self._angle)

        # Ring
        pen = QPen(QColor(COLORS["border"]), 2.5)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(-18, -18, 36, 36)

        # Active arc (top portion)
        pen.setColor(QColor(COLORS["accent"]))
        painter.setPen(pen)
        painter.drawArc(-18, -18, 36, 36, 90 * 16, 80 * 16)

        painter.end()

    def stop(self):
        self._timer.stop()

    def start(self):
        self._timer.start(30)


class ProcessingOverlay(QWidget):
    """Semi-transparent overlay with spinner shown during caption generation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Card container
        card = QFrame()
        card.setStyleSheet(
            f"background: {COLORS['bg_dark']}; border: 1px solid {COLORS['border']}; "
            f"border-radius: 12px; padding: 24px;"
        )
        card_layout = QVBoxLayout(card)
        card_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.setSpacing(12)

        self._spinner = _SpinnerWidget()
        spinner_row = QHBoxLayout()
        spinner_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        spinner_row.addWidget(self._spinner)
        card_layout.addLayout(spinner_row)

        title = QLabel("Analyzing Image")
        title.setStyleSheet(
            f"font-size: 14px; font-weight: 500; color: {COLORS['text_primary']};"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(title)

        subtitle = QLabel("Extracting visual features...")
        subtitle.setStyleSheet(
            f"font-size: 12px; color: {COLORS['text_dim']};"
        )
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(subtitle)

        layout.addWidget(card)

    def paintEvent(self, event):
        """Draw the semi-transparent backdrop."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(9, 9, 11, 100))  # zinc-950/40
        painter.end()
        super().paintEvent(event)

    def show_overlay(self):
        self._spinner.start()
        self.setVisible(True)
        self.raise_()

    def hide_overlay(self):
        self._spinner.stop()
        self.setVisible(False)


class ImageViewer(QFrame):
    """
    Center panel image viewer with canvas toolbar and processing overlay.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "image-viewer")
        self._pixmap: Optional[QPixmap] = None
        self._zoom: float = 1.0
        self._image_path: Optional[Path] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Canvas Toolbar ──
        toolbar = QFrame()
        toolbar.setProperty("class", "canvas-toolbar")
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(16, 0, 16, 0)
        tb_layout.setSpacing(8)

        # Filename
        self.filename_label = QLabel("No image loaded")
        self.filename_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 12px; font-weight: 500;"
        )
        tb_layout.addWidget(self.filename_label)

        # Separator
        sep1 = QFrame()
        sep1.setProperty("class", "v-separator")
        sep1.setFixedSize(1, 16)
        tb_layout.addWidget(sep1)

        # Zoom controls
        self.zoom_out_btn = QPushButton("\u2212")  # minus sign
        self.zoom_out_btn.setProperty("class", "icon-button")
        self.zoom_out_btn.setFixedSize(28, 28)
        self.zoom_out_btn.setToolTip("Zoom Out")
        self.zoom_out_btn.clicked.connect(lambda: self._set_zoom(self._zoom - 0.1))
        tb_layout.addWidget(self.zoom_out_btn)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: 11px; font-family: 'Consolas', monospace; "
            f"min-width: 40px;"
        )
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tb_layout.addWidget(self.zoom_label)

        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setProperty("class", "icon-button")
        self.zoom_in_btn.setFixedSize(28, 28)
        self.zoom_in_btn.setToolTip("Zoom In")
        self.zoom_in_btn.clicked.connect(lambda: self._set_zoom(self._zoom + 0.1))
        tb_layout.addWidget(self.zoom_in_btn)

        tb_layout.addStretch()

        # Reset button
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setProperty("class", "secondary-button")
        self.reset_btn.setFixedHeight(28)
        self.reset_btn.clicked.connect(self._reset_view)
        tb_layout.addWidget(self.reset_btn)

        # Maximize button
        self.maximize_btn = QPushButton("\u26F6")  # square with corners
        self.maximize_btn.setProperty("class", "icon-button")
        self.maximize_btn.setFixedSize(28, 28)
        self.maximize_btn.setToolTip("Maximize")
        tb_layout.addWidget(self.maximize_btn)

        layout.addWidget(toolbar)

        # ── Image display area ──
        self._image_container = QWidget()
        self._image_container.setStyleSheet(f"background: {COLORS['bg_darkest']};")
        container_layout = QVBoxLayout(self._image_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setStyleSheet(f"background: {COLORS['bg_darkest']}; border: none;")
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.setWidgetResizable(False)

        self._image_label = QLabel("Select an image from the sidebar to start captioning")
        self._image_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 14px; background: transparent;"
        )
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self._scroll.setWidget(self._image_label)
        container_layout.addWidget(self._scroll, 1)

        layout.addWidget(self._image_container, 1)

        # ── Processing overlay ──
        self._overlay = ProcessingOverlay(self._image_container)

    def set_image(self, image_path: Path):
        """Load and display an image from a file path."""
        self._image_path = image_path
        self._pixmap = QPixmap(str(image_path))

        if self._pixmap.isNull():
            self._image_label.setText(f"Failed to load: {image_path.name}")
            self.filename_label.setText("Error")
            return

        self.filename_label.setText(
            f"{image_path.name}    "
            f"{self._pixmap.width()}\u00d7{self._pixmap.height()}"
        )

        # Fit to view on first load
        self._fit_to_view()

    def clear(self):
        """Clear the current image."""
        self._pixmap = None
        self._image_path = None
        self._image_label.setPixmap(QPixmap())
        self._image_label.setText("Select an image from the sidebar to start captioning")
        self.filename_label.setText("No image loaded")
        self.zoom_label.setText("100%")
        self._zoom = 1.0

    def set_processing(self, is_processing: bool):
        """Show or hide the processing overlay."""
        if is_processing:
            self._overlay.show_overlay()
        else:
            self._overlay.hide_overlay()

    def _fit_to_view(self):
        """Scale image to fit the current view area."""
        if not self._pixmap or self._pixmap.isNull():
            return

        view_size = self._scroll.size()
        img_w = self._pixmap.width()
        img_h = self._pixmap.height()

        if img_w == 0 or img_h == 0:
            return

        # Calculate scale to fit
        scale_w = (view_size.width() - 20) / img_w
        scale_h = (view_size.height() - 20) / img_h
        self._zoom = min(scale_w, scale_h, 1.0)  # Don't upscale beyond 100%

        self._apply_zoom()

    def _set_zoom(self, new_zoom: float):
        """Set a specific zoom level."""
        self._zoom = max(0.1, min(5.0, new_zoom))
        self._apply_zoom()

    def _apply_zoom(self):
        """Apply the current zoom level to the displayed image."""
        if not self._pixmap or self._pixmap.isNull():
            return

        new_w = int(self._pixmap.width() * self._zoom)
        new_h = int(self._pixmap.height() * self._zoom)

        scaled = self._pixmap.scaled(
            new_w, new_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)
        self._image_label.resize(scaled.size())

        self.zoom_label.setText(f"{int(self._zoom * 100)}%")

    def _reset_view(self):
        """Reset to fit-to-view zoom level."""
        self._fit_to_view()

    def resizeEvent(self, event):
        """Re-fit image when the viewer is resized and keep overlay sized."""
        super().resizeEvent(event)
        if self._pixmap and not self._pixmap.isNull():
            self._fit_to_view()
        # Keep overlay filling the image container
        self._overlay.setGeometry(self._image_container.rect())

    def wheelEvent(self, event: QWheelEvent):
        """Zoom with mouse wheel."""
        if self._pixmap and not self._pixmap.isNull():
            delta = event.angleDelta().y()
            zoom_step = 0.05
            if delta > 0:
                self._set_zoom(self._zoom + zoom_step)
            else:
                self._set_zoom(self._zoom - zoom_step)
            event.accept()
        else:
            super().wheelEvent(event)
