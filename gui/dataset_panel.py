"""
Dataset Panel — VL-CAPTIONER Studio Pro

Shows a table of imported images with metadata:
  filename, dimensions, file size, caption file status, caption preview.
Includes summary stats and a Refresh button.
"""

from pathlib import Path
from typing import List

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
)

from .theme import COLORS


class DatasetPanel(QFrame):
    """Dataset verification view: table of images with caption status."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {COLORS['bg_dark']};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # --- Header Row ---
        header_row = QHBoxLayout()
        title = QLabel("Dataset Overview")
        title.setStyleSheet(
            f"color: {COLORS['text_primary']}; font-size: 16px; font-weight: 600;"
        )
        header_row.addWidget(title)
        header_row.addStretch()

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setProperty("class", "secondary-button")
        self._refresh_btn.setFixedHeight(30)
        self._refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        header_row.addWidget(self._refresh_btn)
        layout.addLayout(header_row)

        # --- Stats Row ---
        self._stats_frame = QFrame()
        self._stats_frame.setStyleSheet(
            f"background: {COLORS['bg_card']}; border: 1px solid {COLORS['border']}; border-radius: 6px;"
        )
        stats_layout = QHBoxLayout(self._stats_frame)
        stats_layout.setContentsMargins(16, 10, 16, 10)
        stats_layout.setSpacing(24)

        self._stat_total = self._make_stat("Total Images", "0")
        self._stat_captioned = self._make_stat("Captioned", "0")
        self._stat_uncaptioned = self._make_stat("Uncaptioned", "0")
        self._stat_coverage = self._make_stat("Coverage", "0%")

        stats_layout.addWidget(self._stat_total)
        stats_layout.addWidget(self._stat_captioned)
        stats_layout.addWidget(self._stat_uncaptioned)
        stats_layout.addWidget(self._stat_coverage)
        stats_layout.addStretch()
        layout.addWidget(self._stats_frame)

        # --- Table ---
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels([
            "Filename", "Dimensions", "Size", "Caption", "Preview"
        ])
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)

        header = self._table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)

        self._table.setStyleSheet(f"""
            QTableWidget {{
                background: {COLORS['bg_darkest']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                gridline-color: {COLORS['border']};
                font-size: 11px;
            }}
            QTableWidget::item {{
                padding: 6px 8px;
            }}
            QTableWidget::item:selected {{
                background: {COLORS['bg_active']};
            }}
            QHeaderView::section {{
                background: {COLORS['bg_surface']};
                color: {COLORS['text_secondary']};
                font-size: 10px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                padding: 6px 8px;
                border: none;
                border-bottom: 1px solid {COLORS['border']};
            }}
            QTableWidget::item:alternate {{
                background: {COLORS['bg_card']};
            }}
        """)

        layout.addWidget(self._table, 1)

        # --- Empty state ---
        self._empty_label = QLabel("Import images in the Project tab to see your dataset here.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: 12px; padding: 40px;"
        )
        self._empty_label.setVisible(False)
        layout.addWidget(self._empty_label)

    def set_refresh_callback(self, callback):
        """Connect the Refresh button to a callback."""
        self._refresh_btn.clicked.connect(callback)

    def populate(self, image_paths: List[Path]):
        """Populate the table with image metadata and caption status."""
        self._table.setRowCount(0)

        if not image_paths:
            self._table.setVisible(False)
            self._empty_label.setVisible(True)
            self._update_stats(0, 0)
            return

        self._table.setVisible(True)
        self._empty_label.setVisible(False)
        self._table.setRowCount(len(image_paths))

        captioned = 0
        for row, img_path in enumerate(sorted(image_paths, key=lambda p: p.name.lower())):
            # Filename
            self._table.setItem(row, 0, QTableWidgetItem(img_path.name))

            # Dimensions
            dims = self._get_image_dims(img_path)
            self._table.setItem(row, 1, QTableWidgetItem(dims))

            # File size
            size_str = self._format_size(img_path)
            self._table.setItem(row, 2, QTableWidgetItem(size_str))

            # Caption file exists?
            txt_path = img_path.with_suffix(".txt")
            has_caption = txt_path.exists()
            if has_caption:
                captioned += 1
            status_item = QTableWidgetItem("Yes" if has_caption else "No")
            status_item.setForeground(
                QPixmap(1, 1).toImage().pixelColor(0, 0)  # placeholder — set via stylesheet
            )
            if has_caption:
                status_item.setForeground(Qt.GlobalColor.green)
            else:
                status_item.setForeground(Qt.GlobalColor.red)
            self._table.setItem(row, 3, status_item)

            # Caption preview
            preview = ""
            if has_caption:
                try:
                    text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
                    preview = text[:120] + ("..." if len(text) > 120 else "")
                except Exception:
                    preview = "(read error)"
            self._table.setItem(row, 4, QTableWidgetItem(preview))

        self._update_stats(len(image_paths), captioned)

    def _update_stats(self, total: int, captioned: int):
        uncaptioned = total - captioned
        pct = int(captioned / total * 100) if total > 0 else 0

        self._stat_total.findChild(QLabel, "stat_value").setText(str(total))
        self._stat_captioned.findChild(QLabel, "stat_value").setText(str(captioned))
        self._stat_uncaptioned.findChild(QLabel, "stat_value").setText(str(uncaptioned))
        self._stat_coverage.findChild(QLabel, "stat_value").setText(f"{pct}%")

    def _make_stat(self, label: str, value: str) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background: transparent;")
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(2)

        lbl = QLabel(label)
        lbl.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: 9px; font-weight: 600; "
            f"text-transform: uppercase; letter-spacing: 0.5px;"
        )
        vl.addWidget(lbl)

        val = QLabel(value)
        val.setObjectName("stat_value")
        val.setStyleSheet(
            f"color: {COLORS['text_primary']}; font-size: 18px; font-weight: 700; "
            f"font-family: 'Consolas', monospace;"
        )
        vl.addWidget(val)
        return w

    @staticmethod
    def _get_image_dims(path: Path) -> str:
        try:
            px = QPixmap(str(path))
            if not px.isNull():
                return f"{px.width()} x {px.height()}"
        except Exception:
            pass
        return "--"

    @staticmethod
    def _format_size(path: Path) -> str:
        try:
            size = path.stat().st_size
            if size < 1024:
                return f"{size} B"
            elif size < 1024 * 1024:
                return f"{size / 1024:.1f} KB"
            else:
                return f"{size / (1024 * 1024):.1f} MB"
        except Exception:
            return "--"
