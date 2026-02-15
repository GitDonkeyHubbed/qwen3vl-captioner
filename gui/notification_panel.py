"""
Notification system for VL-CAPTIONER Studio Pro.

Provides:
  - NotificationEntry: dataclass for individual notifications
  - NotificationStore: QObject that holds up to 50 entries and emits signals
  - NotificationPanel: Popup QFrame dropdown anchored below the bell button
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List

from PyQt6.QtCore import Qt, pyqtSignal, QObject, QPoint
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QWidget, QSizePolicy, QGraphicsDropShadowEffect,
)

from gui.theme import COLORS


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class NotificationEntry:
    """A single notification record."""
    timestamp: datetime
    message: str
    category: str = "info"      # info | error | success | download
    is_read: bool = False


# Category -> dot color
_CATEGORY_COLORS = {
    "info":     COLORS["accent_text"],
    "error":    COLORS["error"],
    "success":  COLORS["success"],
    "download": COLORS["warning"],
}


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class NotificationStore(QObject):
    """Holds the last 50 notifications and emits when one is added."""

    notification_added = pyqtSignal()        # badge / panel can react

    MAX_ENTRIES = 50

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries: List[NotificationEntry] = []

    # -- public API --

    def add(self, message: str, category: str = "info"):
        entry = NotificationEntry(
            timestamp=datetime.now(),
            message=message,
            category=category,
        )
        self._entries.insert(0, entry)      # newest first
        if len(self._entries) > self.MAX_ENTRIES:
            self._entries = self._entries[:self.MAX_ENTRIES]
        self.notification_added.emit()

    def entries(self) -> List[NotificationEntry]:
        return list(self._entries)

    def unread_count(self) -> int:
        return sum(1 for e in self._entries if not e.is_read)

    def mark_all_read(self):
        for e in self._entries:
            e.is_read = True

    def clear(self):
        self._entries.clear()
        self.notification_added.emit()


# ---------------------------------------------------------------------------
# Panel (dropdown popup)
# ---------------------------------------------------------------------------

class NotificationPanel(QFrame):
    """
    Popup panel that appears below the bell icon.
    Shows timestamped notifications with category-colored dots.
    """

    def __init__(self, store: NotificationStore, parent=None):
        super().__init__(parent)
        self._store = store
        self.setWindowFlags(Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setFixedWidth(360)
        self.setMaximumHeight(420)
        self._build_ui()

    # -- layout --

    def _build_ui(self):
        self.setStyleSheet(
            f"NotificationPanel {{"
            f"  background-color: {COLORS['bg_darkest']};"
            f"  border: 1px solid {COLORS['border_light']};"
            f"  border-radius: 10px;"
            f"}}"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- Header ---
        header = QFrame()
        header.setStyleSheet(
            f"background: transparent; border-bottom: 1px solid {COLORS['border']};"
        )
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 12, 12, 10)

        title = QLabel("Notifications")
        title.setStyleSheet(
            f"color: {COLORS['text_primary']}; font-size: 13px; "
            f"font-weight: 700; background: transparent; border: none;"
        )
        header_layout.addWidget(title)
        header_layout.addStretch()

        clear_btn = QPushButton("Clear All")
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.setStyleSheet(
            f"QPushButton {{"
            f"  color: {COLORS['text_dim']}; background: transparent;"
            f"  border: none; font-size: 11px; font-weight: 500; padding: 2px 6px;"
            f"}}"
            f"QPushButton:hover {{ color: {COLORS['text_secondary']}; }}"
        )
        clear_btn.clicked.connect(self._on_clear)
        header_layout.addWidget(clear_btn)

        root.addWidget(header)

        # --- Scrollable content ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self._content_widget = QWidget()
        self._content_widget.setStyleSheet("background: transparent;")
        self._content_layout = QVBoxLayout(self._content_widget)
        self._content_layout.setContentsMargins(8, 4, 8, 8)
        self._content_layout.setSpacing(0)
        self._content_layout.addStretch()

        scroll.setWidget(self._content_widget)
        root.addWidget(scroll)

        # Drop-shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 4)
        shadow.setColor(COLORS["bg_darkest"] if isinstance(COLORS["bg_darkest"], type(shadow.color())) else shadow.color())
        try:
            from PyQt6.QtGui import QColor as _QC
            shadow.setColor(_QC(0, 0, 0, 100))
        except Exception:
            pass
        self.setGraphicsEffect(shadow)

    # -- public --

    def show_below(self, anchor: QWidget):
        """Position the panel just below *anchor* and show it."""
        self._refresh()
        self._store.mark_all_read()

        # Position: below the anchor widget, right-aligned
        global_pos = anchor.mapToGlobal(QPoint(0, anchor.height() + 4))
        # Right-align the panel to the button
        x = global_pos.x() + anchor.width() - self.width()
        self.move(x, global_pos.y())
        self.show()

    # -- internals --

    def _refresh(self):
        """Rebuild the notification items from the store."""
        # Clear existing items (keep the stretch at the end)
        while self._content_layout.count() > 1:
            item = self._content_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        entries = self._store.entries()
        if not entries:
            empty = QLabel("No notifications yet")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setStyleSheet(
                f"color: {COLORS['text_muted']}; font-size: 12px; "
                f"padding: 24px; background: transparent;"
            )
            self._content_layout.insertWidget(0, empty)
            return

        for idx, entry in enumerate(entries):
            row = self._make_row(entry)
            self._content_layout.insertWidget(idx, row)

    def _make_row(self, entry: NotificationEntry) -> QFrame:
        """Build a single notification row widget."""
        row = QFrame()
        row.setStyleSheet(
            f"QFrame {{ background: transparent; border-bottom: 1px solid {COLORS['border']}; }}"
        )
        layout = QHBoxLayout(row)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # Category dot
        dot_color = _CATEGORY_COLORS.get(entry.category, COLORS["text_dim"])
        dot = QLabel("\u2022")
        dot.setFixedWidth(12)
        dot.setStyleSheet(
            f"color: {dot_color}; font-size: 18px; background: transparent; border: none;"
        )
        dot.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(dot)

        # Message + timestamp column
        col = QVBoxLayout()
        col.setSpacing(2)

        msg = QLabel(entry.message)
        msg.setWordWrap(True)
        msg.setStyleSheet(
            f"color: {COLORS['text_primary']}; font-size: 12px; "
            f"background: transparent; border: none;"
        )
        col.addWidget(msg)

        ts = QLabel(entry.timestamp.strftime("%I:%M %p"))
        ts.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 10px; "
            f"background: transparent; border: none;"
        )
        col.addWidget(ts)

        layout.addLayout(col, 1)
        return row

    def _on_clear(self):
        self._store.clear()
        self._refresh()
