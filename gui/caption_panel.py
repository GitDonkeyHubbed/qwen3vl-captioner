"""
Caption Panel (Center Bottom)

Displays the generated caption with:
  - Confidence pill badge (emerald)
  - Copy button with animated checkmark feedback (2s)
  - Regenerate button (blue-600 with glow shadow)
  - Monospace caption textarea
  - Bottom bar: blue dot + "SDXL Format" badge, char/word count
  - Trash/delete button (red hover), Save Changes (inverted light style)

Matches the Figma "VL-CAPTIONER Studio Pro" CaptionWorkspace design.
"""

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QFrame, QApplication, QSizePolicy,
)

from gui.theme import COLORS


class CaptionPanel(QFrame):
    """
    Caption display and editing panel below the image viewer.
    Shows generated caption with controls for copy, regenerate, and save.
    """

    regenerate_requested = pyqtSignal()
    save_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "caption-area")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        # ── Header row: label + confidence + copy + regenerate ──
        header_row = QHBoxLayout()
        header_row.setSpacing(10)

        title_label = QLabel("VISUAL DESCRIPTION / CAPTION")
        title_label.setProperty("class", "section-header")
        header_row.addWidget(title_label)

        self.confidence_badge = QLabel("")
        self.confidence_badge.setProperty("class", "confidence-badge")
        self.confidence_badge.setVisible(False)
        header_row.addWidget(self.confidence_badge)

        header_row.addStretch()

        # Copy button with animated feedback
        self.copy_btn = QPushButton("\U0001F4CB")  # clipboard
        self.copy_btn.setProperty("class", "icon-button")
        self.copy_btn.setFixedSize(32, 28)
        self.copy_btn.setToolTip("Copy caption to clipboard")
        self.copy_btn.clicked.connect(self._copy_caption)
        header_row.addWidget(self.copy_btn)

        # Regenerate button (accent blue with glow)
        self.regenerate_btn = QPushButton("\u2728 Regenerate Caption")
        self.regenerate_btn.setProperty("class", "accent-button")
        self.regenerate_btn.setStyleSheet(
            f"QPushButton {{ "
            f"background-color: {COLORS['accent']}; color: #ffffff; border: none; "
            f"border-radius: 6px; padding: 6px 16px; font-weight: 600; font-size: 12px; "
            f"}} "
            f"QPushButton:hover {{ background-color: {COLORS['accent_hover']}; }} "
            f"QPushButton:disabled {{ background-color: {COLORS['bg_hover']}; color: {COLORS['text_muted']}; }}"
        )
        self.regenerate_btn.clicked.connect(self.regenerate_requested.emit)
        header_row.addWidget(self.regenerate_btn)

        layout.addLayout(header_row)

        # ── Caption text area (monospace) ──
        self.caption_text = QTextEdit()
        self.caption_text.setPlaceholderText(
            "AI will generate a caption here..."
        )
        self.caption_text.setProperty("class", "caption-editor")
        self.caption_text.setMinimumHeight(80)
        self.caption_text.setMaximumHeight(180)
        self.caption_text.setAcceptRichText(False)
        self.caption_text.textChanged.connect(self._update_counts)
        layout.addWidget(self.caption_text)

        # ── Bottom row: format badge + counts | trash + save ──
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(16)

        # Format badge (blue dot + label)
        format_row = QHBoxLayout()
        format_row.setSpacing(6)

        format_dot = QLabel("\u25CF")  # filled circle
        format_dot.setStyleSheet(
            f"color: {COLORS['accent']}; font-size: 8px;"
        )
        format_row.addWidget(format_dot)

        self.format_badge = QLabel("SDXL FORMAT")
        self.format_badge.setProperty("class", "format-badge")
        format_row.addWidget(self.format_badge)

        bottom_row.addLayout(format_row)

        # Character and word count
        self.count_label = QLabel("0 characters \u2022 0 words")
        self.count_label.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: 10px;"
        )
        bottom_row.addWidget(self.count_label)

        bottom_row.addStretch()

        # Feedback label (for copy/save confirmations)
        self.feedback_label = QLabel("")
        self.feedback_label.setStyleSheet(
            f"color: {COLORS['success']}; font-size: 12px;"
        )
        bottom_row.addWidget(self.feedback_label)

        # Trash / delete button
        self.delete_btn = QPushButton("\U0001F5D1")  # wastebasket
        self.delete_btn.setProperty("class", "danger-button")
        self.delete_btn.setFixedSize(32, 28)
        self.delete_btn.setToolTip("Clear caption")
        self.delete_btn.clicked.connect(self.clear_caption)
        bottom_row.addWidget(self.delete_btn)

        # Save button (inverted: light bg, dark text)
        self.save_btn = QPushButton("Save Changes")
        self.save_btn.setProperty("class", "primary-button")
        self.save_btn.clicked.connect(self.save_requested.emit)
        bottom_row.addWidget(self.save_btn)

        layout.addLayout(bottom_row)

    # ─── Public API ───

    def set_caption(self, text: str):
        """Set the caption text (replaces current content)."""
        self.caption_text.setPlainText(text)
        self.feedback_label.setText("")

    def append_token(self, token: str):
        """Append a streaming token to the caption."""
        cursor = self.caption_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(token)
        self.caption_text.setTextCursor(cursor)
        self.caption_text.ensureCursorVisible()

    def clear_caption(self):
        """Clear the caption text."""
        self.caption_text.clear()
        self.confidence_badge.setVisible(False)
        self.feedback_label.setText("")

    def get_caption(self) -> str:
        """Get the current caption text."""
        return self.caption_text.toPlainText().strip()

    def set_confidence(self, score: float):
        """Display a confidence score badge (0-100)."""
        self.confidence_badge.setText(f"{score:.0f}% Confidence")
        self.confidence_badge.setVisible(True)

    def show_feedback(self, message: str, is_success: bool = True):
        """Show feedback message (e.g., 'Saved!' or 'Error')."""
        color = COLORS["success"] if is_success else COLORS["error"]
        icon = "\u2713" if is_success else "\u2717"
        self.feedback_label.setStyleSheet(f"color: {color}; font-size: 12px;")
        self.feedback_label.setText(f"{icon} {message}")

    def set_generating(self, is_generating: bool):
        """Update UI state during generation."""
        self.regenerate_btn.setEnabled(not is_generating)
        self.regenerate_btn.setText(
            "\u23f3 Generating..." if is_generating else "\u2728 Regenerate Caption"
        )

    def set_format_label(self, text: str):
        """Update the format badge text (e.g., when preset changes)."""
        self.format_badge.setText(text.upper() + " FORMAT")

    # ─── Private ───

    def _copy_caption(self):
        """Copy caption text to clipboard with animated feedback."""
        text = self.get_caption()
        if text:
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            # Swap icon to checkmark for 2 seconds
            original_text = self.copy_btn.text()
            self.copy_btn.setText("\u2714")  # heavy checkmark
            self.copy_btn.setStyleSheet(
                f"color: {COLORS['success']}; background: transparent; "
                f"border: none; font-size: 16px; padding: 4px;"
            )
            self.show_feedback("Copied!")

            def _restore():
                self.copy_btn.setText(original_text)
                self.copy_btn.setStyleSheet("")
                self.copy_btn.setProperty("class", "icon-button")
                self.copy_btn.style().unpolish(self.copy_btn)
                self.copy_btn.style().polish(self.copy_btn)
                self.feedback_label.setText("")

            QTimer.singleShot(2000, _restore)

    def _update_counts(self):
        """Update character and word counts."""
        text = self.caption_text.toPlainText()
        chars = len(text)
        words = len(text.split()) if text.strip() else 0
        self.count_label.setText(f"{chars} characters \u2022 {words} words")
