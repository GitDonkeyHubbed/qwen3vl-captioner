"""
Application Settings Dialog for VL-CAPTIONER Studio Pro.

Provides:
  - Dark / Light theme toggle
  - HuggingFace token input (for gated model downloads)
  - Persistent storage via gui.config
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QCheckBox, QFrame, QWidget, QListWidget, QListWidgetItem,
    QFileDialog,
)

from gui.config import load_config, save_config
from gui.theme import COLORS


class AppSettingsDialog(QDialog):
    """Modal settings dialog opened by the gear icon."""

    theme_changed = pyqtSignal(str)   # "dark" or "light"
    paths_changed = pyqtSignal()        # model search paths were modified

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedSize(500, 560)
        self.setStyleSheet(
            f"QDialog {{ background-color: {COLORS['bg_darkest']}; "
            f"border: 1px solid {COLORS['border_light']}; border-radius: 10px; }}"
        )

        self._cfg = load_config()
        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(0)

        # --- Title ---
        title = QLabel("Settings")
        title.setStyleSheet(
            f"color: {COLORS['text_primary']}; font-size: 16px; "
            f"font-weight: 700; background: transparent; margin-bottom: 4px;"
        )
        root.addWidget(title)

        subtitle = QLabel("Configure application preferences")
        subtitle.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: 11px; background: transparent;"
        )
        root.addWidget(subtitle)
        root.addSpacing(16)

        # --- Theme Section ---
        root.addWidget(self._section_header("APPEARANCE"))
        root.addSpacing(6)

        theme_row = QHBoxLayout()
        theme_label = QLabel("Dark Mode")
        theme_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 12px; background: transparent;"
        )
        theme_row.addWidget(theme_label)
        theme_row.addStretch()

        self._dark_mode_cb = QCheckBox()
        self._dark_mode_cb.setChecked(self._cfg.get("theme", "dark") == "dark")
        self._dark_mode_cb.stateChanged.connect(self._on_theme_toggled)
        theme_row.addWidget(self._dark_mode_cb)

        root.addLayout(theme_row)

        theme_note = QLabel("Some changes may require an app restart for full effect.")
        theme_note.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 10px; "
            f"background: transparent; padding-left: 2px;"
        )
        root.addWidget(theme_note)
        root.addSpacing(16)

        # --- Separator ---
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {COLORS['border']};")
        root.addWidget(sep)
        root.addSpacing(16)

        # --- HuggingFace Token Section ---
        root.addWidget(self._section_header("HUGGINGFACE"))
        root.addSpacing(6)

        token_desc = QLabel("API token for downloading gated or private models")
        token_desc.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: 11px; background: transparent;"
        )
        root.addWidget(token_desc)
        root.addSpacing(6)

        token_row = QHBoxLayout()
        token_row.setSpacing(6)

        self._token_input = QLineEdit()
        self._token_input.setPlaceholderText("hf_xxxxxxxxxxxxxxxxxxxxxxxx")
        self._token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._token_input.setText(self._cfg.get("hf_token", ""))
        self._token_input.setStyleSheet(
            f"QLineEdit {{"
            f"  background-color: {COLORS['bg_dark']};"
            f"  color: {COLORS['text_primary']};"
            f"  border: 1px solid {COLORS['border']};"
            f"  border-radius: 6px; padding: 8px;"
            f"  font-family: 'Consolas', monospace; font-size: 12px;"
            f"}}"
            f"QLineEdit:focus {{ border-color: {COLORS['accent']}; }}"
        )
        token_row.addWidget(self._token_input, 1)

        # Show/Hide toggle
        self._show_token_btn = QPushButton("\U0001F441")  # 👁
        self._show_token_btn.setFixedSize(34, 34)
        self._show_token_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._show_token_btn.setToolTip("Show / Hide token")
        self._show_token_btn.setStyleSheet(
            f"QPushButton {{ background: {COLORS['bg_hover']}; border: 1px solid {COLORS['border']}; "
            f"border-radius: 6px; font-size: 14px; }}"
            f"QPushButton:hover {{ background: {COLORS['bg_surface']}; }}"
        )
        self._show_token_btn.clicked.connect(self._toggle_token_visibility)
        token_row.addWidget(self._show_token_btn)

        root.addLayout(token_row)
        root.addSpacing(16)

        # --- Separator ---
        sep2 = QFrame()
        sep2.setFixedHeight(1)
        sep2.setStyleSheet(f"background: {COLORS['border']};")
        root.addWidget(sep2)
        root.addSpacing(16)

        # --- Model Search Paths Section ---
        root.addWidget(self._section_header("MODEL SEARCH PATHS"))
        root.addSpacing(6)

        paths_desc = QLabel("Additional directories to scan for .gguf model files")
        paths_desc.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: 11px; background: transparent;"
        )
        root.addWidget(paths_desc)
        root.addSpacing(6)

        self._paths_list = QListWidget()
        self._paths_list.setStyleSheet(
            f"QListWidget {{"
            f"  background-color: {COLORS['bg_dark']};"
            f"  color: {COLORS['text_primary']};"
            f"  border: 1px solid {COLORS['border']};"
            f"  border-radius: 6px; padding: 4px;"
            f"  font-family: 'Consolas', monospace; font-size: 11px;"
            f"}}"
            f"QListWidget::item {{ padding: 3px 4px; }}"
            f"QListWidget::item:selected {{ background: {COLORS['accent']}; }}"
        )
        self._paths_list.setMinimumHeight(80)
        # Load existing paths
        for p in self._cfg.get("model_search_paths", []):
            self._paths_list.addItem(p)
        root.addWidget(self._paths_list)

        paths_btn_row = QHBoxLayout()
        paths_btn_row.setSpacing(6)

        add_path_btn = QPushButton("+ Add Folder")
        add_path_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_path_btn.setStyleSheet(
            f"QPushButton {{ background: {COLORS['bg_hover']}; color: {COLORS['text_primary']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 6px; padding: 6px 12px; font-size: 11px; }}"
            f"QPushButton:hover {{ background: {COLORS['accent']}; border-color: {COLORS['accent']}; }}"
        )
        add_path_btn.clicked.connect(self._add_search_path)
        paths_btn_row.addWidget(add_path_btn)

        remove_path_btn = QPushButton("− Remove")
        remove_path_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        remove_path_btn.setStyleSheet(
            f"QPushButton {{ background: {COLORS['bg_hover']}; color: {COLORS['text_primary']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 6px; padding: 6px 12px; font-size: 11px; }}"
            f"QPushButton:hover {{ background: {COLORS['error']}; border-color: {COLORS['error']}; }}"
        )
        remove_path_btn.clicked.connect(self._remove_search_path)
        paths_btn_row.addWidget(remove_path_btn)

        paths_btn_row.addStretch()
        root.addLayout(paths_btn_row)

        root.addStretch()

        # --- Footer buttons ---
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setProperty("class", "secondary-button")
        cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        save_btn = QPushButton("  Save")
        save_btn.setProperty("class", "accent-button")
        save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        save_btn.clicked.connect(self._save_and_close)
        btn_row.addWidget(save_btn)

        root.addLayout(btn_row)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _section_header(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-size: 10px; font-weight: 700; "
            f"letter-spacing: 1.5px; background: transparent;"
        )
        return lbl

    def _toggle_token_visibility(self):
        if self._token_input.echoMode() == QLineEdit.EchoMode.Password:
            self._token_input.setEchoMode(QLineEdit.EchoMode.Normal)
            self._show_token_btn.setText("\U0001F576")  # 🕶
        else:
            self._token_input.setEchoMode(QLineEdit.EchoMode.Password)
            self._show_token_btn.setText("\U0001F441")  # 👁

    def _on_theme_toggled(self, state: int):
        """React immediately so the user sees the change in real time."""
        mode = "dark" if self._dark_mode_cb.isChecked() else "light"
        self._cfg["theme"] = mode
        self.theme_changed.emit(mode)

    def _add_search_path(self):
        """Open a folder picker to add a model search directory."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Model Directory", "",
            QFileDialog.Option.ShowDirsOnly,
        )
        if folder:
            # Avoid duplicates
            existing = [self._paths_list.item(i).text() for i in range(self._paths_list.count())]
            if folder not in existing:
                self._paths_list.addItem(folder)

    def _remove_search_path(self):
        """Remove the currently selected search path."""
        row = self._paths_list.currentRow()
        if row >= 0:
            self._paths_list.takeItem(row)

    def _save_and_close(self):
        self._cfg["theme"] = "dark" if self._dark_mode_cb.isChecked() else "light"
        self._cfg["hf_token"] = self._token_input.text().strip()

        # Collect search paths from list widget
        old_paths = self._cfg.get("model_search_paths", [])
        new_paths = [self._paths_list.item(i).text() for i in range(self._paths_list.count())]
        self._cfg["model_search_paths"] = new_paths

        save_config(self._cfg)

        if new_paths != old_paths:
            self.paths_changed.emit()

        self.accept()
