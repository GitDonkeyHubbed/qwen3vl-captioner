"""
Premium theme stylesheet for the VL-CAPTIONER Studio Pro GUI.

Design system: zinc-based dark/light palette, blue-600 accent, emerald status,
rounded corners, smooth hover transitions. Matches the Figma design.

Runtime switching:
  - COLORS is a *mutable* dict that always reflects the active palette.
  - call ``set_theme("dark")`` or ``set_theme("light")`` to swap palettes.
  - call ``get_stylesheet()`` (or ``get_stylesheet("light")``) to regenerate QSS.
"""

from typing import Dict

# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

_DARK_COLORS: Dict[str, str] = {
    # Backgrounds (zinc scale)
    "bg_darkest":     "#09090b",   # zinc-950 — header, sidebars, status bar
    "bg_dark":        "#18181b",   # zinc-900 — main panel bg, cards
    "bg_card":        "#1c1c22",   # slightly lifted card surface
    "bg_input":       "#18181b",   # zinc-900 — input fields
    "bg_hover":       "#27272a",   # zinc-800 — hover states
    "bg_active":      "#1e3a5f",   # blue-tinted active/selected
    "bg_surface":     "#27272a",   # zinc-800 — elevated surfaces

    # Borders
    "border":         "#27272a",   # zinc-800
    "border_light":   "#3f3f46",   # zinc-700
    "border_subtle":  "#27272a",   # same as zinc-800

    # Text
    "text_primary":   "#f4f4f5",   # zinc-100
    "text_secondary": "#a1a1aa",   # zinc-400
    "text_dim":       "#71717a",   # zinc-500
    "text_muted":     "#52525b",   # zinc-600

    # Accent — blue-600
    "accent":         "#2563eb",
    "accent_hover":   "#3b82f6",
    "accent_dim":     "rgba(37, 99, 235, 0.15)",
    "accent_glow":    "rgba(37, 99, 235, 0.25)",
    "accent_text":    "#60a5fa",

    # Status colors
    "success":        "#10b981",
    "success_dim":    "rgba(16, 185, 129, 0.1)",
    "success_border": "rgba(16, 185, 129, 0.2)",
    "warning":        "#f59e0b",
    "error":          "#ef4444",
    "error_dim":      "rgba(239, 68, 68, 0.1)",

    # Badges
    "badge_bg":       "rgba(16, 185, 129, 0.1)",
    "badge_text":     "#10b981",
    "badge_border":   "rgba(16, 185, 129, 0.2)",

    # Inverted (light buttons)
    "inverted_bg":    "#f4f4f5",
    "inverted_text":  "#09090b",
    "inverted_hover": "#ffffff",
}


_LIGHT_COLORS: Dict[str, str] = {
    # Backgrounds (zinc-light scale)
    "bg_darkest":     "#e4e4e7",   # zinc-200 — header, sidebars, status bar
    "bg_dark":        "#f4f4f5",   # zinc-100 — main panel bg, cards
    "bg_card":        "#ffffff",   # white — card surface
    "bg_input":       "#ffffff",   # white — input fields
    "bg_hover":       "#e4e4e7",   # zinc-200 — hover states
    "bg_active":      "#dbeafe",   # blue-100 — active/selected
    "bg_surface":     "#e4e4e7",   # zinc-200 — elevated surfaces

    # Borders
    "border":         "#d4d4d8",   # zinc-300
    "border_light":   "#a1a1aa",   # zinc-400
    "border_subtle":  "#d4d4d8",   # zinc-300

    # Text
    "text_primary":   "#18181b",   # zinc-900
    "text_secondary": "#52525b",   # zinc-600
    "text_dim":       "#71717a",   # zinc-500
    "text_muted":     "#a1a1aa",   # zinc-400

    # Accent — blue-600 (same)
    "accent":         "#2563eb",
    "accent_hover":   "#3b82f6",
    "accent_dim":     "rgba(37, 99, 235, 0.10)",
    "accent_glow":    "rgba(37, 99, 235, 0.15)",
    "accent_text":    "#1d4ed8",   # blue-700 on light bg

    # Status colors (same)
    "success":        "#10b981",
    "success_dim":    "rgba(16, 185, 129, 0.08)",
    "success_border": "rgba(16, 185, 129, 0.15)",
    "warning":        "#f59e0b",
    "error":          "#ef4444",
    "error_dim":      "rgba(239, 68, 68, 0.08)",

    # Badges
    "badge_bg":       "rgba(16, 185, 129, 0.08)",
    "badge_text":     "#059669",   # emerald-600 for light bg
    "badge_border":   "rgba(16, 185, 129, 0.15)",

    # Inverted (dark buttons on light bg)
    "inverted_bg":    "#18181b",
    "inverted_text":  "#f4f4f5",
    "inverted_hover": "#27272a",
}


# ---------------------------------------------------------------------------
# Active palette — mutable dict consumed by the entire app at import time
# ---------------------------------------------------------------------------

COLORS: Dict[str, str] = dict(_DARK_COLORS)


def set_theme(mode: str = "dark"):
    """Swap the active palette in-place. All references to COLORS update."""
    source = _LIGHT_COLORS if mode == "light" else _DARK_COLORS
    COLORS.clear()
    COLORS.update(source)


# ---------------------------------------------------------------------------
# Stylesheet generator
# ---------------------------------------------------------------------------

def get_stylesheet(mode: str = "dark") -> str:
    """Generate the complete application stylesheet for the given mode.

    If *mode* differs from the current COLORS state, set_theme() is called
    first so that any subsequent widget-level ``COLORS[...]`` references
    also match.
    """
    # Ensure COLORS matches the requested mode
    expected_bg = _LIGHT_COLORS["bg_dark"] if mode == "light" else _DARK_COLORS["bg_dark"]
    if COLORS.get("bg_dark") != expected_bg:
        set_theme(mode)

    c = COLORS
    return f"""
    /* === GLOBAL === */
    QMainWindow, QWidget {{
        background-color: {c['bg_dark']};
        color: {c['text_primary']};
        font-family: 'Segoe UI', 'Inter', 'Roboto', sans-serif;
        font-size: 13px;
    }}

    /* === LABELS === */
    QLabel {{
        color: {c['text_primary']};
        background: transparent;
        padding: 0px;
    }}
    QLabel[class="section-header"] {{
        color: {c['text_dim']};
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 8px 0 4px 0;
    }}
    QLabel[class="brand-title"] {{
        color: {c['text_primary']};
        font-size: 13px;
        font-weight: 700;
        letter-spacing: -0.3px;
    }}
    QLabel[class="brand-subtitle"] {{
        color: {c['text_dim']};
        font-size: 10px;
        font-family: 'Consolas', 'Courier New', monospace;
        letter-spacing: -0.5px;
        text-transform: uppercase;
    }}
    QLabel[class="version-label"] {{
        color: {c['text_dim']};
        font-size: 10px;
    }}
    QLabel[class="confidence-badge"] {{
        background-color: {c['badge_bg']};
        color: {c['badge_text']};
        border: 1px solid {c['badge_border']};
        border-radius: 10px;
        padding: 2px 10px;
        font-weight: 600;
        font-size: 10px;
    }}
    QLabel[class="format-badge"] {{
        color: {c['text_dim']};
        font-size: 10px;
        font-weight: 700;
        letter-spacing: -0.3px;
        text-transform: uppercase;
    }}
    QLabel[class="status-dot-green"] {{
        color: {c['success']};
    }}
    QLabel[class="status-dot-yellow"] {{
        color: {c['warning']};
    }}
    QLabel[class="metric-value"] {{
        color: {c['text_primary']};
        font-weight: 600;
    }}
    QLabel[class="metric-label"] {{
        color: {c['success']};
        font-size: 10px;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }}
    QLabel[class="info-label"] {{
        color: {c['accent_text']};
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }}

    /* === BUTTONS === */
    QPushButton {{
        background-color: {c['bg_dark']};
        color: {c['text_primary']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
        min-height: 20px;
    }}
    QPushButton:hover {{
        background-color: {c['bg_hover']};
        border-color: {c['border_light']};
    }}
    QPushButton:pressed {{
        background-color: {c['bg_active']};
    }}
    QPushButton:disabled {{
        background-color: {c['bg_darkest']};
        color: {c['text_muted']};
        border-color: {c['border']};
    }}

    /* Blue accent button (Regenerate, Generate) */
    QPushButton[class="accent-button"] {{
        background-color: {c['accent']};
        color: #ffffff;
        border: none;
        font-weight: 600;
    }}
    QPushButton[class="accent-button"]:hover {{
        background-color: {c['accent_hover']};
    }}
    QPushButton[class="accent-button"]:disabled {{
        background-color: {c['bg_hover']};
        color: {c['text_muted']};
    }}

    /* Inverted button (Batch Caption All, Save Changes) */
    QPushButton[class="primary-button"] {{
        background-color: {c['inverted_bg']};
        color: {c['inverted_text']};
        border: none;
        font-weight: 700;
        padding: 10px 20px;
    }}
    QPushButton[class="primary-button"]:hover {{
        background-color: {c['inverted_hover']};
    }}

    /* Secondary button (Export, Reset) */
    QPushButton[class="secondary-button"] {{
        background-color: {c['bg_dark']};
        color: {c['text_secondary']};
        border: 1px solid {c['border']};
        font-weight: 600;
    }}
    QPushButton[class="secondary-button"]:hover {{
        background-color: {c['bg_hover']};
        color: {c['text_primary']};
        border-color: {c['border_light']};
    }}

    /* Icon-only header button */
    QPushButton[class="icon-button"] {{
        background: transparent;
        border: none;
        border-radius: 6px;
        padding: 6px;
        color: {c['text_dim']};
    }}
    QPushButton[class="icon-button"]:hover {{
        background-color: {c['bg_dark']};
        color: {c['text_primary']};
    }}

    /* Destructive / delete button */
    QPushButton[class="danger-button"] {{
        background: transparent;
        border: none;
        color: {c['text_dim']};
        padding: 4px;
    }}
    QPushButton[class="danger-button"]:hover {{
        background-color: {c['error_dim']};
        color: {c['error']};
    }}

    /* Preset grid button (inactive) */
    QPushButton[class="preset-button"] {{
        background-color: {c['bg_dark']};
        color: {c['text_dim']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 8px 4px;
        font-size: 10px;
        font-weight: 500;
    }}
    QPushButton[class="preset-button"]:hover {{
        border-color: {c['border_light']};
        color: {c['text_secondary']};
    }}

    /* Preset grid button (active) */
    QPushButton[class="preset-button-active"] {{
        background-color: {c['accent_dim']};
        color: {c['accent_text']};
        border: 1px solid {c['accent']};
        border-radius: 6px;
        padding: 8px 4px;
        font-size: 10px;
        font-weight: 500;
    }}

    /* === INPUT FIELDS === */
    QLineEdit, QTextEdit, QPlainTextEdit {{
        background-color: {c['bg_dark']};
        color: {c['text_primary']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 8px;
        selection-background-color: {c['accent_dim']};
        selection-color: {c['text_primary']};
    }}
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
        border-color: {c['accent']};
    }}
    QLineEdit::placeholder {{
        color: {c['text_muted']};
    }}

    /* Monospace caption editor */
    QTextEdit[class="caption-editor"] {{
        background-color: rgba(24, 24, 27, 0.5);
        border: 1px solid {c['border']};
        border-radius: 8px;
        padding: 12px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 13px;
        line-height: 1.6;
    }}
    QTextEdit[class="caption-editor"]:focus {{
        border-color: rgba(37, 99, 235, 0.5);
    }}

    /* Prompt template textarea */
    QTextEdit[class="prompt-template"] {{
        background-color: {c['bg_dark']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 8px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 11px;
    }}

    /* === COMBOBOX === */
    QComboBox {{
        background-color: {c['bg_dark']};
        color: {c['text_primary']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 8px 12px;
        font-weight: 500;
        font-size: 12px;
        min-height: 20px;
    }}
    QComboBox:hover {{
        border-color: {c['border_light']};
    }}
    QComboBox:focus {{
        border-color: {c['accent']};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}
    QComboBox::down-arrow {{
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 6px solid {c['text_secondary']};
        margin-right: 8px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {c['bg_dark']};
        color: {c['text_primary']};
        border: 1px solid {c['border']};
        selection-background-color: {c['bg_active']};
        selection-color: {c['text_primary']};
        outline: none;
        padding: 4px;
    }}

    /* === CHECKBOXES === */
    QCheckBox {{
        color: {c['text_dim']};
        font-size: 10px;
        spacing: 8px;
        padding: 4px 6px;
        border-radius: 4px;
    }}
    QCheckBox:hover {{
        color: {c['text_secondary']};
        background-color: rgba(39, 39, 42, 0.5);
    }}
    QCheckBox::indicator {{
        width: 14px;
        height: 14px;
        border-radius: 3px;
        border: 1px solid {c['border_light']};
        background-color: {c['bg_hover']};
    }}
    QCheckBox::indicator:checked {{
        background-color: {c['accent']};
        border-color: {c['accent']};
    }}

    /* === SLIDERS === */
    QSlider::groove:horizontal {{
        border: none;
        height: 4px;
        background: {c['bg_hover']};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {c['accent']};
        border: none;
        width: 14px;
        height: 14px;
        margin: -5px 0;
        border-radius: 7px;
    }}
    QSlider::handle:horizontal:hover {{
        background: {c['accent_hover']};
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }}
    QSlider::sub-page:horizontal {{
        background: {c['accent']};
        border-radius: 2px;
    }}

    /* === SCROLL AREAS === */
    QScrollArea {{
        border: none;
        background: transparent;
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 6px;
        margin: 0;
        border: none;
    }}
    QScrollBar::handle:vertical {{
        background: {c['border']};
        border-radius: 3px;
        min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {c['border_light']};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: transparent;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 6px;
        margin: 0;
        border: none;
    }}
    QScrollBar::handle:horizontal {{
        background: {c['border']};
        border-radius: 3px;
        min-width: 30px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {c['border_light']};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
    }}
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
        background: transparent;
    }}

    /* === SPLITTER === */
    QSplitter::handle {{
        background-color: {c['border']};
        width: 1px;
    }}

    /* === GROUP BOX === */
    QGroupBox {{
        border: 1px solid {c['border']};
        border-radius: 8px;
        margin-top: 16px;
        padding: 12px;
        background-color: transparent;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 8px;
        color: {c['text_dim']};
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
    }}

    /* === STATUS BAR === */
    QStatusBar {{
        background-color: {c['bg_darkest']};
        color: {c['text_dim']};
        border-top: 1px solid {c['border']};
        font-size: 10px;
        padding: 0px 8px;
        max-height: 24px;
    }}

    /* === PROGRESS BAR === */
    QProgressBar {{
        border: none;
        border-radius: 3px;
        text-align: center;
        color: {c['text_primary']};
        background-color: {c['bg_hover']};
        max-height: 6px;
    }}
    QProgressBar::chunk {{
        background-color: {c['accent']};
        border-radius: 3px;
    }}

    /* === FRAMES (panels) === */
    QFrame[class="sidebar-panel"] {{
        background-color: {c['bg_darkest']};
        border-right: 1px solid {c['border']};
    }}
    QFrame[class="settings-panel"] {{
        background-color: {c['bg_darkest']};
        border-left: 1px solid {c['border']};
    }}
    QFrame[class="nav-bar"] {{
        background-color: {c['bg_darkest']};
        border-bottom: 1px solid {c['border']};
        min-height: 48px;
        max-height: 56px;
    }}
    QFrame[class="canvas-toolbar"] {{
        background-color: rgba(9, 9, 11, 0.5);
        border-bottom: 1px solid {c['border']};
        min-height: 40px;
        max-height: 48px;
    }}
    QFrame[class="image-viewer"] {{
        background-color: {c['bg_darkest']};
    }}
    QFrame[class="caption-area"] {{
        background-color: {c['bg_darkest']};
        border-top: 1px solid {c['border']};
    }}
    QFrame[class="model-status"] {{
        background-color: rgba(37, 99, 235, 0.05);
        border: 1px solid rgba(37, 99, 235, 0.2);
        border-radius: 8px;
        padding: 12px;
    }}
    QFrame[class="extra-options-container"] {{
        background-color: rgba(24, 24, 27, 0.5);
        border: 1px solid {c['border']};
        border-radius: 6px;
    }}
    QFrame[class="gpu-pill"] {{
        background-color: rgba(24, 24, 27, 0.5);
        border: 1px solid {c['border']};
        border-radius: 14px;
        padding: 4px 12px;
    }}

    /* Thumbnail items */
    QFrame[class="thumbnail-item"] {{
        background-color: transparent;
        border-left: 2px solid transparent;
        padding: 4px;
    }}
    QFrame[class="thumbnail-item"]:hover {{
        background-color: {c['bg_dark']};
    }}
    QFrame[class="thumbnail-selected"] {{
        background-color: {c['accent_dim']};
        border-left: 2px solid {c['accent']};
        padding: 4px;
    }}

    /* === TAB-LIKE NAV BUTTONS === */
    QPushButton[class="nav-tab"] {{
        background: transparent;
        border: none;
        border-radius: 6px;
        color: {c['text_secondary']};
        padding: 4px 12px;
        font-weight: 500;
        font-size: 12px;
    }}
    QPushButton[class="nav-tab"]:hover {{
        color: {c['text_primary']};
        background: transparent;
    }}
    QPushButton[class="nav-tab-active"] {{
        background: {c['bg_dark']};
        border: none;
        border-radius: 6px;
        color: {c['text_primary']};
        padding: 4px 12px;
        font-weight: 500;
        font-size: 12px;
    }}

    /* === TOOLTIP === */
    QToolTip {{
        background-color: {c['bg_dark']};
        color: {c['text_primary']};
        border: 1px solid {c['border']};
        border-radius: 4px;
        padding: 6px 10px;
        font-size: 12px;
    }}

    /* === SEPARATOR LINES === */
    QFrame[class="h-separator"] {{
        background-color: {c['border']};
        max-height: 1px;
        min-height: 1px;
    }}
    QFrame[class="v-separator"] {{
        background-color: {c['border']};
        max-width: 1px;
        min-width: 1px;
    }}
    """
