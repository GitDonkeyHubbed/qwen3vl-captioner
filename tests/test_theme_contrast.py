"""Light-mode legibility guards for the theme (gui.theme and friends).

Several controls rendered white-on-white in light mode because an
unselectored `background:` on a container cascaded onto its children and beat
the button rules, or because a colour was frozen inline at construction and so
never followed a runtime theme switch.

Contrast is measured with the WCAG 2.x relative-luminance formula. 3.0:1 is
the WCAG AA floor for large text and UI components; body text wants 4.5:1.
"""

import re

import pytest

from gui import theme
from gui.theme import COLORS, get_stylesheet, set_theme


AA_LARGE = 3.0
AA_TEXT = 4.5


def _rgb(color: str) -> tuple[float, float, float]:
    color = color.strip()
    m = re.fullmatch(r"#([0-9a-fA-F]{6})", color)
    if m:
        h = m.group(1)
        return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    m = re.fullmatch(r"rgba?\(([^)]*)\)", color)
    if m:
        parts = [p.strip() for p in m.group(1).split(",")]
        return tuple(int(p) / 255 for p in parts[:3])
    raise ValueError(f"unparsable color: {color!r}")


def _luminance(color: str) -> float:
    def channel(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(c) for c in _rgb(color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def code_only(src: str) -> str:
    """Strip comments so a test can't be satisfied (or broken) by prose."""
    return "\n".join(
        line.split("#", 1)[0] for line in src.splitlines()
    )


def contrast(a: str, b: str) -> float:
    la, lb = _luminance(a), _luminance(b)
    lighter, darker = max(la, lb), min(la, lb)
    return (lighter + 0.05) / (darker + 0.05)


@pytest.fixture
def light():
    set_theme("light")
    yield COLORS
    set_theme("dark")


def test_contrast_helper_is_calibrated():
    assert contrast("#ffffff", "#000000") == pytest.approx(21.0, abs=0.01)
    assert contrast("#ffffff", "#ffffff") == pytest.approx(1.0, abs=0.01)


# ── The two white-on-white buttons ──────────────────────────────────────

def test_accent_button_is_legible_in_light_mode(light):
    # "Open Folder" — white text on the blue accent, not on the panel bg.
    assert contrast("#ffffff", light["accent"]) >= AA_LARGE


def test_primary_button_is_legible_in_light_mode(light):
    # "Load Model" / "Batch Caption All" use the inverted (dark) button.
    assert contrast(light["inverted_text"], light["inverted_bg"]) >= AA_TEXT


def test_container_backgrounds_are_scoped_to_their_own_widget():
    """An unselectored `background:` cascades onto children in Qt QSS.

    That is what made the accent/primary buttons render white-on-white: the
    frame's own background won over the button rules.
    """
    import inspect

    from gui import file_browser, settings_panel

    for module, marker in (
        (file_browser, "browserActions"),
        (settings_panel, "settingsScroll"),
    ):
        src = inspect.getsource(module)
        assert f"#{marker}" in src, f"{module.__name__} lost its scoped selector"


# ── Colours that must follow the palette, not be frozen ─────────────────

def test_canvas_toolbar_uses_a_palette_token(light):
    qss = get_stylesheet("light")
    toolbar = qss.split('QFrame[class="canvas-toolbar"]', 1)[1].split("}", 1)[0]
    assert "rgba(9, 9, 11" not in toolbar, "hardcoded dark strip in light mode"
    assert light["surface_translucent"] in toolbar


def test_dataset_status_colors_come_from_the_palette(light):
    # Qt.GlobalColor.green (#00ff00) is 1.08:1 on the light table background.
    assert contrast("#00ff00", light["bg_card"]) < AA_LARGE
    assert contrast(light["success"], light["bg_card"]) > contrast(
        "#00ff00", light["bg_card"]
    )

    import inspect

    from gui import dataset_panel
    src = code_only(inspect.getsource(dataset_panel))
    assert "GlobalColor.green" not in src
    assert "GlobalColor.red" not in src


def test_panel_title_and_brand_title_have_no_frozen_inline_color():
    import inspect

    from gui import main_window, settings_panel

    settings_src = inspect.getsource(settings_panel)
    assert 'title.setProperty("class", "panel-title")' in settings_src

    qss = get_stylesheet("light")
    assert 'QLabel[class="panel-title"]' in qss

    window_src = code_only(inspect.getsource(main_window))
    brand = window_src.split("brand_title.setProperty", 1)[1].split("addWidget", 1)[0]
    assert "color:" not in brand


def test_notification_dot_colors_resolve_at_paint_time(light):
    from gui.notification_panel import category_color

    # Resolved against the ACTIVE palette, not a snapshot taken at import.
    assert category_color("info") == light["accent_text"]
    set_theme("dark")
    assert category_color("info") == theme._DARK_COLORS["accent_text"]
    set_theme("light")


def test_warning_category_has_its_own_dot():
    from gui.notification_panel import category_color

    # 'warning' had no mapping and fell through to the dim "unknown" dot,
    # despite being emitted for stem collisions and decode failures.
    assert category_color("warning") == COLORS["warning"]
    assert category_color("warning") != category_color("nonexistent-category")


def test_download_icon_is_repainted_for_its_background(light):
    # accent_text (#1d4ed8) on the accent hover fill (#2563eb) is 1.30:1 —
    # the arrow vanished exactly while the user pointed at it.
    assert contrast(light["accent_text"], light["accent"]) < AA_LARGE
    assert contrast("#ffffff", light["accent"]) >= AA_LARGE

    import inspect

    from gui.settings_panel import SettingsPanel
    src = inspect.getsource(SettingsPanel._refresh_download_icon)
    assert "hovered" in src


def test_thumbnail_internals_are_class_styled():
    """Per-item inline colours froze at construction, so thumbnails imported
    after a theme switch mixed the new palette's text with the old
    palette's background."""
    import inspect

    from gui.file_browser import ThumbnailItem
    src = inspect.getsource(ThumbnailItem)
    assert '"thumb-image"' in src
    assert '"thumb-name"' in src
    assert '"thumb-preview"' in src

    qss = get_stylesheet("light")
    for cls in ("thumb-image", "thumb-name", "thumb-preview", "thumb-preview-active"):
        assert f'QLabel[class="{cls}"]' in qss


# ── Palette-wide sanity ─────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize(
    "fg, bg",
    [
        ("text_primary", "bg_dark"),
        ("text_primary", "bg_darkest"),
        ("text_primary", "bg_card"),
        ("text_secondary", "bg_dark"),
        ("accent_text", "bg_darkest"),
    ],
)
def test_core_text_pairs_meet_aa(mode, fg, bg):
    set_theme(mode)
    try:
        assert contrast(COLORS[fg], COLORS[bg]) >= AA_TEXT
    finally:
        set_theme("dark")
