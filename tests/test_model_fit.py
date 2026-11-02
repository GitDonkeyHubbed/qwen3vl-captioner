"""Model-dropdown fit hints and the busy state they share a panel with.

The dropdown is what a user trusts before a multi-gigabyte download. The
weights-plus-overhead rules decide three different outcomes: a red "won't
fit", an orange "tight fit", and the quant that is selected on its own.
A one-sided comparison, or treating the f16 build as just another quant,
picks a model that cannot run. Rebuilding the list (the window does this
whenever it rescans disk) must not throw away the model the user picked.

The same panel keeps Load, Download, and Batch Caption disabled for the
whole batch, including the last item and the gap between images. Enabling
them there starts a second run while the first is still in flight.
"""

import math
import sys

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtGui import QColor  # noqa: E402
from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from gui.model_download_manager import get_model_info, mlx_backend_supported  # noqa: E402
from gui.settings_panel import SettingsPanel  # noqa: E402
from gui.theme import COLORS  # noqa: E402


# The dropdown's own rules, written out so a change to either threshold
# fails here. Overhead is context and runtime on top of the weights; the
# red warning starts a gigabyte sooner than that, while the weights are
# still loading.
_OVERHEAD_GB = 2.0
_WONT_FIT_SLACK_GB = 1.0

_Q4 = "Qwen3-VL 8B ABL v2 — Q4_K_M (4.68 GB)"
_Q4_S = "Qwen3-VL 8B ABL v2 — Q4_K_S (4.47 GB)"
_Q2 = "Qwen3-VL 8B ABL v2 — Q2_K (3.06 GB)"
_Q6 = "Qwen3-VL 8B ABL v2 — Q6_K (6.26 GB)"
_Q8 = "Qwen3-VL 8B ABL v2 — Q8_0 (8.11 GB)"
_F16 = "Qwen3-VL 8B ABL v2 — f16 (15.26 GB)"
_MLX8 = "Qwen3-VL 8B ABL v2 MLX — 8bit (9.2 GB)"


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


def _size(name: str) -> float:
    info = get_model_info(name)
    assert info is not None, name
    return info["size_gb"]


def _index(panel: SettingsPanel, name: str) -> int:
    for i in range(panel.model_combo.count()):
        if panel.model_combo.itemData(i) == ("registry", name):
            return i
    raise AssertionError(f"{name} is not in the dropdown")


def _foreground(panel: SettingsPanel, index: int):
    brush = panel.model_combo.itemData(index, Qt.ItemDataRole.ForegroundRole)
    if brush is None or not brush.color().isValid():
        return None
    return brush.color().name()


def _tooltip(panel: SettingsPanel, index: int) -> str:
    return panel.model_combo.itemData(index, Qt.ItemDataRole.ToolTipRole) or ""


def _color(key: str) -> str:
    return QColor(COLORS[key]).name()


def _fresh(panel: SettingsPanel):
    """Drop the row __init__ already chose so auto-select can run.

    populate_models restores whatever is current. A panel starts on the
    Q6_K fallback, and a later call would keep that row even when the
    budget says something else.
    """
    panel.model_combo.setCurrentIndex(-1)


def test_fit_warning_changes_at_one_gigabyte_over_the_weights(qapp):
    """Just under size+1 GB is red; the exact boundary is only tight.

    `vram < size + 1` is the won't-fit gate. `<=` would paint the boundary
    red, and dropping the slack would paint a model that still loads.
    """
    size = _size(_Q4)
    needs = f"needs ~{size + _OVERHEAD_GB:.0f} GB"

    panel = SettingsPanel()
    panel.populate_models(
        [], set(), vram_gb=math.nextafter(size + _WONT_FIT_SLACK_GB, 0.0),
        memory_is_shared=False,
    )
    idx = _index(panel, _Q4)
    assert _foreground(panel, idx) == _color("error")
    tip = _tooltip(panel, idx)
    assert tip.startswith("Won't fit:")
    assert needs in tip
    assert "VRAM" in tip and "your GPU has" in tip
    # A smaller quant in the same list still fits, so the warning is per model.
    assert _foreground(panel, _index(panel, _Q2)) is None
    panel.deleteLater()

    panel = SettingsPanel()
    panel.populate_models(
        [], set(), vram_gb=size + _WONT_FIT_SLACK_GB, memory_is_shared=False,
    )
    idx = _index(panel, _Q4)
    assert _foreground(panel, idx) == _color("warning")
    tip = _tooltip(panel, idx)
    assert tip.startswith("Tight fit:")
    assert "Won't fit" not in tip
    assert needs in tip
    panel.deleteLater()


def test_automatic_choice_includes_the_overhead_boundary_and_skips_f16(qapp):
    """size + 2 GB still fits; the next smaller step does not count as room.

    f16 is the largest recommended build. It is skipped on purpose: Q8 is
    enough for captioning and much faster, so a roomy machine must not land
    on the 15 GB weights.
    """
    q4 = _size(_Q4)
    assert _size(_Q4_S) + _OVERHEAD_GB < q4 + _OVERHEAD_GB
    assert _size(_F16) > _size(_MLX8)

    panel = SettingsPanel()
    _fresh(panel)
    panel.populate_models(
        [], set(), vram_gb=q4 + _OVERHEAD_GB, memory_is_shared=False,
    )
    assert panel.get_selected_model() == ("registry", _Q4)
    assert _foreground(panel, _index(panel, _Q4)) is None
    panel.deleteLater()

    panel = SettingsPanel()
    _fresh(panel)
    panel.populate_models(
        [], set(),
        vram_gb=math.nextafter(q4 + _OVERHEAD_GB, 0.0),
        memory_is_shared=False,
    )
    assert panel.get_selected_model() == ("registry", _Q4_S)
    panel.deleteLater()

    # Largest recommended build that is not f16: Q8_0 on this dropdown.
    # MLX 8-bit is larger, and only appears where that backend can run.
    expected = _MLX8 if mlx_backend_supported() else _Q8
    assert "f16" not in expected.lower()
    assert _size(expected) + _OVERHEAD_GB <= _size(_F16) + _OVERHEAD_GB
    assert _size(expected) > _size(_Q4)

    panel = SettingsPanel()
    _fresh(panel)
    panel.populate_models(
        [], set(), vram_gb=_size(_F16) + _OVERHEAD_GB, memory_is_shared=False,
    )
    assert _index(panel, _F16) >= 0
    assert panel.get_selected_model() == ("registry", expected)
    panel.deleteLater()


def test_a_budget_that_fits_nothing_lands_on_q6_not_a_header(qapp):
    """Nothing is comfortable in 1 GB. The fallback is the v2 Q6_K row.

    The first rows are disabled group headers with no model behind them.
    Leaving the combo there makes Load ask for a header label.
    """
    panel = SettingsPanel()
    _fresh(panel)
    panel.populate_models([], set(), vram_gb=1.0, memory_is_shared=False)

    assert panel.get_selected_model() == ("registry", _Q6)
    headers = 0
    for i in range(panel.model_combo.count()):
        if panel.model_combo.itemText(i).startswith("—"):
            headers += 1
            item = panel.model_combo.model().item(i)
            assert item is not None and item.isEnabled() is False
    assert headers > 0
    panel.deleteLater()


def test_rebuilding_the_list_keeps_the_model_the_user_picked(qapp):
    """A rescan must not snap the selection back to whatever fits today."""
    panel = SettingsPanel()
    _fresh(panel)
    panel.populate_models(
        [], set(), vram_gb=_size(_Q4) + _OVERHEAD_GB, memory_is_shared=False,
    )
    assert panel.get_selected_model() == ("registry", _Q4)

    panel.model_combo.setCurrentIndex(_index(panel, _Q2))
    panel.populate_models(
        [], set(), vram_gb=_size(_F16) + _OVERHEAD_GB, memory_is_shared=False,
    )
    assert panel.get_selected_model() == ("registry", _Q2)
    panel.deleteLater()


def test_downloaded_mark_does_not_change_the_registry_key(qapp):
    panel = SettingsPanel()
    panel.populate_models([], {_Q4}, vram_gb=None, memory_is_shared=False)
    idx = _index(panel, _Q4)
    assert panel.model_combo.itemText(idx) == f"✓ {_Q4}"
    assert "Already downloaded" in _tooltip(panel, idx)
    panel.model_combo.setCurrentIndex(idx)
    assert panel.get_selected_model() == ("registry", _Q4)
    panel.deleteLater()


def test_shared_memory_is_not_described_as_a_gpu(qapp, monkeypatch):
    """Apple Silicon reports free unified memory, which moves between launches.

    Omitting the flag follows the platform. Forcing it the other way must
    not keep the Mac wording on Windows, or the GPU wording on a Mac.
    """
    size = _size(_Q8)
    vram = math.nextafter(size + _WONT_FIT_SLACK_GB, 0.0)

    panel = SettingsPanel()
    panel.populate_models([], set(), vram_gb=vram, memory_is_shared=True)
    tip = _tooltip(panel, _index(panel, _Q8))
    assert "unified memory" in tip and "free right now" in tip
    assert "VRAM" not in tip and "your GPU has" not in tip
    panel.deleteLater()

    monkeypatch.setattr(sys, "platform", "darwin")
    panel = SettingsPanel()
    panel.populate_models([], set(), vram_gb=vram, memory_is_shared=None)
    tip = _tooltip(panel, _index(panel, _Q8))
    assert "unified memory" in tip
    panel.deleteLater()

    monkeypatch.setattr(sys, "platform", "win32")
    panel = SettingsPanel()
    panel.populate_models([], set(), vram_gb=vram, memory_is_shared=None)
    tip = _tooltip(panel, _index(panel, _Q8))
    assert "VRAM" in tip and "your GPU has" in tip
    assert "unified memory" not in tip
    panel.deleteLater()


def test_batch_stays_busy_on_the_last_item_and_between_items(qapp):
    """current == total is still in flight, and so is the gap after a caption.

    The batch button used to re-enable on the last item, and again in the
    ~100 ms between images, which started a second batch on top of the first.
    """
    panel = SettingsPanel()
    panel.set_batch_progress(10, 10)
    assert panel.batch_btn.isEnabled() is False
    assert "10/10" in panel.batch_btn.text()
    assert panel._download_btn.isEnabled() is False
    assert panel.load_model_btn.isEnabled() is False
    assert panel.cancel_btn.isHidden() is False

    panel.set_generating(True)
    panel.set_generating(False)
    panel.set_batch_progress(4, 10)
    assert panel.batch_btn.isEnabled() is False
    assert "4/10" in panel.batch_btn.text()
    assert panel._download_btn.isEnabled() is False
    assert panel.load_model_btn.isEnabled() is False
    assert panel.cancel_btn.isHidden() is False

    panel.set_batch_progress(0, 0)
    assert panel.batch_btn.isEnabled() is True
    assert panel._download_btn.isEnabled() is True
    assert panel.cancel_btn.isHidden() is True
    panel.deleteLater()


def test_inference_time_is_the_last_caption_not_an_average(qapp):
    panel = SettingsPanel()
    panel.set_inference_time(2)
    assert panel.inference_time_label.text() == "Last caption: 2.0s"
    assert "Average" not in panel.inference_time_label.text()
    assert panel.inference_time_label.isHidden() is False
    panel.deleteLater()
