"""Guards around memory readouts, image navigation, and the engine swap.

Each case fails when its boundary is loosened: a VRAM total of zero must stay
a reading, keyboard navigation must not wrap, and a second load of the same
backend must not throw away the engine that is already loaded.
"""

import builtins
import sys
import types

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QCoreApplication, QEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from engine.inference import Qwen3VLEngine  # noqa: E402
from engine.mlx_engine import MlxVlmEngine  # noqa: E402
from gui.main_window import MainWindow  # noqa: E402
from gui.settings_panel import TARGET_PRESETS  # noqa: E402
from gui.theme import COLORS  # noqa: E402

_GIB = 1024 ** 3


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def win(qapp, tmp_path):
    window = MainWindow(model_dir=tmp_path)
    window._gpu_timer.stop()
    yield window
    window._gpu_timer.stop()
    window._file_browser._thumb_pool.clear()
    window._file_browser._thumb_pool.waitForDone(1000)
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


class _MemInfo:
    def __init__(self, used, total):
        self.used = used
        self.total = total


class _Nvml:
    def __init__(self, info):
        self._info = info

    def nvmlDeviceGetMemoryInfo(self, _handle):
        if isinstance(self._info, Exception):
            raise self._info
        return self._info


def _band(label) -> str:
    sheet = label.styleSheet()
    for name in ("error", "warning", "success"):
        if COLORS[name] in sheet:
            return name
    raise AssertionError(f"no status colour in {sheet!r}")


@pytest.mark.parametrize(
    ("used_units", "total_units", "pct", "band"),
    [
        (699, 1000, 69, "success"),  # 69.9 floors; rounding would call it 70
        (700, 1000, 70, "warning"),
        (899, 1000, 89, "warning"),
        (900, 1000, 90, "error"),
    ],
)
def test_vram_color_changes_at_70_and_90(
    win, monkeypatch, used_units, total_units, pct, band,
):
    monkeypatch.setattr(sys, "platform", "linux")
    win._pynvml = _Nvml(_MemInfo(used_units * _GIB, total_units * _GIB))
    win._nvml_handle = object()

    win._update_gpu_info()

    assert win._gpu_label.text() == f"GPU: {pct}%"
    assert win._vram_label.text() == f"{used_units:.1f}/{total_units:.0f}GB"
    assert "UNIFIED" not in win._vram_label.text()
    assert _band(win._gpu_label) == band
    assert _band(win._gpu_dot) == band


def test_zero_vram_total_is_still_a_reading(win, monkeypatch):
    """total == 0 must not divide and then fall through to the idle label."""
    monkeypatch.setattr(sys, "platform", "linux")
    win._pynvml = _Nvml(_MemInfo(5, 0))
    win._nvml_handle = object()

    win._update_gpu_info()

    assert win._gpu_label.text() == "GPU: 0%"
    assert win._vram_label.text() == "0.0/0GB"
    assert _band(win._gpu_label) == "success"


def test_nvml_failure_drops_the_reading(win, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    win._gpu_label.setText("GPU: 99%")
    win._vram_label.setText("99/99GB")
    win._pynvml = _Nvml(RuntimeError("nvml down"))
    win._nvml_handle = object()

    win._update_gpu_info()

    assert win._gpu_label.text() == "GPU: Active"
    assert win._vram_label.text() == "VRAM: N/A"


@pytest.mark.parametrize(
    ("percent", "band"),
    [(69, "success"), (70, "warning"), (90, "error")],
)
def test_unified_memory_reports_what_is_free(win, monkeypatch, percent, band):
    """Apple Silicon pressure is free memory, not the process's used figure."""
    monkeypatch.setattr(sys, "platform", "darwin")
    win._nvml_handle = None
    mem = types.SimpleNamespace(
        total=16 * _GIB,
        available=4 * _GIB,  # 12 GiB in use
        used=10 * _GIB,      # a different number, so the two cannot be swapped
        percent=percent,
    )
    monkeypatch.setitem(
        sys.modules, "psutil",
        types.SimpleNamespace(virtual_memory=lambda: mem),
    )

    win._update_gpu_info()

    assert win._gpu_label.text() == f"MEM: {percent}%"
    assert win._vram_label.text() == "12.0/16GB UNIFIED"
    assert _band(win._gpu_label) == band


def _block_psutil(monkeypatch):
    monkeypatch.delitem(sys.modules, "psutil", raising=False)
    real_import = builtins.__import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "psutil":
            raise ImportError("no psutil")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked)


def test_mac_without_psutil_does_not_invent_a_reading(win, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    win._nvml_handle = None
    _block_psutil(monkeypatch)

    win._update_gpu_info()

    assert win._gpu_label.text() == "GPU: Active"
    assert win._vram_label.text() == "VRAM: N/A"


def test_ram_label_uses_used_bytes(win, monkeypatch):
    mem = types.SimpleNamespace(
        total=16 * _GIB,
        available=1 * _GIB,
        used=7 * _GIB // 2,  # 3.5 GiB; total - available would show 15
    )
    monkeypatch.setitem(
        sys.modules, "psutil",
        types.SimpleNamespace(virtual_memory=lambda: mem),
    )

    win._update_ram_info()

    assert win._ram_label.text() == "RAM: 3.5 / 16 GB"


def test_ram_label_clears_when_psutil_is_missing(win, monkeypatch):
    win._ram_label.setText("RAM: stale")
    _block_psutil(monkeypatch)

    win._update_ram_info()

    assert win._ram_label.text() == ""


def test_adjacent_selection_stops_at_the_ends(win, tmp_path):
    """Ctrl+Left on the first image stays there; a missing current opens the first."""
    paths = []
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        path = tmp_path / name
        path.write_bytes(b"stub")
        paths.append(path)
    win._file_browser.add_images(paths)

    win._file_browser.select_item(paths[0])
    win._select_adjacent(-1)
    assert win._current_image == paths[0]

    win._select_adjacent(1)
    assert win._current_image == paths[1]
    win._select_adjacent(1)
    assert win._current_image == paths[2]
    win._select_adjacent(1)
    assert win._current_image == paths[2]

    win._current_image = tmp_path / "gone.jpg"
    win._select_adjacent(1)
    assert win._current_image == paths[0]

    win._file_browser.clear_all()
    stuck = win._current_image
    win._select_adjacent(-1)
    assert win._current_image == stuck


def test_format_badge_follows_the_preset_and_clears(win):
    """Deselecting must not leave the previous preset's name on the badge."""
    assert win._caption_panel.format_badge.text() == "CUSTOM FORMAT"
    stable = next(preset for preset in TARGET_PRESETS if preset["id"] == "sd")

    win._settings_panel._apply_preset(stable)
    assert win._caption_panel.format_badge.text() == "STABLE DIFFUSION FORMAT"

    win._settings_panel._apply_preset(stable)
    assert win._caption_panel.format_badge.text() == "CUSTOM FORMAT"


def test_backend_swap_keeps_a_live_engine_and_names_the_window(win):
    """A second Load of the same backend must not construct a new engine."""
    gguf = win._engine
    assert isinstance(gguf, Qwen3VLEngine)

    win._ensure_engine("gguf")
    assert win._engine is gguf
    assert win.windowTitle() == f"{win._app_title} — GGUF Engine"

    win._ensure_engine("mlx")
    mlx = win._engine
    assert isinstance(mlx, MlxVlmEngine)
    assert mlx is not gguf
    assert win.windowTitle() == f"{win._app_title} — MLX Engine"

    win._ensure_engine("mlx")
    assert win._engine is mlx

    win._ensure_engine("other")
    assert isinstance(win._engine, Qwen3VLEngine)
    assert win.windowTitle() == f"{win._app_title} — GGUF Engine"

    win._ensure_engine("mlx")
    win._engine.model = object()
    win._engine._is_loaded = True
    win._unload_model()
    assert win._engine.is_loaded is False
    assert win.windowTitle() == win._app_title
