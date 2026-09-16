"""The queued vision-encoder download must be keyed to the model being downloaded.

Downloading a second model family into a folder that already holds another
family's mmproj used to skip the new encoder entirely: the check asked
``find_mmproj_file(target_dir)`` with no model, which returns whichever
encoder sorts first. The model then loaded against a foreign encoder, which
llama.cpp does not reject cleanly — it crashes natively.

The window is real; the download itself is stubbed, so nothing touches the
network.
"""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QCoreApplication, QEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from gui.main_window import MainWindow  # noqa: E402
from gui.model_download_manager import MODEL_REGISTRY  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(qapp, tmp_path):
    w = MainWindow(model_dir=tmp_path)
    yield w
    w._gpu_timer.stop()
    w.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


def _registry_entry(prefix):
    """The first registry label whose entry has an mmproj and matching name."""
    for label, info in MODEL_REGISTRY.items():
        if label.startswith(prefix) and info.get("mmproj_filename"):
            return label, info
    pytest.skip(f"no registry entry for {prefix!r}")


@pytest.fixture
def downloads(win, monkeypatch, tmp_path):
    """Auto-confirm the size dialog and record what would be downloaded."""
    started = []
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    # The two call sites pass their arguments differently (the queue path by
    # keyword, the chain path positionally), so bind against the real
    # signature rather than assuming either style.
    import inspect

    sig = inspect.signature(MainWindow._start_file_download)

    def record(*args, **kwargs):
        bound = sig.bind(win, *args, **kwargs)
        bound.apply_defaults()
        started.append({k: v for k, v in bound.arguments.items() if k != "self"})

    monkeypatch.setattr(win, "_start_file_download", record)
    return started


def test_new_family_still_queues_its_own_encoder(win, downloads, tmp_path):
    """A foreign mmproj already on disk must not satisfy this model."""
    label, info = _registry_entry("HauhauCS Gemma-4")
    # A different family's encoder, sorting before the Gemma-4 one.
    (tmp_path / "Qwen3-VL-8B-Instruct-abliterated-v2.mmproj-f16.gguf").write_bytes(b"x")

    win._download_model(label)

    assert downloads and downloads[0]["filename"] == info["filename"]
    assert win._pending_mmproj is not None
    assert win._pending_mmproj[1] == info["mmproj_filename"]


def test_the_models_own_encoder_is_not_downloaded_twice(win, downloads, tmp_path):
    """Exact-name match is still a skip — no redundant re-download."""
    label, info = _registry_entry("HauhauCS Gemma-4")
    (tmp_path / info["mmproj_filename"]).write_bytes(b"x")

    win._download_model(label)

    assert win._pending_mmproj is None


def _run_timers_immediately(monkeypatch):
    """Fire QTimer.singleShot callbacks inline — no event loop runs here."""
    from gui import main_window

    monkeypatch.setattr(
        main_window.QTimer, "singleShot",
        staticmethod(lambda _ms, fn: fn()),
    )


def test_chained_encoder_download_rechecks_by_exact_name(
    win, downloads, tmp_path, monkeypatch
):
    """The post-model re-check must use the same exact-name rule.

    This is the second gate: the model download can take an hour, so the
    folder is re-examined before the encoder is fetched. It asked the same
    model-less question, so the foreign encoder skipped it here too.
    """
    _run_timers_immediately(monkeypatch)
    label, info = _registry_entry("HauhauCS Gemma-4")
    (tmp_path / "Qwen3-VL-8B-Instruct-abliterated-v2.mmproj-f16.gguf").write_bytes(b"x")
    win._pending_mmproj = (info["repo_id"], info["mmproj_filename"], tmp_path)

    win._on_download_finished(str(tmp_path / info["filename"]))

    assert [d["filename"] for d in downloads] == [info["mmproj_filename"]]
    assert win._pending_mmproj is None


def test_chain_skips_when_the_exact_encoder_arrived_meanwhile(
    win, downloads, tmp_path, monkeypatch
):
    """The re-check still prevents a redundant second download."""
    _run_timers_immediately(monkeypatch)
    label, info = _registry_entry("HauhauCS Gemma-4")
    (tmp_path / info["mmproj_filename"]).write_bytes(b"x")
    win._pending_mmproj = (info["repo_id"], info["mmproj_filename"], tmp_path)

    win._on_download_finished(str(tmp_path / info["filename"]))

    assert downloads == []
    assert win._pending_mmproj is None
