"""Tests for the GUI model-load worker (gui.main_window.ModelLoadWorker).

``run()`` is a plain method, so the worker is exercised synchronously — no
QThread needed (conftest's offscreen platform covers the Qt import). A fake
engine records the ``load_model`` call so the tests can assert exactly which
kwargs the worker forwards.
"""

from pathlib import Path

from gui.main_window import ModelLoadWorker


class _RecordingEngine:
    """Stands in for Qwen3VLEngine: records each load_model call's args."""

    def __init__(self):
        self.calls = []

    def load_model(self, model_path, mmproj_path, **kwargs):
        self.calls.append(
            {"model_path": model_path, "mmproj_path": mmproj_path, **kwargs}
        )


def _run_worker(chat_family):
    eng = _RecordingEngine()
    worker = ModelLoadWorker(
        eng,
        Path("/models/model.gguf"),
        Path("/models/mmproj.gguf"),
        chat_family=chat_family,
    )
    outcome = {"finished": 0, "errors": []}
    worker.finished.connect(
        lambda: outcome.__setitem__("finished", outcome["finished"] + 1)
    )
    worker.error.connect(outcome["errors"].append)
    worker.run()
    return eng, outcome


def test_worker_forwards_chat_family_to_load_model():
    eng, outcome = _run_worker(chat_family="qwen35")

    assert outcome["errors"] == []
    assert outcome["finished"] == 1
    assert len(eng.calls) == 1
    call = eng.calls[0]
    assert call["model_path"] == Path("/models/model.gguf")
    assert call["mmproj_path"] == Path("/models/mmproj.gguf")
    assert call["chat_family"] == "qwen35"


def test_worker_omits_chat_family_kwarg_when_none():
    # The MLX engine's load_model has no chat_family parameter, so a None
    # family must not be forwarded at all (even chat_family=None would
    # TypeError there) — the kwarg has to be absent, not merely None.
    eng, outcome = _run_worker(chat_family=None)

    assert outcome["errors"] == []
    assert outcome["finished"] == 1
    assert len(eng.calls) == 1
    assert "chat_family" not in eng.calls[0]
