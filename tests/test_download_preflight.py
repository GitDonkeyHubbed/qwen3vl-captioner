"""Pre-flight guards for the GGUF downloader.

The worker is real; the network, the disk-space syscall, and the transfer
itself are stubbed. These pin the decisions made before any byte is written.
"""

import sys
import types

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from gui.model_download_manager import ModelDownloadWorker  # noqa: E402
import gui.model_download_manager as mdm  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def hub(monkeypatch):
    try:
        import huggingface_hub
    except ImportError:
        huggingface_hub = types.ModuleType("huggingface_hub")
        monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_url",
        lambda **kwargs: "https://example.invalid/file",
        raising=False,
    )


def _worker(tmp_path, connections):
    return ModelDownloadWorker(
        "owner/repo", "m.gguf", tmp_path, max_connections=connections,
    )


def test_disk_check_uses_an_existing_ancestor(qapp, tmp_path, monkeypatch, hub):
    """A not-yet-created destination still has to be measured for free space.

    Skipping the walk up to a real directory makes disk_usage raise, and that
    exception is treated as 'size unknown' — the download then starts.
    """
    nested = tmp_path / "no-such" / "models"
    worker = _worker(nested, 1)
    called = []
    monkeypatch.setattr(worker, "_probe", lambda url: (1024 ** 3, False))
    monkeypatch.setattr(worker, "_run_single", lambda *a: called.append("single"))
    monkeypatch.setattr(worker, "_run_parallel", lambda *a: called.append("parallel"))

    def disk_usage(path):
        if not path.exists():
            raise FileNotFoundError(path)
        return types.SimpleNamespace(total=1, used=1, free=1024)

    monkeypatch.setattr(mdm.shutil, "disk_usage", disk_usage)
    errors = []
    worker.error.connect(errors.append)

    worker.run()

    assert called == []
    assert not nested.exists()
    assert len(errors) == 1
    assert "Not enough free disk space" in errors[0]
    assert "no-such" not in errors[0]
    assert str(tmp_path) in errors[0]


def test_parallel_download_starts_only_for_a_fresh_large_ranged_file(
    qapp, tmp_path, monkeypatch, hub,
):
    threshold = ModelDownloadWorker._PARALLEL_THRESHOLD
    monkeypatch.setattr(
        mdm.shutil, "disk_usage",
        lambda path: types.SimpleNamespace(total=1, used=0, free=10 ** 15),
    )
    cases = [
        ("below-threshold", threshold - 1, 4, None, True, "single"),
        ("one-connection", threshold, 1, None, True, "single"),
        ("zero-connections", threshold, 0, None, True, "single"),
        ("existing-partial", threshold, 4, b"partial", True, "single"),
        ("no-range", threshold, 4, None, False, "single"),
        ("large-ranged", threshold, 4, None, True, "parallel"),
    ]
    for name, total, connections, part_bytes, ranged, expect in cases:
        dest = tmp_path / name
        dest.mkdir()
        worker = _worker(dest, connections)
        if connections <= 0:
            assert worker.max_connections == 1
        if part_bytes is not None:
            (dest / "m.gguf.part").write_bytes(part_bytes)
        called = []
        monkeypatch.setattr(
            worker, "_probe", lambda url, total=total, ranged=ranged: (total, ranged),
        )
        monkeypatch.setattr(
            worker, "_run_single", lambda *a, called=called: called.append("single"),
        )
        monkeypatch.setattr(
            worker, "_run_parallel",
            lambda *a, called=called: called.append("parallel"),
        )
        errors = []
        worker.error.connect(errors.append)

        worker.run()

        assert errors == [], name
        assert called == [expect], name
        if part_bytes is not None:
            assert (dest / "m.gguf.part").read_bytes() == part_bytes


def test_rate_eta_switches_at_ninety_seconds():
    one = 1024 ** 2
    assert ModelDownloadWorker._rate_eta(0, 1, one) == ""
    assert ModelDownloadWorker._rate_eta(one, 0, one) == ""
    assert ModelDownloadWorker._rate_eta(one, 1, 0) == ", 1 MB/s"
    assert ModelDownloadWorker._rate_eta(one, 1, 89 * one) == ", 1 MB/s, ~89s left"
    assert ModelDownloadWorker._rate_eta(one, 1, 90 * one) == ", 1 MB/s, ~2 min left"
