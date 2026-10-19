"""Disk-space, MLX-snapshot, and parallel-retry guards.

No network and no HuggingFace hub. ``run`` / ``_run_snapshot`` /
``_run_parallel`` are driven with stand-ins.
"""

import sys
import types

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

import gui.model_download_manager as mdm  # noqa: E402
from gui.model_download_manager import ModelDownloadWorker  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


def _worker(tmp_path, **kwargs):
    params = dict(
        repo_id="owner/repo",
        filename="m.gguf",
        target_dir=tmp_path,
    )
    params.update(kwargs)
    return ModelDownloadWorker(**params)


def _collect(worker):
    finished, errors = [], []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)
    return finished, errors


def _install_hub(monkeypatch, *, files=(), on_download=None, on_list=None):
    seen = {"downloaded": [], "listed": False}

    class _Api:
        def list_repo_files(self, repo_id, token=None):
            seen["listed"] = True
            seen["list_args"] = (repo_id, token)
            if on_list is not None:
                on_list(repo_id, token)
            return list(files)

    def hf_hub_download(repo_id, filename, local_dir, token=None):
        seen["downloaded"].append((repo_id, filename, local_dir, token))
        if on_download is not None:
            on_download(repo_id, filename, local_dir, token)

    def hf_hub_url(repo_id, filename):
        return f"https://example.invalid/{repo_id}/{filename}"

    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = _Api
    fake.hf_hub_download = hf_hub_download
    fake.hf_hub_url = hf_hub_url
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    return seen


class _Response:
    status = 206

    def __init__(self, body):
        self._body = body

    def read(self, n):
        chunk = self._body[:n]
        self._body = self._body[n:]
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _install_opener(monkeypatch, open_fn):
    class _Opener:
        def open(self, request, timeout=None):
            return open_fn(request, timeout)

    monkeypatch.setattr(
        mdm.urllib.request, "build_opener", lambda *args, **kwargs: _Opener(),
    )


# ── MLX folder downloads ────────────────────────────────────────────────


def test_snapshot_skips_unsafe_filenames(qapp, tmp_path, monkeypatch):
    """A remote listing must not be allowed to write outside the model folder."""
    files = [
        "config.json",
        "../escape.bin",
        "..\\escape.bin",
        "C:\\evil.bin",
        "C:evil.bin",
        "/abs.bin",
        "foo/../../etc.bin",
        "sub/ok.bin",
        "weights.safetensors",
    ]
    seen = _install_hub(monkeypatch, files=files)
    worker = _worker(tmp_path, snapshot_folder="mlx-model", hf_token="tok")
    finished, errors = _collect(worker)

    worker._run_snapshot()

    assert errors == []
    names = [item[1] for item in seen["downloaded"]]
    assert names == ["config.json", "sub/ok.bin", "weights.safetensors"]
    dest = str(tmp_path / "mlx-model")
    assert {item[2] for item in seen["downloaded"]} == {dest}
    assert {item[3] for item in seen["downloaded"]} == {"tok"}
    assert seen["list_args"] == ("owner/repo", "tok")
    assert finished == [dest]


def test_snapshot_cancel_discards_an_incomplete_folder(qapp, tmp_path, monkeypatch):
    worker = _worker(tmp_path, snapshot_folder="mlx-model")

    def on_download(repo_id, filename, local_dir, token):
        dest = tmp_path / "mlx-model"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "partial.bin").write_bytes(b"x")
        worker.cancel()

    seen = _install_hub(
        monkeypatch, files=["a.bin", "b.bin", "c.bin"], on_download=on_download,
    )
    finished, errors = _collect(worker)

    worker._run_snapshot()

    assert finished == []
    assert errors == ["Download cancelled."]
    assert [item[1] for item in seen["downloaded"]] == ["a.bin"]
    assert not (tmp_path / "mlx-model").exists()


def test_snapshot_cancel_keeps_a_complete_model(qapp, tmp_path, monkeypatch):
    """Stop during a re-check must not delete a model that is already usable."""
    dest = tmp_path / "mlx-model"
    dest.mkdir()
    (dest / "config.json").write_text("{}", encoding="utf-8")
    (dest / "model.safetensors").write_bytes(b"weights")
    (dest / "keep-me").write_text("safe", encoding="utf-8")
    worker = _worker(tmp_path, snapshot_folder="mlx-model")

    def on_download(repo_id, filename, local_dir, token):
        worker.cancel()

    seen = _install_hub(
        monkeypatch, files=["a.bin", "b.bin"], on_download=on_download,
    )
    finished, errors = _collect(worker)

    worker._run_snapshot()

    assert finished == []
    assert errors == ["Download cancelled."]
    assert [item[1] for item in seen["downloaded"]] == ["a.bin"]
    assert (dest / "config.json").is_file()
    assert (dest / "model.safetensors").read_bytes() == b"weights"
    assert (dest / "keep-me").read_text(encoding="utf-8") == "safe"


def test_cancel_before_start_does_not_touch_the_repo(qapp, tmp_path, monkeypatch):
    seen = _install_hub(monkeypatch, files=["a.bin"])
    worker = _worker(tmp_path, snapshot_folder="mlx-model")
    called = []
    monkeypatch.setattr(worker, "_run_snapshot", lambda: called.append("snapshot"))
    finished, errors = _collect(worker)
    worker.cancel()

    worker.run()

    assert called == []
    assert seen["listed"] is False
    assert finished == []
    assert errors == ["Download cancelled before starting."]


# ── Disk space ──────────────────────────────────────────────────────────


def _stub_strategy(monkeypatch, worker, total, supports_range):
    monkeypatch.setattr(worker, "_probe", lambda url: (total, supports_range))
    called = []
    monkeypatch.setattr(worker, "_run_parallel", lambda *args: called.append("parallel"))
    monkeypatch.setattr(worker, "_run_single", lambda *args: called.append("single"))
    return called


def test_low_disk_refuses_before_any_download(qapp, tmp_path, monkeypatch):
    _install_hub(monkeypatch)
    worker = _worker(tmp_path)
    called = _stub_strategy(monkeypatch, worker, 2 * 1024 ** 3, True)
    monkeypatch.setattr(
        mdm.shutil, "disk_usage",
        lambda path: types.SimpleNamespace(total=0, used=0, free=100 * 1024 ** 2),
    )
    finished, errors = _collect(worker)

    worker.run()

    assert called == []
    assert finished == []
    assert not (tmp_path / "m.gguf").exists()
    assert not (tmp_path / "m.gguf.part").exists()
    assert errors
    assert "Not enough free disk space" in errors[0]
    assert "m.gguf" in errors[0]


def test_resume_credits_the_partial_toward_free_space(qapp, tmp_path, monkeypatch):
    """A nearly finished .part must not be charged as a second full copy.

    free space here fits the remainder plus the 256 MiB headroom, and does
    not fit the whole file. Ignoring the bytes already on disk refuses a
    resume that should proceed.
    """
    _install_hub(monkeypatch)
    worker = _worker(tmp_path)
    total = 2 * 1024 ** 3
    existing = total - 1024
    called = _stub_strategy(monkeypatch, worker, total, True)
    # Equal to the remainder (1 KiB) plus the 256 MiB headroom. A check that
    # ignores the bytes already on disk asks for the whole 2 GiB and refuses.
    free = 1024 + 256 * 1024 * 1024
    monkeypatch.setattr(
        mdm.shutil, "disk_usage",
        lambda path: types.SimpleNamespace(total=0, used=0, free=free),
    )
    real_stat = mdm.Path.stat

    def fake_stat(self, *args, **kwargs):
        if self.name.endswith(".gguf.part"):
            return types.SimpleNamespace(st_size=existing)
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(mdm.Path, "stat", fake_stat)
    finished, errors = _collect(worker)

    worker.run()

    assert errors == []
    assert called == ["single"]
    assert finished == []


def test_unknown_size_does_not_fail_the_disk_check(qapp, tmp_path, monkeypatch):
    """A failed size probe reports 0. That must not be treated as 'no room'."""
    _install_hub(monkeypatch)
    worker = _worker(tmp_path)
    called = _stub_strategy(monkeypatch, worker, 0, False)
    monkeypatch.setattr(
        mdm.shutil, "disk_usage",
        lambda path: types.SimpleNamespace(total=0, used=0, free=0),
    )
    finished, errors = _collect(worker)

    worker.run()

    assert errors == []
    assert called == ["single"]
    assert finished == []


# ── Parallel retry budget ───────────────────────────────────────────────


def _range_of(request):
    start_s, end_s = request.get_header("Range").split("=", 1)[1].split("-")
    return int(start_s), int(end_s)


def test_parallel_retry_budget_resets_after_progress(qapp, tmp_path, monkeypatch):
    """A byte of progress refreshes the stall budget. Without the reset, the
    later stall below exceeds ``_MAX_RETRIES`` and deletes a download that
    was still moving."""
    monkeypatch.setattr(mdm.time, "sleep", lambda seconds: None)
    payload = b"abcdefgh"
    attempts = {"n": 0}

    def open_fn(request, timeout=None):
        start, end = _range_of(request)
        if start >= 4:
            return _Response(payload[start:end + 1])
        attempts["n"] += 1
        n = attempts["n"]
        if n in (1, 3):
            raise TimeoutError("stall")
        if n == 2:
            return _Response(payload[:1])
        return _Response(payload[start:end + 1])

    _install_opener(monkeypatch, open_fn)
    worker = _worker(tmp_path, max_connections=2)
    worker._MAX_RETRIES = 1
    finished, errors = _collect(worker)
    target = tmp_path / "m.gguf"
    part = tmp_path / "m.gguf.part"

    worker._run_parallel("https://example.invalid/m.gguf", target, part, len(payload))

    assert errors == []
    assert finished == [str(target)]
    assert target.read_bytes() == payload
    assert not part.exists()
    assert not (tmp_path / "m.gguf.part.parallel").exists()


def test_parallel_stall_discards_the_partial(qapp, tmp_path, monkeypatch):
    """No forward progress past the retry budget deletes the hole-filled
    .part. Leaving it would look resumable and be assembled as a model."""
    monkeypatch.setattr(mdm.time, "sleep", lambda seconds: None)

    def open_fn(request, timeout=None):
        raise TimeoutError("stalled")

    _install_opener(monkeypatch, open_fn)
    worker = _worker(tmp_path, max_connections=2)
    worker._MAX_RETRIES = 1
    finished, errors = _collect(worker)
    target = tmp_path / "m.gguf"
    part = tmp_path / "m.gguf.part"

    worker._run_parallel("https://example.invalid/m.gguf", target, part, 8)

    assert finished == []
    assert not target.exists()
    assert not part.exists()
    assert not (tmp_path / "m.gguf.part.parallel").exists()
    assert errors == ["stalled"]
