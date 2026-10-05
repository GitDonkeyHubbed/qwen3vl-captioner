"""Resume and parallel-partial guards in ``ModelDownloadWorker``.

v1.4.1 fixed silent-corruption paths that the happy-path download test
does not exercise:

* A killed multi-connection download leaves a full-size, hole-filled
  ``.part`` plus a ``.parallel`` marker. Size alone looks "complete", so the
  next attempt must discard both instead of resuming or finalizing them.
* A parallel segment that is answered with HTTP 200 (Range ignored) must not
  be written: the body is the whole file, and storing it at that segment's
  offset produces a full-size but corrupt model.
* A single-stream resume must append only on a real 206, replace the file
  when the server ignores Range, and refuse a ``.part`` that belongs to a
  different download or is larger than the remote file. HTTP 416 finalizes
  a partial only when its size matches a known total.

No network and no HuggingFace hub: ``run`` / ``_run_single`` / ``_run_parallel``
are driven with a stand-in opener.
"""

import email.message
import sys
import types
import urllib.error
import urllib.request

import pytest
from PyQt6.QtWidgets import QApplication

from gui.model_download_manager import ModelDownloadWorker

_URL = "https://example.invalid/m.gguf"
_PAYLOAD = b"abcdefghij"  # 10 bytes; halves differ so a bad parallel assemble shows


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


def _worker(tmp_path, *, repo_id="owner/repo", max_connections=2):
    return ModelDownloadWorker(
        repo_id=repo_id,
        filename="m.gguf",
        target_dir=tmp_path,
        max_connections=max_connections,
    )


def _collect(worker):
    finished, errors = [], []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)
    return finished, errors


class _Response:
    def __init__(self, status, body, headers):
        self.status = status
        self.headers = headers
        self._body = body

    def read(self, n):
        chunk = self._body[:n]
        self._body = self._body[n:]
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(request, code):
    return urllib.error.HTTPError(
        request.full_url, code, "error", email.message.Message(), None,
    )


def _install_opener(monkeypatch, open_fn):
    class _Opener:
        def open(self, request, timeout=None):
            return open_fn(request, timeout)

    monkeypatch.setattr(
        urllib.request, "build_opener", lambda *args, **kwargs: _Opener(),
    )


def _paths(tmp_path):
    target = tmp_path / "m.gguf"
    part = tmp_path / "m.gguf.part"
    return target, part


# ── Parallel downloads ──────────────────────────────────────────────────


def test_stale_parallel_marker_is_discarded_before_any_resume(
    qapp, tmp_path, monkeypatch,
):
    """A crash mid pre-allocation must not be resumed or finalized.

    ``run`` deletes the hole-filled ``.part`` and its ``.parallel`` marker
    before it asks HuggingFace for a URL, so a later resume cannot append
    onto bytes that were never downloaded.
    """
    target, part = _paths(tmp_path)
    marker = tmp_path / "m.gguf.part.parallel"
    part.write_bytes(b"\x00" * 64)
    marker.write_text("1")

    def refuse_url(**kwargs):
        raise RuntimeError("stop after discard")

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_url = refuse_url
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    worker = _worker(tmp_path)
    finished, errors = _collect(worker)
    worker.run()

    assert finished == []
    assert errors
    assert "not installed" not in errors[0]
    assert not part.exists()
    assert not marker.exists()
    assert not target.exists()


def test_parallel_http_200_is_not_assembled_into_a_model(
    qapp, tmp_path, monkeypatch,
):
    """A Range-ignored 200 is the whole file. Writing it at a segment offset
    fills every segment with the *start* of the file and still looks complete.
    """
    target, part = _paths(tmp_path)

    def open_fn(request, timeout=None):
        # What a server that ignored Range actually returns: the entire object.
        return _Response(200, _PAYLOAD, {"Content-Length": str(len(_PAYLOAD))})

    _install_opener(monkeypatch, open_fn)
    worker = _worker(tmp_path)
    worker._MAX_RETRIES = 0  # fail the segment on the first non-206
    finished, errors = _collect(worker)

    worker._run_parallel(_URL, target, part, len(_PAYLOAD))

    assert finished == []
    assert not target.exists()
    assert not part.exists()
    assert not (tmp_path / "m.gguf.part.parallel").exists()
    assert errors
    assert "ignored Range" in errors[0]
    assert "status 200" in errors[0]


def test_parallel_cancel_discards_the_partial(qapp, tmp_path, monkeypatch):
    """Stop must not leave an offset-written ``.part`` that a retry could trust."""
    target, part = _paths(tmp_path)

    def open_fn(request, timeout=None):
        raise AssertionError("a cancelled download must not open a connection")

    _install_opener(monkeypatch, open_fn)
    worker = _worker(tmp_path)
    worker.cancel()
    finished, errors = _collect(worker)

    worker._run_parallel(_URL, target, part, len(_PAYLOAD))

    assert finished == []
    assert errors == ["Download cancelled."]
    assert not target.exists()
    assert not part.exists()
    assert not (tmp_path / "m.gguf.part.parallel").exists()


# ── Single-stream resume ────────────────────────────────────────────────


def _serve(monkeypatch, worker, *, probe_total, respond):
    """Stub the size probe and the one GET ``_run_single`` issues."""
    monkeypatch.setattr(worker, "_probe", lambda url: (probe_total, True))
    seen = []

    def open_fn(request, timeout=None):
        seen.append(request.get_header("Range"))
        return respond(request)

    _install_opener(monkeypatch, open_fn)
    return seen


def test_single_resume_appends_only_the_remainder(qapp, tmp_path, monkeypatch):
    """A pre-sidecar partial smaller than the remote file still resumes.

    Upgrading must not throw away a multi-GB ``.part`` that has no identity
    sidecar, and the bytes already on disk must stay in front of the tail.
    """
    target, part = _paths(tmp_path)
    prefix = _PAYLOAD[:4]
    part.write_bytes(prefix)
    worker = _worker(tmp_path)
    seen = _serve(
        monkeypatch, worker, probe_total=len(_PAYLOAD),
        respond=lambda request: _Response(
            206, _PAYLOAD[4:],
            {"Content-Range": f"bytes 4-{len(_PAYLOAD) - 1}/{len(_PAYLOAD)}"},
        ),
    )
    finished, errors = _collect(worker)

    worker._run_single(_URL, target, part)

    assert errors == []
    assert seen == ["bytes=4-"]
    assert finished == [str(target)]
    assert target.read_bytes() == _PAYLOAD
    assert not part.exists()
    assert not worker._part_meta_path(part).exists()


def test_single_ignored_range_replaces_instead_of_appending(
    qapp, tmp_path, monkeypatch,
):
    """HTTP 200 on a resume means the body is the whole file. Appending it
    onto the ``.part`` would bake the prefix in twice."""
    target, part = _paths(tmp_path)
    part.write_bytes(b"XXXX")
    worker = _worker(tmp_path)
    worker._write_part_identity(part, len(_PAYLOAD))
    seen = _serve(
        monkeypatch, worker, probe_total=len(_PAYLOAD),
        respond=lambda request: _Response(
            200, _PAYLOAD, {"Content-Length": str(len(_PAYLOAD))},
        ),
    )
    finished, errors = _collect(worker)

    worker._run_single(_URL, target, part)

    assert errors == []
    assert seen == ["bytes=4-"]
    assert finished == [str(target)]
    assert target.read_bytes() == _PAYLOAD


def test_single_does_not_resume_a_partial_from_another_file(
    qapp, tmp_path, monkeypatch,
):
    target, part = _paths(tmp_path)
    part.write_bytes(b"STALE-PARTIAL")
    _worker(tmp_path, repo_id="someone/else")._write_part_identity(
        part, len(_PAYLOAD),
    )
    worker = _worker(tmp_path)
    seen = _serve(
        monkeypatch, worker, probe_total=len(_PAYLOAD),
        respond=lambda request: _Response(
            200, _PAYLOAD, {"Content-Length": str(len(_PAYLOAD))},
        ),
    )
    finished, errors = _collect(worker)

    worker._run_single(_URL, target, part)

    assert errors == []
    assert seen == [None]
    assert finished == [str(target)]
    assert target.read_bytes() == _PAYLOAD


def test_single_discards_a_partial_larger_than_the_remote(
    qapp, tmp_path, monkeypatch,
):
    """A leftover ``.part`` bigger than the remote object cannot be this file,
    even when it predates the identity sidecar."""
    target, part = _paths(tmp_path)
    part.write_bytes(b"X" * (len(_PAYLOAD) + 15))
    worker = _worker(tmp_path)
    seen = _serve(
        monkeypatch, worker, probe_total=len(_PAYLOAD),
        respond=lambda request: _Response(
            200, _PAYLOAD, {"Content-Length": str(len(_PAYLOAD))},
        ),
    )
    finished, errors = _collect(worker)

    worker._run_single(_URL, target, part)

    assert errors == []
    assert seen == [None]
    assert finished == [str(target)]
    assert target.read_bytes() == _PAYLOAD


@pytest.mark.parametrize(
    ("probe_total", "on_disk", "should_finalize"),
    [
        ("match", "full", True),
        ("match", "short", False),
        ("unknown", "full", True),
    ],
    ids=["exact-size", "short-partial", "probe-failed"],
)
def test_single_416_finalizes_only_a_verified_partial(
    qapp, tmp_path, monkeypatch, probe_total, on_disk, should_finalize,
):
    """416 means the requested range is past EOF. That is 'already complete'
    only when the ``.part`` size equals a known total — the live probe, or,
    when the probe failed, the total stored in the identity sidecar.
    """
    target, part = _paths(tmp_path)
    part.write_bytes(_PAYLOAD if on_disk == "full" else _PAYLOAD[:4])
    worker = _worker(tmp_path)
    worker._write_part_identity(part, len(_PAYLOAD))
    reported = len(_PAYLOAD) if probe_total == "match" else 0

    def respond(request):
        raise _http_error(request, 416)

    _serve(monkeypatch, worker, probe_total=reported, respond=respond)
    finished, errors = _collect(worker)

    worker._run_single(_URL, target, part)

    if should_finalize:
        assert errors == []
        assert finished == [str(target)]
        assert target.read_bytes() == _PAYLOAD
        assert not part.exists()
        assert not worker._part_meta_path(part).exists()
    else:
        assert finished == []
        assert not target.exists()
        assert not part.exists()
        assert not worker._part_meta_path(part).exists()
        assert errors
        assert "could not be verified" in errors[0]


def test_single_dropped_connection_keeps_the_partial_for_resume(
    qapp, tmp_path, monkeypatch,
):
    target, part = _paths(tmp_path)
    worker = _worker(tmp_path)
    _install_opener(
        monkeypatch,
        lambda request, timeout=None: _Response(
            200, b"abc", {"Content-Length": "100"},
        ),
    )
    finished, errors = _collect(worker)

    worker._run_single(_URL, target, part)

    assert finished == []
    assert not target.exists()
    assert part.read_bytes() == b"abc"
    assert worker._part_identity_matches(part, 100)
    assert errors
    assert "resume" in errors[0]


def test_single_cancel_discards_the_partial(qapp, tmp_path, monkeypatch):
    """Cancel throws the ``.part`` away; a dropped connection does not."""
    target, part = _paths(tmp_path)
    worker = _worker(tmp_path)

    def open_fn(request, timeout=None):
        response = _Response(200, b"abc", {"Content-Length": "100"})
        original = response.read

        def read(n):
            worker.cancel()
            return original(n)

        response.read = read
        return response

    _install_opener(monkeypatch, open_fn)
    finished, errors = _collect(worker)

    worker._run_single(_URL, target, part)

    assert finished == []
    assert errors == ["Download cancelled."]
    assert not target.exists()
    assert not part.exists()
    assert not worker._part_meta_path(part).exists()
