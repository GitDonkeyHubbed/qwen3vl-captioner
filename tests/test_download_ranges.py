"""Download probe parsing, request headers, and parallel byte ranges.

A parallel download of a file whose size does not divide evenly has to cover
the tail exactly once. That tail is the end of a GGUF; the pre-allocation
writes a 0 there, and leaving it corrupts the model.
"""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

import gui.model_download_manager as mdm  # noqa: E402
from gui.model_download_manager import ModelDownloadWorker  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


def _worker(tmp_path, **kwargs):
    return ModelDownloadWorker(
        repo_id="owner/repo",
        filename="m.gguf",
        target_dir=tmp_path,
        **kwargs,
    )


class _Resp:
    def __init__(self, status, headers):
        self.status = status
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Body:
    def __init__(self, data, status=206):
        self._data = data
        self.status = status

    def read(self, n=-1):
        if n < 0 or n > len(self._data):
            n = len(self._data)
        chunk, self._data = self._data[:n], self._data[n:]
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _install_probe(monkeypatch, response):
    def build_opener(*args, **kwargs):
        class Opener:
            def open(self, request, timeout=None):
                assert request.get_header("Range") == "bytes=0-0"
                assert timeout == 30
                return response

        return Opener()

    monkeypatch.setattr(mdm.urllib.request, "build_opener", build_opener)


def test_probe_reads_the_complete_length_from_content_range(tmp_path, monkeypatch):
    """A 206 to the 1-byte probe carries Content-Length 1. The file size is
    the Content-Range total; using the length downloads a one-byte model."""
    worker = _worker(tmp_path)
    _install_probe(
        monkeypatch,
        _Resp(206, {"Content-Range": "bytes 0-0/4096", "Content-Length": "1"}),
    )

    assert worker._probe("https://example.invalid/m.gguf") == (4096, True)


def test_probe_does_not_treat_a_full_response_as_range_support(tmp_path, monkeypatch):
    worker = _worker(tmp_path)
    _install_probe(
        monkeypatch,
        _Resp(200, {"Content-Length": "4096", "Content-Range": "bytes 0-0/1"}),
    )

    assert worker._probe("https://example.invalid/m.gguf") == (4096, False)


def test_probe_unknown_length_is_not_a_range_download(tmp_path, monkeypatch):
    """`bytes 0-0/*` means the server did not say how big the file is.
    Treating that as a ranged download of an unknown size must not raise."""
    worker = _worker(tmp_path)
    _install_probe(monkeypatch, _Resp(206, {"Content-Range": "bytes 0-0/*"}))

    assert worker._probe("https://example.invalid/m.gguf") == (0, False)


def test_request_omits_authorization_without_a_token(tmp_path):
    headers = _worker(tmp_path, hf_token="")._base_headers()
    assert "Authorization" not in headers
    assert headers["User-Agent"] == "qwen3vl-captioner"


def test_request_sends_the_bearer_token(tmp_path):
    headers = _worker(tmp_path, hf_token="hf_secret")._base_headers()
    assert headers["Authorization"] == "Bearer hf_secret"


@pytest.mark.parametrize("text", ["HTTP Error 401: Unauthorized", "403 Forbidden"])
def test_auth_errors_point_at_the_token_setting(text):
    message = ModelDownloadWorker._format_error(Exception(text))
    assert "HuggingFace token" in message
    assert text in message


def test_non_auth_errors_stay_verbatim():
    message = ModelDownloadWorker._format_error(Exception("HTTP 400 Bad Request"))
    assert message == "HTTP 400 Bad Request"


def _range_of(request):
    start_s, end_s = request.get_header("Range").split("=", 1)[1].split("-")
    return int(start_s), int(end_s)


def test_parallel_ranges_tile_an_uneven_file(qapp, tmp_path, monkeypatch):
    """10 bytes over 3 connections leaves a remainder on the last segment.

    Ending that segment at ``start + seg - 1`` drops the last byte, which the
    pre-allocation left as 0. Overlapping segments write the same offset twice.
    """
    monkeypatch.setattr(mdm.time, "sleep", lambda seconds: None)
    payload = bytes(range(1, 11))  # no zero, so a leftover pad byte is visible
    seen = []

    def open_fn(request, timeout=None):
        start, end = _range_of(request)
        seen.append((start, end))
        return _Body(payload[start:end + 1])

    def build_opener(*args, **kwargs):
        class Opener:
            def open(self, request, timeout=None):
                return open_fn(request, timeout)

        return Opener()

    monkeypatch.setattr(mdm.urllib.request, "build_opener", build_opener)
    worker = _worker(tmp_path, max_connections=3)
    finished = []
    errors = []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)
    target = tmp_path / "m.gguf"
    part = tmp_path / "m.gguf.part"

    worker._run_parallel("https://example.invalid/m.gguf", target, part, len(payload))

    assert errors == []
    assert finished == [str(target)]
    assert target.read_bytes() == payload
    assert not part.exists()
    assert not (tmp_path / "m.gguf.part.parallel").exists()

    covered = [0] * len(payload)
    for start, end in seen:
        assert 0 <= start <= end < len(payload)
        for index in range(start, end + 1):
            covered[index] += 1
    assert covered == [1] * len(payload)
