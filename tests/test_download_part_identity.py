"""Tests for the .part identity-sidecar logic (gui.model_download_manager).

The sidecar (.part.meta) records which repo/file/size a partial download
belongs to, so a resume can never append onto a stale partial left under the
same name by a *different* file — the silent-corruption case fixed in v1.4.3.
These tests pin the sidecar unit contract: write/match, reject on mismatch,
absent-sidecar detection (the grandfathering split), and the recorded-total
fallback used by the HTTP 416 finalize path.

ModelDownloadWorker is a plain QObject — no QApplication, threads, or network
are needed to exercise these helpers.
"""

import json

import pytest

from gui.model_download_manager import ModelDownloadWorker, is_unsafe_repo_filename


@pytest.fixture
def worker(tmp_path):
    return ModelDownloadWorker(
        repo_id="owner/repo",
        filename="model.Q6_K.gguf",
        target_dir=tmp_path,
    )


@pytest.fixture
def part(tmp_path):
    p = tmp_path / "model.Q6_K.gguf.part"
    p.write_bytes(b"\x00" * 128)
    return p


def test_identity_round_trip(worker, part):
    worker._write_part_identity(part, 1000)
    assert worker._part_identity_matches(part, 1000) is True
    assert worker._recorded_part_total(part) == 1000


def test_mismatched_repo_is_rejected(worker, part, tmp_path):
    other = ModelDownloadWorker(
        repo_id="someone/else", filename="model.Q6_K.gguf", target_dir=tmp_path,
    )
    other._write_part_identity(part, 1000)
    assert worker._part_identity_matches(part, 1000) is False
    assert worker._recorded_part_total(part) == 0  # wrong owner — no fallback


def test_mismatched_filename_is_rejected(worker, part, tmp_path):
    other = ModelDownloadWorker(
        repo_id="owner/repo", filename="different.Q2_K.gguf", target_dir=tmp_path,
    )
    other._write_part_identity(part, 1000)
    assert worker._part_identity_matches(part, 1000) is False


def test_mismatched_total_is_rejected(worker, part):
    worker._write_part_identity(part, 1000)
    # Remote file changed size since the partial was written — cannot resume.
    assert worker._part_identity_matches(part, 2000) is False


def test_absent_sidecar_does_not_match(worker, part):
    """No sidecar => not a verified match (the resume path then applies the
    pre-1.4.3 size-only grandfathering heuristic instead of discarding)."""
    assert not worker._part_meta_path(part).exists()
    assert worker._part_identity_matches(part, 1000) is False
    assert worker._recorded_part_total(part) == 0


def test_recorded_total_survives_probe_failure(worker, part):
    """The 416 finalize path falls back to the recorded total when the live
    size probe fails (probed_total == 0)."""
    worker._write_part_identity(part, 4096)
    assert worker._recorded_part_total(part) == 4096
    # Sidecar with unknown total records 0 — no false finalize
    worker._write_part_identity(part, 0)
    assert worker._recorded_part_total(part) == 0


def test_corrupt_sidecar_is_rejected(worker, part):
    worker._part_meta_path(part).write_text("{not json", encoding="utf-8")
    assert worker._part_identity_matches(part, 1000) is False
    assert worker._recorded_part_total(part) == 0


def test_sidecar_content_shape(worker, part):
    worker._write_part_identity(part, 555)
    data = json.loads(worker._part_meta_path(part).read_text(encoding="utf-8"))
    assert data == {
        "repo_id": "owner/repo",
        "filename": "model.Q6_K.gguf",
        "total": 555,
    }


@pytest.mark.parametrize(
    "fname",
    [
        "/etc/passwd",             # POSIX absolute
        "../outside.bin",          # POSIX traversal
        "sub/../../outside.bin",   # nested POSIX traversal
        "..\\evil.dll",            # Windows traversal (backslash separator)
        "sub\\..\\..\\evil.dll",   # nested Windows traversal
        "C:\\Windows\\evil.dll",   # Windows absolute with drive
        "C:/Windows/evil.dll",     # drive with forward slashes
        "C:evil.dll",              # drive-relative (no separator)
        "\\\\server\\share\\x",    # UNC path
    ],
)
def test_unsafe_repo_filenames_rejected(fname):
    assert is_unsafe_repo_filename(fname) is True


@pytest.mark.parametrize(
    "fname",
    [
        "config.json",
        "model-00001-of-00002.safetensors",
        "sub/dir/tokenizer.json",
        "weights..half.bin",       # '..' inside a NAME is fine
    ],
)
def test_normal_repo_filenames_allowed(fname):
    assert is_unsafe_repo_filename(fname) is False
