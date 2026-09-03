"""Tests for video frame extraction (engine.video).

The fixtures encode tiny synthetic mp4s with cv2.VideoWriter where each
frame is a solid color whose blue channel stores the frame index (index*8).
Decoding that value back from the sampled PIL images lets the tests assert
*which* frames were picked — count, temporal order, midpoint spacing — with
tolerance for mp4v's lossy compression. If this platform's OpenCV can't
encode mp4v, the encoding-dependent tests skip.

``engine.video`` imports cv2 lazily, so the missing-cv2 error path is
exercised here too by poisoning ``sys.modules``.
"""

import sys
from pathlib import Path

import pytest

from PIL import Image

from engine.video import (
    VIDEO_EXTENSIONS,
    first_frame,
    is_video_file,
    sample_frames,
)

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

FRAME_SIZE = (64, 48)  # (width, height)
TOTAL_FRAMES = 30


def _write_video(path, n_frames=TOTAL_FRAMES):
    w, h = FRAME_SIZE
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h)
    )
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter cannot encode mp4v on this platform")
    try:
        for i in range(n_frames):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:, :, 0] = i * 8  # BGR: blue channel encodes the index
            writer.write(frame)
    finally:
        writer.release()
    return path


def _frame_index(img: Image.Image) -> float:
    """Recover the encoded frame index from a sampled RGB image."""
    blue = np.asarray(img)[:, :, 2].astype(float).mean()
    return blue / 8.0


@pytest.fixture
def video_path(tmp_path):
    return _write_video(tmp_path / "clip.mp4")


def test_sample_frames_count_and_type(video_path):
    frames = sample_frames(video_path, num_frames=8)
    assert len(frames) == 8
    for frame in frames:
        assert isinstance(frame, Image.Image)
        assert frame.mode == "RGB"
        assert frame.size == FRAME_SIZE


def test_sample_frames_temporal_order(video_path):
    frames = sample_frames(video_path, num_frames=8)
    indices = [_frame_index(f) for f in frames]
    # Adjacent samples are ~30/8 indices apart; compression error is well
    # under one index, so strict inequality is safe.
    assert all(b > a for a, b in zip(indices, indices[1:]))


def test_sample_frames_midpoint_spacing(video_path):
    """3 of 30 frames must land mid-span (~5, 15, 25), not start-anchored."""
    frames = sample_frames(video_path, num_frames=3)
    assert len(frames) == 3
    indices = [_frame_index(f) for f in frames]
    for got, expected in zip(indices, (5, 15, 25)):
        assert got == pytest.approx(expected, abs=2)
    # Start-anchored sampling (round(i*total/n)) would pick {0, 10, 20};
    # the midpoint of the first span never touches the intro frames.
    assert indices[0] >= 3


def test_sample_frames_more_than_total(video_path):
    frames = sample_frames(video_path, num_frames=100)
    assert 1 <= len(frames) <= TOTAL_FRAMES
    indices = [_frame_index(f) for f in frames]
    assert all(b > a for a, b in zip(indices, indices[1:]))


def test_sample_frames_sequential_fallback(video_path, monkeypatch):
    """A capture reporting frame count 0 must trigger the sequential pass."""
    real_capture = cv2.VideoCapture

    class NoCountCapture:
        def __init__(self, source):
            self._cap = real_capture(source)

        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 0.0
            return self._cap.get(prop)

        def __getattr__(self, name):
            return getattr(self._cap, name)

    monkeypatch.setattr(cv2, "VideoCapture", NoCountCapture)
    frames = sample_frames(video_path, num_frames=8)
    assert 1 <= len(frames) <= 8
    indices = [_frame_index(f) for f in frames]
    assert all(b > a for a, b in zip(indices, indices[1:]))


def test_sample_frames_zero_raises(video_path):
    with pytest.raises(ValueError):
        sample_frames(video_path, num_frames=0)
    with pytest.raises(ValueError):
        sample_frames(video_path, num_frames=-3)


def test_unreadable_file_raises(tmp_path):
    bogus = tmp_path / "garbage.mp4"
    bogus.write_bytes(b"this is not a video file")
    with pytest.raises(RuntimeError, match="garbage.mp4"):
        sample_frames(bogus)


def test_missing_cv2_raises_runtime_error(tmp_path, monkeypatch):
    """The lazy import must surface a clear setup hint, not an ImportError."""
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(RuntimeError, match="opencv-python-headless"):
        sample_frames(tmp_path / "any.mp4")
    with pytest.raises(RuntimeError, match="opencv-python-headless"):
        first_frame(tmp_path / "any.mp4")


def test_first_frame(video_path):
    frame = first_frame(video_path)
    assert isinstance(frame, Image.Image)
    assert frame.mode == "RGB"
    assert frame.size == FRAME_SIZE
    # ~10% into 30 frames = index 3
    assert _frame_index(frame) == pytest.approx(3, abs=1.5)


def test_first_frame_unreadable_raises(tmp_path):
    bogus = tmp_path / "garbage.mp4"
    bogus.write_bytes(b"still not a video file")
    with pytest.raises(RuntimeError, match="garbage.mp4"):
        first_frame(bogus)


@pytest.mark.parametrize("ext", sorted(VIDEO_EXTENSIONS))
def test_is_video_file_supported(ext):
    assert is_video_file(Path("clip" + ext))
    assert is_video_file(Path("clip" + ext.upper()))


@pytest.mark.parametrize("name", ["photo.jpg", "clip.gif", "notes.txt", "clip"])
def test_is_video_file_rejected(name):
    assert not is_video_file(Path(name))
