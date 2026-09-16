"""Tests for video frame extraction (engine.video).

The fixtures encode tiny synthetic mp4s with cv2.VideoWriter where each
frame is a solid color whose blue channel stores the frame index (index*8).
Decoding that value back from the sampled PIL images lets the tests assert
*which* frames were picked — count, temporal order, midpoint spacing — with
tolerance for mp4v's lossy compression. If this platform's OpenCV can't
encode mp4v, the encoding-dependent tests skip.

Broken containers are simulated by patching ``cv2.VideoCapture`` with
delegating fakes: ``_break_capture`` for a missing frame count or a capture
that refuses to seek, and ``_lying_header_capture`` for a file shorter than
its own header claims (seeks past the real end succeed, the read after them
fails). Both forward real reads, so the sampled frames keep their encoded
index and coverage stays assertable.

``engine.video`` imports cv2 lazily, so the missing-cv2 error path is
exercised here too by poisoning ``sys.modules``.
"""

import contextlib
import sys
import warnings
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
    assert all(b > a for a, b in zip(indices, indices[1:], strict=False))


def test_sample_frames_midpoint_spacing(video_path):
    """3 of 30 frames must land mid-span (~5, 15, 25), not start-anchored."""
    frames = sample_frames(video_path, num_frames=3)
    assert len(frames) == 3
    indices = [_frame_index(f) for f in frames]
    for got, expected in zip(indices, (5, 15, 25), strict=True):
        assert got == pytest.approx(expected, abs=2)
    # Start-anchored sampling (round(i*total/n)) would pick {0, 10, 20};
    # the midpoint of the first span never touches the intro frames.
    assert indices[0] >= 3


def test_sample_frames_more_than_total(video_path):
    frames = sample_frames(video_path, num_frames=100)
    assert 1 <= len(frames) <= TOTAL_FRAMES
    indices = [_frame_index(f) for f in frames]
    assert all(b > a for a, b in zip(indices, indices[1:], strict=False))


def _break_capture(monkeypatch, *, frame_count=None, seekable=True):
    """Patch cv2.VideoCapture with a delegating fake that misbehaves.

    ``frame_count`` overrides CAP_PROP_FRAME_COUNT (0.0 and -1.0 are what
    VFR webm files really report); ``seekable=False`` makes every
    CAP_PROP_POS_FRAMES seek fail, which is how an unseekable container
    behaves. Everything else is forwarded to the real capture, so the
    fallback paths still decode genuine frames.
    """
    real_capture = cv2.VideoCapture

    class BrokenCapture:
        def __init__(self, source):
            self._cap = real_capture(source)

        def get(self, prop):
            if frame_count is not None and prop == cv2.CAP_PROP_FRAME_COUNT:
                return frame_count
            return self._cap.get(prop)

        def set(self, prop, value):
            if not seekable and prop == cv2.CAP_PROP_POS_FRAMES:
                return False
            return self._cap.set(prop, value)

        def __getattr__(self, name):
            return getattr(self._cap, name)

    monkeypatch.setattr(cv2, "VideoCapture", BrokenCapture)


def test_sample_frames_sequential_fallback(video_path, monkeypatch):
    """A capture reporting frame count 0 must trigger the sequential pass.

    The sequential sampler counts frames itself, so it owes the caller the
    full ``num_frames`` — the old rolling-halving implementation returned
    fewer, and ``<= 8`` let that pass.
    """
    _break_capture(monkeypatch, frame_count=0.0)
    frames = sample_frames(video_path, num_frames=8)
    assert len(frames) == 8
    indices = [_frame_index(f) for f in frames]
    assert all(b > a for a, b in zip(indices, indices[1:], strict=False))


def test_sequential_fallback_keeps_midpoint_coverage(video_path, monkeypatch):
    """The sequential pass must span the whole clip, not just its start.

    Rolling-halving kept whichever frames survived its decimation, which
    skewed the sample toward the intro. Counting first and then decoding the
    midpoint indices gives the same {5, 15, 25} the seeking path picks.
    """
    _break_capture(monkeypatch, frame_count=-1.0)
    frames = sample_frames(video_path, num_frames=3)
    assert len(frames) == 3
    indices = [_frame_index(f) for f in frames]
    for got, expected in zip(indices, (5, 15, 25), strict=True):
        assert got == pytest.approx(expected, abs=2)


def test_sample_frames_unseekable_falls_back(video_path, monkeypatch):
    """A header-claimed frame count plus failing seeks must still sample.

    _sample_by_index bails out the moment a seek fails, because reads after a
    failed seek just decode wherever the decoder happens to sit. sample_frames
    then re-runs the sequential pass — a path nothing covered before, so a
    regression there would have shipped silently. It is worth a warning too,
    but its own: see test_unseekable_warning_does_not_blame_the_header.
    """
    _break_capture(monkeypatch, seekable=False)
    with pytest.warns(RuntimeWarning, match="refused to seek"):
        frames = sample_frames(video_path, num_frames=4)
    assert len(frames) == 4
    indices = [_frame_index(f) for f in frames]
    assert all(b > a for a, b in zip(indices, indices[1:], strict=False))
    # Midpoints of 4 equal spans over 30 frames: ~4, 11, 19, 26.
    for got, expected in zip(indices, (4, 11, 19, 26), strict=True):
        assert got == pytest.approx(expected, abs=2)


def test_unseekable_warning_does_not_blame_the_header(video_path, monkeypatch):
    """An unseekable container's header is truthful — don't call it a liar.

    This clip really has TOTAL_FRAMES frames and its header says so; only
    set(CAP_PROP_POS_FRAMES) fails. Both faults came out of one sentence, so
    a healthy file read as "header claims 30 frames but only 0 of 4 sampled
    indices decoded" — pointing the reader at a frame count that is correct
    and away from the container that actually refused. They now branch.
    """
    _break_capture(monkeypatch, seekable=False)
    with pytest.warns(RuntimeWarning) as caught:
        sample_frames(video_path, num_frames=4)

    messages = [
        str(w.message) for w in caught if w.category is RuntimeWarning
    ]
    assert len(messages) == 1
    message = messages[0]
    assert "refused to seek" in message
    assert "none of the 4 sampled indices" in message
    # The header is not the fault here and must not be named as one.
    assert "header claims" not in message
    assert "decoded" not in message
    assert "re-sampled sequentially and got 4" in message


class _CaptureStats:
    """How many captures the sampler opened, and how many at once."""

    def __init__(self):
        self.opened = 0
        self.live = 0
        self.max_live = 0


def _lying_header_capture(monkeypatch, *, header, real):
    """Patch cv2.VideoCapture with a file that is shorter than it claims.

    ``header`` is what CAP_PROP_FRAME_COUNT reports; ``real`` is how many
    frames actually decode. Seeks past ``real`` still succeed — that is what
    a real truncated AVI does, and why the under-sampling was silent: only
    the read after such a seek fails, and a skipped read used to just shrink
    the sample. Reads and grabs below ``real`` are forwarded to the real
    capture, so the decoded frames still carry their blue-channel index and
    coverage stays assertable.
    """
    real_capture = cv2.VideoCapture
    stats = _CaptureStats()

    class LyingHeaderCapture:
        def __init__(self, source):
            self._cap = real_capture(source)
            self._pos = 0
            stats.opened += 1
            stats.live += 1
            stats.max_live = max(stats.max_live, stats.live)

        def isOpened(self):  # cv2's own spelling
            return self._cap.isOpened()

        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return float(header)
            return self._cap.get(prop)

        def set(self, prop, value):
            if prop == cv2.CAP_PROP_POS_FRAMES:
                self._pos = int(value)
                if self._pos < real:
                    return self._cap.set(prop, value)
                return True  # accepted, but the next read will fail
            return self._cap.set(prop, value)

        def grab(self):
            if self._pos >= real:
                return False
            self._pos += 1
            return self._cap.grab()

        def read(self):
            if self._pos >= real:
                self._pos += 1
                return False, None
            self._pos += 1
            return self._cap.read()

        def retrieve(self):
            return self._cap.retrieve()

        def release(self):
            stats.live -= 1
            self._cap.release()

    monkeypatch.setattr(cv2, "VideoCapture", LyingHeaderCapture)
    return stats


@contextlib.contextmanager
def _quiet():
    """Swallow the short-sample warning, which has its own test.

    Tests about counts and coverage assert on frames, not on warnings — so a
    regression fails them on the thing they are named for, and the suite
    still runs clean.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def _spy_on_sequential(monkeypatch):
    """Count sequential-fallback runs without changing what it returns."""
    import engine.video as video_module

    calls = []
    real_sequential = video_module._sample_sequential

    def counting_sequential(cv2_mod, path, num_frames):
        calls.append(num_frames)
        return real_sequential(cv2_mod, path, num_frames)

    monkeypatch.setattr(video_module, "_sample_sequential", counting_sequential)
    return calls


@pytest.mark.parametrize("num_frames", [4, 8, 16])
def test_lying_header_returns_full_count(video_path, monkeypatch, num_frames):
    """A header that over-reports must not silently shrink the sample.

    Measured on a truncated AVI (header 50, 32 real frames): 4 requested
    frames came back as 3, 8 as 5, 16 as 10. The indexed pass seeks into the
    part of the index space the file never had, those reads fail, and the
    failures were simply skipped. The same shape reproduces here with a
    50-frame header over 20 real frames (4 -> 2, 8 -> 3, 16 -> 6).
    """
    _lying_header_capture(monkeypatch, header=50, real=20)
    with _quiet():
        frames = sample_frames(video_path, num_frames=num_frames)

    assert len(frames) == num_frames
    indices = [_frame_index(f) for f in frames]
    assert all(b > a for a, b in zip(indices, indices[1:], strict=False))
    # Coverage must span the real 20-frame clip, not the surviving slice of a
    # 50-frame index space: midpoints of 20 start at ~2 and end at ~18, while
    # the buggy path started at ~6 (first index of a 50-frame layout) and
    # stopped wherever the file ran out.
    assert indices[0] <= 4
    assert 14 <= indices[-1] < 20


def test_lying_header_warns_with_counts(video_path, monkeypatch):
    """The re-sample is silent to the caller, so say so on stderr once."""
    _lying_header_capture(monkeypatch, header=50, real=20)
    with pytest.warns(RuntimeWarning, match=r"clip\.mp4: header claims 50"):
        sample_frames(video_path, num_frames=4)


def test_lying_header_opens_one_capture_at_a_time(video_path, monkeypatch):
    """Falling back must not leave the first capture open on the file."""
    stats = _lying_header_capture(monkeypatch, header=50, real=20)
    with _quiet():
        sample_frames(video_path, num_frames=4)

    assert stats.opened >= 2  # indexed pass, then count + decode passes
    assert stats.max_live == 1
    assert stats.live == 0


@pytest.mark.parametrize("sampler", [sample_frames, first_frame])
def test_cannot_open_releases_its_capture(tmp_path, monkeypatch, sampler):
    """The cannot-open exit must not leak the capture it just constructed.

    Both entry points build the VideoCapture *before* the try/finally that
    owns release(), so the isOpened() bail-out returned straight past it:
    measured opened=1, released=0, live=1 for every unreadable file. The
    error is correct, the handle is not — and this is exactly the path a
    batch run hits repeatedly when a folder holds a few broken clips.
    """
    bogus = tmp_path / "garbage.mp4"
    bogus.write_bytes(b"this is not a video file")
    stats = _lying_header_capture(monkeypatch, header=50, real=20)

    with pytest.raises(RuntimeError, match="garbage.mp4"):
        sampler(bogus)

    assert stats.opened == 1
    assert stats.live == 0


def test_lying_header_keeps_seeked_frames_if_resample_finds_less(
    video_path, monkeypatch
):
    """Best result wins — a worse fallback must not empty the sample."""
    import engine.video as video_module

    _lying_header_capture(monkeypatch, header=50, real=20)
    calls = []

    def dud_sequential(cv2_mod, path, n):
        calls.append(n)
        return []

    monkeypatch.setattr(video_module, "_sample_sequential", dud_sequential)
    with _quiet():
        frames = sample_frames(video_path, num_frames=4)

    assert calls == [4]  # the short sample did trigger the fallback
    # Of indices {6, 19, 31, 44} only 6 and 19 exist in the real clip.
    assert len(frames) == 2
    indices = [_frame_index(f) for f in frames]
    assert all(b > a for a, b in zip(indices, indices[1:], strict=False))


def test_short_clip_returns_what_exists(tmp_path, monkeypatch):
    """3 real frames asked for 8: return 3, and don't pay for a second pass.

    The index list is clamped to the clip's own length, so a full 3-frame
    sample is not "short" — the fallback rule keys off the index count, not
    num_frames, precisely so honest little clips stay on the fast path.
    """
    clip = _write_video(tmp_path / "short.mp4", n_frames=3)
    calls = _spy_on_sequential(monkeypatch)

    frames = sample_frames(clip, num_frames=8)

    assert len(frames) == 3
    assert calls == []
    indices = [_frame_index(f) for f in frames]
    assert all(b > a for a, b in zip(indices, indices[1:], strict=False))


def test_honest_header_skips_sequential_pass(video_path, monkeypatch):
    """The common case must still be one seek-and-decode pass."""
    calls = _spy_on_sequential(monkeypatch)
    frames = sample_frames(video_path, num_frames=8)
    assert len(frames) == 8
    assert calls == []


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
