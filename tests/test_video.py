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

from engine.base import VideoCancelled
from engine.video import (
    VIDEO_EXTENSIONS,
    first_frame,
    is_video_file,
    sample_frames,
)

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from engine import video  # noqa: E402  (needs cv2 present)


class _FakeCv2:
    """Stands in for the lazily-imported cv2 module in the unit tests below.

    Only what ``_sample_by_index`` touches is needed, and the real ones do
    the job — what these tests fake is the *capture*, not the library.
    """

    CAP_PROP_POS_FRAMES = cv2.CAP_PROP_POS_FRAMES
    COLOR_BGR2RGB = cv2.COLOR_BGR2RGB

    @staticmethod
    def cvtColor(frame, code):
        return cv2.cvtColor(frame, code)


def _solid_frame(value: int):
    """A BGR frame whose blue channel encodes *value*, as the fixtures do."""
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[:, :, 0] = value % 256
    return frame

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


def _break_capture(monkeypatch, *, frame_count=None, seekable=True,
                   lying_seek=False):
    """Patch cv2.VideoCapture with a delegating fake that misbehaves.

    ``frame_count`` overrides CAP_PROP_FRAME_COUNT (0.0 and -1.0 are what
    VFR webm files really report); ``seekable=False`` makes every
    CAP_PROP_POS_FRAMES seek fail, which is how an unseekable container
    behaves; ``lying_seek=True`` makes the seek *claim* success and do
    nothing, which is how some backends and containers behave and is the
    nastier fault — the reads that follow return the right number of frames
    from the wrong part of the clip. Everything else is forwarded to the real
    capture, so the fallback paths still decode genuine frames.
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
            if prop == cv2.CAP_PROP_POS_FRAMES:
                if not seekable:
                    return False
                if lying_seek:
                    # Claim the seek worked, leave the decoder where it was.
                    return True
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
            if prop == cv2.CAP_PROP_POS_FRAMES:
                # Report the position this capture believes it is at, not the
                # real file's. A truncated container accepts the seek and
                # reports the requested index; the fault only surfaces on the
                # read that follows, which is the whole point of this fake.
                return float(self._pos)
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

    def dud_sequential(cv2_mod, path, n, cancel_check=None, max_dim=None):
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


# ── Cancellation during extraction ───────────────────────────────────────
#
# Extraction can run for seconds before any token loop exists to notice a
# cancel: a header-less file is scanned twice end to end. These drive the
# real sampler against a real (short) clip, so they assert the wiring, and
# the counter checks prove the scan stopped rather than merely reporting so.


def test_sample_frames_raises_when_cancelled_on_the_indexed_path(video_path):
    calls = []

    def cancel_check():
        calls.append(1)
        return True

    with pytest.raises(VideoCancelled):
        sample_frames(video_path, num_frames=8, cancel_check=cancel_check)
    assert calls, "cancel_check was never consulted"


def test_sample_frames_raises_when_cancelled_on_the_sequential_path(
    video_path, monkeypatch
):
    """The counting pass is the longest run with nothing else watching."""
    _break_capture(monkeypatch, frame_count=0.0)
    with pytest.raises(VideoCancelled):
        sample_frames(video_path, num_frames=8, cancel_check=lambda: True)


def test_sample_frames_completes_when_cancel_check_stays_false(video_path):
    """A live-but-false predicate must not disturb a normal run."""
    polled = []
    frames = sample_frames(
        video_path, num_frames=8,
        cancel_check=lambda: (polled.append(1), False)[1],
    )
    assert len(frames) == 8
    assert polled, "cancel_check was never consulted"


def test_sample_frames_without_cancel_check_is_unchanged(video_path):
    """The argument is optional; omitting it keeps the original behaviour."""
    assert len(sample_frames(video_path, num_frames=8)) == 8


def test_count_frames_polls_cancel_during_a_long_scan(video_path, monkeypatch):
    """The grab-only count checks periodically, not just at the ends.

    The 30-frame fixture is shorter than the 64-frame poll interval, so drop
    the interval to 1 to prove the check is inside the loop rather than
    bolted on after it.
    """
    import engine.video as video_module

    grabs = []
    real_capture = cv2.VideoCapture

    class CountingCapture:
        def __init__(self, source):
            self._cap = real_capture(source)

        def grab(self):
            grabs.append(1)
            return self._cap.grab()

        def __getattr__(self, name):
            return getattr(self._cap, name)

    monkeypatch.setattr(cv2, "VideoCapture", CountingCapture)
    monkeypatch.setattr(video_module, "CANCEL_POLL_INTERVAL", 1)

    with pytest.raises(VideoCancelled):
        video_module._count_frames(cv2, video_path, cancel_check=lambda: True)
    # Stopped on the first poll rather than grabbing all 30 frames first.
    assert 0 < len(grabs) < TOTAL_FRAMES


# ── Index generation returns one index per frame it can ──────────────────

@pytest.mark.parametrize("total,num_frames", [
    (8, 8), (3, 3), (10, 10), (16, 16), (2, 2), (1, 1),   # total == n
    (30, 8), (100, 8), (5, 4), (30, 3), (30, 4),          # total > n
    (3, 8), (1, 8), (30, 100),                            # total < n
])
def test_midpoint_indices_returns_min_total_num_frames(total, num_frames):
    """The count invariant: exactly min(total, num_frames) distinct indices.

    round() breaks ties to even, so with total == num_frames the midpoints
    0.5, 1.5, 2.5 ... collapsed in pairs and dedup then discarded them for
    good: an 8-frame clip asked for 8 frames yielded [0, 2, 4, 6, 7], and a
    3-frame clip asked for 3 yielded [0, 2]. Truncation has no ties.
    """
    from engine.video import _midpoint_indices

    indices = _midpoint_indices(total, num_frames)
    assert len(indices) == min(total, num_frames)
    assert len(set(indices)) == len(indices), "duplicate indices"
    assert indices == sorted(indices), "not in temporal order"
    assert all(0 <= i < total for i in indices), "index out of range"


def test_midpoint_indices_are_exhaustive_when_total_equals_num_frames():
    """Asking for every frame of a clip must return every frame."""
    from engine.video import _midpoint_indices

    assert _midpoint_indices(8, 8) == list(range(8))
    assert _midpoint_indices(3, 3) == [0, 1, 2]


def test_midpoint_indices_still_land_mid_span():
    """Truncation must not reintroduce the start-anchored bias."""
    from engine.video import _midpoint_indices

    # 3 of 30: the middles of [0,10), [10,20), [20,30).
    assert _midpoint_indices(30, 3) == [5, 15, 25]
    # Never frame 0 when the first span is wide enough to have a middle.
    assert _midpoint_indices(30, 8)[0] > 0


def test_sample_frames_returns_every_frame_of_an_exact_length_clip(tmp_path):
    """End to end: an 8-frame clip asked for 8 frames returns 8, not 5."""
    path = _write_video(tmp_path / "eight.mp4", n_frames=8)
    frames = sample_frames(path, num_frames=8)
    assert len(frames) == 8
    indices = [_frame_index(f) for f in frames]
    assert all(b > a for a, b in zip(indices, indices[1:], strict=False))


# ── Frames are clamped at decode, not after the list is built ────────────

def test_sample_frames_clamps_during_extraction(tmp_path):
    """max_dim must bound PEAK memory, not just the returned images.

    The sampler builds the whole list before returning, so a caller that
    downscales afterwards has already paid for every frame at native
    resolution: 16 frames of 4K RGB is 400 MB, of 8K is 1.6 GB, held for the
    length of the inference. Clamped at decode the same 16 frames are 11 MB.
    """
    path = _write_video(tmp_path / "big.mp4")  # 64x48 fixture
    frames = sample_frames(path, num_frames=4, max_dim=32)
    assert len(frames) == 4
    for f in frames:
        assert max(f.size) <= 32


def test_sample_frames_without_max_dim_keeps_native_size(tmp_path):
    """The parameter is opt-in; omitting it preserves the old contract."""
    path = _write_video(tmp_path / "native.mp4")
    for f in sample_frames(path, num_frames=3):
        assert f.size == FRAME_SIZE


def test_sequential_path_clamps_too(video_path, monkeypatch):
    """Both decode paths go through _to_pil, so both must honour max_dim."""
    _break_capture(monkeypatch, frame_count=0.0)
    frames = sample_frames(video_path, num_frames=4, max_dim=24)
    assert len(frames) == 4
    for f in frames:
        assert max(f.size) <= 24


def test_first_frame_is_not_clamped(video_path):
    """Thumbnails want the real image; first_frame passes no max_dim."""
    assert first_frame(video_path).size == FRAME_SIZE


# ── A seek that reports success but does not land ───────────────────────
#
# `cap.set(CAP_PROP_POS_FRAMES, idx)` returning True is not proof the
# decoder moved. When it lies, the reads that follow decode consecutive
# frames from wherever the decoder actually sits, and nothing downstream can
# tell: the frame *count* is right, so neither the short-sample check nor the
# refused-seek check fires. The caller is handed a cluster of near-identical
# frames from one part of the clip, described to the model as spanning the
# whole video.

def test_a_seek_that_lies_falls_back_to_the_sequential_pass(
    video_path, monkeypatch
):
    """Coverage must be preserved, not silently lost."""
    _break_capture(monkeypatch, lying_seek=True)

    frames = sample_frames(video_path, num_frames=8)

    assert len(frames) == 8
    indices = [_frame_index(f) for f in frames]
    # The sequential fallback decodes real frames, so these must span the
    # clip rather than bunch up at the decoder's resting position.
    assert all(b > a for a, b in zip(indices, indices[1:], strict=False))
    assert max(indices) - min(indices) > 1


def test_a_lying_seek_is_detected_before_any_frame_is_returned(monkeypatch):
    """The indexed sampler itself reports the refusal, at the first index."""
    reads = []

    class LyingCap:
        def set(self, prop, value):
            return True  # "seeked", but see get() below

        def get(self, prop):
            return 0.0  # never actually moves

        def read(self):
            reads.append(1)
            raise AssertionError("must not read after an unverified seek")

    frames, seek_refused = video._sample_by_index(
        _FakeCv2(), LyingCap(), [300, 600, 900], None, None
    )

    assert frames == []
    assert seek_refused is True
    assert reads == [], "no frame should be decoded once the seek is doubted"


def test_a_keyframe_snap_within_tolerance_is_accepted(monkeypatch):
    """Landing a little short of the target is normal, not a failure.

    A seek that snaps to the nearest keyframe still shows the intended
    moment. Falling back to a full sequential scan for that would make every
    ordinary long video pay for a second pass.
    """
    landed = []

    class SnappingCap:
        def __init__(self):
            self._pos = 0

        def set(self, prop, value):
            self._pos = max(0, value - 5)  # snapped back to a keyframe
            return True

        def get(self, prop):
            return float(self._pos)

        def read(self):
            landed.append(self._pos)
            return True, _solid_frame(self._pos)

    frames, seek_refused = video._sample_by_index(
        _FakeCv2(), SnappingCap(), [0, 500, 1000], None, None
    )

    assert seek_refused is False
    assert len(frames) == 3
    assert landed == [0, 495, 995]


@pytest.mark.parametrize(
    "indices, expected",
    [
        ([], 0),                # nothing to preserve: demand exactness
        ([7], 0),               # a lone index has no spacing to protect
        ([0, 1000], 500),       # half the spacing
        ([0, 400, 1000], 200),  # the *tightest* gap sets the tolerance
        ([2, 6, 10, 14], 2),    # a short clip gets a correspondingly tight
                                # bound — a fixed GOP floor would be wider
                                # than the whole video and hide the fault
        ([0, 1], 0),            # every frame sampled: nothing may drift
    ],
)
def test_seek_tolerance_scales_with_the_spacing_being_sampled(indices, expected):
    """Tolerance is half the tightest gap, with no constant floor.

    Scaling with the request rather than fixing a frame count is what keeps
    the rule meaning the same thing on a 30-frame clip and a two-hour film.
    """
    assert video._seek_tolerance(indices) == expected


def test_an_unreadable_position_counts_as_a_failed_seek():
    """A backend that cannot report a position gets the correct-but-slow path."""
    class MutePosCap:
        def set(self, prop, value):
            return True

        def get(self, prop):
            raise RuntimeError("backend does not support this property")

        def read(self):
            raise AssertionError("must not read")

    frames, seek_refused = video._sample_by_index(
        _FakeCv2(), MutePosCap(), [0, 500], None, None
    )
    assert frames == [] and seek_refused is True
