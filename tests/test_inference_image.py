"""Tests for the GGUF engine's image preprocessing (engine.inference).

Covers the two silent-wrongness bugs found in the v1.4.3 QC audit:

1. EXIF orientation — phone/camera JPEGs carrying Orientation 3/6/8 must be
   transposed before encoding, or the model captions a sideways scene (the Qt
   preview auto-rotates, so the user can't tell).
2. Extreme aspect ratios — resizing must clamp both dimensions to >= 1 px so a
   10000x2 strip can't crash resize with a zero dimension (which also aborted
   the rest of a batch).

3. Size clamping — the longest side must be capped for every backend, and a
   JPEG source must be decoded through ``draft()`` so a 40 MP photo is not
   fully decoded just to be scaled down.

These exercise ``image_to_data_uri`` / ``load_image_for_inference`` only,
which are pure PIL — no llama_cpp, no Qt, no network.
"""

import base64
import io

from PIL import Image

from engine.base import load_image_for_inference
from engine.inference import image_to_data_uri


def _decode_data_uri(uri: str) -> Image.Image:
    # JPEG q95, not PNG: the chat handler re-encodes whatever it receives to
    # JPEG q95 anyway, so a PNG here only cost a slow compression pass and a
    # several-times-larger base64 payload on every caption.
    assert uri.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(uri.split(",", 1)[1])
    return Image.open(io.BytesIO(raw))


def test_exif_orientation_is_applied(tmp_path):
    """A JPEG stored rotated with Orientation=6 must be transposed upright."""
    path = tmp_path / "rotated.jpg"
    img = Image.new("RGB", (100, 60), "red")
    exif = Image.Exif()
    exif[274] = 6  # Orientation tag: 90° CW rotation required for display
    img.save(path, "JPEG", exif=exif.tobytes())

    out = _decode_data_uri(image_to_data_uri(path))
    # Transposing a 100x60 image by orientation 6 yields 60x100
    assert out.size == (60, 100)


def test_no_exif_image_unchanged(tmp_path):
    path = tmp_path / "plain.png"
    Image.new("RGB", (120, 80), "blue").save(path)

    out = _decode_data_uri(image_to_data_uri(path))
    assert out.size == (120, 80)


def test_extreme_aspect_ratio_does_not_crash(tmp_path):
    """A 5000x2 strip must downscale without a zero-height resize crash."""
    path = tmp_path / "strip.png"
    Image.new("RGB", (5000, 2), "green").save(path)

    out = _decode_data_uri(image_to_data_uri(path, max_dim=1280))
    assert out.size[0] == 1280
    assert out.size[1] >= 1  # clamped, not zero


def test_resize_keeps_aspect_for_normal_images(tmp_path):
    path = tmp_path / "big.png"
    Image.new("RGB", (2560, 1280), "white").save(path)

    out = _decode_data_uri(image_to_data_uri(path, max_dim=1280))
    assert out.size == (1280, 640)


def test_jpeg_source_uses_draft_decoding(tmp_path, monkeypatch):
    """A large JPEG must be downscaled during decode, not after it.

    Without draft() a 40 MP photo is fully decoded (~160 MB of pixels) and
    then LANCZOS-resized from full resolution on the caption worker.
    """
    path = tmp_path / "huge.jpg"
    Image.new("RGB", (4000, 3000), "red").save(path, "JPEG")

    # Record the size the LANCZOS pass starts from. With draft() it is the
    # decoder's downscaled output; without it, the full 4000x3000 source.
    seen = {}
    real_resize = Image.Image.resize

    def spy(self, size, *args, **kwargs):
        seen.setdefault("from", self.size)
        return real_resize(self, size, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "resize", spy)
    out = load_image_for_inference(path, max_dim=1280)

    # libjpeg scales by a power-of-two fraction, so the decoded image is
    # smaller than the source but still at least the requested 1280x960.
    assert seen["from"] == (2000, 1500)
    assert out.size == (1280, 960)


def test_clamp_is_a_no_op_for_small_images(tmp_path):
    path = tmp_path / "small.png"
    Image.new("RGB", (320, 200), "white").save(path)
    assert load_image_for_inference(path, max_dim=1280).size == (320, 200)
