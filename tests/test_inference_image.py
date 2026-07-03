"""Tests for the GGUF engine's image preprocessing (engine.inference).

Covers the two silent-wrongness bugs found in the v1.4.3 QC audit:

1. EXIF orientation — phone/camera JPEGs carrying Orientation 3/6/8 must be
   transposed before encoding, or the model captions a sideways scene (the Qt
   preview auto-rotates, so the user can't tell).
2. Extreme aspect ratios — resizing must clamp both dimensions to >= 1 px so a
   10000x2 strip can't crash resize with a zero dimension (which also aborted
   the rest of a batch).

These exercise ``image_to_data_uri`` only, which is pure PIL — no llama_cpp,
no Qt, no network.
"""

import base64
import io

from PIL import Image

from engine.inference import image_to_data_uri


def _decode_data_uri(uri: str) -> Image.Image:
    assert uri.startswith("data:image/png;base64,")
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
