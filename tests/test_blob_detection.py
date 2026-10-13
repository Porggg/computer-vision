import numpy as np
import pytest

from core.blob_detection import blob_detection, blob_overlay


def _disc(shape, center, radius, value=1.0):
    """A bright disc on black, center given as (x, y)."""
    image = np.zeros(shape)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    image[(yy - center[1]) ** 2 + (xx - center[0]) ** 2 <= radius**2] = value
    return image


def _closest(blobs, x, y):
    return blobs[np.argmin(np.hypot(blobs[:, 0] - x, blobs[:, 1] - y))]


def test_blob_detection_returns_x_y_sigma_rows():
    blobs = blob_detection(_disc((64, 64), (32, 32), 6))
    assert blobs.ndim == 2 and blobs.shape[1] == 3
    assert blobs.dtype == np.float64


def test_blob_detection_finds_a_small_disc_at_its_center():
    blobs = blob_detection(_disc((64, 64), (40, 24), 6))
    x, y, _ = _closest(blobs, 40, 24)
    assert (x, y) == (40, 24)


@pytest.mark.parametrize("radius", [6, 20])
def test_blob_detection_sigma_follows_the_radius(radius):
    # a disc of radius r answers the most at sigma = r / sqrt(2); with k = sqrt(2)
    # the scales are coarse, so only check the right ballpark
    blobs = blob_detection(_disc((128, 128), (64, 64), radius))
    _, _, sigma = _closest(blobs, 64, 64)
    assert radius / np.sqrt(2) / 2 <= sigma <= radius / np.sqrt(2) * 2


def test_blob_detection_brings_the_upper_octaves_back_to_the_original_size():
    # a large disc is found in octave 2: without the 2**i its center would be
    # reported 4 times too close to the origin
    blobs = blob_detection(_disc((128, 128), (40, 90), 20))
    biggest = blobs[np.argmax(blobs[:, 2])]
    assert biggest[0] == pytest.approx(40, abs=4)
    assert biggest[1] == pytest.approx(90, abs=4)


def test_blob_detection_finds_nothing_on_a_flat_image():
    assert len(blob_detection(np.full((64, 64), 0.5))) == 0


def test_blob_detection_finds_dark_blobs_too():
    # |DoG| is searched, so a dark disc on white is found like a bright one on black
    bright = blob_detection(_disc((64, 64), (32, 32), 6))
    dark = blob_detection(1.0 - _disc((64, 64), (32, 32), 6))
    x, y, sigma = _closest(dark, 32, 32)
    assert (x, y) == (32, 32)
    assert sigma == _closest(bright, 32, 32)[2]


def test_blob_detection_accepts_a_color_image():
    gray = _disc((64, 64), (32, 32), 6)
    color = np.repeat(gray[..., None], 3, axis=-1)
    assert np.allclose(blob_detection(color), blob_detection(gray))


def test_blob_detection_stays_inside_the_image():
    image = np.random.default_rng(0).random((100, 140))
    blobs = blob_detection(image)
    assert np.all((blobs[:, 0] >= 0) & (blobs[:, 0] < 140))
    assert np.all((blobs[:, 1] >= 0) & (blobs[:, 1] < 100))


def test_blob_overlay_circles_the_disc_in_red():
    out = blob_overlay(_disc((64, 64), (32, 32), 6))
    assert out.shape == (64, 64, 3)
    red = np.all(out == [1.0, 0.0, 0.0], axis=-1)
    assert red.any()
