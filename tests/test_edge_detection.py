import numpy as np
import pytest

from core.edge_detection import edge_detection, gradient_magnitude, non_max_suppression
from core.filters import (
    convolve,
    derivative_of_gaussian,
    derivative_of_gaussian_kernels,
    to_grayscale,
)


def test_edge_detection_is_the_magnitude_then_the_suppression():
    image = np.random.default_rng(0).random((32, 32))
    I_x, I_y = derivative_of_gaussian(image)
    expected = non_max_suppression(gradient_magnitude(image), I_x, I_y)
    assert np.allclose(edge_detection(image), expected)


def test_edge_detection_thins_a_vertical_edge_to_one_pixel_per_row():
    image = np.zeros((24, 24))
    image[:, 12:] = 1
    edges = edge_detection(image)
    assert np.count_nonzero(edges) == 24          # one per row, every row
    assert np.count_nonzero(edges[12]) == 1


def test_edge_detection_thins_a_horizontal_edge_to_one_pixel_per_column():
    image = np.zeros((24, 24))
    image[12:, :] = 1
    edges = edge_detection(image)
    assert np.count_nonzero(edges) == 24
    assert np.count_nonzero(edges[:, 12]) == 1


def test_edge_detection_zero_on_constant_image():
    assert np.allclose(edge_detection(np.full((24, 24), 0.5)), 0.0)


def test_edge_detection_accepts_color_and_returns_one_channel():
    edges = edge_detection(np.random.default_rng(0).random((20, 24, 3)))
    assert edges.shape == (20, 24)


def test_edge_detection_respects_the_image_contract():
    edges = edge_detection(np.random.default_rng(0).random((17, 23)))
    assert edges.shape == (17, 23)
    assert edges.dtype == np.float64
    assert edges.min() >= 0.0 and edges.max() <= 1.0


# --- gradient_magnitude -----------------------------------------

def test_gradient_magnitude_zero_on_constant_image():
    assert np.allclose(gradient_magnitude(np.full((8, 8), 0.5)), 0.0)


def test_gradient_magnitude_is_never_negative():
    image = np.random.default_rng(0).random((16, 16))
    assert (gradient_magnitude(image) >= 0).all()


def test_gradient_magnitude_is_the_hypotenuse_of_the_dog_responses():
    image = np.random.default_rng(0).random((16, 16))
    dog_x, dog_y = derivative_of_gaussian_kernels(9, 1.4)
    gray = to_grayscale(image)
    expected = np.hypot(convolve(gray, dog_x), convolve(gray, dog_y))
    assert np.allclose(gradient_magnitude(image), expected / expected.max())


def test_gradient_magnitude_is_normalized_to_one():
    image = np.random.default_rng(0).random((16, 16))
    assert gradient_magnitude(image).max() == pytest.approx(1.0)


def test_gradient_magnitude_detects_a_vertical_edge():
    image = np.zeros((24, 24))
    image[:, 12:] = 1
    edges = gradient_magnitude(image)
    assert edges.max() == pytest.approx(1.0)
    assert np.allclose(edges[:, 11:13], 1.0)  # strongest on the edge
    assert np.allclose(edges[:, :7], 0.0)     # flat regions stay at zero
    assert np.allclose(edges[:, 17:], 0.0)
    # the gaussian spreads the response over the kernel width, symmetrically
    assert np.allclose(edges[12, 7:12], edges[12, 12:17][::-1])


def test_gradient_magnitude_peaks_on_an_edge():
    # the image has to be wider than the kernel for the flat parts to stay flat
    image = np.zeros((24, 24))
    image[:, 12:] = 1
    magnitude = gradient_magnitude(image)
    assert np.allclose(magnitude[:, 11:13], magnitude.max())
    assert np.allclose(magnitude[:, :7], 0.0)
    assert np.allclose(magnitude[:, 17:], 0.0)


def test_gradient_magnitude_ignores_the_edge_orientation():
    # gx and gy swap roles under a transpose, so the magnitude only transposes.
    vertical = np.zeros((24, 24))
    vertical[:, 12:] = 1
    horizontal = vertical.T
    assert np.allclose(
        gradient_magnitude(horizontal),
        gradient_magnitude(vertical).T,
    )


def test_gradient_magnitude_does_not_depend_on_the_sign_of_the_edge():
    # a dark-to-bright edge and its bright-to-dark mirror have the same strength
    image = np.zeros((24, 24))
    image[:, 12:] = 1
    assert np.allclose(
        gradient_magnitude(image),
        gradient_magnitude(1 - image),
    )


def test_gradient_magnitude_collapses_color_to_one_channel():
    # to_grayscale runs first, so a color image gives a single edge map
    assert gradient_magnitude(np.zeros((8, 9, 3))).shape == (8, 9)


# --- non_max_suppression ----------------------------------------------------

def _suppressed(image):
    """non_max_suppression on its own inputs, without going through the pipeline."""
    I_x, I_y = derivative_of_gaussian(image)
    return non_max_suppression(gradient_magnitude(image), I_x, I_y)


def _edge_image(vertical=True, size=24):
    image = np.zeros((size, size))
    if vertical:
        image[:, size // 2:] = 1
    else:
        image[size // 2:, :] = 1
    return image


def test_nms_thins_a_vertical_edge_to_one_pixel():
    thinned = _suppressed(_edge_image())
    assert np.count_nonzero(thinned[12]) == 1         # one pixel per row
    assert np.count_nonzero(thinned) == 24            # one per row, all 24 rows


def test_nms_thins_a_horizontal_edge_to_one_pixel():
    thinned = _suppressed(_edge_image(vertical=False))
    assert np.count_nonzero(thinned[:, 12]) == 1
    assert np.count_nonzero(thinned) == 24


def test_nms_keeps_the_peak_of_the_ridge():
    image = _edge_image()
    magnitude = gradient_magnitude(image)
    thinned = _suppressed(image)
    kept = np.flatnonzero(thinned[12])
    assert magnitude[12, kept[0]] == pytest.approx(magnitude[12].max())


def test_nms_only_keeps_or_zeroes_never_changes_a_value():
    image = np.random.default_rng(0).random((32, 32))
    magnitude = gradient_magnitude(image)
    thinned = _suppressed(image)
    survivors = thinned != 0
    assert np.allclose(thinned[survivors], magnitude[survivors])  # values untouched
    assert np.all(thinned[~survivors] == 0.0)                     # the rest are zero
    assert np.all(thinned <= magnitude)


def test_nms_removes_pixels():
    image = np.random.default_rng(0).random((32, 32))
    assert np.count_nonzero(_suppressed(image)) < np.count_nonzero(gradient_magnitude(image))


def test_nms_zero_on_constant_image():
    assert np.allclose(_suppressed(np.full((24, 24), 0.5)), 0.0)


def test_nms_is_idempotent():
    # a survivor now has zeroed neighbours so it survives again, and a zero can
    # never beat a neighbour with the strict >
    image = np.random.default_rng(0).random((32, 32))
    I_x, I_y = derivative_of_gaussian(image)
    once = non_max_suppression(gradient_magnitude(image), I_x, I_y)
    twice = non_max_suppression(once, I_x, I_y)
    assert np.allclose(once, twice)


def test_nms_suppresses_a_border_pixel_pointing_outside():
    # the inf padding means a comparison that reaches off the image always loses
    image = _edge_image()
    magnitude = gradient_magnitude(image)
    I_x, I_y = derivative_of_gaussian(image)
    thinned = non_max_suppression(magnitude, I_x, I_y)
    horizontal = np.abs(I_y) < np.abs(I_x)      # gradient points left or right
    assert np.all(thinned[:, 0][horizontal[:, 0]] == 0.0)
    assert np.all(thinned[:, -1][horizontal[:, -1]] == 0.0)


def test_nms_keeps_the_shape_and_the_contract():
    thinned = _suppressed(np.random.default_rng(0).random((17, 23)))
    assert thinned.shape == (17, 23)
    assert thinned.min() >= 0.0 and thinned.max() <= 1.0
