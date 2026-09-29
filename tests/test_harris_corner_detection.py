import numpy as np
import pytest

from core.filters import non_max_suppression_neighborhood, threshold
from core.harris_corner_detection import harris_corner_detection_heatmap, haaris_corner_detection


def _square(size=48, start=12, stop=36):
    """A bright square on black: four corners, four edges and flat regions.

    Every landmark sits 12 pixels from the others, further than the DoG (radius 5)
    and the structure tensor blur (radius 4) reach together, so none of them leaks
    into another.
    """
    image = np.zeros((size, size))
    image[start:stop, start:stop] = 1.0
    return image


CORNERS = [(12, 12), (12, 35), (35, 12), (35, 35)]
MID_EDGES = [(24, 12), (24, 35), (12, 24), (35, 24)]
FLAT = [(24, 24), (2, 2), (45, 45)]


# --- harris_corner_detection_heatmap ----------------------------------------

def test_harris_response_is_positive_on_the_corners():
    response = harris_corner_detection_heatmap(_square())
    for r, c in CORNERS:
        assert response[r - 2:r + 2, c - 2:c + 2].max() > 0.5


def test_harris_response_is_negative_on_the_edges():
    # one derivative is zero on a straight edge, so det = 0 and R = -k trace^2
    response = harris_corner_detection_heatmap(_square())
    for r, c in MID_EDGES:
        assert response[r, c] < 0


def test_harris_response_is_zero_on_flat_regions():
    response = harris_corner_detection_heatmap(_square())
    for r, c in FLAT:
        assert response[r, c] == pytest.approx(0.0)


def test_harris_response_follows_a_transpose():
    # I_x and I_y swap roles, which swaps M_11 and M_22: det and trace do not move
    image = np.random.default_rng(0).random((32, 32))
    assert np.allclose(
        harris_corner_detection_heatmap(image.T),
        harris_corner_detection_heatmap(image).T,
    )


def test_harris_response_does_not_depend_on_the_contrast_sign():
    # inverting the image flips I_x and I_y, and M only holds their products
    image = _square()
    assert np.allclose(
        harris_corner_detection_heatmap(1 - image),
        harris_corner_detection_heatmap(image),
    )


def test_harris_response_stays_in_the_signed_unit_range():
    response = harris_corner_detection_heatmap(np.random.default_rng(0).random((32, 32)))
    assert response.min() >= -1.0 and response.max() <= 1.0
    assert response.min() < 0 < response.max()  # both signs survive the scaling


def test_harris_response_accepts_color_and_returns_one_channel():
    response = harris_corner_detection_heatmap(np.random.default_rng(0).random((20, 24, 3)))
    assert response.shape == (20, 24)
    assert response.dtype == np.float64


# --- haaris_corner_detection ------------------------------------------------

def test_harris_detection_is_the_thresholded_response_then_the_suppression():
    image = np.random.default_rng(0).random((32, 32))
    expected = non_max_suppression_neighborhood(threshold(harris_corner_detection_heatmap(image), 0.6))
    assert np.allclose(haaris_corner_detection(image), expected)


def test_harris_detection_keeps_fewer_pixels_than_the_threshold_alone():
    image = _square()
    thresholded = threshold(harris_corner_detection_heatmap(image), 0.6)
    assert np.count_nonzero(haaris_corner_detection(image)) < np.count_nonzero(thresholded)


def test_harris_detection_survivors_are_local_maxima():
    corners = haaris_corner_detection(np.random.default_rng(0).random((32, 32)))
    assert np.array_equal(non_max_suppression_neighborhood(corners), corners)


def test_harris_detection_survivors_sit_on_the_corners():
    rows, cols = np.nonzero(haaris_corner_detection(_square()))
    for r, c in zip(rows, cols):
        distance = min(max(abs(r - cr), abs(c - cc)) for cr, cc in CORNERS)
        assert distance <= 4


def test_harris_detection_is_either_zero_or_above_0_6():
    corners = haaris_corner_detection(np.random.default_rng(0).random((32, 32)))
    kept = corners[corners != 0]
    assert kept.size > 0
    assert (kept > 0.6).all()


def test_harris_detection_keeps_the_corners_only():
    # a wider window than for the response: the suppression keeps only the peak,
    # which is not necessarily on the pixel next to the corner
    corners = haaris_corner_detection(_square())
    for r, c in CORNERS:
        assert corners[r - 3:r + 4, c - 3:c + 4].max() > 0.6
    for r, c in MID_EDGES + FLAT:
        assert corners[r, c] == 0.0


def test_harris_detection_respects_the_image_contract():
    corners = haaris_corner_detection(np.random.default_rng(0).random((17, 23)))
    assert corners.shape == (17, 23)
    assert corners.dtype == np.float64
    assert corners.min() >= 0.0 and corners.max() <= 1.0
