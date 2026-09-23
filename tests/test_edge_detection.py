import numpy as np
import pytest

from core.edge_detection import edge_detection
from core.filters import (
    convolve,
    derivative_of_gaussian,
    gradient_magnitude,
    non_max_suppression,
    derivative_of_gaussian_kernels,
    to_grayscale,
)


def test_edge_detection_is_the_magnitude_then_the_suppression():
    image = np.random.default_rng(0).random((32, 32))
    magnitude, I_x, I_y = gradient_magnitude(image)
    expected = non_max_suppression(magnitude, I_x, I_y)
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
