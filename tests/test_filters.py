import numpy as np
import pytest

from core.filters import (
    convolve,
    gaussian_blur,
    gaussian_kernel,
    sobel,
    x_derivative,
    y_derivative,
    to_grayscale,
)


# --- to_grayscale -----------------------------------------------------------

def test_grayscale_white_stays_white():
    assert np.allclose(to_grayscale(np.ones((3, 3, 3))), 1.0)


def test_grayscale_uses_luma_weights():
    red, green, blue = np.eye(3).reshape(3, 1, 1, 3)
    assert to_grayscale(red)[0, 0] == pytest.approx(0.299)
    assert to_grayscale(green)[0, 0] == pytest.approx(0.587)
    assert to_grayscale(blue)[0, 0] == pytest.approx(0.114)


def test_grayscale_passthrough_for_2d():
    image = np.random.default_rng(0).random((4, 4))
    assert to_grayscale(image) is image


# --- convolve ---------------------------------------------------------------

def test_convolve_identity_kernel():
    image = np.random.default_rng(0).random((6, 7))
    kernel = np.zeros((3, 3))
    kernel[1, 1] = 1
    assert np.allclose(convolve(image, kernel), image)


def test_convolve_impulse_returns_kernel():
    # A true convolution of an impulse gives the kernel back unflipped
    # (a correlation would give it rotated by 180 degrees).
    image = np.zeros((5, 5))
    image[2, 2] = 1
    kernel = np.arange(9, dtype=np.float64).reshape(3, 3)
    assert np.allclose(convolve(image, kernel)[1:4, 1:4], kernel)


def test_convolve_keeps_shape():
    assert convolve(np.zeros((10, 12)), np.ones((5, 5))).shape == (10, 12)


def test_convolve_normalized_kernel_keeps_constant_image():
    image = np.full((6, 6), 0.4)
    assert np.allclose(convolve(image, np.ones((3, 3)) / 9), 0.4)


# --- gaussian_kernel --------------------------------------------------------

def test_gaussian_kernel_rejects_even_size():
    with pytest.raises(ValueError):
        gaussian_kernel(10)


@pytest.mark.parametrize("size, sigma", [(3, 0.5), (5, 1.0), (7, 2.0)])
def test_gaussian_kernel_properties(size, sigma):
    kernel = gaussian_kernel(size, sigma)
    assert kernel.shape == (size, size)
    assert kernel.sum() == pytest.approx(1.0)
    assert np.allclose(kernel, kernel.T)
    assert np.allclose(kernel, kernel[::-1, ::-1])
    assert kernel.argmax() == (size * size) // 2  # peak at the center


# --- gaussian_blur ----------------------------------------------------------

def test_blur_keeps_constant_image():
    assert np.allclose(gaussian_blur(np.full((8, 8, 3), 0.7)), 0.7)


def test_blur_reduces_variance():
    image = np.random.default_rng(0).random((32, 32))
    assert gaussian_blur(image).var() < image.var()


def test_blur_keeps_the_color_channels():
    assert gaussian_blur(np.zeros((8, 9, 3))).shape == (8, 9, 3)
    assert gaussian_blur(np.zeros((8, 9))).shape == (8, 9)


def test_blur_does_not_mix_channels():
    image = np.zeros((8, 8, 3))
    image[:, :, 0] = 1.0  # a fully red image stays red
    blurred = gaussian_blur(image)
    assert np.allclose(blurred[:, :, 0], 1.0)
    assert np.allclose(blurred[:, :, 1:], 0.0)


# --- sobel ------------------------------------------------------------------

def test_sobel_zero_on_constant_image():
    assert np.allclose(sobel(np.full((8, 8), 0.5)), 0.0)


def test_sobel_detects_vertical_edge():
    image = np.zeros((8, 8))
    image[:, 4:] = 1
    edges = sobel(image)
    assert edges.max() == pytest.approx(1.0)
    assert np.allclose(edges[:, 3:5], 1.0)  # strongest on the edge
    assert np.allclose(edges[:, :2], 0.0)   # flat regions stay at zero
    assert np.allclose(edges[:, 6:], 0.0)


# --- x_derivative / y_derivative --------------------------------------------

def test_x_derivative_reacts_to_vertical_edges_only():
    image = np.zeros((8, 8))
    image[:, 4:] = 1
    assert np.abs(x_derivative(image)).max() > 0
    assert np.allclose(y_derivative(image), 0.0)


def test_y_derivative_reacts_to_horizontal_edges_only():
    image = np.zeros((8, 8))
    image[4:, :] = 1
    assert np.abs(y_derivative(image)).max() > 0
    assert np.allclose(x_derivative(image), 0.0)


def test_derivative_sign_follows_the_edge_direction():
    # convolve flips the kernel, so SOBEL_X measures left minus right:
    # a dark-to-bright edge is negative, and mirroring it flips the sign.
    image = np.zeros((8, 8))
    image[:, 4:] = 1  # dark on the left, bright on the right
    assert x_derivative(image)[4, 3] < 0
    assert x_derivative(image[:, ::-1])[4, 4] > 0


def test_derivatives_keep_the_color_channels():
    assert x_derivative(np.zeros((8, 9, 3))).shape == (8, 9, 3)
    assert y_derivative(np.zeros((8, 9, 3))).shape == (8, 9, 3)
