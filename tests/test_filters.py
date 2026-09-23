import numpy as np
import pytest

from core.filters import (
    box_blur,
    box_kernel,
    convolve,
    convolve_channels,
    derivative_of_gaussian,
    D_x,
    D_y,
    gaussian_blur,
    derivative_of_gaussian_kernels,
    gaussian_kernel,
    gradient_magnitude,
    non_max_suppression,
    x_derivative,
    y_derivative,
    threshold,
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


# --- box_kernel / box_blur --------------------------------------------------

def test_box_kernel_rejects_even_size():
    with pytest.raises(ValueError):
        box_kernel(4)


@pytest.mark.parametrize("size", [3, 5, 9])
def test_box_kernel_is_uniform_and_normalized(size):
    kernel = box_kernel(size)
    assert kernel.shape == (size, size)
    assert kernel.sum() == pytest.approx(1.0)
    assert len(np.unique(kernel)) == 1              # every weight identical
    assert kernel[0, 0] == pytest.approx(1 / size**2)


def test_box_blur_keeps_a_constant_image():
    assert np.allclose(box_blur(np.full((8, 8, 3), 0.7), 5), 0.7)


def test_box_blur_averages_its_window():
    # an impulse spreads into an even square of 1/size^2
    image = np.zeros((9, 9))
    image[4, 4] = 1.0
    blurred = box_blur(image, 3)
    assert np.allclose(blurred[3:6, 3:6], 1 / 9)
    assert np.allclose(blurred[:2, :], 0.0)


def test_box_blur_keeps_the_color_channels():
    assert box_blur(np.zeros((8, 9, 3)), 5).shape == (8, 9, 3)
    assert box_blur(np.zeros((8, 9)), 5).shape == (8, 9)


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


# --- x_derivative / y_derivative --------------------------------------------

def test_x_derivative_reacts_to_vertical_edges_only():
    # the "no response" half is checked on the raw convolution: a flat result
    # has no peak for to_display to scale against
    image = np.zeros((8, 8))
    image[:, 4:] = 1
    assert np.abs(x_derivative(image) - 0.5).max() > 0
    assert np.allclose(convolve_channels(image, D_y), 0.0)


def test_y_derivative_reacts_to_horizontal_edges_only():
    image = np.zeros((8, 8))
    image[4:, :] = 1
    assert np.abs(y_derivative(image) - 0.5).max() > 0
    assert np.allclose(convolve_channels(image, D_x), 0.0)


def test_derivative_sign_follows_the_edge_direction():
    # convolve flips the kernel, so SOBEL_X measures left minus right:
    # a dark-to-bright edge is negative, and mirroring it flips the sign.
    image = np.zeros((8, 8))
    image[:, 4:] = 1  # dark on the left, bright on the right
    assert x_derivative(image)[4, 3] < 0.5          # darker than mid gray
    assert x_derivative(image[:, ::-1])[4, 4] > 0.5  # brighter


def test_derivatives_keep_the_color_channels():
    # not a flat image: to_display has no peak to scale a flat one against
    image = np.random.default_rng(0).random((8, 9, 3))
    assert x_derivative(image).shape == (8, 9, 3)
    assert y_derivative(image).shape == (8, 9, 3)


# --- derivative_of_gaussian_kernels --------------------------------------------

def test_derivative_of_gaussian_kernels_reject_even_size():
    with pytest.raises(ValueError):
        derivative_of_gaussian_kernels(8)


@pytest.mark.parametrize("size, sigma", [(3, 1.0), (5, 1.0), (9, 1.4)])
def test_derivative_of_gaussian_kernels_properties(size, sigma):
    dog_x, dog_y = derivative_of_gaussian_kernels(size, sigma)
    # D_x is 3 wide, so the full support of the combined kernel is size + 2
    assert dog_x.shape == (size + 2, size + 2)
    assert dog_y.shape == (size + 2, size + 2)
    assert dog_x.sum() == pytest.approx(0.0)        # no response on a flat image
    assert np.allclose(dog_x, -dog_x[:, ::-1])      # antisymmetric across x
    assert np.allclose(dog_y, dog_x.T)              # D_y is D_x.T and G is symmetric


def test_derivative_of_gaussian_kernels_keep_the_full_support():
    # without the pad, convolve would clip the result back to the 3x3 of D_x
    dog_x, _ = derivative_of_gaussian_kernels(9, 1.4)
    assert dog_x.shape == (11, 11)
    assert np.abs(dog_x[0]).max() > 0     # the outer ring survived
    assert np.abs(dog_x[:, 0]).max() > 0


def test_derivative_of_gaussian_kernels_equal_blurring_then_deriving():
    # the whole point of the construction: (D_x * G) * I == D_x * (G * I).
    # Only the interior, since the two step version edge pads twice.
    image = np.random.default_rng(0).random((40, 40))
    dog_x, dog_y = derivative_of_gaussian_kernels(9, 1.4)
    smoothed = gaussian_blur(image, 9, 1.4)
    # against the raw convolution, since x_derivative now display maps its result
    assert np.allclose(convolve(image, dog_x)[7:-7, 7:-7], convolve(smoothed, D_x)[7:-7, 7:-7])
    assert np.allclose(convolve(image, dog_y)[7:-7, 7:-7], convolve(smoothed, D_y)[7:-7, 7:-7])


def test_derivative_of_gaussian_kernels_follow_the_d_x_sign_convention():
    # built from D_x, so the sign matches x_derivative: convolve flips the
    # kernel, which makes it measure left minus right
    image = np.zeros((24, 24))
    image[:, 12:] = 1  # dark on the left, bright on the right
    dog_x, _ = derivative_of_gaussian_kernels(9, 1.4)
    assert convolve(image, dog_x)[12, 11] < 0
    assert x_derivative(image)[12, 11] < 0.5  # display mapped, so below mid gray


# --- threshold --------------------------------------------------------------

def test_threshold_keeps_what_is_above_and_zeroes_the_rest():
    image = np.array([[0.1, 0.4, 0.6, 0.9]])
    assert np.allclose(threshold(image, 0.5), [[0.0, 0.0, 0.6, 0.9]])


def test_threshold_keeps_the_surviving_values_unchanged():
    # it selects, it does not rescale or binarize
    image = np.random.default_rng(0).random((16, 16))
    kept = threshold(image, 0.5)
    survivors = kept != 0
    assert np.allclose(kept[survivors], image[survivors])
    assert np.array_equal(survivors, image > 0.5)


def test_threshold_is_strict_on_the_level_itself():
    # a value exactly on the level is dropped, since the test is >
    assert np.allclose(threshold(np.array([[0.5]]), 0.5), [[0.0]])


def test_threshold_below_the_minimum_changes_nothing():
    image = np.random.default_rng(0).random((8, 8))
    assert np.allclose(threshold(image, -1.0), image)


def test_threshold_above_the_maximum_zeroes_everything():
    image = np.random.default_rng(0).random((8, 8))
    assert np.allclose(threshold(image, 1.0), 0.0)


def test_threshold_drops_negative_values_with_a_level_of_zero():
    # a signed map, like a harris response, keeps only its positive side
    image = np.array([[-0.8, -0.1, 0.0, 0.3]])
    assert np.allclose(threshold(image, 0.0), [[0.0, 0.0, 0.0, 0.3]])


def test_threshold_does_not_modify_its_input():
    image = np.random.default_rng(0).random((8, 8))
    before = image.copy()
    threshold(image, 0.5)
    assert np.allclose(image, before)


def test_threshold_keeps_the_shape_and_the_dtype():
    image = np.random.default_rng(0).random((5, 7))
    result = threshold(image, 0.5)
    assert result.shape == (5, 7)
    assert result.dtype == np.float64


# --- gradient_magnitude -----------------------------------------

def test_gradient_magnitude_zero_on_constant_image():
    assert np.allclose(gradient_magnitude(np.full((8, 8), 0.5))[0], 0.0)


def test_gradient_magnitude_returns_the_derivatives_it_used():
    image = np.random.default_rng(0).random((16, 16))
    magnitude, I_x, I_y = gradient_magnitude(image)
    expected_x, expected_y = derivative_of_gaussian(image)
    assert np.allclose(I_x, expected_x)
    assert np.allclose(I_y, expected_y)
    assert np.allclose(magnitude * np.hypot(I_x, I_y).max(), np.hypot(I_x, I_y))


def test_gradient_magnitude_is_never_negative():
    image = np.random.default_rng(0).random((16, 16))
    assert (gradient_magnitude(image)[0] >= 0).all()


def test_gradient_magnitude_is_the_hypotenuse_of_the_dog_responses():
    image = np.random.default_rng(0).random((16, 16))
    dog_x, dog_y = derivative_of_gaussian_kernels(9, 1.4)
    gray = to_grayscale(image)
    expected = np.hypot(convolve(gray, dog_x), convolve(gray, dog_y))
    assert np.allclose(gradient_magnitude(image)[0], expected / expected.max())


def test_gradient_magnitude_is_normalized_to_one():
    image = np.random.default_rng(0).random((16, 16))
    assert gradient_magnitude(image)[0].max() == pytest.approx(1.0)


def test_gradient_magnitude_detects_a_vertical_edge():
    image = np.zeros((24, 24))
    image[:, 12:] = 1
    edges = gradient_magnitude(image)[0]
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
    magnitude = gradient_magnitude(image)[0]
    assert np.allclose(magnitude[:, 11:13], magnitude.max())
    assert np.allclose(magnitude[:, :7], 0.0)
    assert np.allclose(magnitude[:, 17:], 0.0)


def test_gradient_magnitude_ignores_the_edge_orientation():
    # gx and gy swap roles under a transpose, so the magnitude only transposes.
    vertical = np.zeros((24, 24))
    vertical[:, 12:] = 1
    horizontal = vertical.T
    assert np.allclose(
        gradient_magnitude(horizontal)[0],
        gradient_magnitude(vertical)[0].T,
    )


def test_gradient_magnitude_does_not_depend_on_the_sign_of_the_edge():
    # a dark-to-bright edge and its bright-to-dark mirror have the same strength
    image = np.zeros((24, 24))
    image[:, 12:] = 1
    assert np.allclose(
        gradient_magnitude(image)[0],
        gradient_magnitude(1 - image)[0],
    )


def test_gradient_magnitude_collapses_color_to_one_channel():
    # to_grayscale runs first, so a color image gives a single edge map
    assert gradient_magnitude(np.zeros((8, 9, 3)))[0].shape == (8, 9)


# --- non_max_suppression ----------------------------------------------------

def _suppressed(image):
    """non_max_suppression on its own inputs, without going through the pipeline."""
    magnitude, I_x, I_y = gradient_magnitude(image)
    return non_max_suppression(magnitude, I_x, I_y)


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
    magnitude = gradient_magnitude(image)[0]
    thinned = _suppressed(image)
    kept = np.flatnonzero(thinned[12])
    assert magnitude[12, kept[0]] == pytest.approx(magnitude[12].max())


def test_nms_only_keeps_or_zeroes_never_changes_a_value():
    image = np.random.default_rng(0).random((32, 32))
    magnitude = gradient_magnitude(image)[0]
    thinned = _suppressed(image)
    survivors = thinned != 0
    assert np.allclose(thinned[survivors], magnitude[survivors])  # values untouched
    assert np.all(thinned[~survivors] == 0.0)                     # the rest are zero
    assert np.all(thinned <= magnitude)


def test_nms_removes_pixels():
    image = np.random.default_rng(0).random((32, 32))
    assert np.count_nonzero(_suppressed(image)) < np.count_nonzero(gradient_magnitude(image)[0])


def test_nms_zero_on_constant_image():
    assert np.allclose(_suppressed(np.full((24, 24), 0.5)), 0.0)


def test_nms_is_idempotent():
    # a survivor now has zeroed neighbours so it survives again, and a zero can
    # never beat a neighbour with the strict >
    image = np.random.default_rng(0).random((32, 32))
    magnitude, I_x, I_y = gradient_magnitude(image)
    once = non_max_suppression(magnitude, I_x, I_y)
    twice = non_max_suppression(once, I_x, I_y)
    assert np.allclose(once, twice)


def test_nms_suppresses_a_border_pixel_pointing_outside():
    # the inf padding means a comparison that reaches off the image always loses
    image = _edge_image()
    magnitude = gradient_magnitude(image)[0]
    I_x, I_y = derivative_of_gaussian(image)
    thinned = non_max_suppression(magnitude, I_x, I_y)
    horizontal = np.abs(I_y) < np.abs(I_x)      # gradient points left or right
    assert np.all(thinned[:, 0][horizontal[:, 0]] == 0.0)
    assert np.all(thinned[:, -1][horizontal[:, -1]] == 0.0)


def test_nms_keeps_the_shape_and_the_contract():
    thinned = _suppressed(np.random.default_rng(0).random((17, 23)))
    assert thinned.shape == (17, 23)
    assert thinned.min() >= 0.0 and thinned.max() <= 1.0
