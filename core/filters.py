"""Basic image filters, implemented by hand with numpy."""

import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale."""
    if image.ndim == 2:
        return image
    return image @ np.array([0.299, 0.587, 0.114])
    # note : those weigths are choose because the eyes are not sensitive to every colors equally
    # so we cant simply take (R+G+B)/3


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D convolution of a grayscale image, with edge padding."""
    kh, kw = kernel.shape
    padded = np.pad(image, ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel[::-1, ::-1])


def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Normalized 2D gaussian kernel. The size must be odd to have a center."""
    if size % 2 == 0:
        raise ValueError(f"kernel size must be odd, got {size}")
    ax = np.arange(size) - size // 2
    g = np.exp(-(ax**2) / (2 * sigma**2))
    kernel = np.outer(g, g)
    return kernel / kernel.sum()


def convolve_channels(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve a grayscale image, or each channel of a color image on its own."""
    if image.ndim == 2:
        return convolve(image, kernel)
    channels = [convolve(image[:, :, c], kernel) for c in range(image.shape[2])]
    return np.stack(channels, axis=-1)


def gaussian_blur(image: np.ndarray, size: int = 11, sigma: float = 1.0) -> np.ndarray:
    """Gaussian blur. Colors are kept: each channel is blurred on its own."""
    return convolve_channels(image, gaussian_kernel(size, sigma))


# Sobel operators
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64) / 8
SOBEL_Y = SOBEL_X.T


def x_derivative(image: np.ndarray) -> np.ndarray:
    """Horizontal derivative gx: it reacts to vertical edges. Values are signed."""
    return convolve_channels(image, SOBEL_X)

def y_derivative(image: np.ndarray) -> np.ndarray:
    """Vertical derivative gy: it reacts to horizontal edges. Values are signed."""
    return convolve_channels(image, SOBEL_Y)

def sobel(image: np.ndarray) -> np.ndarray:
    """Gradient magnitude using Sobel operators, normalized to [0, 1]."""
    gray = to_grayscale(image)
    magnitude = np.hypot(x_derivative(gray), y_derivative(gray))
    return magnitude / magnitude.max() if magnitude.max() > 0 else magnitude
