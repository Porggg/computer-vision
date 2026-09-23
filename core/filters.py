"""Basic image filters, implemented by hand with numpy."""

import numpy as np

from core.image_io import to_display

### CONVOLUTIONS

def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D convolution of a grayscale image, with edge padding."""
    kh, kw = kernel.shape
    padded = np.pad(image, ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel[::-1, ::-1])

def convolve_channels(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve a grayscale image, or each channel of a color image on its own."""
    if image.ndim == 2:
        return convolve(image, kernel)
    channels = [convolve(image[:, :, c], kernel) for c in range(image.shape[2])]
    return np.stack(channels, axis=-1)

### KERNELS 

def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Normalized 2D gaussian kernel. The size must be odd to have a center.
    Notation: G
    """
    if size % 2 == 0:
        raise ValueError(f"kernel size must be odd, got {size}")
    ax = np.arange(size) - size // 2
    g = np.exp(-(ax**2) / (2 * sigma**2))
    kernel = np.outer(g, g)
    return kernel / kernel.sum()

def box_kernel(size: int = 5) -> np.ndarray:
    """Uniform window, every weight the same and the whole thing summing to 1."""
    if size % 2 == 0:
        raise ValueError(f"kernel size must be odd, got {size}")
    return np.full((size, size), 1.0 / (size * size))

# Sobel operators
D_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64) / 8
D_y = D_x.T

def derivative_of_gaussian_kernels(size: int = 9, sigma: float = 1.4) -> tuple[np.ndarray, np.ndarray]:
    """D_x and D_y convolved with a gaussian, as a pair."""
    smoothing = gaussian_kernel(size, sigma)
    pad = size // 2
    return convolve(np.pad(D_x, pad), smoothing), convolve(np.pad(D_y, pad), smoothing)

### FILTERS

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale."""
    if image.ndim == 2:
        return image
    return image @ np.array([0.299, 0.587, 0.114])
    # note : those weigths are choose because the eyes are not sensitive to every colors equally
    # so we cant simply take (R+G+B)/3

def box_blur(image: np.ndarray, size: int = 5) -> np.ndarray:
    """Average over a square window. Colors are kept: one channel at a time."""
    return convolve_channels(image, box_kernel(size))

def gaussian_blur(image: np.ndarray, size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Gaussian blur. Colors are kept: each channel is blurred on its own.
    Notation: I*G
    """
    return convolve_channels(image, gaussian_kernel(size, sigma))

def x_derivative(image: np.ndarray) -> np.ndarray:
    """Horizontal derivative gx: it reacts to vertical edges.
    Notation: I * D_x

    The raw derivative is signed, so it goes through to_display before being
    returned: 0 becomes mid gray, negative is darker and positive is brighter.
    """
    return to_display(convolve_channels(image, D_x))

def y_derivative(image: np.ndarray) -> np.ndarray:
    """Vertical derivative gy: it reacts to horizontal edges.
    Notation: I*D_y

    Display mapped like x_derivative, so 0 sits at mid gray.
    """
    return to_display(convolve_channels(image, D_y))

def gradient_magnitude(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gradient magnitude normalized to [0, 1], with the derivatives behind it.

    I_x and I_y come back with it so that a caller does not have to convolve a
    second time to get them: non_max_suppression needs exactly that pair.
    """
    I_x, I_y = derivative_of_gaussian(image)

    magnitude = np.hypot(I_x, I_y)
    peak = magnitude.max()
    normalized = magnitude / peak if peak > 1e-12 else np.zeros_like(magnitude)
    return normalized, I_x, I_y

def derivative_of_gaussian(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """I_x and I_y of the image, as a pair.
    Notation: I_x = D_x * (G * I) = (D_x * G) * I"""
    gray = to_grayscale(image)
    dog_x, dog_y = derivative_of_gaussian_kernels(9, 1.4)
    return convolve(gray, dog_x), convolve(gray, dog_y)

def threshold(image: np.ndarray, level: float) -> np.ndarray:
    """Keep the values strictly above level, set the rest to 0."""
    return np.where(image > level, image, 0.0)

def non_max_suppression(magnitude: np.ndarray, I_x: np.ndarray, I_y: np.ndarray) -> np.ndarray:
    """check is a pixel is the max of his neighboors in the grad direction"""

    theta = np.arctan2(I_y, I_x)
    theta_positive = theta % (2 * np.pi)
    angle = np.pi / 4
    angle_index = np.round(theta_positive / angle).astype(int) % 4

    keep = np.zeros(magnitude.shape, dtype=bool)

    # windows is a list of all possible 3x3 windows
    # w[0,0] being the 3x3 window centered on pixel 0,0 (with padding)
    padded = np.pad(magnitude, 1, mode="constant", constant_values=np.inf)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))

    # which neighbours to visit for each angle indexes
    NEIGHBOURS = {
        0: (0, 1), 
        1: (1, 1), 
        2: (1, 0), 
        3: (1, -1)
    }

    # instead of exploring the 4 angle state on every pixel,
    # we compare every pixel for every state
    for case, (dr, dc) in NEIGHBOURS.items():
        # this is the image translated by (dr, dc)
        pos_neighbours  = windows[..., 1 + dr, 1 + dc]   # neighbour along the gradient
        neg_neighbours = windows[..., 1 - dr, 1 - dc]    # and the opposite one

        keep |= (angle_index == case) & (magnitude >= neg_neighbours) & (magnitude > pos_neighbours)

    return magnitude * keep
