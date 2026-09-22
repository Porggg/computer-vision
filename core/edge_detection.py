"""Canny style edge detection, built on the operators in filters.py."""

import numpy as np

from core.filters import derivative_of_gaussian

def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Gradient magnitude of a smoothed image, normalized to [0, 1]."""
    I_x, I_y = derivative_of_gaussian(image)

    magnitude = np.hypot(I_x, I_y)
    peak = magnitude.max()
    return magnitude / peak if peak > 1e-12 else np.zeros_like(magnitude)

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

def edge_detection(image: np.ndarray) -> np.ndarray:
    """Edges thinned to one pixel wide: gradient magnitude, then suppression."""
    I_x, I_y = derivative_of_gaussian(image)
    return non_max_suppression(gradient_magnitude(image), I_x, I_y)
