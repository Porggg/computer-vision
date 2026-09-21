"""Canny style edge detection, built on the operators in filters.py."""

import numpy as np

from core.filters import derivative_of_gaussian, gradient_magnitude, non_max_suppression

def edge_detection(image: np.ndarray) -> np.ndarray:
    """Edges thinned to one pixel wide: gradient magnitude, then suppression."""
    I_x, I_y = derivative_of_gaussian(image)
    return non_max_suppression(gradient_magnitude(image), I_x, I_y)
