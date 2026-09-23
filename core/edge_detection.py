import numpy as np

from core.filters import gradient_magnitude, non_max_suppression

def edge_detection(image: np.ndarray) -> np.ndarray:
    """Edges thinned to one pixel wide: gradient magnitude, then suppression."""
    magnitude, I_x, I_y = gradient_magnitude(image)
    return non_max_suppression(magnitude, I_x, I_y)
