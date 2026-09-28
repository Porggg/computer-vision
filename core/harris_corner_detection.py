import numpy as np

from core.filters import gaussian_blur, derivative_of_gaussian, threshold
from core.image_io import to_heatmap

def harris_corner_detection_heatmap(image: np.ndarray, k: float = 0.05) -> np.ndarray:
    """return a grayscale [-1, 1] image with -1 = edge, 0 = flat and 1 = corner"
    pass it in to_heatmap to make the RGB image corresponding"""
    I_x, I_y = derivative_of_gaussian(image)
    I_xx, I_xy, I_yy = I_x*I_x, I_y*I_x, I_y*I_y
    
    M_11 = gaussian_blur(I_xx, 9, 1.5)
    M_22 = gaussian_blur(I_yy, 9, 1.5)
    M_12 = gaussian_blur(I_xy, 9, 1.5)

    det = M_11*M_22 - M_12**2
    trace = M_11 + M_22
    R = det - k * trace**2

    scale = np.percentile(np.abs(R), 99.5)
    signed = np.clip(R / scale, -1.0, 1.0)
    return np.sign(signed) * np.sqrt(np.abs(signed)) # the sqrt augment the contrast

def haaris_corner_detection(image: np.ndarray, k: float = 0.05) -> np.ndarray:
    return threshold(harris_corner_detection_heatmap(image, k), 0.6)