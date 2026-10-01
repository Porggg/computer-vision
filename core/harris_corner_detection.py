import numpy as np

from core.filters import gaussian_blur, derivative_of_gaussian, threshold, non_max_suppression_neighborhood

def harris_corner_detection_heatmap(image: np.ndarray, k: float = 0.05) -> np.ndarray:
    '''return a grayscale [-1, 1] image with -1 = edge, 0 = flat and 1 = corner"
    pass it in to_heatmap to make the RGB image corresponding'''
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

def final_image(threshold_image: np.ndarray, image: np.ndarray) -> np.ndarray:
    rgb = np.repeat(image[..., None], 3, axis=-1) if image.ndim == 2 else image.copy()

    return np.where(threshold_image[..., None] > 0, [1.0, 0.0, 0.0], rgb)

def haaris_corner_detection(image: np.ndarray, k: float = 0.05) -> np.ndarray:
    return threshold(non_max_suppression_neighborhood(harris_corner_detection_heatmap(image, k)), 0.6)

def harris_corner_overlay(image: np.ndarray, k: float = 0.05) -> np.ndarray:
    return final_image(haaris_corner_detection(image, k), image)
