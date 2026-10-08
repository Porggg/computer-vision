import numpy as np

from core.filters import gaussian_blur, to_grayscale, downsample, non_max_suppression_neighborhood_3d
from core.image_io import draw_circles

def blob_detection(image: np.ndarray) -> np.ndarray:
    """return the list of (x, y, sigma) blobs in the image"""
    image = to_grayscale(image)

    octave_nb = 4
    stack_size = 5
    sigma = 1.6
    k = np.sqrt(2)
    sigma_stack = np.array([k**i * sigma for i in range(0, stack_size)])
    size = int(2*np.ceil(5*k*sigma)+1)

    downsample_images = [image]
    for i in range(octave_nb-1):
        downsample_images.append(downsample(downsample_images[-1]))

    octaves = []
    for j in range(octave_nb):
        stack = []
        for i in range(stack_size):
            stack.append(gaussian_blur(downsample_images[j], size, sigma_stack[i]))
        octaves.append(np.stack(stack))

    possible_points = []
    for i in range(octave_nb):
        dogs = octaves[i][1:] - octaves[i][:-1]
        maxima = non_max_suppression_neighborhood_3d(np.abs(dogs), np.inf)
        coords = np.argwhere(maxima > 0.1 * maxima.max())  # columns (s, y, x)

        sigmas = sigma_stack[coords[:, 0]]
        possible_points.append(np.column_stack((coords[:, 2], coords[:, 1], sigmas)) * 2**i)

    return np.concatenate(possible_points)

def blob_overlay(image: np.ndarray) -> np.ndarray:
    """the image with every blob circled in red, radius = sqrt(2) * sigma"""
    blobs = blob_detection(image)
    circles = np.column_stack((blobs[:, 0], blobs[:, 1], np.sqrt(2) * blobs[:, 2]))
    return draw_circles(image, circles)
