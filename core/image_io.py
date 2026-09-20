"""Load and save images as numpy arrays."""

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path, gray: bool = False) -> np.ndarray:
    """Load an image as a float array in [0, 1] (H, W) or (H, W, 3)."""
    img = Image.open(path).convert("L" if gray else "RGB")
    return np.asarray(img, dtype=np.float64) / 255.0


def save_image(image: np.ndarray, path: str | Path) -> None:
    """Save a float image in [0, 1] to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (np.clip(image, 0.0, 1.0) * 255).round().astype(np.uint8)
    Image.fromarray(data).save(path)


def to_display(image: np.ndarray) -> np.ndarray:
    """Map a signed image to [0, 1] so it can be saved, with 0 shown as mid gray.

    Derivatives are negative on one side of an edge and positive on the other,
    so 0 has to sit in the middle of the scale instead of at black.
    """
    peak = np.abs(image).max()
    if peak == 0:
        return np.full_like(image, 0.5)
    return image / (2 * peak) + 0.5
