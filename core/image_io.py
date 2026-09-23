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
    """
    peak = np.abs(image).max()
    return image / (2 * peak) + 0.5

_HEAT_NEGATIVE = (0x2a, 0x78, 0xd6)  # blue
_HEAT_NEUTRAL = (0xf0, 0xef, 0xec)  # near white
_HEAT_POSITIVE = (0xd0, 0x3b, 0x3b)  # red


def _to_linear(srgb: np.ndarray) -> np.ndarray:
    """sRGB to linear light, so that blending two colors does not go muddy."""
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)

def _to_srgb(linear: np.ndarray) -> np.ndarray:
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)

def to_heatmap(image: np.ndarray) -> np.ndarray:
    """Color a signed [-1, 1] map: blue at -1, near white at 0, red at +1."""
    signed = np.clip(image, -1.0, 1.0)
    weight = np.abs(signed)[..., None]

    neutral = _to_linear(np.array(_HEAT_NEUTRAL, dtype=np.float64) / 255.0)
    negative = _to_linear(np.array(_HEAT_NEGATIVE, dtype=np.float64) / 255.0)
    positive = _to_linear(np.array(_HEAT_POSITIVE, dtype=np.float64) / 255.0)

    pole = np.where(signed[..., None] < 0, negative, positive)
    return np.clip(_to_srgb(neutral * (1 - weight) + pole * weight), 0.0, 1.0)
