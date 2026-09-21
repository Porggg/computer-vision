"""Run an algorithm on an image from input/ and write the result to output/."""

import argparse
from pathlib import Path

from core.filters import (
    gaussian_blur,
    gradient_magnitude,
    to_grayscale,
    x_derivative,
    y_derivative,
)
from core.edge_detection import edge_detection
from core.image_io import load_image, save_image, to_display

ALGORITHMS = {
    "grayscale": to_grayscale,
    "blur": gaussian_blur,
    "gradient_magnitude": gradient_magnitude,
    "edge_detection": edge_detection,
    "dx": lambda image: to_display(x_derivative(image)),
    "dy": lambda image: to_display(y_derivative(image)),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="path to the input image")
    parser.add_argument("-a", "--algo", choices=ALGORITHMS, default="grayscale")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()

    image = load_image(args.image)
    result = ALGORITHMS[args.algo](image)

    out_path = args.output_dir / f"{args.image.stem}_{args.algo}.png"
    save_image(result, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
