"""Run an algorithm on an image from input/ and write the result to output/."""

import argparse
from functools import partial
from pathlib import Path

from core.filters import (
    gaussian_blur,
    gradient_magnitude,
    to_grayscale,
    x_derivative,
    y_derivative,
)
from core.edge_detection import edge_detection
from core.harris_corner_detection import harris_corner_detection_heatmap
from core.image_io import load_image, save_image

ALGORITHMS = {
    "grayscale": to_grayscale,
    "blur": gaussian_blur,
    "gradient_magnitude": lambda image: gradient_magnitude(image)[0],
    "edge_detection": edge_detection,
    "harris_heatmap": harris_corner_detection_heatmap,
    "dx": x_derivative,
    "dy": y_derivative,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="path to the input image")
    parser.add_argument("-a", "--algo", choices=ALGORITHMS, default="grayscale")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "-k",
        type=float,
        default=0.05,
        help="harris sensitivity, usually 0.04 to 0.06 (harris only)",
    )
    args = parser.parse_args()

    algorithm = ALGORITHMS[args.algo]
    suffix = ""
    if args.algo.startswith("harris"):
        algorithm = partial(algorithm, k=args.k)
        suffix = f"_k{args.k:g}"  # so two runs with different k do not collide

    image = load_image(args.image)
    result = algorithm(image)

    out_path = args.output_dir / f"{args.image.stem}_{args.algo}{suffix}.png"
    save_image(result, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
