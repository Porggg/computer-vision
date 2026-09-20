# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project intent

School project: classic computer vision algorithms implemented by hand with
numpy. **Do not reach for OpenCV, scipy.ndimage or skimage** — the point is the
implementation itself. Only numpy and pillow are dependencies.


## Working style

Answer questions with explanations; do not implement the algorithm on the
user's behalf unless asked explicitly. The user writes the maths.

## Commands

```sh
make install                    # create .venv (needs python3.13) + install requirements
make test                       # pytest
make run ALGO=sobel IMAGE=input/pinhole.jpeg
make clean                      # wipe output/ and caches
make fclean                     # clean + delete .venv
```

Run a single test: `.venv/bin/python -m pytest tests/test_filters.py::test_sobel_zero_on_constant_image`

`pytest.ini` sets `pythonpath = .`, so `core` imports resolve without an install.
The Makefile's `PY ?= python3.13` picks the interpreter used to build the venv.

## Architecture

**The image contract.** Every function in `core/` takes and returns a numpy
array of float64 in `[0, 1]`, shaped `(H, W)` for grayscale or `(H, W, 3)` for
RGB. `load_image` and `save_image` in `core/image_io.py` are the only places
that touch the filesystem or uint8; `main.py` is the only caller of both.
Algorithms never open or write files, which is what makes them directly
testable on synthetic arrays.

**Adding an algorithm.** Write the function in a module under `core/`, then
register it in the `ALGORITHMS` dict in `main.py`. The CLI derives the output
name from it: `output/<input stem>_<algo key>.png`.

**Convolution details that bite.**

- `convolve` handles 2D only. Color input goes through `convolve_channels`,
  which applies the same kernel to each channel independently.
- `convolve` flips the kernel, so it is a true convolution, not the correlation
  most libraries implement. Consequence: `x_derivative` measures left minus
  right, and its sign is the opposite of OpenCV's. Magnitudes are unaffected.
  `test_derivative_sign_follows_the_edge_direction` pins this down.
- `gaussian_kernel` rejects even sizes: with no center pixel, the `kh // 2`
  padding would make the output one pixel larger than the input.

**Signed results.** Derivatives are negative on one side of an edge. Passing
them straight to `save_image` clips every negative to black, so they go through
`to_display` first, which puts 0 at mid gray. The `dx` and `dy` CLI entries show
the pattern.

