# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project intent

School project: classic computer vision algorithms implemented by hand with
numpy. **Do not reach for OpenCV, scipy.ndimage or skimage** — the point is the
implementation itself. Only numpy and pillow are dependencies.

The one exception is `ground-truth/`, which compares an implementation against
OpenCV as a check. It is gitignored, absent from `requirements.txt`, and nothing
under `core/` may import it.

## Working style

Answer questions with explanations; do not implement the algorithm on the
user's behalf unless asked explicitly. The user writes the maths.

## Commands

```sh
make install                    # create .venv (needs python3.13) + install requirements
make test                       # pytest
make run ALGO=edge_detection IMAGE=input/mandrill.jpeg
make run ALGO=harris_heatmap K=0.04
make run-edge                   # dx, dy, gradient_magnitude, edge_detection on one image
make run-harris                 # harris_heatmap on input/hlm.jpeg
make clean                      # wipe output/ and caches
make fclean                     # clean + delete .venv
```

Run a single test: `.venv/bin/python -m pytest tests/test_filters.py::test_convolve_identity_kernel`

`pytest.ini` sets `pythonpath = .`, so `core` imports resolve without an install.
The Makefile's `PY ?= python3.13` picks the interpreter used to build the venv.
`.github/workflows/tests.yml` runs `make install PY=python` then `make test` on
every push; `PY=python` is needed because a runner has no `python3.13` binary.

## Architecture

**The image contract.** Every function in `core/` takes and returns a numpy
array of float64 in `[0, 1]`, shaped `(H, W)` for grayscale or `(H, W, 3)` for
RGB. `load_image` and `save_image` in `core/image_io.py` are the only places
that touch the filesystem or uint8; `main.py` is the only caller of both.
Algorithms never open or write files, which is what makes them directly
testable on synthetic arrays.

Two functions step outside it deliberately. `gradient_magnitude` returns the
triplet `(magnitude, I_x, I_y)`, and `to_heatmap` turns `(H, W)` into
`(H, W, 3)`.

**Module layout.** `filters.py` holds the operators: convolution, kernels,
derivatives, `gradient_magnitude`, `non_max_suppression`, `threshold`.
`edge_detection.py` and `harris_corner_detection.py` hold one composed pipeline
each, built from those operators. `image_io.py` holds the filesystem and the two
display mappings.

**Adding an algorithm.** Write the function in a module under `core/`, then
register it in the `ALGORITHMS` dict in `main.py`. Entries are bare function
references wherever possible. The CLI derives the output name:
`output/<input stem>_<algo key>.png`, plus `_k<value>` for harris, so that two
runs with different `k` do not overwrite each other.

**Convolution details that bite.**

- `convolve` handles 2D only. Color input goes through `convolve_channels`,
  which applies the same kernel to each channel independently.
- `convolve` flips the kernel, so it is a true convolution, not the correlation
  most libraries implement. Consequence: `D_x` measures left minus right, and
  its sign is the opposite of OpenCV's. Magnitudes are unaffected.
  `test_derivative_of_gaussian_kernels_follow_the_d_x_sign_convention` pins
  this down.
- `convolve` returns the size of its *first* argument. Convolving two kernels to
  combine them therefore clips the result unless the smaller one is padded
  first, which is what `derivative_of_gaussian_kernels` does with `np.pad`.
  Without it the gaussian's outer ring is silently thrown away.
- `gaussian_kernel` and `box_kernel` reject even sizes: with no center pixel,
  the `kh // 2` padding would make the output one pixel larger than the input.

**Signed results.** A derivative is negative on one side of an edge, and a
harris response is negative on an edge and positive on a corner. Passing either
straight to `save_image` clips every negative to black, so `image_io.py` has two
mappings:

- `to_display(image)` — any signed range to `[0, 1]`, 0 at mid gray, scaled by
  `max|image|`. Applied *inside* `x_derivative` and `y_derivative`, which is why
  those two return a picture rather than a derivative: neither the sign nor the
  scale survives. Code that needs the real values wants
  `convolve_channels(image, D_x)`.
- `to_heatmap(image)` — a signed `[-1, 1]` map to an RGB diverging ramp, blue at
  -1, near white at 0, red at +1. The caller must scale symmetrically, by the
  largest magnitude and not by stretching min to max, or 0 stops landing on the
  neutral. `harris_corner_detection_heatmap` does this with a 99.5th percentile
  and a signed square root.

**Known assumption: no flat images.** `to_display` and
`harris_corner_detection_heatmap` both divide by a peak taken from the data. A
constant image makes that peak 0 and the result is `NaN`, with only a
RuntimeWarning. The guards were removed on purpose. `np.abs(image).max() or 1.0`
restores the behaviour in one line if it ever matters.
