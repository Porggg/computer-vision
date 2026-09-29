# CLAUDE.md

School project: classic computer vision implemented by hand with numpy.
**No OpenCV, scipy.ndimage or skimage**; only numpy and pillow.

**Working style:** answer questions with explanations. Do not implement an
algorithm unless explicitly asked: the user writes the maths.

## Commands

```sh
make install / make test / make clean / make fclean
make run ALGO=edge_detection IMAGE=input/mandrill.jpeg
make run ALGO=harris_corner_detection K=0.04
make run-edge      # dx, dy, gradient_magnitude, edge_detection
make run-harris    # harris_corner_detection on input/hlm.jpeg
.venv/bin/python -m pytest tests/test_filters.py::test_convolve_identity_kernel
```

CI runs `make install PY=python`: runners have no `python3.13` binary.

## Architecture

Every function in `core/` takes and returns float64 in `[0, 1]`, `(H, W)` or
`(H, W, 3)`. Only `main.py` touches files, via `load_image` / `save_image`.
Algorithms are registered as bare function references in `ALGORITHMS` in
`main.py`; `tests/test_main.py` checks every entry against the contract.

Deliberate exceptions: `gradient_magnitude` returns `(magnitude, I_x, I_y)`;
`to_heatmap` returns `(H, W, 3)`; `harris_corner_detection_heatmap` returns the
**signed `[-1, 1]` response**, not a colored heatmap, despite its name.

## Tricky things

- **`haaris_corner_detection`** (double a) is the real function name.
- **`convolve` flips the kernel** (true convolution). `D_x` measures left minus
  right, the opposite sign of OpenCV. Magnitudes are unaffected.
- **`convolve` returns the size of its first argument.** Combining two kernels
  clips the result unless the smaller one is padded first
  (`derivative_of_gaussian_kernels` does it with `np.pad`).
- **Kernel sizes must be odd**: even sizes are rejected, otherwise the output
  would be one pixel larger than the input.
- **`x_derivative` / `y_derivative` return pictures**, passed through
  `to_display` (0 at mid gray): sign and scale are lost. For real values use
  `convolve_channels(image, D_x)`.
- **`save_image` silently clips to `[0, 1]`**: a signed map loses its negatives.
  Map it first with `to_display` or `to_heatmap`. `to_heatmap` needs symmetric
  scaling (by the largest magnitude, not min to max), or 0 is not on the neutral.
- **Two NMS with opposite padding:**
  - `non_max_suppression_line` (edges): compares 2 neighbours along the
    gradient, pads with `+inf`, so a border pixel pointing outside loses.
  - `non_max_suppression_neighborhood` (corners): compares all 8 neighbours,
    pads with `-inf`, so border pixels can win. It uses `==`, so plateaus keep
    every tied pixel. The harris response is clipped to 1.0, which creates
    plateaus. It ignores the sign, so negatives must be removed by `threshold`.
- **Flat images give `NaN`** in `to_display` and
  `harris_corner_detection_heatmap` (division by a zero peak). The guards were
  removed on purpose: `np.abs(image).max() or 1.0` restores them if needed.
- **Synthetic test images must be wider than the kernel reach.** DoG radius 5
  plus structure tensor blur radius 4: harris landmarks need about 10 pixels of
  separation, or they leak into each other.
