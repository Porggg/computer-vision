# Computer Vision

Hand-written computer vision algorithms with numpy.

## Setup

```sh
make install          # creates .venv (Python 3.13, `brew install python@3.13`) and installs requirements
```

## Usage

```sh
make run                                   # grayscale on input/sample.png
make run ALGO=sobel IMAGE=input/photo.jpg  # algos: grayscale, blur, sobel
make test                                  # run the tests
make clean                                 # remove output/ results and caches
make fclean                                # clean + remove .venv
```

## Layout

- `core/`   — algorithms (`filters.py`) and image I/O (`image_io.py`)
- `input/`  — source images
- `output/` — generated results
- `tests/`  — pytest tests for each function in `core/`
- `main.py` — command-line entry point
