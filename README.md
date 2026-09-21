# Computer Vision

Hand-written computer vision algorithms with numpy.

## Setup

```sh
make install          # creates .venv (Python 3.13, `brew install python@3.13`) and installs requirements
```

## Usage

```sh
make run ALGO=edge_detection IMAGE=input/photo.jpg
make test                                  
make clean                                                            
```

## Layout

- `core/`   — algorithms (`filters.py`) and image I/O (`image_io.py`)
- `input/`  — source images
- `output/` — generated results
- `tests/`  — pytest tests for each function in `core/`
- `main.py` — command-line entry point
