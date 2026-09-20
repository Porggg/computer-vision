VENV   := .venv
PY     ?= python3.13
PYTHON := $(VENV)/bin/python
IMAGE  ?= input/pinhole.jpeg
ALGO   ?= blur	

.PHONY: all install run test clean fclean

all: install

$(VENV):
	$(PY) -m venv $(VENV)

install: $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) main.py $(IMAGE) --algo $(ALGO)

test:
	$(PYTHON) -m pytest -v

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
	find output -type f ! -name ".gitkeep" -delete

fclean: clean
	rm -rf $(VENV)
