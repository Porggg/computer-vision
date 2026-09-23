VENV   := .venv
PY     ?= python3.13
PYTHON := $(VENV)/bin/python
IMAGE  ?= input/mandrill.jpeg
ALGO   ?= harris_heatmap
K      ?= 0.04

EDGE_IMAGE   ?= input/mandrill.jpeg
EDGE_ALGOS   := dx dy gradient_magnitude edge_detection
HARRIS_IMAGE ?= input/hlm.jpeg

.PHONY: all install run run-edge run-harris test clean fclean

all: install

$(VENV):
	$(PY) -m venv $(VENV)

install: $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) main.py $(IMAGE) --algo $(ALGO) -k $(K)

run-edge:
	@for algo in $(EDGE_ALGOS); do \
		$(PYTHON) main.py $(EDGE_IMAGE) --algo $$algo || exit 1; \
	done

run-harris:
	$(PYTHON) main.py $(HARRIS_IMAGE) --algo harris_heatmap -k $(K)

test:
	$(PYTHON) -m pytest -v

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
	find output -type f ! -name ".gitkeep" -delete

fclean: clean
	rm -rf $(VENV)
