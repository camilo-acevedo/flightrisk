PYTHON := .venv/Scripts/python.exe
PIP := $(PYTHON) -m pip

.PHONY: help install install-dev install-dl data features train-risk train-survival train-uplift simulate report serve demo test lint format clean

help:
	@echo "Targets:"
	@echo "  install         - install runtime deps"
	@echo "  install-dev     - install runtime + dev deps"
	@echo "  install-dl      - install deep learning extras (torch, pycox)"
	@echo "  data            - pull raw datasets via DVC"
	@echo "  features        - build feature matrices"
	@echo "  train-risk      - train Track A"
	@echo "  train-survival  - train Track B"
	@echo "  train-uplift    - train Track C"
	@echo "  simulate        - run business ROI simulator"
	@echo "  report          - end-to-end pipeline + figures"
	@echo "  serve           - launch FastAPI scoring service"
	@echo "  demo            - launch Streamlit demo"
	@echo "  test            - run pytest with coverage"
	@echo "  lint            - run ruff + mypy"
	@echo "  format          - apply ruff formatting"
	@echo "  clean           - remove caches and build artifacts"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

install-dl:
	$(PIP) install -e ".[deeplearning]"

data:
	$(PYTHON) -m flightrisk.cli data pull

features:
	$(PYTHON) -m flightrisk.cli features build

train-risk:
	$(PYTHON) -m flightrisk.cli train risk

train-survival:
	$(PYTHON) -m flightrisk.cli train survival

train-uplift:
	$(PYTHON) -m flightrisk.cli train uplift

simulate:
	$(PYTHON) -m flightrisk.cli simulate

report: features train-risk train-survival train-uplift simulate
	@echo "Report generated under reports/"

serve:
	$(PYTHON) -m uvicorn flightrisk.serving.api:app --host 0.0.0.0 --port 8000

demo:
	$(PYTHON) -m streamlit run flightrisk/serving/streamlit_app.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check flightrisk tests
	$(PYTHON) -m mypy flightrisk

format:
	$(PYTHON) -m ruff format flightrisk tests
	$(PYTHON) -m ruff check --fix flightrisk tests

clean:
	-rmdir /S /Q .pytest_cache .mypy_cache .ruff_cache build dist .coverage htmlcov 2>nul || true
	-find . -type d -name "__pycache__" -exec rm -rf {} + 2>nul || true
