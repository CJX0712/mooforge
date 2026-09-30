# MOOForge Makefile — common dev / CI tasks
# All commands run from the repo root (the directory containing this file).

PY ?= python
PIP := $(PY) -m pip

.PHONY: help install dev lint format test smoke benchmark clean

help:
	@echo "MOOForge targets:"
	@echo "  make install   install runtime deps"
	@echo "  make dev        install dev deps (pytest, ruff) + editable package"
	@echo "  make lint       ruff check"
	@echo "  make format     ruff format"
	@echo "  make test       full pytest suite"
	@echo "  make smoke      tiny-budget benchmark (fast CI check)"
	@echo "  make benchmark  full deterministic benchmark -> benchmark.json"
	@echo "  make clean      remove caches / scratch outputs"

install:
	$(PIP) install -r requirements.txt

dev:
	$(PIP) install -e .
	$(PIP) install pytest pytest-cov ruff

lint:
	$(PY) -m ruff check mooforge tests

format:
	$(PY) -m ruff format mooforge tests

test:
	$(PY) -m pytest -q

smoke:
	$(PY) -m mooforge.cli run --out _smoke.json --n-evals 300 --pop-size 24 --n-seeds 2 --ablation-seeds 1

benchmark:
	$(PY) -m mooforge.cli run --out benchmark.json

clean:
	$(PY) -m ruff clean 2>/dev/null || true
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -f _smoke.json .coverage
