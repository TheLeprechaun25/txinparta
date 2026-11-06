# Shortcuts. Run from the repo root.
PY ?= python

.PHONY: install test lint format pultsito demo clean

install:          ## editable install with development tools
	$(PY) -m pip install -e ".[dev]"

test:             ## run all tests
	$(PY) -m pytest -q

lint:             ## static checks
	$(PY) -m ruff check .

format:           ## apply automatic lint fixes
	$(PY) -m ruff check . --fix

pultsito:         ## build the C backend (build/libdarwinsnn_pultsito.so)
	native/build_pultsito.sh

demo:             ## random vs evolved networks in about a minute
	$(PY) examples/quickstart.py

clean:            ## remove build artefacts and caches
	rm -rf build .pytest_cache .ruff_cache src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
