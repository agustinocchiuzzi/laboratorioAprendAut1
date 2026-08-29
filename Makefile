.PHONY: setup clean-data test lint experiment experiment-full notebook clean

setup:
	python3.12 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -e ".[dev,notebook]"

clean-data:
	.venv/bin/aa-clean-data

test:
	.venv/bin/python -m pytest

lint:
	.venv/bin/python -m ruff check .

experiment:
	.venv/bin/aa-run-experiments --profile quick

experiment-full:
	.venv/bin/aa-run-experiments --profile full

notebook:
	.venv/bin/python -m jupyter lab notebooks/01_experiments.ipynb

clean:
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache results/*

