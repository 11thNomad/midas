PYTHON ?= python

.PHONY: help install-dev lint format type test test-q check health smoke clean-reports

help:
	@echo "Available targets:"
	@echo "  install-dev     Install project with dev dependencies"
	@echo "  lint            Run Ruff lint checks"
	@echo "  format          Run Ruff formatter"
	@echo "  type            Run mypy on src/"
	@echo "  test            Run full pytest suite"
	@echo "  test-q          Run pytest in quiet mode"
	@echo "  check           Run lint + type + test-q"
	@echo "  health          Run connectivity and environment health checks"
	@echo "  smoke           Alias for health"
	@echo "  clean-reports   Delete generated report artifacts"

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	ruff check .

format:
	ruff format .

type:
	mypy src

test:
	pytest

test-q:
	pytest -q

check: lint type test-q

health:
	$(PYTHON) scripts/health_check.py

smoke: health

clean-reports:
	rm -rf data/reports/*
