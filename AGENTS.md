# Repository Guidelines

## Project Structure & Module Organization
- Core code lives in `src/` and is split by domain:
  `src/data`, `src/regime`, `src/signals`, `src/strategies`, `src/backtest`, `src/execution`, and `src/risk`.
- CLI and operational workflows live in `scripts/` (for example, `run_backtest.py`, `run_paper.py`, `daily_maintenance.py`).
- Tests mirror runtime modules in `tests/` (`tests/test_backtest`, `tests/test_data`, `tests/test_scripts`, etc.).
- Runtime configuration is in `config/settings.yaml`; local secrets are loaded from `.env`.
- Generated artifacts and cached datasets are stored under `data/` (typically `data/cache` and `data/reports`).

## Build, Test, and Development Commands
- Environment baseline:
  `source .venv3.13/bin/activate`
  Always run tooling with Python 3.13 by default. If a Python 3.13-specific issue appears, stop and
  decide explicitly before switching to 3.12/3.11.
- Install with dev tooling:
  `python -m pip install -e ".[dev]"`
- Run full tests:
  `python -m pytest`
- Run a focused test module:
  `python -m pytest tests/test_scripts/test_run_paper.py`
- Lint and format:
  `ruff check .` and `ruff format .`
- Type-check (strict mode):
  `mypy src`
- Common local workflows:
  `python scripts/health_check.py`
  `python scripts/run_backtest.py --symbol NIFTY --timeframe 1d --from 2025-01-01 --to 2025-12-31`

## Coding Style & Naming Conventions
- Target Python 3.13 with 4-space indentation and type hints for new/changed code.
- Follow Ruff defaults configured in `pyproject.toml` (line length `100`; rule sets include `E/F/W/I/N/UP/B/SIM`).
- Keep modules/functions in `snake_case`; classes in `PascalCase`; constants in `UPPER_SNAKE_CASE`.
- Prefer small, composable functions and keep script CLI args explicit and documented.
- Use `rg`/`rg --files` for text/file search in the repo.

## Testing Guidelines
- Framework: `pytest` with `pytest-asyncio` (`asyncio_mode = auto`).
- Place tests in mirrored paths and name files `test_*.py`.
- For bug fixes, add or update a regression test in the closest relevant test package.
- Before opening a PR, run at least `pytest`, `ruff check .`, and `mypy src`.

## Commit & Pull Request Guidelines
- Use concise, imperative commit subjects; repository history favors Conventional Commit prefixes like `feat:`, `fix:`, and `refactor:`.
- Keep commits scoped to one logical change and include tests with behavior changes.
- PRs should include:
  1. What changed and why.
  2. Risk/impact areas (data, fills, regime outputs, reports).
  3. Validation evidence (commands run and key outputs).
  4. Relevant artifacts/screenshots for report or visualization changes.

## Security & Configuration Tips
- Never commit `.env` or API tokens.
- Validate external connectivity with `python scripts/health_check.py` before running paper/live-like flows.
- Prefer config updates in `config/settings.yaml` over hard-coded strategy/risk parameters.
