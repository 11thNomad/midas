# Parity Check Report

Date: 2026-02-22
Branch: `regime-classifier`
Python: `3.13.2` (via `pyenv local 3.13.2`)

## 1) Command Status (pass/fail)

| Step | Command | Status | Notes |
|---|---|---|---|
| A1 | `git status -sb` | PASS | repo state captured |
| A2 | `git diff --name-only` | PASS | changed files captured |
| A3 | `sed -n '1,240p' OPENAI_HANDOFF.md` | FAIL | file missing: `OPENAI_HANDOFF.md` not found in workspace |
| B1 | `ruff check scripts/run_backtest.py src/backtest/__init__.py src/backtest/engine.py tests/test_backtest/test_engine.py` | PASS | all checks passed |
| B2 | `pytest tests/test_backtest/test_engine.py tests/test_backtest/test_report.py tests/test_strategies/test_iron_condor.py tests/test_backtest/test_simulator.py -q` | PASS | `22 passed in 2.61s` |
| C0 | clean parity dirs (`data/reports/parity_dynamic`, `data/reports/parity_precompute`) | PASS | cleaned via Python `shutil.rmtree` + recreate |
| C1 | dynamic run (`--no-precompute`) with `/usr/bin/time -p` | PASS | exit `0` |
| C2 | precompute run (default) with `/usr/bin/time -p` | PASS | exit `0` |
| D1 | pandas parity compare (metrics + CSV artifacts) | PASS | all compared artifacts identical |

## 2) Validation Results

- Ruff: PASS
- Pytest: PASS (`22 passed in 2.61s`)

## 3) Runtime Comparison (wall clock)

- Dynamic (`--no-precompute`): `real 401.69s`
- Precompute (default): `real 303.37s`
- Delta: precompute faster by `98.32s` (`~24.48%` faster vs dynamic)

## 4) Artifact Parity Result

Overall parity: **IDENTICAL**

Compared artifacts:
- `data/reports/parity_dynamic/iron_condor_nifty_1d_2022-01-01_2025-12-31_backtest_metrics.json`
- `data/reports/parity_precompute/iron_condor_nifty_1d_2022-01-01_2025-12-31_backtest_metrics.json`
- `data/reports/parity_dynamic/iron_condor_nifty_1d_2022-01-01_2025-12-31_backtest_fills.csv`
- `data/reports/parity_precompute/iron_condor_nifty_1d_2022-01-01_2025-12-31_backtest_fills.csv`
- `data/reports/parity_dynamic/iron_condor_nifty_1d_2022-01-01_2025-12-31_backtest_equity.csv`
- `data/reports/parity_precompute/iron_condor_nifty_1d_2022-01-01_2025-12-31_backtest_equity.csv`
- `data/reports/parity_dynamic/iron_condor_nifty_1d_2022-01-01_2025-12-31_backtest_regimes.csv`
- `data/reports/parity_precompute/iron_condor_nifty_1d_2022-01-01_2025-12-31_backtest_regimes.csv`
- `data/reports/parity_dynamic/iron_condor_nifty_1d_2022-01-01_2025-12-31_backtest_signal_snapshots.csv`
- `data/reports/parity_precompute/iron_condor_nifty_1d_2022-01-01_2025-12-31_backtest_signal_snapshots.csv`
- `data/reports/parity_dynamic/iron_condor_nifty_1d_2022-01-01_2025-12-31_backtest_decisions.csv`
- `data/reports/parity_precompute/iron_condor_nifty_1d_2022-01-01_2025-12-31_backtest_decisions.csv`

Comparison method:
- Metrics JSON: flattened field-by-field compare with absolute numeric tolerance `1e-9`
- CSVs: row-count compare + full equality after sorting by `timestamp` and then all columns

Row counts:
- fills: `572` vs `572`
- equity: `903` vs `903`
- regimes: `992` vs `992`
- signal_snapshots: `992` vs `992`
- decisions: `721` vs `721`

## 5) Exact Fields/Files That Differ

- None. No metric fields differed beyond `1e-9` tolerance.
- No CSV row differences after sorted full-frame comparison.

## OPENAI_HANDOFF.md Summary (5 bullets)

Blocked because `OPENAI_HANDOFF.md` is not present in this workspace.

1. File lookup at repo root failed (`No such file or directory`).
2. Recursive search for `OPENAI_HANDOFF.md` / `*handoff*.md` returned no matches.
3. No alternate handoff artifact with that filename was available to read.
4. Therefore no content-level summary could be produced from the requested file.
5. This is documented as a non-fatal step failure; remaining validation/parity steps were completed.

## 6) Final Verdict

**safe to commit**

## Exact Command List Executed

```bash
git status -sb
git diff --name-only
git diff --name-only --cached
sed -n '1,240p' OPENAI_HANDOFF.md
rg --files -g 'OPENAI_HANDOFF.md' -g '*handoff*.md'
ruff check scripts/run_backtest.py src/backtest/__init__.py src/backtest/engine.py tests/test_backtest/test_engine.py
pytest tests/test_backtest/test_engine.py tests/test_backtest/test_report.py tests/test_strategies/test_iron_condor.py tests/test_backtest/test_simulator.py -q
python - <<'PY'  # clean parity dirs via shutil.rmtree + mkdir
/usr/bin/time -p -o /tmp/parity_dynamic.time python scripts/run_backtest.py --symbol NIFTY --timeframe 1d --from 2022-01-01 --to 2025-12-31 --indicator-warmup-days 60 --strategy iron_condor --output-dir data/reports/parity_dynamic --no-timestamp-subdir --no-precompute
/usr/bin/time -p -o /tmp/parity_precompute.time python scripts/run_backtest.py --symbol NIFTY --timeframe 1d --from 2022-01-01 --to 2025-12-31 --indicator-warmup-days 60 --strategy iron_condor --output-dir data/reports/parity_precompute --no-timestamp-subdir
python - <<'PY'  # pandas/json parity comparison
```

