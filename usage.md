# Usage Guide (Execution Order + Command Reference)

This file is the runbook for this repo.

It is sequenced in the order you should run things, and it explicitly separates:
- event-driven `run_backtest.py` (strategy realism)
- vectorized `run_vectorbt_research.py` (fast schedule research)
- `--hybrid` (vector schedule replayed in event-driven engine)

## 0) What Each Engine Actually Tests

### A) `run_backtest.py` (event-driven strategy backtest)
Use when you want to test real strategy logic (`baseline_trend`, `iron_condor`, `jade_lizard`, etc.) with bar-by-bar signal generation, regime gating, and fill simulation.

What it computes each bar (no-lookahead):
- Feature context from historical slice up to current bar:
  - ADX(14), ATR(14), RSI(14), Bollinger width
  - VIX level + 5d ROC
  - FII rolling 3d/5d
  - USDINR 1d/3d ROC
  - PCR + OI support/resistance
  - IV surface shift/tilt
  - ATM Greeks via mibian (`delta/gamma/theta/vega/rho`)
- Regime classification using `config/settings.yaml -> regime`
- Strategy entry/exit decisions
- Fill simulation with costs from `config/settings.yaml -> backtest`

Outputs per run:
- `<run_name>_metrics.json`
- `<run_name>_fills.csv`
- `<run_name>_equity.csv`
- `<run_name>_regimes.csv`
- `<run_name>_signal_snapshots.csv`
- `<run_name>_report.html`

### B) `run_vectorbt_research.py` (vectorized schedule research)
Use when you want fast regime-gate research and parameter sweeps.

What it computes:
- Builds/loads snapshots
- Creates entry/exit schedule from gate:
  - `entry = regime in entry_regimes AND adx_14 >= adx_min AND (optional vix <= vix_max)`
  - `exit = gate closes`
- Runs vectorbt portfolio on underlying close series (not multi-leg options)

Outputs:
- `vectorbt_metrics.json`, `vectorbt_schedule.csv`, `vectorbt_equity.csv`, `vectorbt_trades.csv`, `vectorbt_sensitivity.csv`
- with `--walk-forward`: fold/summary files

### C) `run_vectorbt_research.py --hybrid`
Use when you want vectorbt-generated schedule replayed through event-driven simulator/costing.

What it does:
- Reuses vectorbt schedule entry/exit timestamps
- Executes schedule via `ScheduleDrivenStrategy` in event-driven backtest engine
- Produces fill/equity/regime artifacts with simulator costs/circuit-breaker path

Important:
- Hybrid currently validates schedule execution realism.
- It does **not** yet model iron-condor leg construction/lifecycle in hybrid path.
- For iron-condor strategy behavior, use `run_backtest.py --strategy iron_condor`.

## 1) First-Time Setup (Do Once / Per Token Refresh)

1. Install deps
```bash
pip install -e ".[dev]"
```

2. Kite auth bootstrap
```bash
python scripts/kite_bootstrap.py login-url
python scripts/kite_bootstrap.py exchange --request-token <TOKEN> --save-env --verify
```

3. Health check
```bash
python scripts/health_check.py --quick
```

Achievement:
- Confirms env vars, packages, folder structure, and Kite auth/profile access.

## 2) Data Bootstrap + Warmup

Use enough pre-history for indicator warmup (for your 2022+ test window, keep at least Nov-Dec 2021; more is better).

1. Download core history
```bash
python scripts/download_historical.py --symbol NIFTY --timeframe 1d --start 2021-11-01 --end 2025-12-31
```

2. Optional full preset (NIFTY + BANKNIFTY, 1d + 5m)
```bash
python scripts/download_historical.py --full --days 1500
```

3. FII explicit refresh (optional)
```bash
python scripts/download_fii_dii.py --days 1500
```

Achievement:
- Precomputes cache inputs in `data/cache` used by backtest/vectorbt/paper.

## 3) Data Quality + Regime Sanity

0. Build curated candle cache (dedupe + canonical daily timestamps)
```bash
python scripts/cleanup_curated_data.py --symbol NIFTY --symbol BANKNIFTY --timeframe 1d --strict
```

1. Quality gate
```bash
python scripts/data_quality_report.py --strict
```

2. Visual regime review
```bash
python scripts/regime_visual_review.py --symbol NIFTY --timeframe 1d --start 2022-01-01 --end 2025-12-31 --indicator-warmup-days 60
```

3. Optional replay report
```bash
python scripts/replay_regime.py --symbol NIFTY --timeframe 1d --start 2022-01-01 --end 2025-12-31 --indicator-warmup-days 60
python scripts/regime_transition_report.py --symbol NIFTY --start 2022-01-01 --end 2025-12-31
```

Achievement:
- Verifies raw data integrity and regime labeling stability before strategy comparison.
- Produces reproducible curated candle data in `data/curated_cache` for downstream runs.
- Backtest/vectorbt/replay pipelines now prefer curated candles and fallback to raw if curated is unavailable.

## 4) Event-Driven Strategy Backtests (Primary Comparison)

### A) Backtest baseline vs iron condor
```bash
python scripts/run_backtest.py --symbol NIFTY --timeframe 1d --from 2022-01-01 --to 2025-12-31 --indicator-warmup-days 60 --strategy baseline_trend --strategy iron_condor
```

### B) Walk-forward baseline vs iron condor
```bash
python scripts/run_backtest.py --symbol NIFTY --timeframe 1d --from 2022-01-01 --to 2025-12-31 --indicator-warmup-days 60 --walk-forward --strategy baseline_trend --strategy iron_condor
```

What strategy logic is tested here:

- `baseline_trend`
  - Entry:
    - no position open
    - regime in `strategies.baseline_trend.active_regimes`
    - `adx_14 >= adx_min`
    - if configured, `vix <= vix_max`
  - Exit:
    - gate closes (regime/ADX/VIX no longer valid)

- `iron_condor`
  - Entry:
    - no position open
    - strategy active in regime
    - option chain available and expiry DTE in `[dte_min, dte_max]`
    - short legs selected by target delta (`call_delta`, `put_delta`) if available
    - hedge wings from `wing_width`
  - Exit:
    - profit target (`close_debit <= entry_credit * (1 - profit_target_pct/100)`)
    - stop loss (`close_debit >= entry_credit * (1 + stop_loss_pct/100)`)
    - DTE exit (`dte <= dte_exit`)
    - regime invalidation

Achievement:
- Produces apples-to-apples strategy comparison and walk-forward robustness with realistic costs.

## 5) VectorBT Research + Hybrid (Schedule Layer)

### A) Single vectorbt run
```bash
python scripts/run_vectorbt_research.py --symbol NIFTY --timeframe 1d --from 2022-01-01 --to 2025-12-31
```

### B) Walk-forward vectorbt
```bash
python scripts/run_vectorbt_research.py --symbol NIFTY --timeframe 1d --walk-forward --from 2022-01-01 --to 2025-12-31
```

### C) Vectorbt + hybrid replay
```bash
python scripts/run_vectorbt_research.py --symbol NIFTY --timeframe 1d --from 2022-01-01 --to 2025-12-31 --hybrid
```

Achievement:
- Fast gate-level research (`vectorbt`) plus schedule execution realism cross-check (`hybrid`).

## 6) Parameter-Set Lab + Ranking

Run named parameter sets from `config/vectorbt_parameter_sets.yaml` (currently `baseline_trend`, `conservative_trend`) across fee profiles.

1. Batch (no walk-forward)
```bash
python scripts/run_vectorbt_parameter_sets.py --symbol NIFTY --timeframe 1d --from 2022-01-01 --to 2025-12-31
```

2. Batch with walk-forward
```bash
python scripts/run_vectorbt_parameter_sets.py --symbol NIFTY --timeframe 1d --walk-forward --from 2022-01-01 --to 2025-12-31
```

Ranking files produced:
- `vectorbt_parameter_set_results.csv`
- `vectorbt_parameter_set_leaderboard.csv`
- `vectorbt_parameter_set_robustness.csv`
- `vectorbt_parameter_set_top_sets.csv`
- `vectorbt_parameter_set_promotion_gate.csv`
- `vectorbt_parameter_set_summary.json`

How ranking works:
- Eligibility:
  - min trades (`--min-trades`, default 3)
  - optional drawdown cap (`--max-drawdown-pct`)
- Sort:
  - by selected metric (`--rank-by`; default return metric)
- Robustness:
  - set-level aggregation across fee profiles
  - favors high eligible-profile share + strong worst-case metric
- Promotion gate:
  - thresholds from `backtest.vectorbt_promotion_gate`

## 7) Dashboards (Review + Compare)

1. Backtest / walk-forward / hybrid dashboard
```bash
streamlit run scripts/strategy_results_dashboard.py
```

2. Parameter-set dashboard
```bash
streamlit run scripts/vectorbt_paramsets_dashboard.py
```

Achievement:
- Visual comparison of returns, drawdowns, folds, fills, and hybrid artifacts.

## 8) Paper Loop + Daily Ops

1. Run paper loop
```bash
python scripts/run_paper.py --symbol NIFTY --timeframe 5m --iterations 20 --sleep-seconds 30
```

2. Paper fills PnL summary
```bash
python scripts/paper_fills_report.py --symbol NIFTY
```

3. Daily maintenance (incremental append + quality)
```bash
python scripts/daily_maintenance.py --days 2 --symbols NIFTY,BANKNIFTY --timeframes 1d,5m --strict-quality
```

4. Ops gate checks
```bash
python scripts/paper_ops_runbook.py --phase open --symbol NIFTY --timeframe 1d --run-health-check
python scripts/paper_ops_runbook.py --phase intraday --symbol NIFTY --timeframe 1d
python scripts/paper_ops_runbook.py --phase close --symbol NIFTY --timeframe 1d --run-maintenance --run-vectorbt --run-paper-fills-report
```

Achievement:
- Validates runtime readiness/freshness and summarizes execution economics.

## 9) Settings That Control Runs

Primary settings live in `config/settings.yaml`:
- `market`:
  - decision clock (`decision_timeframe`: `15m`)
  - defensive checks (`defensive_check_timeframe`: `5m`)
  - entry cutoff (`last_new_entry_time`: `15:15`)
- `risk`:
  - `initial_capital` (currently `150000`)
  - drawdown/daily loss caps
  - margin utilization band (`20%` to `40%`)
- `regime`:
  - VIX/ADX thresholds + hysteresis + smoothing + stress overrides
- `strategies.baseline_trend` and `strategies.iron_condor`:
  - entry/exit and sizing parameters
- `backtest`:
  - date defaults, fee/slippage assumptions, walk-forward window sizes, vectorbt fee profiles, promotion gate

## 10) Backtest vs Hybrid: When To Use What

Use `run_backtest.py` when:
- comparing actual strategy implementations
- validating option-structure behavior (`iron_condor` exits, DTE handling)
- reviewing fills and signal snapshots per strategy

Use `run_vectorbt_research.py` when:
- researching gate logic quickly
- running sensitivity and walk-forward cheaply

Use `--hybrid` when:
- you want to sanity-check if vectorized schedule still behaves under event-driven fill/cost path
- you need hybrid artifacts (`hybrid_metrics/equity/fills/regimes`)

## 11) Practical Ranking Order You Should Follow

1. Event-driven walk-forward (`run_backtest.py --walk-forward`) for `baseline_trend` and `iron_condor`.
2. Reject any strategy with poor drawdown/consistency even if headline return is high.
3. Use vectorbt parameter-set lab to tune gates and test fee-profile robustness.
4. Promote only sets passing promotion gate across profiles.
5. Confirm candidate behavior in paper loop before any live decision.

## 12) Full Command Catalog (Current Scripts)

```bash
python scripts/health_check.py [--quick] [--skip-broker-checks] [--check-truedata]
python scripts/kite_auth.py login-url|exchange ...
python scripts/kite_bootstrap.py login-url|exchange ...
python scripts/kite_ws_stream.py --tokens ... [--mode ltp|quote|full]

python scripts/download_historical.py ...
python scripts/download_fii_dii.py ...
python scripts/data_quality_report.py [--strict]
python scripts/daily_maintenance.py ...

python scripts/replay_regime.py ...
python scripts/regime_transition_report.py ...
python scripts/regime_visual_review.py ...

python scripts/run_backtest.py ...
python scripts/run_vectorbt_research.py ...
python scripts/run_vectorbt_parameter_sets.py ...

python scripts/run_paper.py ...
python scripts/paper_fills_report.py ...
python scripts/paper_ops_runbook.py --phase open|intraday|close ...

streamlit run scripts/strategy_results_dashboard.py
streamlit run scripts/vectorbt_paramsets_dashboard.py
```
