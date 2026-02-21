# Backtest Report (Iron Condor Sweep - 2026-02-21)

## Scope

- Symbol: `NIFTY`
- Timeframe: `1d`
- Analysis window: `2022-01-01` to `2025-12-31`
- Indicator preload: `60` days (`load_start=2021-11-02`)
- Initial capital: `150000`
- Entry/exit pricing: real daily option prices from option-chain snapshots (no underlying fallback for option symbols)
- Fees: Zerodha-style stack configured in `config/settings.yaml`

Command run:

```bash
python scripts/run_backtest.py --symbol NIFTY --timeframe 1d --from 2022-01-01 --to 2025-12-31 --indicator-warmup-days 60 --strategy iron_condor
python scripts/run_backtest.py --symbol NIFTY --timeframe 1d --from 2022-01-01 --to 2025-12-31 --indicator-warmup-days 60 --walk-forward --strategy iron_condor
```

Artifacts:

- Backtest: `data/reports/backtest_20260221_034109/`
- Walk-forward: `data/reports/walkforward_20260221_034109/`

## Core Results

### Backtest (`iron_condor_nifty_1d_2022-01-01_2025-12-31_backtest_metrics.json`)

- Total return: `15.22%`
- Final equity: `172823.99`
- Max drawdown: `1.10%`
- Sharpe: `-1.305`
- Profit factor: `1.018`
- Win rate: `50.0%`
- Estimated trades: `72`
- Fill count: `572`
- Fees paid: `16195.91`

### Walk-forward (`iron_condor_nifty_1d_2022-01-01_2025-12-31_walkforward_summary.json`)

- Folds: `11`
- Mean fold return: `2.16%`
- Median fold return: `2.04%`
- Mean max drawdown: `0.80%`
- Mean Sharpe: `0.029`
- Mean profit factor: `1.040`
- Mean estimated trades/fold: `9.18`
- Mean fees/fold: `2070.82`
- Positive folds: `10`
- Negative folds: `1`

## Iron Condor Construction (Current Logic)

Source: `src/strategies/iron_condor.py`

1. Entry gates:
- Regime must be active.
- Day must be in `entry_days`.
- VIX must be within `[min_entry_vix, max_entry_vix]`.
- Expiry selected only if DTE is within `[dte_min, dte_max]`.
- Credit per unit must pass `min_premium`.

2. Strike targeting:
- Target offset is `max(wing_width, ATR14 * atr_multiple)`.
- Short legs are selected OTM near target levels.
- Hedge legs are selected farther OTM than shorts.

3. Greeks/IV:
- Upstream chain delta is not trusted for selection.
- IV solved from daily option price using mibian implied-vol solve.
- If IV solve fails, `fallback_iv_pct` is used.
- Delta is then computed locally with mibian using solved/fallback IV.

4. Exits:
- Profit target (`profit_target_pct`).
- Stop loss (`stop_loss_pct`).
- DTE exit (`dte_exit`).
- Calendar/time exit (`enable_time_exit`, `time_exit_day/week_day`).
- Regime-change exit.

## Lifecycle Integrity Snapshot

From `..._backtest_decisions.csv` and `..._backtest_fills.csv`:

- Actionable entry signals: `72`
- Entry fill timestamps: `72`
- Actionable exit signals: `160`
- Exit fill timestamps: `71`
- End-of-window open position: `1`

Interpretation:
- We no longer see the prior large unmatched-exit explosion in fills.
- Exit attempts can still occur without fills (for example, missing close prices for one or more legs on that day), which is why exit decisions are higher than exit fills.

## Worst Closed Trades (Sample)

All from `data/reports/backtest_20260221_034109/`.

### 1) 2024-06-24 -> 2024-06-26
- PnL: `-1182.96`
- Exit reason: `Calendar time exit gate reached`
- Entry VIX/ADX: `14.06 / 11.88`
- Exit VIX/ADX: `14.05 / 10.92`
- Legs:
  - Call short: `NIFTY_20240704_24000CE`
  - Call hedge: `NIFTY_20240704_24100CE`
  - Put short: `NIFTY_20240704_23100PE`
  - Put hedge: `NIFTY_20240704_23000PE`
- Wings: call `100`, put `100`
- Local greeks at entry:
  - Call delta/IV: `0.1868 / 11.5891`
  - Put delta/IV: `-0.1890 / 14.5712`

### 2) 2024-01-29 -> 2024-01-31
- PnL: `-776.69`
- Exit reason: `Calendar time exit gate reached`
- Entry VIX/ADX: `15.68 / 26.66`
- Exit VIX/ADX: `16.05 / 23.19`
- Legs:
  - Call short: `NIFTY_20240208_22100CE`
  - Call hedge: `NIFTY_20240208_22200CE`
  - Put short: `NIFTY_20240208_21350PE`
  - Put hedge: `NIFTY_20240208_21250PE`
- Wings: call `100`, put `100`
- Local greeks at entry:
  - Call delta/IV: `0.2155 / 10.8795`
  - Put delta/IV: `-0.3156 / 26.6252`

### 3) 2023-10-09 -> 2023-10-11
- PnL: `-626.70`
- Exit reason: `Calendar time exit gate reached`
- Entry VIX/ADX: `11.40 / 21.68`
- Exit VIX/ADX: `10.99 / 20.34`
- Legs:
  - Call short: `NIFTY_20231019_19750CE`
  - Call hedge: `NIFTY_20231019_19850CE`
  - Put short: `NIFTY_20231019_19250PE`
  - Put hedge: `NIFTY_20231019_19150PE`
- Wings: call `100`, put `100`
- Local greeks at entry:
  - Call delta/IV: `0.3324 / 13.4764`
  - Put delta/IV: `-0.1561 / 9.4757`

## Notes

- Baseline-trend rerun that started during this sweep was interrupted before artifacts were written, so this report is iron-condor focused.
- New decision artifact is now emitted automatically:
  - `..._backtest_decisions.csv`
