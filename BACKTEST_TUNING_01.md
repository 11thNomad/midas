# Backtest Tuning 01

Date: 2026-02-22
Status: Planned

## Objectives
- Fix backtest execution/measurement integrity before any strategy tuning.
- Establish a clean, repeatable baseline.
- Run controlled tuning only after Phase 1 passes all gates.

## Locked Decisions
- Forced liquidation at final bar is required.
- Forced liquidation exit reason must be `backtest_window_end`.
- Symbol-fix regression invariant:
  - `forced_liquidation_count <= number_of_symbols_in_run`
  - Flag when `forced_liquidation_count > number_of_symbols_in_run`
- Forced liquidation report must include both count and impacted symbols.
- Forced liquidation and unfilled-exit integrity diagnostics live inside metrics JSON under
  `run_integrity` (not as a separate artifact).
- Metrics JSON must include all four Sharpe variants:
  - `sharpe_daily_rf7`
  - `sharpe_daily_rf0`
  - `sharpe_trade_rf7`
  - `sharpe_trade_rf0`
- Summary header should surface:
  - `sharpe_daily_rf7`
  - `sharpe_trade_rf0`
- Phase 1 Tuesday/Wednesday behavior is audit-only (no rule changes).
- Add `early_exit_opportunity` flag to decisions CSV in Phase 1.
- `early_exit_opportunity` audit is broad (all eligible pre-exit days), not tied to Tuesday-only logic.

## Global Promotion Rule
- Do not start Phase 2 until all Phase 1 gates are green.

## Phase 1: Execution + Measurement Integrity

### 1A) Exit Integrity
- [ ] Add forced liquidation on final bar for all open positions.
- [ ] Tag forced exits with reason `backtest_window_end`.
- [ ] Add run-level forced liquidation report:
  - [ ] `count`
  - [ ] `symbols` (unique list)
  - [ ] `threshold` (`number_of_symbols_in_run`)
  - [ ] `flag` (`count > threshold`)
- [ ] Ensure exit telemetry records:
  - [ ] `exit_attempted_count`
  - [ ] `exit_filled_count`
  - [ ] `exit_unfilled_count`
  - [ ] failure reasons (grouped counts)
- [ ] Write these diagnostics into metrics JSON under `run_integrity`:
  - [ ] `forced_liquidations`
  - [ ] `unfilled_exits`

### 1B) Metric Integrity
- [ ] Add trade-level win rate and profit factor to metrics JSON.
- [ ] Keep fill-level win rate and profit factor, clearly labeled.
- [ ] Add all four Sharpe variants to metrics JSON:
  - [ ] daily RF=7%
  - [ ] daily RF=0%
  - [ ] trade RF=7%
  - [ ] trade RF=0%
- [ ] Update summary output/report header to show:
  - [ ] `sharpe_daily_rf7`
  - [ ] `sharpe_trade_rf0`
- [ ] Add detailed section with the other two Sharpe variants.

### 1C) Early Exit Audit (No Behavior Change)
- [ ] Add `early_exit_opportunity` boolean field to decisions CSV.
- [ ] Flag broadly when profit could have been taken before actual exit (any eligible day).
- [ ] Add audit fields to decisions CSV:
  - [ ] `earliest_exit_day`
  - [ ] `earliest_exit_pnl`
  - [ ] `actual_exit_pnl`
  - [ ] `pnl_delta_vs_earliest_exit`
- [ ] Verify flag is present on all decision rows.

### 1D) Baseline Re-run
- [ ] Re-run full baseline backtest window.
- [ ] Export clean artifacts and metrics.
- [ ] Verify no silent exit failures.

### Phase 1 Sanity Tests
- [ ] Unit test: canonical leg matching survives symbol-format changes.
- [ ] Unit test: forced liquidation occurs only on final bar.
- [ ] Unit test: forced liquidation reason is `backtest_window_end`.
- [ ] Unit test: forced liquidation flag logic uses `count > number_of_symbols`.
- [ ] Unit test: forced liquidation report includes impacted symbols list.
- [ ] Unit test: Sharpe fields and trade-level metrics exist in metrics JSON.
- [ ] Integration test: decisions/fills consistency for exits (attempted vs filled).
- [ ] Integration test: `early_exit_opportunity` appears in decisions CSV.

### Phase 1 Exit Gate (All Required)
- [ ] `forced_liquidation_count <= number_of_symbols_in_run`
- [ ] forced liquidation report populated and accurate
- [ ] no unexplained exit fill failures
- [ ] dual-level metrics (fill + trade) validated
- [ ] all four Sharpe variants present
- [ ] baseline artifacts generated and archived

## Phase 2: Strategy Tuning (After Phase 1 Gate)

### 2A) Entry/Selection Tuning
- [ ] Add hard post-selection delta cap at 0.20.
- [ ] Rebalance picker scoring toward delta dominance.
- [ ] Increase `atr_multiple` to 2.0 and compare vs baseline.
- [ ] Test removing `low_vol_trending` from `iron_condor` active regimes.

### 2B) Exit Rule Variants
- [ ] Implement Tuesday/Wednesday exit variants only now (not in Phase 1).
- [ ] Include Tuesday threshold variant at 40% in experiment matrix.

### 2C) Evaluation Checklist
- [ ] Compare against Phase 1 baseline:
  - [ ] Sharpe (daily_rf7, trade_rf0)
  - [ ] trade-level PF / win rate
  - [ ] drawdown
  - [ ] worst-trade tail behavior
  - [ ] forced liquidation and exit-fill integrity unchanged
- [ ] Promote only if both sanity and outcome improve.

## Phase 3: Paper-Phase Readiness
- [ ] Integrate FII data path.
- [ ] Add price-shock override for gap-risk handling.
- [ ] Run strict regime replay + visual review cycle.
- [ ] Capture before/after classifier metrics for any threshold tweak.
- [ ] Promote only changes that improve both classifier sanity and strategy outcomes.

## Suggested Forced Liquidation Report Shape
```json
{
  "run_integrity": {
    "forced_liquidations": {
      "count": 0,
      "symbols": [],
      "threshold": 1,
      "flag": false
    },
    "unfilled_exits": {
      "attempted": 0,
      "filled": 0,
      "unfilled": 0
    }
  }
}
```

## Run Checklist Template
- [ ] Branch clean and reproducible environment active (`.venv3.13`)
- [ ] Phase 1 implementation complete
- [ ] Phase 1 tests green
- [ ] Baseline rerun complete
- [ ] Baseline report archived
- [ ] Phase 2 experiments executed one-at-a-time
- [ ] Comparison summary produced
- [ ] Promotion decision recorded
