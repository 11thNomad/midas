# NiftyQuant

A modular algorithmic trading system for Indian markets (NSE), focused on Nifty and Bank Nifty options strategies.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     MONITORING LAYER                     │
│            Logs · Alerts · Performance Dashboard         │
└──────────────┬──────────────────────┬───────────────────┘
               │                      │
┌──────────────▼──────────┐ ┌────────▼───────────────────┐
│     STRATEGY ENGINE     │ │     REGIME CLASSIFIER      │
│  Signal generation      │ │  VIX · ADX · FII Flows     │
│  Entry/exit rules       │ │  PCR · Moving Averages     │
│  Strategy router        │ │  Outputs: market state     │
└──────────────┬──────────┘ └────────┬───────────────────┘
               │                      │
┌──────────────▼──────────────────────▼───────────────────┐
│                    DATA LAYER                            │
│ Kite Connect (primary) + NSE datasets + TrueData (optional standby) │
└──────────────┬──────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────┐
│                  EXECUTION LAYER                         │
│              Zerodha Kite Connect API                    │
│         Order management · Position tracking             │
└──────────────┬──────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────┐
│                 RISK MANAGEMENT                          │
│     Max daily loss · Position sizing · Circuit breaker   │
│     Drawdown limits · Regime-based exposure control      │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
nifty-quant/
├── README.md
├── pyproject.toml
├── .env.example              # API keys template (never commit .env)
├── config/
│   └── settings.yaml         # Strategy params, risk limits, regime thresholds
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/                 # Data layer — fetching + storage
│   │   ├── __init__.py
│   │   ├── kite_feed.py      # Kite API wrapper (historical + live snapshots)
│   │   ├── fii.py            # NSE FII/DII data pipeline
│   │   ├── truedata_feed.py  # Optional standby adapter (not on primary path)
│   │   ├── store.py          # Local data storage (SQLite/Parquet)
│   │   └── schemas.py        # Data models — Candle, OptionChain, Greeks
│   │
│   ├── regime/               # Regime classifier
│   │   ├── __init__.py
│   │   ├── classifier.py     # Core regime detection logic
│   │   ├── signals.py        # Individual signal computations (VIX, ADX, PCR, FII)
│   │   └── config.py         # Regime thresholds and state definitions
│   │
│   ├── strategies/           # Strategy implementations
│   │   ├── __init__.py
│   │   ├── base.py           # Abstract base class for all strategies
│   │   ├── iron_condor.py    # Example: Nifty iron condor
│   │   ├── credit_spread.py  # Example: Bank Nifty credit spread
│   │   ├── momentum.py       # Example: EMA crossover trend following
│   │   └── mean_reversion.py # Example: RSI/Bollinger mean reversion
│   │
│   ├── execution/            # Order execution
│   │   ├── __init__.py
│   │   ├── broker.py         # Kite Connect wrapper — place/modify/cancel orders
│   │   ├── paper.py          # Paper trading engine (simulated fills)
│   │   └── order_manager.py  # Order state machine, retry logic, fill tracking
│   │
│   ├── risk/                 # Risk management
│   │   ├── __init__.py
│   │   ├── position_sizer.py # Kelly criterion, fractional sizing
│   │   ├── circuit_breaker.py# Max daily loss, drawdown limits, kill switch
│   │   └── greeks.py         # Options Greeks computation (uses py_vollib)
│   │
│   ├── backtest/             # Backtesting engine
│   │   ├── __init__.py
│   │   ├── engine.py         # Walk-forward backtesting loop
│   │   ├── simulator.py      # Fill simulation with slippage + commissions
│   │   ├── metrics.py        # Sharpe, Sortino, max drawdown, win rate, etc.
│   │   └── report.py         # Generate backtest reports (HTML/PDF)
│   │
│   └── monitoring/           # Observability
│       ├── __init__.py
│       ├── logger.py         # Structured logging (JSON)
│       ├── alerts.py         # Telegram/email alerts for anomalies
│       ├── dashboard.py      # Streamlit dashboard for live monitoring
│       └── trade_journal.py  # Trade log with entry/exit reasons
│
├── tests/                    # Test suite
│   ├── test_data/
│   ├── test_regime/
│   ├── test_strategies/
│   ├── test_execution/
│   ├── test_risk/
│   └── test_backtest/
│
├── notebooks/                # Exploratory analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_regime_analysis.ipynb
│   ├── 03_strategy_backtest.ipynb
│   └── 04_live_performance.ipynb
│
└── scripts/                  # Utility scripts
    ├── download_historical.py    # Bulk download + cache historical data
    ├── run_backtest.py           # CLI to run backtests
    ├── run_paper.py              # CLI to start paper trading
    ├── run_live.py               # CLI to start live trading
    └── health_check.py           # Verify API connections + data freshness
```

## Getting Started

### Prerequisites
- Python 3.11+
- Zerodha account with Kite Connect API access (execution)
- TrueData account (optional, only if you want secondary-feed validation)

### Installation

```bash
git clone https://github.com/yourusername/nifty-quant.git
cd nifty-quant
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your API keys
```

### First Steps (in order)

```bash
# 1. Verify API connections
python scripts/health_check.py

# 2. Download historical data
python scripts/download_historical.py --symbol NIFTY --days 365

# 3. Explore data in notebook
jupyter lab notebooks/01_data_exploration.ipynb

# 4. Run first backtest
python scripts/run_backtest.py --strategy iron_condor --from 2023-01-01 --to 2025-01-01

# 5. Paper trade
python scripts/run_paper.py --symbol NIFTY --timeframe 5m --iterations 20 --sleep-seconds 30

# 6. Only after validating paper results — go live
python scripts/run_live.py --strategy iron_condor --capital 200000
```

### Scheduling (Cron)

```bash
# Run ad-hoc daily workflow now
python scripts/daily_maintenance.py --days 2 --symbols NIFTY,BANKNIFTY --timeframes 1d,5m

# Generate daily paper fills P&L summary (gross/net/fees by strategy)
python scripts/paper_fills_report.py --symbol NIFTY

# Generate visual regime-review artifact bundle (CSV + PNG + HTML + checklist)
python scripts/regime_visual_review.py --symbol NIFTY --timeframe 1d --days 365

# Explore vectorbt parameter-set outputs in Streamlit
streamlit run scripts/vectorbt_paramsets_dashboard.py

# Install the sample crontab (edit repo path first if needed)
crontab config/cron/daily_maintenance.crontab.example
```

## Design Principles

1. **Data and execution are strictly separated.** Kite data feed and Kite execution are distinct layers.
   Never mix these concerns.

2. **Every strategy inherits from `base.py`.** Enforces a consistent interface:
   `generate_signal()`, `compute_position_size()`, `get_exit_conditions()`.

3. **Risk management is not optional.** The circuit breaker runs independently and can
   kill all positions regardless of what the strategy thinks.

4. **Paper trading uses the same code path as live.** Only the execution layer swaps out.
   `paper.py` and `broker.py` implement the same interface.

5. **All trades are logged with reasons.** Every entry and exit records: the signal that
   triggered it, the regime at the time, the Greeks snapshot, and the risk parameters.

6. **Configuration over code.** Strategy parameters live in `settings.yaml`, not hardcoded.
   Changing a parameter should never require a code change.
