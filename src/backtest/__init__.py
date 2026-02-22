"""Backtest layer exports."""

from src.backtest.engine import BacktestEngine, BacktestPrecomputedData, BacktestResult
from src.backtest.fee_profiles import (
    VectorBTFeeProfile,
    parse_vectorbt_fee_profiles,
    resolve_vectorbt_costs,
    select_vectorbt_fee_profiles,
)
from src.backtest.hybrid import HybridConfig, run_hybrid_schedule_backtest
from src.backtest.metrics import (
    monte_carlo_permutation_pvalue,
    regime_segmented_returns,
    summarize_backtest,
)
from src.backtest.promotion_gate import (
    VectorBTPromotionGateConfig,
    evaluate_vectorbt_promotion_gate,
    parse_vectorbt_promotion_gate_config,
)
from src.backtest.report import write_backtest_report, write_walkforward_report
from src.backtest.simulator import FillSimulator
from src.backtest.vectorbt_dashboard_data import (
    list_vectorbt_paramset_runs,
    load_detail_artifact,
    load_run_summary,
    load_run_table,
    load_walkforward_folds,
    parse_artifact_filename,
    parse_fold_filename,
)
from src.backtest.vectorbt_parameter_sets import (
    VectorBTParameterSet,
    apply_parameter_set,
    parse_parameter_sets,
    rank_parameter_results,
)
from src.backtest.vectorbt_research import (
    VectorBTResearchConfig,
    VectorBTResearchResult,
    build_snapshots_from_market_data,
    build_vectorbt_schedule,
    run_hybrid_from_schedule,
    run_hybrid_from_schedule_result,
    run_vectorbt_research,
    run_vectorbt_sensitivity,
    run_vectorbt_walk_forward,
)
from src.backtest.vectorbt_trade_attribution import build_trade_attribution
from src.backtest.walkforward import (
    WalkForwardWindow,
    aggregate_cross_instrument_results,
    aggregate_walk_forward_metrics,
    build_sensitivity_variants,
    generate_walk_forward_windows,
    summarize_sensitivity_results,
)

__all__ = [
    "BacktestEngine",
    "BacktestPrecomputedData",
    "BacktestResult",
    "HybridConfig",
    "FillSimulator",
    "WalkForwardWindow",
    "generate_walk_forward_windows",
    "aggregate_walk_forward_metrics",
    "build_sensitivity_variants",
    "summarize_sensitivity_results",
    "aggregate_cross_instrument_results",
    "regime_segmented_returns",
    "monte_carlo_permutation_pvalue",
    "summarize_backtest",
    "VectorBTPromotionGateConfig",
    "parse_vectorbt_promotion_gate_config",
    "evaluate_vectorbt_promotion_gate",
    "write_backtest_report",
    "write_walkforward_report",
    "VectorBTResearchConfig",
    "VectorBTResearchResult",
    "build_snapshots_from_market_data",
    "build_vectorbt_schedule",
    "run_vectorbt_research",
    "run_vectorbt_sensitivity",
    "run_vectorbt_walk_forward",
    "run_hybrid_schedule_backtest",
    "run_hybrid_from_schedule",
    "run_hybrid_from_schedule_result",
    "list_vectorbt_paramset_runs",
    "load_detail_artifact",
    "load_run_summary",
    "load_run_table",
    "load_walkforward_folds",
    "parse_artifact_filename",
    "parse_fold_filename",
    "build_trade_attribution",
    "VectorBTFeeProfile",
    "parse_vectorbt_fee_profiles",
    "select_vectorbt_fee_profiles",
    "resolve_vectorbt_costs",
    "VectorBTParameterSet",
    "parse_parameter_sets",
    "apply_parameter_set",
    "rank_parameter_results",
]
