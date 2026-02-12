"""Data layer exports."""

from src.data.contracts import NormalizedCandleDTO, NormalizedFiiDiiDTO, NormalizedOptionRowDTO
from src.data.feed import CandleRequest, DataFeed
from src.data.fii import FiiDownloadError, NseFiiClient, fetch_fii_dii, load_or_fetch_fii_dii
from src.data.free_feed import DataFeedError, DataUnavailableError, FreeFeed
from src.data.kite_feed import KiteFeed
from src.data.quality import (
    CandleQualityReport,
    CandleQualityThresholds,
    QualityGateResult,
    assess_candle_quality,
    evaluate_quality_gate,
    summarize_issue_count,
    thresholds_from_config,
)
from src.data.store import DataStore
from src.data.truedata_feed import TrueDataFeed

__all__ = [
    "CandleRequest",
    "DataFeed",
    "NormalizedCandleDTO",
    "NormalizedOptionRowDTO",
    "NormalizedFiiDiiDTO",
    "FiiDownloadError",
    "NseFiiClient",
    "DataFeedError",
    "DataUnavailableError",
    "CandleQualityReport",
    "CandleQualityThresholds",
    "QualityGateResult",
    "DataStore",
    "FreeFeed",
    "KiteFeed",
    "TrueDataFeed",
    "assess_candle_quality",
    "fetch_fii_dii",
    "load_or_fetch_fii_dii",
    "evaluate_quality_gate",
    "summarize_issue_count",
    "thresholds_from_config",
]
