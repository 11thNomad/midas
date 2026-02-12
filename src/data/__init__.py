"""Data layer exports."""

from src.data.contracts import (
    DTOValidationError,
    NormalizedCandleDTO,
    NormalizedFiiDiiDTO,
    NormalizedOptionRowDTO,
    candle_dtos_from_frame,
    fii_dtos_from_frame,
    frame_from_candle_dtos,
    frame_from_fii_dtos,
    option_dtos_from_chain,
)
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
    "DTOValidationError",
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
    "candle_dtos_from_frame",
    "frame_from_candle_dtos",
    "option_dtos_from_chain",
    "fii_dtos_from_frame",
    "frame_from_fii_dtos",
]
