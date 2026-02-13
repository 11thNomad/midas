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

try:
    from src.data.kite_feed import KiteFeed
except Exception:  # pragma: no cover - optional dependency path
    KiteFeed = None  # type: ignore[assignment]

try:
    from src.data.truedata_feed import TrueDataFeed
except Exception:  # pragma: no cover - optional dependency path
    TrueDataFeed = None  # type: ignore[assignment]

__all__ = [
    "CandleRequest",
    "DataFeed",
    "NormalizedCandleDTO",
    "NormalizedOptionRowDTO",
    "NormalizedFiiDiiDTO",
    "DTOValidationError",
    "FiiDownloadError",
    "NseFiiClient",
    "CandleQualityReport",
    "CandleQualityThresholds",
    "QualityGateResult",
    "DataStore",
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

if KiteFeed is not None:
    __all__.append("KiteFeed")

if TrueDataFeed is not None:
    __all__.append("TrueDataFeed")
