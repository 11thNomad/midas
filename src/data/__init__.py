"""Data layer exports."""

from importlib import import_module
from typing import Any

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


def _load_optional_attr(module_name: str, attr_name: str) -> Any | None:
    try:
        module = import_module(module_name)
        return getattr(module, attr_name, None)
    except Exception:  # pragma: no cover - optional dependency path
        return None


KiteFeed: Any | None = _load_optional_attr("src.data.kite_feed", "KiteFeed")
TrueDataFeed: Any | None = _load_optional_attr("src.data.truedata_feed", "TrueDataFeed")

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
