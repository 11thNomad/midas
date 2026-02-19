"""Quality checks for option-chain snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class OptionChainQualityReport:
    rows: int
    duplicate_contract_rows: int
    invalid_option_type_rows: int
    has_calls: bool
    has_puts: bool
    unique_call_strikes: int
    unique_put_strikes: int
    negative_oi_rows: int
    nonpositive_ltp_rows: int
    nonpositive_ltp_share: float
    out_of_dte_rows: int
    out_of_dte_share: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "duplicate_contract_rows": self.duplicate_contract_rows,
            "invalid_option_type_rows": self.invalid_option_type_rows,
            "has_calls": self.has_calls,
            "has_puts": self.has_puts,
            "unique_call_strikes": self.unique_call_strikes,
            "unique_put_strikes": self.unique_put_strikes,
            "negative_oi_rows": self.negative_oi_rows,
            "nonpositive_ltp_rows": self.nonpositive_ltp_rows,
            "nonpositive_ltp_share": self.nonpositive_ltp_share,
            "out_of_dte_rows": self.out_of_dte_rows,
            "out_of_dte_share": self.out_of_dte_share,
        }


@dataclass(frozen=True)
class OptionChainQualityThresholds:
    min_rows: int = 1
    max_duplicate_contract_rows: int = 0
    max_invalid_option_type_rows: int = 0
    require_both_option_types: bool = True
    min_unique_strikes_per_side: int = 3
    max_negative_oi_rows: int = 0
    max_nonpositive_ltp_share: float = 1.0
    max_out_of_dte_share: float = 1.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> OptionChainQualityThresholds:
        return cls(
            min_rows=int(config.get("min_rows", 1)),
            max_duplicate_contract_rows=int(config.get("max_duplicate_contract_rows", 0)),
            max_invalid_option_type_rows=int(config.get("max_invalid_option_type_rows", 0)),
            require_both_option_types=bool(config.get("require_both_option_types", True)),
            min_unique_strikes_per_side=int(config.get("min_unique_strikes_per_side", 3)),
            max_negative_oi_rows=int(config.get("max_negative_oi_rows", 0)),
            max_nonpositive_ltp_share=float(config.get("max_nonpositive_ltp_share", 1.0)),
            max_out_of_dte_share=float(config.get("max_out_of_dte_share", 1.0)),
        )


@dataclass(frozen=True)
class OptionChainQualityGateResult:
    status: str
    issue_count: int
    violations: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "issue_count": self.issue_count,
            "violations": self.violations,
        }


def assess_option_chain_quality(
    chain_df: pd.DataFrame | None,
    *,
    asof: datetime | None = None,
    dte_min: int | None = None,
    dte_max: int | None = None,
) -> OptionChainQualityReport:
    if chain_df is None or chain_df.empty:
        return OptionChainQualityReport(
            rows=0,
            duplicate_contract_rows=0,
            invalid_option_type_rows=0,
            has_calls=False,
            has_puts=False,
            unique_call_strikes=0,
            unique_put_strikes=0,
            negative_oi_rows=0,
            nonpositive_ltp_rows=0,
            nonpositive_ltp_share=0.0,
            out_of_dte_rows=0,
            out_of_dte_share=0.0,
        )

    out = chain_df.copy()
    out["option_type"] = out.get("option_type", "").astype(str).str.upper()
    out["strike"] = pd.to_numeric(out.get("strike"), errors="coerce")
    out["ltp"] = pd.to_numeric(out.get("ltp", 0.0), errors="coerce")
    out["oi"] = pd.to_numeric(out.get("oi", 0.0), errors="coerce")
    out["expiry"] = pd.to_datetime(out.get("expiry"), errors="coerce")

    rows = int(len(out))
    valid_types = out["option_type"].isin(["CE", "PE"])
    invalid_option_type_rows = int((~valid_types).sum())
    out = out.loc[valid_types].copy()

    duplicate_contract_rows = 0
    dedup_cols = [col for col in ["expiry", "strike", "option_type"] if col in out.columns]
    if dedup_cols:
        duplicate_contract_rows = int(out.duplicated(subset=dedup_cols, keep=False).sum())

    calls = out.loc[out["option_type"] == "CE"]
    puts = out.loc[out["option_type"] == "PE"]
    has_calls = not calls.empty
    has_puts = not puts.empty
    unique_call_strikes = int(calls["strike"].dropna().nunique()) if has_calls else 0
    unique_put_strikes = int(puts["strike"].dropna().nunique()) if has_puts else 0

    negative_oi_rows = int((out["oi"] < 0).sum())
    nonpositive_ltp_rows = int((out["ltp"] <= 0).sum())
    nonpositive_ltp_share = float(nonpositive_ltp_rows / max(rows, 1))

    out_of_dte_rows = 0
    if (
        asof is not None
        and "expiry" in out.columns
        and (dte_min is not None or dte_max is not None)
    ):
        anchor = pd.Timestamp(asof).normalize()
        dte = (out["expiry"].dt.normalize() - anchor).dt.days
        mask = pd.Series(False, index=out.index)
        if dte_min is not None:
            mask = mask | (dte < int(dte_min))
        if dte_max is not None:
            mask = mask | (dte > int(dte_max))
        out_of_dte_rows = int(mask.fillna(True).sum())
    out_of_dte_share = float(out_of_dte_rows / max(rows, 1))

    return OptionChainQualityReport(
        rows=rows,
        duplicate_contract_rows=duplicate_contract_rows,
        invalid_option_type_rows=invalid_option_type_rows,
        has_calls=has_calls,
        has_puts=has_puts,
        unique_call_strikes=unique_call_strikes,
        unique_put_strikes=unique_put_strikes,
        negative_oi_rows=negative_oi_rows,
        nonpositive_ltp_rows=nonpositive_ltp_rows,
        nonpositive_ltp_share=nonpositive_ltp_share,
        out_of_dte_rows=out_of_dte_rows,
        out_of_dte_share=out_of_dte_share,
    )


def evaluate_option_chain_quality(
    report: OptionChainQualityReport,
    thresholds: OptionChainQualityThresholds,
) -> OptionChainQualityGateResult:
    if report.rows == 0:
        return OptionChainQualityGateResult(status="no_data", issue_count=0, violations=[])

    violations: list[str] = []

    if report.rows < thresholds.min_rows:
        violations.append(f"rows={report.rows} < {thresholds.min_rows}")
    if report.duplicate_contract_rows > thresholds.max_duplicate_contract_rows:
        violations.append(
            "duplicate_contract_rows="
            f"{report.duplicate_contract_rows} > {thresholds.max_duplicate_contract_rows}"
        )
    if report.invalid_option_type_rows > thresholds.max_invalid_option_type_rows:
        violations.append(
            "invalid_option_type_rows="
            f"{report.invalid_option_type_rows} > {thresholds.max_invalid_option_type_rows}"
        )
    if thresholds.require_both_option_types and (not report.has_calls or not report.has_puts):
        violations.append("missing_option_side")
    if report.unique_call_strikes < thresholds.min_unique_strikes_per_side:
        violations.append(
            "unique_call_strikes="
            f"{report.unique_call_strikes} < {thresholds.min_unique_strikes_per_side}"
        )
    if report.unique_put_strikes < thresholds.min_unique_strikes_per_side:
        violations.append(
            "unique_put_strikes="
            f"{report.unique_put_strikes} < {thresholds.min_unique_strikes_per_side}"
        )
    if report.negative_oi_rows > thresholds.max_negative_oi_rows:
        violations.append(
            f"negative_oi_rows={report.negative_oi_rows} > {thresholds.max_negative_oi_rows}"
        )
    if report.nonpositive_ltp_share > thresholds.max_nonpositive_ltp_share:
        violations.append(
            "nonpositive_ltp_share="
            f"{report.nonpositive_ltp_share:.4f} > {thresholds.max_nonpositive_ltp_share:.4f}"
        )
    if report.out_of_dte_share > thresholds.max_out_of_dte_share:
        violations.append(
            "out_of_dte_share="
            f"{report.out_of_dte_share:.4f} > {thresholds.max_out_of_dte_share:.4f}"
        )

    issue_count = len(violations)
    status = "ok" if issue_count == 0 else "failed_thresholds"
    return OptionChainQualityGateResult(
        status=status,
        issue_count=issue_count,
        violations=violations,
    )
