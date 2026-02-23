"""Build a clean daily FII equity net cache from NSE latest + NSDL archive files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from html import unescape
from pathlib import Path
from typing import Any

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.calendar import nse_calendar

NSE_BASE_URL = "https://www.nseindia.com"
NSE_REACT_URL = f"{NSE_BASE_URL}/api/fiidiiTradeReact"
DATE_RE = re.compile(r"^\d{2}-[A-Za-z]{3}-\d{4}$")
TD_RE = re.compile(r"<td[^>]*>(.*?)</td>", flags=re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class BuildReport:
    nse_rows: int
    nsdl_rows: int
    output_rows: int
    first_date: str | None
    last_date: str | None
    validation_start: str
    validation_end: str
    expected_trading_days: int
    covered_trading_days: int
    missing_trading_days: int
    missing_pct: float
    missing_by_year: dict[str, dict[str, float | int]]
    years_above_10pct_gap: list[int]
    values_abs_gt_15000: int
    bearish_days_lt_m1000: int
    neutral_days_between: int
    bullish_days_gt_p1000: int
    source_counts: dict[str, int]
    nsdl_files_seen: int
    nsdl_files_parsed: int
    nsdl_parse_failures: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FII daily equity net cache.")
    parser.add_argument(
        "--nsdl-dir",
        default="data/raw/fii_nsdl",
        help="Directory containing NSDL monthly HTML-in-XLS files",
    )
    parser.add_argument(
        "--output",
        default="data/cache/fii/fii_equity_daily.csv",
        help="Output CSV path for cleaned daily cache",
    )
    parser.add_argument(
        "--start",
        default="2022-01-01",
        help="Validation window start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        default="2025-12-31",
        help="Validation window end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--report-json",
        default="",
        help="Optional path to write build report JSON",
    )
    return parser.parse_args()


def _clean_cell(cell_html: str) -> str:
    text = TAG_RE.sub("", cell_html)
    text = unescape(text).replace("\xa0", " ")
    return " ".join(text.split()).strip()


def _parse_amount(value: str) -> float | None:
    text = value.strip()
    if text in {"", "-", "--", "NA", "N/A"}:
        return None

    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    text = text.replace(",", "")
    text = text.replace("₹", "").replace("Rs.", "").replace("Crores", "")

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None

    parsed = float(match.group(0))
    return -abs(parsed) if negative else parsed


def _decode_bytes(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="ignore")


def parse_nsdl_file(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    text = _decode_bytes(raw)
    cells = [_clean_cell(cell) for cell in TD_RE.findall(text)]

    rows: list[dict[str, Any]] = []
    for idx, cell in enumerate(cells):
        if not DATE_RE.match(cell):
            continue
        try:
            parsed_date = datetime.strptime(cell, "%d-%b-%Y").date()
        except ValueError:
            continue
        net_idx = idx + 16
        if net_idx >= len(cells):
            continue
        net_value = _parse_amount(cells[net_idx])
        if net_value is None:
            continue
        # NSDL archive exports include cumulative totals in the same stream.
        if abs(net_value) > 20_000:
            continue
        rows.append(
            {
                "date": parsed_date,
                "fii_equity_net_cr": float(net_value),
                "source": "nsdl_archive",
                "source_file": path.name,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["date", "fii_equity_net_cr", "source", "source_file"])
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return out.reset_index(drop=True)


def fetch_nse_latest_equity_net(timeout: int = 20) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    session.get(NSE_BASE_URL, timeout=timeout)

    response = session.get(
        NSE_REACT_URL,
        headers={
            "User-Agent": session.headers["User-Agent"],
            "Accept": "application/json,text/plain,*/*",
            "Referer": f"{NSE_BASE_URL}/reports/fii-dii",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        return pd.DataFrame(columns=["date", "fii_equity_net_cr", "source", "source_file"])

    rows: list[dict[str, Any]] = []
    for item in payload:
        category = str(item.get("category", "")).strip().upper()
        if category != "FII/FPI":
            continue
        date_text = str(item.get("date", "")).strip()
        net_text = str(item.get("netValue", "")).strip()
        if not date_text:
            continue
        try:
            parsed_date = datetime.strptime(date_text, "%d-%b-%Y").date()
        except ValueError:
            continue
        net_value = _parse_amount(net_text)
        if net_value is None:
            continue
        rows.append(
            {
                "date": parsed_date,
                "fii_equity_net_cr": float(net_value),
                "source": "nse_latest_api",
                "source_file": "nse_api_fiidiiTradeReact",
            }
        )

    if not rows:
        return pd.DataFrame(columns=["date", "fii_equity_net_cr", "source", "source_file"])
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return out.reset_index(drop=True)


def build_report(
    cache: pd.DataFrame, start: date, end: date, *, meta: dict[str, Any]
) -> BuildReport:
    expected_days = nse_calendar.trading_days_between(start, end)
    expected_set = set(expected_days)

    if cache.empty:
        covered_set: set[date] = set()
        first_date = None
        last_date = None
        in_window = pd.DataFrame(columns=cache.columns)
    else:
        cache = cache.copy()
        cache["date"] = pd.to_datetime(cache["date"]).dt.date
        first_date = str(cache["date"].min())
        last_date = str(cache["date"].max())
        in_window = cache[(cache["date"] >= start) & (cache["date"] <= end)]
        covered_set = set(in_window["date"].tolist())

    missing = sorted(expected_set - covered_set)
    missing_by_year: dict[str, dict[str, float | int]] = {}
    years_above_10pct_gap: list[int] = []
    for year in range(start.year, end.year + 1):
        expected_year = [d for d in expected_days if d.year == year]
        if not expected_year:
            continue
        missing_year = [d for d in missing if d.year == year]
        pct = (len(missing_year) / len(expected_year)) * 100.0
        if pct > 10.0:
            years_above_10pct_gap.append(year)
        missing_by_year[str(year)] = {
            "expected_days": len(expected_year),
            "missing_days": len(missing_year),
            "missing_pct": round(pct, 4),
        }

    values = pd.to_numeric(
        in_window.get("fii_equity_net_cr", pd.Series(dtype=float)), errors="coerce"
    )
    values = values.dropna()
    bearish = int((values < -1000.0).sum())
    bullish = int((values > 1000.0).sum())
    neutral = int(len(values) - bearish - bullish)

    source_counts = (
        cache["source"].value_counts().to_dict()
        if not cache.empty and "source" in cache.columns
        else {}
    )

    return BuildReport(
        nse_rows=int(meta["nse_rows"]),
        nsdl_rows=int(meta["nsdl_rows"]),
        output_rows=int(len(cache)),
        first_date=first_date,
        last_date=last_date,
        validation_start=str(start),
        validation_end=str(end),
        expected_trading_days=len(expected_days),
        covered_trading_days=len(covered_set),
        missing_trading_days=len(missing),
        missing_pct=round((len(missing) / len(expected_days) * 100.0) if expected_days else 0.0, 4),
        missing_by_year=missing_by_year,
        years_above_10pct_gap=years_above_10pct_gap,
        values_abs_gt_15000=int((values.abs() > 15000.0).sum()),
        bearish_days_lt_m1000=bearish,
        neutral_days_between=neutral,
        bullish_days_gt_p1000=bullish,
        source_counts={str(k): int(v) for k, v in source_counts.items()},
        nsdl_files_seen=int(meta["nsdl_files_seen"]),
        nsdl_files_parsed=int(meta["nsdl_files_parsed"]),
        nsdl_parse_failures=meta["nsdl_parse_failures"],
    )


def main() -> int:
    args = parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    nsdl_dir = REPO_ROOT / args.nsdl_dir
    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nsdl_frames: list[pd.DataFrame] = []
    parse_failures: dict[str, str] = {}
    nsdl_files = sorted(nsdl_dir.glob("fii_*.xls"))
    for file_path in nsdl_files:
        try:
            frame = parse_nsdl_file(file_path)
            if not frame.empty:
                nsdl_frames.append(frame)
        except Exception as exc:  # pragma: no cover - file-specific corruption path
            parse_failures[file_path.name] = repr(exc)

    nsdl_df = (
        pd.concat(nsdl_frames, ignore_index=True)
        if nsdl_frames
        else pd.DataFrame(columns=["date", "fii_equity_net_cr", "source", "source_file"])
    )
    if not nsdl_df.empty:
        nsdl_df["date"] = pd.to_datetime(nsdl_df["date"])
        nsdl_df = nsdl_df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    try:
        nse_df = fetch_nse_latest_equity_net()
    except Exception as exc:
        print(f"[WARN] NSE latest fetch failed: {exc}")
        nse_df = pd.DataFrame(columns=["date", "fii_equity_net_cr", "source", "source_file"])

    frames = [frame for frame in (nsdl_df, nse_df) if not frame.empty]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if combined.empty:
        combined = pd.DataFrame(columns=["date", "fii_equity_net_cr", "source", "source_file"])
    else:
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
        combined["fii_equity_net_cr"] = pd.to_numeric(
            combined["fii_equity_net_cr"], errors="coerce"
        )
        combined = combined.dropna(subset=["date", "fii_equity_net_cr"])
        combined = combined.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    final = combined[["date", "fii_equity_net_cr"]].copy()
    final["date"] = pd.to_datetime(final["date"]).dt.strftime("%Y-%m-%d")
    final.to_csv(output_path, index=False)

    report = build_report(
        combined,
        start,
        end,
        meta={
            "nse_rows": len(nse_df),
            "nsdl_rows": len(nsdl_df),
            "nsdl_files_seen": len(nsdl_files),
            "nsdl_files_parsed": len(nsdl_frames),
            "nsdl_parse_failures": parse_failures,
        },
    )

    print(f"[OK] Cache written: {output_path.relative_to(REPO_ROOT)} rows={len(final)}")
    print(
        "[INFO] Coverage "
        f"{report.validation_start}..{report.validation_end}: "
        f"{report.covered_trading_days}/{report.expected_trading_days} "
        f"({100.0 - report.missing_pct:.2f}% covered)"
    )
    if report.years_above_10pct_gap:
        print(f"[WARN] Gap >10% in years: {report.years_above_10pct_gap}")
    if report.values_abs_gt_15000 > 0:
        print(f"[WARN] Found {report.values_abs_gt_15000} rows outside ±15000 Cr")

    report_payload = asdict(report)
    report_payload["generated_at_utc"] = datetime.now(UTC).isoformat()

    if args.report_json:
        report_path = REPO_ROOT / args.report_json
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report_payload, indent=2))
        print(f"[OK] Report JSON: {report_path.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
