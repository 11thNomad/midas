"""FII/DII data ingestion helpers using NSE public API."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests


class FiiDownloadError(RuntimeError):
    """Raised when FII/DII download fails."""


@dataclass
class NseFiiClient:
    """Client for NSE FII/DII trade activity endpoint."""

    base_url: str = "https://www.nseindia.com"
    timeout: int = 15
    session: requests.Session = field(init=False, repr=False)
    headers: dict[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
            "Referer": f"{self.base_url}/all-reports",
        }

    @staticmethod
    def _format_date(value: datetime) -> str:
        return value.strftime("%d-%m-%Y")

    def fetch_raw(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        url = (
            f"{self.base_url}/api/fiidiiTradeReact"
            f"?fromDate={self._format_date(start)}&toDate={self._format_date(end)}"
        )
        try:
            response = self.session.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise FiiDownloadError(f"NSE request failed: {exc}") from exc
        except ValueError as exc:
            raise FiiDownloadError("NSE response was not valid JSON.") from exc

        if not isinstance(payload, list):
            raise FiiDownloadError("Unexpected NSE response shape (expected list).")
        return payload


def normalize_fii_payload(payload: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert NSE category rows into a date-indexed FII/DII table."""
    if not payload:
        return pd.DataFrame(
            columns=["date", "fii_buy", "fii_sell", "fii_net", "dii_buy", "dii_sell", "dii_net"]
        )

    rows: list[dict[str, Any]] = []
    for item in payload:
        category = str(item.get("category", "")).strip().upper()
        date_text = item.get("date")
        if not date_text:
            continue
        rows.append(
            {
                "date": pd.to_datetime(date_text, format="%d-%b-%Y", errors="coerce"),
                "category": category,
                "buy": float(item.get("buyValue", 0.0) or 0.0),
                "sell": float(item.get("sellValue", 0.0) or 0.0),
                "net": float(item.get("netValue", 0.0) or 0.0),
            }
        )

    raw = pd.DataFrame(rows)
    if raw.empty:
        return pd.DataFrame(
            columns=["date", "fii_buy", "fii_sell", "fii_net", "dii_buy", "dii_sell", "dii_net"]
        )

    raw = raw.dropna(subset=["date"])

    def pivot_category(cat_label: str, prefix: str) -> pd.DataFrame:
        part = raw[raw["category"] == cat_label][["date", "buy", "sell", "net"]].copy()
        return part.rename(
            columns={"buy": f"{prefix}_buy", "sell": f"{prefix}_sell", "net": f"{prefix}_net"}
        )

    fii = pivot_category("FII/FPI", "fii")
    dii = pivot_category("DII", "dii")

    merged = pd.merge(fii, dii, on="date", how="outer")
    merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    for col in ["fii_buy", "fii_sell", "fii_net", "dii_buy", "dii_sell", "dii_net"]:
        if col not in merged.columns:
            merged[col] = 0.0

    merged = merged[["date", "fii_buy", "fii_sell", "fii_net", "dii_buy", "dii_sell", "dii_net"]]
    return merged.reset_index(drop=True)


def fetch_fii_dii(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch and normalize FII/DII data from NSE."""
    client = NseFiiClient()
    payload = client.fetch_raw(start=start, end=end)
    return normalize_fii_payload(payload)


def load_or_fetch_fii_dii(path: str | Path, start: datetime, end: datetime) -> pd.DataFrame:
    """Load local FII/DII CSV if available; otherwise fetch and persist from NSE."""
    file_path = Path(path)
    if file_path.exists():
        df = pd.read_csv(file_path)
        if "date" not in df.columns:
            raise FiiDownloadError(f"Invalid FII CSV at {file_path}: missing 'date' column")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        return df.reset_index(drop=True)

    df = fetch_fii_dii(start=start, end=end)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)
    return df
