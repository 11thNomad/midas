"""NSE F&O bhavcopy download and normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from io import BytesIO, StringIO
from zipfile import ZipFile

import pandas as pd
import requests

from src.data.calendar import nse_calendar

NSE_FO_ROUTE_CUTOFF = date(2024, 7, 8)


class NSEFOBhavcopyError(RuntimeError):
    """Raised when NSE F&O bhavcopy download or parsing fails."""


@dataclass
class NSEFOBhavcopyClient:
    """Download NSE F&O bhavcopy across old/new archive routes."""

    timeout: int = 20
    session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "text/csv,application/zip,*/*",
                "Accept-Encoding": "gzip, deflate",
            }
        )

    @staticmethod
    def _route_for_day(day: date) -> str:
        if day >= NSE_FO_ROUTE_CUTOFF:
            return (
                "https://nsearchives.nseindia.com/content/fo/"
                f"BhavCopy_NSE_FO_0_0_0_{day.strftime('%Y%m%d')}_F_0000.csv.zip"
            )
        return (
            "https://nsearchives.nseindia.com/content/historical/DERIVATIVES/"
            f"{day.year}/{day.strftime('%b').upper()}/"
            f"fo{day.strftime('%d')}{day.strftime('%b').upper()}{day.year}bhav.csv.zip"
        )

    def fetch_day_raw(self, day: date) -> pd.DataFrame:
        """Fetch one F&O bhavcopy day as raw DataFrame."""
        url = self._route_for_day(day)
        response = self.session.get(url, timeout=self.timeout)
        if response.status_code != 200:
            raise NSEFOBhavcopyError(
                f"bhavcopy request failed status={response.status_code} day={day} url={url}"
            )
        if not response.content.startswith(b"PK"):
            preview = response.text[:120].replace("\n", " ")
            raise NSEFOBhavcopyError(
                f"bhavcopy payload is not zip day={day} url={url} preview={preview}"
            )

        csv_bytes = self._extract_csv_bytes(response.content)
        try:
            return pd.read_csv(StringIO(csv_bytes.decode("utf-8", errors="ignore")))
        except Exception as exc:
            raise NSEFOBhavcopyError(f"failed to parse csv day={day}: {exc}") from exc

    @staticmethod
    def _extract_csv_bytes(payload: bytes) -> bytes:
        with ZipFile(BytesIO(payload)) as zf:
            first = zf.namelist()[0]
            raw = zf.read(first)
        # Older files can be nested zip-inside-zip.
        if first.lower().endswith(".zip") or raw.startswith(b"PK"):
            with ZipFile(BytesIO(raw)) as zf_nested:
                nested = zf_nested.namelist()[0]
                return zf_nested.read(nested)
        return raw


def fetch_option_chain_history(
    *,
    symbol: str,
    start: datetime,
    end: datetime,
    client: NSEFOBhavcopyClient | None = None,
) -> pd.DataFrame:
    """Download and normalize daily option-chain rows for a symbol."""
    client = client or NSEFOBhavcopyClient()
    out_frames: list[pd.DataFrame] = []

    day = start.date()
    end_day = end.date()
    while day <= end_day:
        if nse_calendar.is_trading_day(day):
            try:
                raw = client.fetch_day_raw(day)
                norm = normalize_fo_bhavcopy_to_option_chain(
                    raw, symbol=symbol, trade_date=day, source="nse_bhavcopy_fo"
                )
                if not norm.empty:
                    out_frames.append(norm)
            except NSEFOBhavcopyError:
                # Missing day/file happens around holidays/special sessions.
                pass
        day += timedelta(days=1)

    if not out_frames:
        return _empty_option_chain_frame()
    return pd.concat(out_frames, ignore_index=True)


def normalize_fo_bhavcopy_to_option_chain(
    raw: pd.DataFrame,
    *,
    symbol: str,
    trade_date: date | None = None,
    source: str = "nse_bhavcopy_fo",
) -> pd.DataFrame:
    """Map old/new NSE F&O bhavcopy formats into option_chain schema."""
    if raw.empty:
        return _empty_option_chain_frame()

    normalized_symbol = symbol.upper().strip()
    if "INSTRUMENT" in raw.columns:
        return _normalize_legacy_fo(
            raw, symbol=normalized_symbol, trade_date=trade_date, source=source
        )
    if "FinInstrmTp" in raw.columns:
        return _normalize_udiff_fo(
            raw, symbol=normalized_symbol, trade_date=trade_date, source=source
        )
    raise NSEFOBhavcopyError("Unsupported F&O bhavcopy format (known columns not found).")


def _normalize_legacy_fo(
    raw: pd.DataFrame,
    *,
    symbol: str,
    trade_date: date | None,
    source: str,
) -> pd.DataFrame:
    frame = raw.copy()
    frame["INSTRUMENT"] = frame["INSTRUMENT"].astype(str).str.upper()
    frame["SYMBOL"] = frame["SYMBOL"].astype(str).str.upper()
    frame = frame[(frame["INSTRUMENT"] == "OPTIDX") & (frame["SYMBOL"] == symbol)]
    if frame.empty:
        return _empty_option_chain_frame()

    frame["expiry"] = pd.to_datetime(frame["EXPIRY_DT"], errors="coerce")
    frame["strike"] = pd.to_numeric(frame["STRIKE_PR"], errors="coerce")
    frame["option_type"] = frame["OPTION_TYP"].astype(str).str.upper()
    frame["ltp"] = pd.to_numeric(frame.get("CLOSE"), errors="coerce").fillna(0.0)
    frame["volume"] = pd.to_numeric(frame.get("CONTRACTS"), errors="coerce").fillna(0.0)
    frame["oi"] = pd.to_numeric(frame.get("OPEN_INT"), errors="coerce").fillna(0.0)
    frame["change_in_oi"] = pd.to_numeric(frame.get("CHG_IN_OI"), errors="coerce").fillna(0.0)

    if trade_date is not None:
        frame["timestamp"] = pd.to_datetime(trade_date) + pd.Timedelta(hours=18, minutes=30)
    else:
        date_series = pd.to_datetime(frame.get("TIMESTAMP"), errors="coerce")
        frame["timestamp"] = date_series + pd.Timedelta(hours=18, minutes=30)
    frame["underlying"] = symbol
    frame["underlying_price"] = 0.0
    frame["symbol"] = frame.apply(
        lambda row: (
            f"{symbol}_{row['expiry'].strftime('%Y%m%d')}_{int(row['strike'])}{row['option_type']}"
            if pd.notna(row["expiry"]) and pd.notna(row["strike"])
            else f"{symbol}_UNKNOWN"
        ),
        axis=1,
    )
    return _to_option_chain_output(frame, source=source)


def _normalize_udiff_fo(
    raw: pd.DataFrame,
    *,
    symbol: str,
    trade_date: date | None,
    source: str,
) -> pd.DataFrame:
    frame = raw.copy()
    frame["FinInstrmTp"] = frame["FinInstrmTp"].astype(str).str.upper()
    frame["TckrSymb"] = frame["TckrSymb"].astype(str).str.upper()
    frame = frame[(frame["FinInstrmTp"] == "IDO") & (frame["TckrSymb"] == symbol)]
    if frame.empty:
        return _empty_option_chain_frame()

    frame["expiry"] = pd.to_datetime(frame["XpryDt"], errors="coerce")
    frame["strike"] = pd.to_numeric(frame["StrkPric"], errors="coerce")
    frame["option_type"] = frame["OptnTp"].astype(str).str.upper()
    frame["ltp"] = pd.to_numeric(frame.get("LastPric"), errors="coerce").fillna(0.0)
    close_px = pd.to_numeric(frame.get("ClsPric"), errors="coerce").fillna(0.0)
    frame["ltp"] = frame["ltp"].where(frame["ltp"] > 0, close_px)
    frame["volume"] = pd.to_numeric(frame.get("TtlTradgVol"), errors="coerce").fillna(0.0)
    frame["oi"] = pd.to_numeric(frame.get("OpnIntrst"), errors="coerce").fillna(0.0)
    frame["change_in_oi"] = pd.to_numeric(frame.get("ChngInOpnIntrst"), errors="coerce").fillna(
        0.0
    )
    frame["underlying_price"] = pd.to_numeric(frame.get("UndrlygPric"), errors="coerce").fillna(0.0)

    if trade_date is not None:
        frame["timestamp"] = pd.to_datetime(trade_date) + pd.Timedelta(hours=18, minutes=30)
    else:
        date_series = pd.to_datetime(frame.get("TradDt"), errors="coerce")
        frame["timestamp"] = date_series + pd.Timedelta(hours=18, minutes=30)
    frame["underlying"] = symbol
    frame["symbol"] = frame.get("FinInstrmNm", "").astype(str).str.strip()
    frame["symbol"] = frame["symbol"].where(
        frame["symbol"] != "",
        frame.apply(
            lambda row: (
                f"{symbol}_{row['expiry'].strftime('%Y%m%d')}_{int(row['strike'])}{row['option_type']}"
                if pd.notna(row["expiry"]) and pd.notna(row["strike"])
                else f"{symbol}_UNKNOWN"
            ),
            axis=1,
        ),
    )
    return _to_option_chain_output(frame, source=source)


def _to_option_chain_output(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["timestamp"], errors="coerce"),
            "underlying": frame["underlying"].astype(str).str.upper(),
            "underlying_price": (
                pd.to_numeric(frame["underlying_price"], errors="coerce").fillna(0.0)
            ),
            "expiry": pd.to_datetime(frame["expiry"], errors="coerce"),
            "strike": pd.to_numeric(frame["strike"], errors="coerce"),
            "option_type": frame["option_type"].astype(str).str.upper(),
            "symbol": frame["symbol"].astype(str),
            "ltp": pd.to_numeric(frame["ltp"], errors="coerce").fillna(0.0),
            "bid": 0.0,
            "ask": 0.0,
            "volume": pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0),
            "oi": pd.to_numeric(frame["oi"], errors="coerce").fillna(0.0),
            "change_in_oi": pd.to_numeric(frame["change_in_oi"], errors="coerce").fillna(0.0),
            "iv": 0.0,
            "delta": 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0,
            "source": source,
        }
    )
    out = out.dropna(subset=["timestamp", "expiry", "strike"])
    out = out[out["option_type"].isin(["CE", "PE"])]
    out = out.sort_values(["timestamp", "expiry", "strike", "option_type"]).reset_index(drop=True)
    return out


def _empty_option_chain_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timestamp",
            "underlying",
            "underlying_price",
            "expiry",
            "strike",
            "option_type",
            "symbol",
            "ltp",
            "bid",
            "ask",
            "volume",
            "oi",
            "change_in_oi",
            "iv",
            "delta",
            "gamma",
            "theta",
            "vega",
            "rho",
            "source",
        ]
    )
