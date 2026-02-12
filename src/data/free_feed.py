"""Free data provider implementation.

Priority order:
1. yfinance for index candles/VIX/intraday
2. jugaad-data for NSE equity EOD candles
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.data.contracts import candle_dtos_from_frame, fii_dtos_from_frame, frame_from_candle_dtos, frame_from_fii_dtos
from src.data.feed import DataFeed
from src.data.fii import load_or_fetch_fii_dii
from src.data.schemas import OptionChain


class DataFeedError(RuntimeError):
    """Base error for data feed failures."""


class DataUnavailableError(DataFeedError):
    """Raised when a dataset is unavailable from the current provider."""


@dataclass
class FreeFeed(DataFeed):
    """Free/low-cost data feed for bootstrapping development and backtests."""

    data_root: str = "data"
    name: str = "free"

    _YF_SYMBOL_MAP = {
        "NIFTY": "^NSEI",
        "NIFTY50": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "INDIAVIX": "^INDIAVIX",
    }

    _INTERVAL_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "60m": "60m",
        "1h": "60m",
        "1d": "1d",
        "1wk": "1wk",
    }

    def _normalize_symbol(self, symbol: str) -> str:
        return self._YF_SYMBOL_MAP.get(symbol.upper(), symbol)

    def _normalize_timeframe(self, timeframe: str) -> str:
        tf = timeframe.lower().strip()
        if tf not in self._INTERVAL_MAP:
            raise ValueError(f"Unsupported timeframe '{timeframe}'.")
        return self._INTERVAL_MAP[tf]

    def _load_yfinance(self):
        try:
            import yfinance as yf
        except ImportError as exc:
            raise DataUnavailableError(
                "yfinance is not installed. Install it with: pip install yfinance"
            ) from exc
        return yf

    @staticmethod
    def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        normalized = df.copy()
        if isinstance(normalized.columns, pd.MultiIndex):
            normalized.columns = [c[0] for c in normalized.columns]

        rename_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close",
        }
        normalized = normalized.rename(columns=rename_map)
        normalized = normalized.reset_index().rename(columns={"Date": "timestamp", "Datetime": "timestamp"})

        if "timestamp" not in normalized.columns:
            raise DataFeedError("Unable to normalize candle timestamps from source data.")

        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True).dt.tz_convert(None)

        required = ["open", "high", "low", "close"]
        for col in required:
            if col not in normalized.columns:
                raise DataFeedError(f"Missing expected OHLC column '{col}'.")

        if "volume" not in normalized.columns:
            normalized["volume"] = 0

        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        out = normalized[cols].dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(
            subset=["timestamp"], keep="last"
        )
        out["volume"] = out["volume"].fillna(0)
        return out

    def _yf_download(
        self,
        yf,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        # yfinance intraday windows are limited; chunk long requests.
        intraday_intervals = {"1m", "2m", "5m", "15m", "30m", "60m", "90m"}
        if interval not in intraday_intervals:
            raw = yf.download(
                symbol,
                start=start,
                end=end + timedelta(days=1),
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            return self._normalize_ohlcv(raw)

        all_chunks: list[pd.DataFrame] = []
        chunk_start = start
        chunk_span = timedelta(days=58)

        while chunk_start < end:
            chunk_end = min(chunk_start + chunk_span, end)
            raw = yf.download(
                symbol,
                start=chunk_start,
                end=chunk_end + timedelta(days=1),
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            chunk = self._normalize_ohlcv(raw)
            if not chunk.empty:
                all_chunks.append(chunk)
            chunk_start = chunk_end + timedelta(days=1)

        if not all_chunks:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        merged = pd.concat(all_chunks, ignore_index=True)
        merged = merged.drop_duplicates(subset=["timestamp"], keep="last")
        return merged.sort_values("timestamp").reset_index(drop=True)

    def _jugaad_daily(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        try:
            from jugaad_data.nse import stock_df
        except ImportError as exc:
            raise DataUnavailableError("jugaad-data is not installed.") from exc

        df = stock_df(
            symbol=symbol,
            from_date=start.date(),
            to_date=end.date(),
            series="EQ",
        )
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        rename_map = {
            "DATE": "timestamp",
            "OPEN": "open",
            "HIGH": "high",
            "LOW": "low",
            "CLOSE": "close",
            "VOLUME": "volume",
        }
        out = df.rename(columns=rename_map)
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out = out[["timestamp", "open", "high", "low", "close", "volume"]]
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
        return out.reset_index(drop=True)

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        yf_symbol = self._normalize_symbol(symbol)
        interval = self._normalize_timeframe(timeframe)

        try:
            yf = self._load_yfinance()
            data = self._yf_download(yf, yf_symbol, interval, start, end)
            if not data.empty:
                return frame_from_candle_dtos(
                    candle_dtos_from_frame(
                        data,
                        source=self.name,
                        symbol=symbol,
                        timeframe=timeframe,
                    )
                )
        except DataUnavailableError:
            # fall through to jugaad when possible
            pass

        if interval == "1d" and symbol.upper() not in self._YF_SYMBOL_MAP:
            data = self._jugaad_daily(symbol, start, end)
            return frame_from_candle_dtos(
                candle_dtos_from_frame(
                    data,
                    source=self.name,
                    symbol=symbol,
                    timeframe=timeframe,
                )
            )

        raise DataUnavailableError(
            f"No free candle source available for symbol={symbol}, timeframe={timeframe}."
        )

    def get_option_chain(
        self,
        symbol: str,
        expiry: datetime,
        timestamp: datetime | None = None,
    ) -> OptionChain:
        raise DataUnavailableError(
            "Free option-chain history with reliable Greeks is not implemented yet. "
            "Use TrueData or ingest NSE snapshots into local storage first."
        )

    def get_vix(self, start: datetime, end: datetime) -> pd.DataFrame:
        yf = self._load_yfinance()
        data = self._yf_download(yf, self._normalize_symbol("INDIAVIX"), "1d", start, end)
        if data.empty:
            raise DataUnavailableError("Unable to fetch India VIX from free source.")
        return frame_from_candle_dtos(
            candle_dtos_from_frame(
                data,
                source=self.name,
                symbol="INDIAVIX",
                timeframe="1d",
            )
        )

    def get_fii_data(self, start: datetime, end: datetime) -> pd.DataFrame:
        fii_path = Path(self.data_root) / "raw" / "fii_dii.csv"
        try:
            out = load_or_fetch_fii_dii(fii_path, start=start, end=end)
        except Exception as exc:
            raise DataUnavailableError(f"Unable to load/fetch FII/DII data: {exc}") from exc

        out = out.dropna(subset=["date"]).sort_values("date")
        mask = (out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))
        filtered = out.loc[mask].reset_index(drop=True)
        return frame_from_fii_dtos(fii_dtos_from_frame(filtered, source=self.name))
