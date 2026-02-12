"""Kite data feed implementation for historical candles and option-chain snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Iterable

import pandas as pd
from kiteconnect import KiteConnect

from src.data.contracts import candle_dtos_from_frame, frame_from_candle_dtos, option_dtos_from_chain
from src.data.feed import DataFeed
from src.data.schemas import InstrumentType, OptionChain, OptionContract


class KiteFeedError(RuntimeError):
    """Raised when Kite data calls fail or return invalid data."""


@dataclass
class KiteFeed(DataFeed):
    """Supplementary broker feed using Kite Connect APIs."""

    api_key: str
    access_token: str
    name: str = "kite"
    _kite: KiteConnect = field(init=False, repr=False)
    _instruments: list[dict] | None = field(default=None, init=False, repr=False)

    _INTERVAL_MAP = {
        "1m": "minute",
        "3m": "3minute",
        "5m": "5minute",
        "10m": "10minute",
        "15m": "15minute",
        "30m": "30minute",
        "60m": "60minute",
        "1h": "60minute",
        "1d": "day",
    }

    _MAX_CHUNK_DAYS = {
        "minute": 30,
        "3minute": 60,
        "5minute": 90,
        "10minute": 120,
        "15minute": 150,
        "30minute": 180,
        "60minute": 365,
        "day": 3650,
    }

    _INDEX_QUOTE_MAP = {
        "NIFTY": "NSE:NIFTY 50",
        "NIFTY50": "NSE:NIFTY 50",
        "BANKNIFTY": "NSE:NIFTY BANK",
        "INDIAVIX": "NSE:INDIA VIX",
    }

    def __post_init__(self):
        self._kite = KiteConnect(api_key=self.api_key)
        self._kite.set_access_token(self.access_token)

    def _normalize_interval(self, timeframe: str) -> str:
        key = timeframe.lower().strip()
        if key not in self._INTERVAL_MAP:
            raise ValueError(f"Unsupported timeframe '{timeframe}' for Kite feed")
        return self._INTERVAL_MAP[key]

    def _load_instruments(self) -> list[dict]:
        if self._instruments is None:
            self._instruments = self._kite.instruments()
        return self._instruments

    def _resolve_instrument_token(self, symbol: str) -> int:
        needle = symbol.upper()
        instruments = self._load_instruments()

        for row in instruments:
            if str(row.get("tradingsymbol", "")).upper() == needle:
                return int(row["instrument_token"])

        # Fallback for common index aliases.
        mapped = {
            "NIFTY": "NIFTY 50",
            "NIFTY50": "NIFTY 50",
            "BANKNIFTY": "NIFTY BANK",
            "INDIAVIX": "INDIA VIX",
        }.get(needle)
        if mapped:
            for row in instruments:
                if str(row.get("tradingsymbol", "")).upper() == mapped.upper():
                    return int(row["instrument_token"])

        raise KiteFeedError(f"No Kite instrument token found for symbol='{symbol}'")

    def _iter_chunks(self, start: datetime, end: datetime, interval: str) -> Iterable[tuple[datetime, datetime]]:
        max_days = self._MAX_CHUNK_DAYS.get(interval, 30)
        cursor = start
        step = timedelta(days=max_days)
        while cursor < end:
            chunk_end = min(cursor + step, end)
            yield cursor, chunk_end
            cursor = chunk_end + timedelta(seconds=1)

    @staticmethod
    def _normalize_candles(rows: list[dict]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])

        frame = pd.DataFrame(rows)
        frame = frame.rename(columns={"date": "timestamp"})
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(None)
        if "oi" not in frame.columns:
            frame["oi"] = 0
        if "volume" not in frame.columns:
            frame["volume"] = 0

        cols = ["timestamp", "open", "high", "low", "close", "volume", "oi"]
        frame = frame[cols].dropna(subset=["timestamp"]).sort_values("timestamp")
        frame = frame.drop_duplicates(subset=["timestamp"], keep="last")
        return frame.reset_index(drop=True)

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        token = self._resolve_instrument_token(symbol)
        interval = self._normalize_interval(timeframe)

        all_rows: list[dict] = []
        for chunk_start, chunk_end in self._iter_chunks(start, end, interval):
            rows = self._kite.historical_data(
                instrument_token=token,
                from_date=chunk_start,
                to_date=chunk_end,
                interval=interval,
                oi=True,
            )
            all_rows.extend(rows)

        data = self._normalize_candles(all_rows)
        return frame_from_candle_dtos(
            candle_dtos_from_frame(
                data,
                source=self.name,
                symbol=symbol,
                timeframe=timeframe,
            )
        )

    @staticmethod
    def _instrument_type(opt_type: str) -> InstrumentType:
        return InstrumentType.CALL if opt_type.upper() == "CE" else InstrumentType.PUT

    @staticmethod
    def _batched(values: list[str], size: int = 250) -> Iterable[list[str]]:
        for i in range(0, len(values), size):
            yield values[i : i + size]

    def _get_underlying_price(self, symbol: str) -> float:
        quote_symbol = self._INDEX_QUOTE_MAP.get(symbol.upper())
        if not quote_symbol:
            return 0.0
        payload = self._kite.quote([quote_symbol])
        data = payload.get(quote_symbol, {})
        return float(data.get("last_price", 0.0) or 0.0)

    def get_option_chain(
        self,
        symbol: str,
        expiry: datetime,
        timestamp: datetime | None = None,
    ) -> OptionChain:
        target_name = symbol.upper()
        target_expiry = expiry.date()

        candidates = []
        for row in self._load_instruments():
            if row.get("segment") != "NFO-OPT":
                continue
            if str(row.get("name", "")).upper() != target_name:
                continue
            instrument_expiry = row.get("expiry")
            if instrument_expiry is None:
                continue
            if pd.Timestamp(instrument_expiry).date() != target_expiry:
                continue
            candidates.append(row)

        if not candidates:
            raise KiteFeedError(f"No option contracts found for {symbol} expiry={target_expiry}")

        quote_keys = [f"NFO:{row['tradingsymbol']}" for row in candidates]
        quotes: dict[str, dict] = {}
        for chunk in self._batched(quote_keys, size=200):
            quotes.update(self._kite.quote(chunk))

        contracts: list[OptionContract] = []
        for row in candidates:
            key = f"NFO:{row['tradingsymbol']}"
            quote = quotes.get(key, {})
            depth = quote.get("depth", {})
            buy_depth = depth.get("buy", [])
            sell_depth = depth.get("sell", [])
            bid = float(buy_depth[0].get("price", 0.0)) if buy_depth else 0.0
            ask = float(sell_depth[0].get("price", 0.0)) if sell_depth else 0.0

            contracts.append(
                OptionContract(
                    symbol=str(row["tradingsymbol"]),
                    instrument_type=self._instrument_type(str(row.get("instrument_type", "CE"))),
                    strike=float(row.get("strike", 0.0) or 0.0),
                    expiry=pd.Timestamp(row["expiry"]).to_pydatetime(),
                    ltp=float(quote.get("last_price", 0.0) or 0.0),
                    bid=bid,
                    ask=ask,
                    volume=int(quote.get("volume", 0) or 0),
                    oi=int(quote.get("oi", 0) or 0),
                    change_in_oi=int(quote.get("oi_day_high", 0) or 0),  # TODO: replace with true delta OI.
                )
            )

        chain = OptionChain(
            underlying=symbol,
            underlying_price=self._get_underlying_price(symbol),
            timestamp=timestamp or datetime.now(UTC).replace(tzinfo=None),
            expiry=expiry,
            contracts=contracts,
        )
        option_dtos_from_chain(chain, source=self.name)
        return chain

    def get_vix(self, start: datetime, end: datetime) -> pd.DataFrame:
        # Kite exposes India VIX as an index instrument.
        return self.get_candles(symbol="INDIAVIX", timeframe="1d", start=start, end=end)

    def get_fii_data(self, start: datetime, end: datetime) -> pd.DataFrame:
        # TODO: Kite does not provide FII/DII cash-flow dataset. Keep using NSE pipeline.
        raise NotImplementedError("FII flow data is not available via Kite. Use src.data.fii pipeline.")
