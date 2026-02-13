"""TrueData feed implementation via dynamic SDK adapter.

The vendor SDK has multiple package/module variants across installs.
This module uses runtime introspection to bind whichever compatible client class
is available and then normalizes outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import import_module
from typing import Any

import pandas as pd

from src.data.contracts import (
    candle_dtos_from_frame,
    fii_dtos_from_frame,
    frame_from_candle_dtos,
    frame_from_fii_dtos,
    option_dtos_from_chain,
)
from src.data.feed import DataFeed
from src.data.fii import fetch_fii_dii
from src.data.schemas import InstrumentType, OptionChain, OptionContract, OptionGreeks


class TrueDataFeedError(RuntimeError):
    """Raised when TrueData SDK calls fail or return incompatible payloads."""


@dataclass
class TrueDataFeed(DataFeed):
    """Primary paid feed adapter for TrueData historical/chain APIs."""

    username: str
    password: str
    name: str = "truedata"
    _client: Any | None = field(default=None, init=False, repr=False)

    _INTERVAL_MAP = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "60m": "60min",
        "1h": "60min",
        "1d": "eod",
    }

    _CLIENT_IMPORT_CANDIDATES = [
        ("truedata_ws.websocket.TD", "TD"),
        ("truedata_ws.ws_td", "TD"),
        ("truedata_ws", "TD"),
    ]

    def _normalize_interval(self, timeframe: str) -> str:
        key = timeframe.lower().strip()
        if key not in self._INTERVAL_MAP:
            raise ValueError(f"Unsupported timeframe '{timeframe}' for TrueData feed")
        return self._INTERVAL_MAP[key]

    def _resolve_client_class(self) -> Any:
        for module_name, class_name in self._CLIENT_IMPORT_CANDIDATES:
            try:
                module = import_module(module_name)
                cls = getattr(module, class_name)
                return cls
            except Exception:
                continue
        raise TrueDataFeedError(
            "TrueData SDK client class not found. "
            "Install the official truedata_ws package and verify imports."
        )

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client

        client_cls = self._resolve_client_class()

        # TODO: adjust constructor args once your subscription plan/client class is finalized.
        # This covers common variants observed in SDK usage.
        attempts = [
            {"username": self.username, "password": self.password},
            {"uname": self.username, "pwd": self.password},
            {"user": self.username, "password": self.password},
            {"id": self.username, "password": self.password},
            {"username": self.username, "password": self.password, "live_port": 8082},
        ]

        for kwargs in attempts:
            try:
                self._client = client_cls(**kwargs)
                return self._client
            except TypeError:
                continue
            except Exception as exc:
                raise TrueDataFeedError(f"TrueData client init failed: {exc}") from exc

        raise TrueDataFeedError(
            "Unable to initialize TrueData client with known constructor variants. "
            "TODO: map constructor exactly for your installed SDK version."
        )

    @staticmethod
    def _normalize_candles(payload: Any) -> pd.DataFrame:
        if payload is None:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume", "oi"]
            )

        if isinstance(payload, pd.DataFrame):
            frame = payload.copy()
        else:
            try:
                frame = pd.DataFrame(payload)
            except Exception as exc:
                raise TrueDataFeedError(
                    f"Unsupported candle payload type: {type(payload)}"
                ) from exc

        if frame.empty:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume", "oi"]
            )

        rename_map = {
            "datetime": "timestamp",
            "date": "timestamp",
            "Date": "timestamp",
            "time": "timestamp",
            "open": "open",
            "Open": "open",
            "high": "high",
            "High": "high",
            "low": "low",
            "Low": "low",
            "close": "close",
            "Close": "close",
            "volume": "volume",
            "Volume": "volume",
            "oi": "oi",
            "OI": "oi",
        }
        frame = frame.rename(columns=rename_map)

        required = ["timestamp", "open", "high", "low", "close"]
        for col in required:
            if col not in frame.columns:
                raise TrueDataFeedError(
                    f"TrueData candle payload missing '{col}'. Columns={list(frame.columns)}"
                )

        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
        if "volume" not in frame.columns:
            frame["volume"] = 0
        if "oi" not in frame.columns:
            frame["oi"] = 0

        out = frame[["timestamp", "open", "high", "low", "close", "volume", "oi"]]
        out = out.drop_duplicates(subset=["timestamp"], keep="last")
        return out.reset_index(drop=True)

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        client = self._ensure_client()
        interval = self._normalize_interval(timeframe)

        method_candidates = ["get_historic_data", "get_historical_data", "get_hist_data"]
        last_error: Exception | None = None

        for method_name in method_candidates:
            method = getattr(client, method_name, None)
            if method is None:
                continue
            try:
                payload = method(symbol, start, end, interval)
                normalized = self._normalize_candles(payload)
                return frame_from_candle_dtos(
                    candle_dtos_from_frame(
                        normalized,
                        source=self.name,
                        symbol=symbol,
                        timeframe=timeframe,
                    )
                )
            except TypeError:
                # Alternate signature variant: keyword arguments.
                try:
                    payload = method(
                        symbol=symbol,
                        start_time=start,
                        end_time=end,
                        bar_size=interval,
                    )
                    normalized = self._normalize_candles(payload)
                    return frame_from_candle_dtos(
                        candle_dtos_from_frame(
                            normalized,
                            source=self.name,
                            symbol=symbol,
                            timeframe=timeframe,
                        )
                    )
                except Exception as exc:
                    last_error = exc
            except Exception as exc:
                last_error = exc

        raise TrueDataFeedError(
            "No compatible TrueData historical method succeeded. "
            "TODO: bind exact historical endpoint for your SDK build."
        ) from last_error

    @staticmethod
    def _to_option_contract(row: dict[Any, Any]) -> OptionContract:
        opt_type_raw = str(row.get("option_type", row.get("type", "CE"))).upper()
        instrument_type = InstrumentType.CALL if opt_type_raw == "CE" else InstrumentType.PUT

        greeks = OptionGreeks(
            iv=float(row.get("iv", 0.0) or 0.0),
            delta=float(row.get("delta", 0.0) or 0.0),
            gamma=float(row.get("gamma", 0.0) or 0.0),
            theta=float(row.get("theta", 0.0) or 0.0),
            vega=float(row.get("vega", 0.0) or 0.0),
            rho=float(row.get("rho", 0.0) or 0.0),
        )

        expiry_raw = row.get("expiry")
        expiry = (
            pd.Timestamp(expiry_raw).to_pydatetime()
            if expiry_raw is not None
            else datetime.now(UTC).replace(tzinfo=None)
        )

        return OptionContract(
            symbol=str(row.get("symbol", row.get("tradingsymbol", ""))),
            instrument_type=instrument_type,
            strike=float(row.get("strike", 0.0) or 0.0),
            expiry=expiry,
            ltp=float(row.get("ltp", row.get("last_price", 0.0)) or 0.0),
            bid=float(row.get("bid", 0.0) or 0.0),
            ask=float(row.get("ask", 0.0) or 0.0),
            volume=int(row.get("volume", 0) or 0),
            oi=int(row.get("oi", 0) or 0),
            change_in_oi=int(row.get("change_in_oi", 0) or 0),
            greeks=greeks,
        )

    def get_option_chain(
        self,
        symbol: str,
        expiry: datetime,
        timestamp: datetime | None = None,
    ) -> OptionChain:
        client = self._ensure_client()

        method_candidates = ["get_option_chain", "option_chain", "get_chain"]
        payload = None
        last_error: Exception | None = None

        for method_name in method_candidates:
            method = getattr(client, method_name, None)
            if method is None:
                continue
            try:
                payload = method(symbol=symbol, expiry=expiry)
                break
            except TypeError:
                try:
                    payload = method(symbol, expiry)
                    break
                except Exception as exc:
                    last_error = exc
            except Exception as exc:
                last_error = exc

        if payload is None:
            raise TrueDataFeedError(
                "No compatible TrueData option-chain method succeeded. "
                "TODO: bind chain endpoint for your SDK build."
            ) from last_error

        if isinstance(payload, pd.DataFrame):
            records = payload.to_dict(orient="records")
        elif isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict) and "contracts" in payload:
            records = payload["contracts"]
        else:
            raise TrueDataFeedError(f"Unsupported option chain payload type: {type(payload)}")

        contracts = [self._to_option_contract(row) for row in records]

        underlying_price = 0.0
        if isinstance(payload, dict):
            underlying_price = float(payload.get("underlying_price", 0.0) or 0.0)
        if underlying_price == 0.0 and contracts:
            # TODO: replace with explicit underlying snapshot from SDK.
            underlying_price = contracts[0].strike

        chain = OptionChain(
            underlying=symbol,
            underlying_price=underlying_price,
            timestamp=timestamp or datetime.now(UTC).replace(tzinfo=None),
            expiry=expiry,
            contracts=contracts,
        )
        option_dtos_from_chain(chain, source=self.name)
        return chain

    def get_vix(self, start: datetime, end: datetime) -> pd.DataFrame:
        # TODO: confirm exact TrueData symbol alias for India VIX in your account.
        symbol_candidates = ["INDIAVIX", "INDIA VIX", "VIX"]
        last_error: Exception | None = None
        for symbol in symbol_candidates:
            try:
                return self.get_candles(symbol=symbol, timeframe="1d", start=start, end=end)
            except Exception as exc:
                last_error = exc
        raise TrueDataFeedError("Unable to fetch VIX from TrueData symbol aliases.") from last_error

    def get_fii_data(self, start: datetime, end: datetime) -> pd.DataFrame:
        # TODO: if your TrueData plan includes FII/DII endpoints, wire them here.
        data = fetch_fii_dii(start=start, end=end)
        return frame_from_fii_dtos(fii_dtos_from_frame(data, source=self.name))
