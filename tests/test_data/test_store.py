from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.data.store import DataStore


def test_write_and_read_timeseries(tmp_path):
    store = DataStore(base_dir=str(tmp_path / "cache"))

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-02", "2026-01-03"]),
            "open": [1.0, 2.0],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.2, 2.2],
            "volume": [10, 20],
        }
    )

    written = store.write_time_series("candles", df, symbol="TEST", timeframe="1d")
    assert written == 2

    loaded = store.read_time_series(
        "candles",
        symbol="TEST",
        timeframe="1d",
        start=datetime(2026, 1, 1),
        end=datetime(2026, 1, 31),
    )
    assert len(loaded) == 2
    assert list(loaded.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_upsert_deduplicates_and_reports_net_new_rows(tmp_path):
    store = DataStore(base_dir=str(tmp_path / "cache"))

    first = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-02", "2026-01-03"]),
            "value": [1, 2],
        }
    )
    second = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-03", "2026-01-04"]),
            "value": [3, 4],
        }
    )

    assert store.write_time_series("custom", first) == 2
    assert store.write_time_series("custom", second) == 1

    loaded = store.read_time_series("custom")
    assert len(loaded) == 3
    # 2026-01-03 should have been replaced by value=3 from second write
    assert loaded.loc[loaded["timestamp"] == pd.Timestamp("2026-01-03"), "value"].iloc[0] == 3


def test_upsert_with_composite_dedup_columns(tmp_path):
    store = DataStore(base_dir=str(tmp_path / "cache"))

    first = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-03 10:00", "2026-01-03 10:00"]),
            "expiry": pd.to_datetime(["2026-01-08", "2026-01-08"]),
            "strike": [25000.0, 25100.0],
            "option_type": ["CE", "CE"],
            "oi": [100, 200],
        }
    )
    second = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-03 10:00", "2026-01-03 10:00"]),
            "expiry": pd.to_datetime(["2026-01-08", "2026-01-08"]),
            "strike": [25100.0, 25200.0],
            "option_type": ["CE", "CE"],
            "oi": [220, 150],
        }
    )

    dedup_cols = ["timestamp", "expiry", "strike", "option_type"]
    assert store.write_time_series("option_chain", first, dedup_cols=dedup_cols) == 2
    assert store.write_time_series("option_chain", second, dedup_cols=dedup_cols) == 1

    loaded = store.read_time_series("option_chain")
    assert len(loaded) == 3
    replaced = loaded.loc[
        (loaded["timestamp"] == pd.Timestamp("2026-01-03 10:00"))
        & (loaded["expiry"] == pd.Timestamp("2026-01-08"))
        & (loaded["strike"] == 25100.0)
        & (loaded["option_type"] == "CE"),
        "oi",
    ]
    assert len(replaced) == 1
    assert int(replaced.iloc[0]) == 220


def test_write_time_series_raises_for_missing_dedup_column(tmp_path):
    store = DataStore(base_dir=str(tmp_path / "cache"))
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-03"]),
            "value": [1],
        }
    )
    try:
        store.write_time_series("custom", df, dedup_cols=["timestamp", "missing_col"])
    except ValueError as exc:
        assert "missing_col" in str(exc)
        return
    raise AssertionError("Expected ValueError for missing dedup column")


def test_count_rows_reads_parquet_metadata(tmp_path):
    store = DataStore(base_dir=str(tmp_path / "cache"))
    directory = store._dataset_dir("row_count_test")
    pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]), "v": [1, 2]}).to_parquet(
        directory / "2026.parquet", index=False
    )
    pd.DataFrame({"timestamp": pd.to_datetime(["2027-01-01"]), "v": [3]}).to_parquet(
        directory / "2027.parquet", index=False
    )
    assert store._count_rows(directory) == 3


def test_read_time_series_prunes_partitions_by_year(tmp_path, monkeypatch):
    store = DataStore(base_dir=str(tmp_path / "cache"))
    directory = store._dataset_dir("candles", symbol="TEST", timeframe="1d")
    pd.DataFrame({"timestamp": pd.to_datetime(["2025-01-01"]), "close": [1.0]}).to_parquet(
        directory / "2025.parquet", index=False
    )
    pd.DataFrame({"timestamp": pd.to_datetime(["2026-06-01"]), "close": [2.0]}).to_parquet(
        directory / "2026.parquet", index=False
    )
    pd.DataFrame({"timestamp": pd.to_datetime(["2027-01-01"]), "close": [3.0]}).to_parquet(
        directory / "2027.parquet", index=False
    )

    called: list[str] = []
    original = pd.read_parquet

    def _spy(path, *args, **kwargs):
        called.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", _spy)
    loaded = store.read_time_series(
        "candles",
        symbol="TEST",
        timeframe="1d",
        start=datetime(2026, 1, 1),
        end=datetime(2026, 12, 31),
    )
    assert len(loaded) == 1
    assert loaded.iloc[0]["close"] == 2.0
    assert any("2026.parquet" in p for p in called)
    assert all("2025.parquet" not in p for p in called)
    assert all("2027.parquet" not in p for p in called)
