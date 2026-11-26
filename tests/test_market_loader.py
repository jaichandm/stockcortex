from pathlib import Path
from unittest.mock import patch

import pandas as pd

from stockcortex.ingestion.market_loader import MarketDataLoader
from stockcortex.storage.duck_store import DuckStore
from stockcortex.storage.sqlite_ledger import SQLiteLedger


def _sample_raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-11-18", "2026-11-18", "2026-11-19", "2026-11-20"],
            "open": [100.0, 100.5, 102.0, 105.0],
            "high": [103.0, 104.0, 99.0, 108.0],   # Row index 2 has High (99) < Low (101) -> invalid!
            "low": [99.0, 99.5, 101.0, 104.0],
            "close": [102.0, 103.0, 100.0, 107.0],
            "volume": [50000.0, 52000.0, 40000.0, 61000.0],
            "dividends": [0.0, 0.0, 0.0, 0.0],
            "stock_splits": [0.0, 0.0, 0.0, 0.0],
            "symbol": ["INFY", "INFY", "INFY", "INFY"],
            "source_ticker": ["INFY.BO", "INFY.BO", "INFY.BO", "INFY.BO"],
        }
    )


def test_medallion_fallback_and_ohlc_invariants(tmp_path: Path):
    ledger = SQLiteLedger(db_path=tmp_path / "test.db")
    duck = DuckStore(memory_limit="2GB", threads=2)
    loader = MarketDataLoader(
        duck_store=duck,
        ledger=ledger,
        bronze_dir=tmp_path / "bronze",
        silver_dir=tmp_path / "silver",
    )

    with patch.object(loader, "_fetch_yfinance") as mock_yf:
        # Simulate .NS returning empty, .BO returning sample data with 1 duplicate + 1 invalid OHLC bar
        mock_yf.side_effect = [pd.DataFrame(), _sample_raw_df()]
        silver_path = loader.ingest_symbol("INFY", period="1y", jitter_sec=(0.0, 0.0))

    assert silver_path.exists()
    bronze_files = list((tmp_path / "bronze").glob("INFY_ohlcv_*.parquet"))
    assert len(bronze_files) == 1

    silver_df = duck.read_parquet(silver_path)
    # 4 raw rows -> 1 duplicate removed -> 1 invalid High<Low removed -> 2 valid rows remain
    assert len(silver_df) == 2
    assert list(silver_df["date"]) == ["2026-11-18", "2026-11-20"]
    assert silver_df.loc[0, "close"] == 103.0

    with ledger.connect() as conn:
        logs = conn.execute("SELECT layer, source_provider, status, anomaly_details FROM ingestion_audit_log").fetchall()
    assert len(logs) == 2
    assert logs[1]["status"] == "SUCCESS_WITH_WARNINGS"
    assert "dropped_duplicates=1" in logs[1]["anomaly_details"]
    assert "dropped_invalid_ohlc=1" in logs[1]["anomaly_details"]
