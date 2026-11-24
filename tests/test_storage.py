import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from stockcortex.storage.duck_store import DuckStore
from stockcortex.storage.sqlite_ledger import SQLiteLedger


def test_sqlite_wal_and_immutable_ledger(tmp_path: Path):
    db_file = tmp_path / "test_stockcortex.db"
    ledger = SQLiteLedger(db_path=db_file)

    pragmas = ledger.get_pragmas()
    assert pragmas["journal_mode"] == "wal"
    assert pragmas["foreign_keys"] == 1

    audit_id = ledger.log_ingestion(
        layer="SILVER",
        symbol="RELIANCE",
        source_provider="yfinance",
        status="SUCCESS",
        rows_processed=1237,
    )
    assert audit_id == 1

    ledger.record_prediction(
        prediction_id="pred_rel_1m_001",
        symbol="RELIANCE",
        horizon="1M",
        model_version="lgbm-q-v0.1",
        as_of_date="2026-10-01",
        target_date="2026-11-02",
        ref_price=2950.0,
        pred_p10=2810.0,
        pred_p50=3040.0,
        pred_p90=3220.0,
        recommendation="Consider Investing",
        feature_snapshot={"rsi_14": 54.2, "ret_21d": 0.031},
    )

    # Duplicate prediction_id must raise IntegrityError (immutability guarantee)
    with pytest.raises(sqlite3.IntegrityError):
        ledger.record_prediction(
            prediction_id="pred_rel_1m_001",
            symbol="RELIANCE",
            horizon="1M",
            model_version="lgbm-q-v0.1",
            as_of_date="2026-10-01",
            target_date="2026-11-02",
            ref_price=2950.0,
            pred_p10=2800.0,
            pred_p50=3000.0,
            pred_p90=3200.0,
            recommendation="Hold",
            feature_snapshot={},
        )

    pending = ledger.fetch_pending_evaluations(current_date="2026-11-15")
    assert len(pending) == 1
    assert pending[0]["symbol"] == "RELIANCE"
    assert pending[0]["pred_p50"] == 3040.0


def test_duck_store_memory_limit_and_parquet_roundtrip(tmp_path: Path):
    store = DuckStore(memory_limit="6GB", threads=4)
    runtime = store.get_runtime_settings()
    assert runtime["threads"] == "4"
    assert "GiB" in runtime["memory_limit"] or "GB" in runtime["memory_limit"]

    df = pd.DataFrame(
        {
            "date": ["2026-11-20", "2026-11-21"],
            "symbol": ["TCS", "TCS"],
            "close": [4120.5, 4155.0],
        }
    )
    pq_path = tmp_path / "tcs_silver.parquet"
    store.write_parquet(df, pq_path)

    loaded = store.read_parquet(pq_path)
    assert len(loaded) == 2
    assert list(loaded["close"]) == [4120.5, 4155.0]
