from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from stockcortex.features.gold_builder import GoldFeatureBuilder
from stockcortex.ingestion.news_loader import EntityNewsLoader, map_published_to_market_date
from stockcortex.storage.duck_store import DuckStore
from stockcortex.storage.sqlite_ledger import SQLiteLedger


def test_post_1530_ist_news_timestamp_shift():
    # 15:25 IST = 09:55 UTC on Wednesday 2026-11-25 -> stays on 2026-11-25
    pre_close_utc = datetime(2026, 11, 25, 9, 55, tzinfo=timezone.utc)
    assert map_published_to_market_date(pre_close_utc) == "2026-11-25"

    # 15:35 IST = 10:05 UTC on Wednesday 2026-11-25 -> shifts to Thursday 2026-11-26
    post_close_utc = datetime(2026, 11, 25, 10, 5, tzinfo=timezone.utc)
    assert map_published_to_market_date(post_close_utc) == "2026-11-26"

    # Friday 16:00 IST = 10:30 UTC on Friday 2026-11-27 -> shifts across weekend to Monday 2026-11-30
    friday_post_close_utc = datetime(2026, 11, 27, 10, 30, tzinfo=timezone.utc)
    assert map_published_to_market_date(friday_post_close_utc) == "2026-11-30"


def test_entity_news_filtering(tmp_path: Path):
    ledger = SQLiteLedger(db_path=tmp_path / "test.db")
    duck = DuckStore(memory_limit="2GB", threads=2)
    loader = EntityNewsLoader(
        duck_store=duck,
        ledger=ledger,
        bronze_dir=tmp_path / "bronze",
        silver_dir=tmp_path / "silver",
    )

    mock_articles = [
        {
            "title": "Reliance Industries announces green energy expansion",
            "description": "RIL board approves capex.",
            "source_name": "Economic Times",
            "url": "https://example.com/1",
            "published_utc": "2026-11-25T10:15:00+00:00",  # 15:45 IST -> shifts to 2026-11-26
            "provider": "google_news_rss",
        },
        {
            "title": "Unrelated global airline merger update",
            "description": "European carriers consolidate.",
            "source_name": "Reuters",
            "url": "https://example.com/2",
            "published_utc": "2026-11-25T08:00:00+00:00",
            "provider": "google_news_rss",
        },
    ]

    with patch.object(loader, "_fetch_newsdata_io", return_value=[]), patch.object(
        loader, "_fetch_google_news_rss", return_value=mock_articles
    ):
        out_path = loader.ingest_symbol_news("RELIANCE")

    df = duck.read_parquet(out_path)
    assert len(df) == 1
    assert df.loc[0, "market_date"] == "2026-11-26"


def test_gold_technical_causality(tmp_path: Path):
    """Perturbing future prices after day t must NOT alter day t Gold features."""
    dates = pd.bdate_range("2025-01-01", periods=250).strftime("%Y-%m-%d").tolist()
    rng = np.random.default_rng(42)
    prices = 1000.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, size=250)))

    df_orig = pd.DataFrame(
        {
            "date": dates,
            "open": prices * 0.998,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": rng.integers(100_000, 1_000_000, size=250).astype(float),
            "symbol": "TCS",
        }
    )

    df_perturbed = df_orig.copy()
    # Spike prices drastically on the last 10 days (days 240..249)
    df_perturbed.loc[240:, ["open", "high", "low", "close"]] *= 5.0

    gold_orig = GoldFeatureBuilder.compute_technical_features(df_orig)
    gold_pert = GoldFeatureBuilder.compute_technical_features(df_perturbed)

    feature_cols = [c for c in gold_orig.columns if c not in {"date", "open", "high", "low", "close", "volume", "symbol"}]
    pd.testing.assert_frame_equal(gold_orig.loc[:239, feature_cols], gold_pert.loc[:239, feature_cols])
