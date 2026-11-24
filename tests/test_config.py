from stockcortex.config import STARTER_UNIVERSE, settings


def test_hardware_calibration():
    assert settings.duckdb_memory_limit == "6GB"
    assert settings.duckdb_threads == 4
    assert settings.finbert_batch_size == 16
    assert settings.finbert_max_length == 128


def test_medallion_paths_exist():
    assert settings.bronze_dir.exists()
    assert settings.silver_dir.exists()
    assert settings.gold_dir.exists()
    assert settings.sqlite_db_path.name == "stockcortex.db"
    assert "RELIANCE" in STARTER_UNIVERSE
