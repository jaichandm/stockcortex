from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Zero-Cost API Credentials
    gemini_api_key: str = ""
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    newsdata_api_key: str = ""
    alphavantage_api_key: str = ""
    hf_token: str = ""

    # Hardware Calibration (Windows 11 | 16 GB Physical RAM)
    duckdb_memory_limit: str = "6GB"
    duckdb_threads: int = 4
    ml_n_jobs: int = 4
    finbert_batch_size: int = 16
    finbert_max_length: int = 128

    # Project Paths
    base_dir: Path = Path(__file__).resolve().parents[2]

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def bronze_dir(self) -> Path:
        return self.data_dir / "bronze"

    @property
    def silver_dir(self) -> Path:
        return self.data_dir / "silver"

    @property
    def gold_dir(self) -> Path:
        return self.data_dir / "gold"

    @property
    def sqlite_db_path(self) -> Path:
        return self.data_dir / "stockcortex.db"


# Curated Starter NIFTY Universe (expandable to full NIFTY 100)
STARTER_UNIVERSE: dict[str, str] = {
    "RELIANCE": "Reliance Industries Ltd",
    "TCS": "Tata Consultancy Services Ltd",
    "HDFCBANK": "HDFC Bank Ltd",
    "INFY": "Infosys Ltd",
    "ICICIBANK": "ICICI Bank Ltd",
    "BHARTIARTL": "Bharti Airtel Ltd",
    "SBIN": "State Bank of India",
    "LT": "Larsen & Toubro Ltd",
    "ITC": "ITC Ltd",
    "HINDUNILVR": "Hindustan Unilever Ltd",
}

settings = Settings()
