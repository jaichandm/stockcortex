from datetime import datetime, timezone
from pathlib import Path
import random
import time

import httpx
import pandas as pd
import yfinance as yf

from stockcortex.config import STARTER_UNIVERSE, settings
from stockcortex.storage.duck_store import DuckStore
from stockcortex.storage.sqlite_ledger import SQLiteLedger


class MarketDataLoader:
    """Medallion Bronze -> Silver OHLCV ingestion with .NS -> .BO -> Alpha Vantage fallback."""

    def __init__(
        self,
        duck_store: DuckStore | None = None,
        ledger: SQLiteLedger | None = None,
        bronze_dir: Path | None = None,
        silver_dir: Path | None = None,
    ) -> None:
        self.duck = duck_store or DuckStore()
        self.ledger = ledger or SQLiteLedger()
        self.bronze_dir = bronze_dir or settings.bronze_dir
        self.silver_dir = silver_dir or settings.silver_dir

    @staticmethod
    def _normalize_yfinance_df(raw_df: pd.DataFrame, symbol: str, ticker_used: str) -> pd.DataFrame:
        if raw_df is None or raw_df.empty:
            return pd.DataFrame()

        df = raw_df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        df["date"] = pd.to_datetime(df[date_col]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")

        col_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Dividends": "dividends",
            "Stock Splits": "stock_splits",
        }
        df = df.rename(columns=col_map)
        for optional_col in ("dividends", "stock_splits"):
            if optional_col not in df.columns:
                df[optional_col] = 0.0

        required = ["date", "open", "high", "low", "close", "volume", "dividends", "stock_splits"]
        df = df[required].copy()
        df["symbol"] = symbol.upper()
        df["source_ticker"] = ticker_used
        return df

    def _fetch_yfinance(self, symbol: str, suffix: str, period: str = "5y") -> pd.DataFrame:
        ticker_str = f"{symbol.upper()}{suffix}"
        hist = yf.Ticker(ticker_str).history(period=period, interval="1d", auto_adjust=True)
        return self._normalize_yfinance_df(hist, symbol=symbol, ticker_used=ticker_str)

    def _fetch_alphavantage_bse(self, symbol: str) -> pd.DataFrame:
        api_key = settings.alphavantage_api_key
        if not api_key:
            return pd.DataFrame()

        ticker_str = f"{symbol.upper()}.BSE"
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker_str,
            "outputsize": "full",
            "apikey": api_key,
        }
        with httpx.Client(timeout=20.0) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()

        series = payload.get("Time Series (Daily)", {})
        if not series:
            return pd.DataFrame()

        rows = []
        for dt_str, vals in series.items():
            rows.append(
                {
                    "date": dt_str,
                    "open": float(vals["1. open"]),
                    "high": float(vals["2. high"]),
                    "low": float(vals["3. low"]),
                    "close": float(vals["4. close"]),
                    "volume": float(vals["5. volume"]),
                    "dividends": 0.0,
                    "stock_splits": 0.0,
                    "symbol": symbol.upper(),
                    "source_ticker": ticker_str,
                }
            )
        return pd.DataFrame(rows)

    def clean_and_validate_silver(self, raw_df: pd.DataFrame, symbol: str, provider: str) -> pd.DataFrame:
        """Deduplicates dates and enforces strict OHLCV mathematical invariants."""
        if raw_df.empty:
            raise ValueError(f"Empty OHLCV dataset received for {symbol} from {provider}")

        df = raw_df.copy()
        initial_count = len(df)

        # Deduplicate keeping the latest record per trading date
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        dup_dropped = initial_count - len(df)

        # Enforce strict OHLCV invariants: High >= Low, Open > 0, Close > 0, Volume >= 0
        valid_mask = (
            df["open"].notna()
            & df["high"].notna()
            & df["low"].notna()
            & df["close"].notna()
            & (df["open"] > 0)
            & (df["high"] > 0)
            & (df["low"] > 0)
            & (df["close"] > 0)
            & (df["high"] >= df["low"])
            & (df["high"] >= df["open"])
            & (df["high"] >= df["close"])
            & (df["low"] <= df["open"])
            & (df["low"] <= df["close"])
            & (df["volume"].fillna(0) >= 0)
        )

        invalid_count = int((~valid_mask).sum())
        clean_df = df.loc[valid_mask].reset_index(drop=True)

        anomalies: list[str] = []
        if dup_dropped > 0:
            anomalies.append(f"dropped_duplicates={dup_dropped}")
        if invalid_count > 0:
            anomalies.append(f"dropped_invalid_ohlc={invalid_count}")

        self.ledger.log_ingestion(
            layer="SILVER",
            symbol=symbol.upper(),
            source_provider=provider,
            status="SUCCESS" if not anomalies else "SUCCESS_WITH_WARNINGS",
            rows_processed=len(clean_df),
            anomaly_details="; ".join(anomalies) if anomalies else None,
        )
        return clean_df

    def ingest_symbol(self, symbol: str, period: str = "5y", jitter_sec: tuple[float, float] = (0.3, 0.9)) -> Path:
        """Fetches OHLCV via .NS -> .BO -> Alpha Vantage, saves Bronze snapshot, and writes Silver Parquet."""
        clean_sym = symbol.upper().strip()
        if jitter_sec[1] > 0:
            time.sleep(random.uniform(*jitter_sec))

        raw_df = pd.DataFrame()
        provider_used = ""

        # Tier 1: NSE via yfinance
        try:
            raw_df = self._fetch_yfinance(clean_sym, ".NS", period=period)
            if not raw_df.empty:
                provider_used = "yfinance_nse"
        except Exception as exc:
            self.ledger.log_ingestion("BRONZE", clean_sym, "yfinance_nse", "FALLBACK_TRIGGERED", 0, str(exc))

        # Tier 2: BSE via yfinance
        if raw_df.empty:
            try:
                raw_df = self._fetch_yfinance(clean_sym, ".BO", period=period)
                if not raw_df.empty:
                    provider_used = "yfinance_bse"
            except Exception as exc:
                self.ledger.log_ingestion("BRONZE", clean_sym, "yfinance_bse", "FALLBACK_TRIGGERED", 0, str(exc))

        # Tier 3: Alpha Vantage BSE REST API
        if raw_df.empty:
            try:
                raw_df = self._fetch_alphavantage_bse(clean_sym)
                if not raw_df.empty:
                    provider_used = "alphavantage_bse"
            except Exception as exc:
                self.ledger.log_ingestion("BRONZE", clean_sym, "alphavantage_bse", "FAILED", 0, str(exc))

        if raw_df.empty:
            self.ledger.log_ingestion("BRONZE", clean_sym, "ALL_PROVIDERS", "FAILED", 0, "No data returned")
            raise RuntimeError(f"All OHLCV providers failed for {clean_sym}")

        # 1. Write immutable UTC-timestamped Bronze snapshot
        ts_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        bronze_path = self.bronze_dir / f"{clean_sym}_ohlcv_{ts_tag}.parquet"
        self.duck.write_parquet(raw_df, bronze_path)
        self.ledger.log_ingestion("BRONZE", clean_sym, provider_used, "SUCCESS", len(raw_df))

        # 2. Clean, validate invariants, and write Silver Parquet
        silver_df = self.clean_and_validate_silver(raw_df, symbol=clean_sym, provider=provider_used)
        silver_path = self.silver_dir / f"{clean_sym}_ohlcv.parquet"
        return self.duck.write_parquet(silver_df, silver_path)

    def ingest_universe(self, symbols: list[str] | None = None, period: str = "5y") -> dict[str, Path]:
        target_symbols = symbols or list(STARTER_UNIVERSE.keys())
        outputs: dict[str, Path] = {}
        for sym in target_symbols:
            outputs[sym] = self.ingest_symbol(sym, period=period)
        return outputs

    def ingest_nifty50_benchmark(self, period: str = "5y") -> Path:
        """Fetches NIFTY 50 (^NSEI) daily bars into Silver Parquet for rolling Beta and Relative Strength."""
        hist = yf.Ticker("^NSEI").history(period=period, interval="1d", auto_adjust=True)
        raw_df = self._normalize_yfinance_df(hist, symbol="NIFTY50", ticker_used="^NSEI")
        if raw_df.empty:
            raise RuntimeError("Failed to fetch NIFTY 50 (^NSEI) benchmark data")

        ts_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.duck.write_parquet(raw_df, self.bronze_dir / f"NIFTY50_ohlcv_{ts_tag}.parquet")
        silver_df = self.clean_and_validate_silver(raw_df, symbol="NIFTY50", provider="yfinance_index")
        return self.duck.write_parquet(silver_df, self.silver_dir / "NIFTY50_ohlcv.parquet")
