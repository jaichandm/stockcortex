from pathlib import Path

import numpy as np
import pandas as pd

from stockcortex.config import STARTER_UNIVERSE, settings
from stockcortex.storage.duck_store import DuckStore
from stockcortex.storage.sqlite_ledger import SQLiteLedger


class GoldFeatureBuilder:
    """Builds strictly causal, dimensionless technical and cross-sectional ML features."""

    def __init__(
        self,
        duck_store: DuckStore | None = None,
        ledger: SQLiteLedger | None = None,
        silver_dir: Path | None = None,
        gold_dir: Path | None = None,
    ) -> None:
        self.duck = duck_store or DuckStore()
        self.ledger = ledger or SQLiteLedger()
        self.silver_dir = silver_dir or settings.silver_dir
        self.gold_dir = gold_dir or settings.gold_dir

    @staticmethod
    def compute_technical_features(df: pd.DataFrame, benchmark_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Transforms Silver OHLCV bars into stationary, leakage-free Gold features."""
        out = df.sort_values("date").reset_index(drop=True).copy()
        close = out["close"].astype(float)
        open_ = out["open"].astype(float)
        high = out["high"].astype(float)
        low = out["low"].astype(float)
        volume = out["volume"].astype(float)

        # 1. Multi-Horizon Log Returns & Momentum (strictly backward rolling)
        log_close = np.log(close)
        for w in (1, 5, 21, 63, 126):
            out[f"log_ret_{w}d"] = log_close.diff(w)

        for w in (10, 20, 50, 200):
            sma = close.rolling(window=w, min_periods=w).mean()
            out[f"price_to_sma_{w}"] = (close / sma) - 1.0

        ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        out["macd_norm"] = macd / close
        out["macd_hist_norm"] = (macd - macd_signal) / close
        out["roc_10d"] = close.pct_change(10)

        # 2. Volatility & Microstructure
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        out["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

        bb_mid = close.rolling(window=20, min_periods=20).mean()
        bb_std = close.rolling(window=20, min_periods=20).std(ddof=0)
        bb_upper = bb_mid + 2.0 * bb_std
        bb_lower = bb_mid - 2.0 * bb_std
        bb_width = (bb_upper - bb_lower).replace(0.0, np.nan)
        out["bb_pct_b"] = (close - bb_lower) / bb_width
        out["bb_bandwidth"] = (bb_upper - bb_lower) / bb_mid

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_14 = tr.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
        out["atr_14_norm"] = atr_14 / close

        # Garman-Klass 21d Annualized Volatility
        log_hl = np.log((high / low).clip(lower=1e-9))
        log_co = np.log((close / open_).clip(lower=1e-9))
        gk_daily_var = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
        out["gk_vol_21d"] = np.sqrt(gk_daily_var.clip(lower=0.0).rolling(21, min_periods=21).mean() * 252.0)

        # Volume Z-Score (20d) with zero-volume days masked
        vol_nonzero = volume.replace(0.0, np.nan)
        vol_mean = vol_nonzero.rolling(20, min_periods=10).mean()
        vol_std = vol_nonzero.rolling(20, min_periods=10).std(ddof=0).replace(0.0, np.nan)
        out["volume_zscore_20d"] = (vol_nonzero - vol_mean) / vol_std

        # 3. Cross-Sectional Market Regime (Rolling 63d Beta & 21d Relative Strength vs NIFTY 50)
        if benchmark_df is not None and not benchmark_df.empty:
            bench = benchmark_df[["date", "close"]].drop_duplicates("date").sort_values("date").copy()
            bench["bench_log_ret_1d"] = np.log(bench["close"].astype(float)).diff(1)
            bench["bench_log_ret_21d"] = np.log(bench["close"].astype(float)).diff(21)
            out = out.merge(bench[["date", "bench_log_ret_1d", "bench_log_ret_21d"]], on="date", how="left")

            cov_63 = out["log_ret_1d"].rolling(63, min_periods=42).cov(out["bench_log_ret_1d"])
            var_63 = out["bench_log_ret_1d"].rolling(63, min_periods=42).var().replace(0.0, np.nan)
            out["beta_nifty_63d"] = cov_63 / var_63
            out["rel_strength_nifty_21d"] = out["log_ret_21d"] - out["bench_log_ret_21d"]
            out = out.drop(columns=["bench_log_ret_1d", "bench_log_ret_21d"])
        else:
            out["beta_nifty_63d"] = np.nan
            out["rel_strength_nifty_21d"] = np.nan

        return out

    def build_symbol_features(self, symbol: str) -> Path:
        clean_sym = symbol.upper().strip()
        silver_path = self.silver_dir / f"{clean_sym}_ohlcv.parquet"
        df = self.duck.read_parquet(silver_path)

        bench_path = self.silver_dir / "NIFTY50_ohlcv.parquet"
        bench_df = self.duck.read_parquet(bench_path) if bench_path.exists() else None

        gold_df = self.compute_technical_features(df, benchmark_df=bench_df)
        gold_path = self.gold_dir / f"{clean_sym}_features.parquet"
        self.duck.write_parquet(gold_df, gold_path)

        self.ledger.log_ingestion(
            layer="GOLD",
            symbol=clean_sym,
            source_provider="gold_technical_builder",
            status="SUCCESS",
            rows_processed=len(gold_df),
        )
        return gold_path

    def build_universe(self, symbols: list[str] | None = None) -> dict[str, Path]:
        target_symbols = symbols or list(STARTER_UNIVERSE.keys())
        return {sym: self.build_symbol_features(sym) for sym in target_symbols}
