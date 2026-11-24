from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import duckdb
import pandas as pd

from stockcortex.config import settings


class DuckStore:
    """Hardware-calibrated DuckDB analytical engine for Bronze/Silver/Gold Parquet files."""

    def __init__(
        self,
        memory_limit: str | None = None,
        threads: int | None = None,
    ) -> None:
        self.memory_limit = memory_limit or settings.duckdb_memory_limit
        self.threads = threads or settings.duckdb_threads

    @contextmanager
    def session(self) -> Iterator[duckdb.DuckDBPyConnection]:
        con = duckdb.connect(database=":memory:")
        try:
            con.execute(f"SET memory_limit='{self.memory_limit}';")
            con.execute(f"SET threads={self.threads};")
            yield con
        finally:
            con.close()

    def get_runtime_settings(self) -> dict[str, str]:
        with self.session() as con:
            mem = con.execute("SELECT current_setting('memory_limit');").fetchone()[0]
            thr = con.execute("SELECT current_setting('threads');").fetchone()[0]
        return {"memory_limit": str(mem), "threads": str(thr)}

    def write_parquet(self, df: pd.DataFrame, target_path: Path) -> Path:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(target_path, index=False, engine="pyarrow", compression="snappy")
        return target_path

    def query_parquet(self, sql: str) -> pd.DataFrame:
        with self.session() as con:
            return con.execute(sql).df()

    def read_parquet(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Parquet artifact not found: {path}")
        escaped = path.as_posix()
        return self.query_parquet(f"SELECT * FROM read_parquet('{escaped}')")
