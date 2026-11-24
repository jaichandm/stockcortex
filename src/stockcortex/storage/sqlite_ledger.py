import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from stockcortex.config import settings


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS prediction_ledger (
    prediction_id TEXT PRIMARY KEY,
    created_at_utc TEXT NOT NULL,
    symbol TEXT NOT NULL,
    horizon TEXT NOT NULL,
    model_version TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    target_date TEXT NOT NULL,
    ref_price REAL NOT NULL,
    pred_p10 REAL NOT NULL,
    pred_p50 REAL NOT NULL,
    pred_p90 REAL NOT NULL,
    recommendation TEXT NOT NULL,
    feature_snapshot_json TEXT NOT NULL,
    evaluation_status TEXT NOT NULL DEFAULT 'PENDING',
    actual_close REAL,
    evaluated_at_utc TEXT
);

CREATE INDEX IF NOT EXISTS idx_ledger_symbol_horizon
    ON prediction_ledger (symbol, horizon, target_date);

CREATE INDEX IF NOT EXISTS idx_ledger_eval_status
    ON prediction_ledger (evaluation_status, target_date);

CREATE TABLE IF NOT EXISTS ingestion_audit_log (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at_utc TEXT NOT NULL,
    layer TEXT NOT NULL,
    symbol TEXT NOT NULL,
    source_provider TEXT NOT NULL,
    status TEXT NOT NULL,
    rows_processed INTEGER NOT NULL DEFAULT 0,
    anomaly_details TEXT
);

CREATE TABLE IF NOT EXISTS portfolios (
    position_id INTEGER PRIMARY KEY AUTOINCREMENT,
    opened_at_utc TEXT NOT NULL,
    symbol TEXT NOT NULL,
    shares INTEGER NOT NULL CHECK (shares > 0),
    entry_price REAL NOT NULL CHECK (entry_price > 0),
    stop_loss_price REAL,
    horizon TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'OPEN'
);
"""


class SQLiteLedger:
    """Transactional WAL-mode SQLite store for immutable predictions, audit logs, and paper portfolios."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or settings.sqlite_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)

    def get_pragmas(self) -> dict[str, Any]:
        with self.connect() as conn:
            journal = conn.execute("PRAGMA journal_mode;").fetchone()[0]
            fk = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        return {"journal_mode": str(journal).lower(), "foreign_keys": int(fk)}

    def log_ingestion(
        self,
        layer: str,
        symbol: str,
        source_provider: str,
        status: str,
        rows_processed: int = 0,
        anomaly_details: str | None = None,
    ) -> int:
        now_utc = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO ingestion_audit_log
                    (logged_at_utc, layer, symbol, source_provider, status, rows_processed, anomaly_details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (now_utc, layer, symbol, source_provider, status, rows_processed, anomaly_details),
            )
            return int(cur.lastrowid)

    def record_prediction(
        self,
        prediction_id: str,
        symbol: str,
        horizon: str,
        model_version: str,
        as_of_date: str,
        target_date: str,
        ref_price: float,
        pred_p10: float,
        pred_p50: float,
        pred_p90: float,
        recommendation: str,
        feature_snapshot: dict[str, Any],
    ) -> None:
        """Inserts an immutable prediction record; raises IntegrityError on duplicate ID."""
        now_utc = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO prediction_ledger (
                    prediction_id, created_at_utc, symbol, horizon, model_version,
                    as_of_date, target_date, ref_price, pred_p10, pred_p50, pred_p90,
                    recommendation, feature_snapshot_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction_id,
                    now_utc,
                    symbol,
                    horizon,
                    model_version,
                    as_of_date,
                    target_date,
                    float(ref_price),
                    float(pred_p10),
                    float(pred_p50),
                    float(pred_p90),
                    recommendation,
                    json.dumps(feature_snapshot, sort_keys=True),
                ),
            )

    def fetch_pending_evaluations(self, current_date: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM prediction_ledger
                WHERE evaluation_status = 'PENDING' AND target_date <= ?
                ORDER BY target_date ASC
                """,
                (current_date,),
            ).fetchall()
        return [dict(r) for r in rows]
