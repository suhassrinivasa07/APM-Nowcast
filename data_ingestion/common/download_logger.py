"""
Dataset inventory manager — JSON-lines append-only log + SQLite index.

Usage::

    from data_ingestion.common.download_logger import InventoryLogger
    inv = InventoryLogger(Path("data/inventory"))
    inv.record(record)   # DatasetInventoryRecord
    records = inv.query(source="ERA5")
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from common.logging_config import get_logger
from common.schemas import DatasetInventoryRecord, IngestionStatus

log = get_logger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS inventory (
    record_id        TEXT PRIMARY KEY,
    source           TEXT NOT NULL,
    dataset_type     TEXT NOT NULL,
    file_path        TEXT NOT NULL,
    file_size_bytes  INTEGER DEFAULT 0,
    checksum_md5     TEXT,
    status           TEXT NOT NULL,
    valid_time_start TEXT,
    valid_time_end   TEXT,
    variables        TEXT,
    ingested_at      TEXT NOT NULL,
    error_message    TEXT,
    metadata_json    TEXT
);
CREATE INDEX IF NOT EXISTS idx_source   ON inventory(source);
CREATE INDEX IF NOT EXISTS idx_status   ON inventory(status);
CREATE INDEX IF NOT EXISTS idx_ingested ON inventory(ingested_at);
"""


class InventoryLogger:
    """Thread-safe (single-process) JSON-lines + SQLite inventory."""

    def __init__(self, inventory_dir: Path) -> None:
        self._dir = inventory_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = self._dir / "inventory.jsonl"
        self._db    = self._dir / "inventory.db"
        self._conn  = sqlite3.connect(str(self._db), check_same_thread=False)
        self._conn.executescript(_DDL)
        self._conn.commit()
        log.info("InventoryLogger initialised", extra={"db": str(self._db)})

    def record(self, rec: DatasetInventoryRecord) -> None:
        """Append record to both JSON-lines and SQLite."""
        data = rec.model_dump(mode="json")
        with open(self._jsonl, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(data, default=str) + "\n")

        self._conn.execute(
            """
            INSERT OR REPLACE INTO inventory
            (record_id, source, dataset_type, file_path, file_size_bytes,
             checksum_md5, status, valid_time_start, valid_time_end,
             variables, ingested_at, error_message, metadata_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                rec.record_id,
                rec.source,
                rec.dataset_type,
                rec.file_path,
                rec.file_size_bytes,
                rec.checksum_md5,
                rec.status.value,
                rec.valid_time_start.isoformat() if rec.valid_time_start else None,
                rec.valid_time_end.isoformat()   if rec.valid_time_end   else None,
                json.dumps(rec.variables),
                rec.ingested_at.isoformat(),
                rec.error_message,
                json.dumps(rec.metadata, default=str),
            ),
        )
        self._conn.commit()
        log.info(
            "Inventory record saved",
            extra={"record_id": rec.record_id, "source": rec.source, "status": rec.status},
        )

    def update_status(self, record_id: str, status: IngestionStatus, error: str = "") -> None:
        self._conn.execute(
            "UPDATE inventory SET status=?, error_message=? WHERE record_id=?",
            (status.value, error or None, record_id),
        )
        self._conn.commit()

    def query(
        self,
        source: Optional[str] = None,
        status: Optional[IngestionStatus] = None,
        dataset_type: Optional[str] = None,
    ) -> list[dict]:
        """Return matching inventory rows as plain dicts."""
        where, params = [], []
        if source:
            where.append("source = ?"); params.append(source)
        if status:
            where.append("status = ?"); params.append(status.value)
        if dataset_type:
            where.append("dataset_type = ?"); params.append(dataset_type)
        clause = "WHERE " + " AND ".join(where) if where else ""
        cur = self._conn.execute(
            f"SELECT * FROM inventory {clause} ORDER BY ingested_at DESC", params
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def summary(self) -> dict:
        """Return counts grouped by source and status."""
        rows = self._conn.execute(
            "SELECT source, status, COUNT(*) AS cnt FROM inventory GROUP BY source, status"
        ).fetchall()
        result: dict = {}
        for src, sts, cnt in rows:
            result.setdefault(src, {})[sts] = cnt
        return result

    def close(self) -> None:
        self._conn.close()
