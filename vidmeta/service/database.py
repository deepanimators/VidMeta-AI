from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from vidmeta.settings import AppSettings, database_path


SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS uploads (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    content_type TEXT,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    expected_size_bytes INTEGER,
    path TEXT NOT NULL,
    storage_backend TEXT NOT NULL DEFAULT 'local_disk',
    status TEXT NOT NULL DEFAULT 'created',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_path TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'single',
    status TEXT NOT NULL,
    stage TEXT NOT NULL DEFAULT 'queued',
    progress INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    request_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS job_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    progress INTEGER NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS transcripts (
    job_id TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS metadata_outputs (
    job_id TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    analysis TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    raw_output TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_job_events_job_id ON job_events(job_id, id);
CREATE INDEX IF NOT EXISTS idx_uploads_created_at ON uploads(created_at);
"""

# Settings fields encrypted at rest.
_SENSITIVE_PATHS: list[tuple[str, ...]] = [
    ("provider_settings", "api_key"),
    ("storage_settings", "s3_access_key_id"),
    ("storage_settings", "s3_secret_access_key"),
]


def _encrypt_settings(data: dict[str, Any]) -> dict[str, Any]:
    from vidmeta.security import encrypt_secret
    data = json.loads(json.dumps(data))
    for *parents, field in _SENSITIVE_PATHS:
        node = data
        for key in parents:
            node = node.get(key, {})
        if isinstance(node, dict) and node.get(field):
            node[field] = encrypt_secret(node[field])
    return data


def _decrypt_settings(data: dict[str, Any]) -> dict[str, Any]:
    from vidmeta.security import decrypt_secret
    data = json.loads(json.dumps(data))
    for *parents, field in _SENSITIVE_PATHS:
        node = data
        for key in parents:
            node = node.get(key, {})
        if isinstance(node, dict) and node.get(field):
            node[field] = decrypt_secret(node[field])
    return data


class Database:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or database_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def get_settings(self) -> AppSettings:
        with self.connect() as conn:
            row = conn.execute("SELECT value_json FROM settings WHERE key = 'app'").fetchone()
        if not row:
            return AppSettings()
        raw = json.loads(row["value_json"])
        decrypted = _decrypt_settings(raw)
        return AppSettings.model_validate(decrypted)

    def save_settings(self, settings: AppSettings) -> AppSettings:
        raw = json.loads(settings.model_dump_json())
        encrypted = _encrypt_settings(raw)
        payload = json.dumps(encrypted)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO settings (key, value_json, updated_at)
                VALUES ('app', ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (payload,),
            )
        return settings

    def create_upload(self, data: dict[str, Any]) -> dict[str, Any]:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO uploads
                    (id, filename, content_type, size_bytes, expected_size_bytes, path, storage_backend, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["id"],
                    data["filename"],
                    data.get("content_type", ""),
                    data.get("size_bytes", 0),
                    data.get("expected_size_bytes"),
                    data["path"],
                    data.get("storage_backend", "local_disk"),
                    data.get("status", "created"),
                ),
            )
        return data

    def update_upload(self, upload_id: str, **values: Any) -> None:
        if not values:
            return
        assignments: list[str] = []
        params: list[Any] = []
        for key, value in values.items():
            if value == "CURRENT_TIMESTAMP":
                assignments.append(f"{key} = CURRENT_TIMESTAMP")
            else:
                assignments.append(f"{key} = ?")
                params.append(value)
        params.append(upload_id)
        with self.connect() as conn:
            conn.execute(f"UPDATE uploads SET {', '.join(assignments)} WHERE id = ?", params)

    def get_upload(self, upload_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
        return dict(row) if row else None

    def delete_expired_uploads(self, retention_days: int) -> int:
        """Delete upload records older than retention_days. Returns count deleted."""
        with self.connect() as conn:
            result = conn.execute(
                """
                DELETE FROM uploads
                WHERE created_at < datetime('now', ? || ' days')
                AND status IN ('complete', 'stored', 'failed')
                """,
                (f"-{retention_days}",),
            )
        return result.rowcount

    def create_job(self, data: dict[str, Any]) -> dict[str, Any]:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs
                    (id, source_type, source_path, mode, status, stage, progress, request_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["id"],
                    data["source_type"],
                    data["source_path"],
                    data.get("mode", "single"),
                    data.get("status", "queued"),
                    data.get("stage", "queued"),
                    data.get("progress", 0),
                    json.dumps(data.get("request", {})),
                ),
            )
            conn.execute(
                """
                INSERT INTO job_events (job_id, stage, progress, message, details_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    data["id"],
                    data.get("stage", "queued"),
                    data.get("progress", 0),
                    data.get("message", "Job queued"),
                    json.dumps(data.get("details", {})),
                ),
            )
        data["events"] = self.list_job_events(data["id"])
        return data

    def update_job(self, job_id: str, **values: Any) -> None:
        if not values:
            return
        values["updated_at"] = "CURRENT_TIMESTAMP"
        assignments: list[str] = []
        params: list[Any] = []
        for key, value in values.items():
            if value == "CURRENT_TIMESTAMP":
                assignments.append(f"{key} = CURRENT_TIMESTAMP")
            else:
                assignments.append(f"{key} = ?")
                params.append(value)
        params.append(job_id)
        with self.connect() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(assignments)} WHERE id = ?", params)

    def delete_job(self, job_id: str) -> bool:
        """Delete job and all cascade-linked records (events, transcript, metadata)."""
        with self.connect() as conn:
            result = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        return result.rowcount > 0

    def add_job_event(
        self,
        job_id: str,
        stage: str,
        progress: int,
        message: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO job_events (job_id, stage, progress, message, details_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job_id, stage, progress, message, json.dumps(details or {})),
            )

    def list_job_events(self, job_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, job_id, stage, progress, message, details_json, created_at
                FROM job_events
                WHERE job_id = ?
                ORDER BY id ASC
                """,
                (job_id,),
            ).fetchall()
        events: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["details"] = json.loads(item.pop("details_json") or "{}")
            events.append(item)
        return events

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._with_events(self._decode_job(row)) if row else None

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._with_events(self._decode_job(row)) for row in rows]

    def save_result(self, job_id: str, transcript: str, analysis: str, metadata: dict[str, Any], raw: str | None) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO transcripts (job_id, text)
                VALUES (?, ?)
                ON CONFLICT(job_id) DO UPDATE SET text = excluded.text
                """,
                (job_id, transcript),
            )
            conn.execute(
                """
                INSERT INTO metadata_outputs (job_id, analysis, metadata_json, raw_output)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    analysis = excluded.analysis,
                    metadata_json = excluded.metadata_json,
                    raw_output = excluded.raw_output
                """,
                (job_id, analysis, json.dumps(metadata), raw),
            )

    def get_result(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT j.*, t.text AS transcript, m.analysis, m.metadata_json, m.raw_output
                FROM jobs j
                LEFT JOIN transcripts t ON t.job_id = j.id
                LEFT JOIN metadata_outputs m ON m.job_id = j.id
                WHERE j.id = ?
                """,
                (job_id,),
            ).fetchone()
        if not row:
            return None
        data = self._decode_job(row)
        data["transcript"] = row["transcript"] or ""
        data["analysis"] = row["analysis"] or ""
        data["metadata"] = json.loads(row["metadata_json"]) if row["metadata_json"] else None
        data["raw_output"] = row["raw_output"]
        return self._with_events(data)

    def _with_events(self, job: dict[str, Any]) -> dict[str, Any]:
        events = self.list_job_events(job["id"])
        if not events:
            events = [
                {
                    "id": 0,
                    "job_id": job["id"],
                    "stage": job.get("stage", "queued"),
                    "progress": int(job.get("progress") or 0),
                    "message": _fallback_event_message(job),
                    "details": {"source_type": job.get("source_type"), "mode": job.get("mode")},
                    "created_at": job.get("updated_at") or job.get("created_at"),
                }
            ]
        job["events"] = events
        return job

    @staticmethod
    def _decode_job(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["request"] = json.loads(data.pop("request_json") or "{}")
        return data


def _fallback_event_message(job: dict[str, Any]) -> str:
    status = str(job.get("status") or "queued")
    stage = str(job.get("stage") or status)
    if status == "completed":
        return "Job completed. Detailed timeline was not recorded for this job."
    if status == "failed":
        return str(job.get("error_message") or "Job failed. Detailed timeline was not recorded for this job.")
    return f"Job is {status} at {stage}. Detailed timeline has not been recorded yet."
