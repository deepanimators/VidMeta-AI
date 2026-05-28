from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from vidmeta.settings import AppSettings, database_path
from vidmeta.settings import force_desktop_local_settings


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
CREATE INDEX IF NOT EXISTS idx_uploads_status ON uploads(status);
"""

# Settings fields encrypted at rest.
_SENSITIVE_PATHS: list[tuple[str, ...]] = [
    ("provider_settings", "api_key"),
    ("storage_settings", "s3_access_key_id"),
    ("storage_settings", "s3_secret_access_key"),
]

# Max events stored per job. Oldest events beyond this are dropped during insert.
_MAX_EVENTS_PER_JOB = 200


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
        # timeout: wait up to 10 s when another thread holds the write lock
        conn = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
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
            return force_desktop_local_settings(AppSettings())
        raw = json.loads(row["value_json"])
        decrypted = _decrypt_settings(raw)
        return force_desktop_local_settings(AppSettings.model_validate(decrypted))

    def save_settings(self, settings: AppSettings) -> AppSettings:
        settings = force_desktop_local_settings(settings)
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

    def delete_orphaned_uploads(self, max_age_hours: int = 24) -> int:
        """Delete uploads stuck in 'created' state older than max_age_hours (interrupted uploads)."""
        with self.connect() as conn:
            result = conn.execute(
                """
                DELETE FROM uploads
                WHERE status = 'created'
                AND created_at < datetime('now', ? || ' hours')
                """,
                (f"-{max_age_hours}",),
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
            # Prune oldest events beyond cap to prevent unbounded growth on long batch jobs.
            conn.execute(
                """
                DELETE FROM job_events
                WHERE job_id = ?
                AND id NOT IN (
                    SELECT id FROM job_events WHERE job_id = ? ORDER BY id DESC LIMIT ?
                )
                """,
                (job_id, job_id, _MAX_EVENTS_PER_JOB),
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
        return _decode_events(rows)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if not row:
                return None
            # Load events in same connection to avoid extra round-trip
            event_rows = conn.execute(
                """
                SELECT id, job_id, stage, progress, message, details_json, created_at
                FROM job_events WHERE job_id = ? ORDER BY id ASC
                """,
                (job_id,),
            ).fetchall()
        job = self._decode_job(row)
        events = _decode_events(event_rows)
        job["events"] = events or [_fallback_event(job)]
        return job

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            if not rows:
                return []
            job_ids = [row["id"] for row in rows]
            placeholders = ",".join("?" * len(job_ids))
            event_rows = conn.execute(
                f"""
                SELECT id, job_id, stage, progress, message, details_json, created_at
                FROM job_events
                WHERE job_id IN ({placeholders})
                ORDER BY id ASC
                """,
                job_ids,
            ).fetchall()

        events_by_job: dict[str, list[dict[str, Any]]] = {row["id"]: [] for row in rows}
        for event in _decode_events(event_rows):
            # Strip large thumbnail arrays from list queries to keep payload small.
            events_by_job.setdefault(event["job_id"], []).append(_strip_large_details(event))

        result: list[dict[str, Any]] = []
        for row in rows:
            job = self._decode_job(row)
            events = events_by_job.get(job["id"], [])
            job["events"] = events or [_fallback_event(job)]
            result.append(job)
        return result

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
            event_rows = conn.execute(
                """
                SELECT id, job_id, stage, progress, message, details_json, created_at
                FROM job_events WHERE job_id = ? ORDER BY id ASC
                """,
                (job_id,),
            ).fetchall()
        data = self._decode_job(row)
        data["transcript"] = row["transcript"] or ""
        data["analysis"] = row["analysis"] or ""
        data["metadata"] = json.loads(row["metadata_json"]) if row["metadata_json"] else None
        data["raw_output"] = row["raw_output"]
        events = _decode_events(event_rows)
        data["events"] = events or [_fallback_event(data)]
        return data

    @staticmethod
    def _decode_job(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["request"] = json.loads(data.pop("request_json") or "{}")
        return data


def _strip_large_details(event: dict[str, Any]) -> dict[str, Any]:
    """Remove large thumbnail arrays from event details for list queries."""
    details = event.get("details")
    if not details or "thumbnails" not in details:
        return event
    stripped = {k: v for k, v in details.items() if k != "thumbnails"}
    return {**event, "details": stripped}


def _decode_events(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["details"] = json.loads(item.pop("details_json") or "{}")
        events.append(item)
    return events


def _fallback_event(job: dict[str, Any]) -> dict[str, Any]:
    status = str(job.get("status") or "queued")
    stage = str(job.get("stage") or status)
    if status == "completed":
        message = "Job completed. Detailed timeline was not recorded for this job."
    elif status == "failed":
        message = str(job.get("error_message") or "Job failed. Detailed timeline was not recorded for this job.")
    else:
        message = f"Job is {status} at {stage}. Detailed timeline has not been recorded yet."
    return {
        "id": 0,
        "job_id": job["id"],
        "stage": stage,
        "progress": int(job.get("progress") or 0),
        "message": message,
        "details": {"source_type": job.get("source_type"), "mode": job.get("mode")},
        "created_at": job.get("updated_at") or job.get("created_at"),
    }
