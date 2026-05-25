from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.jobs import JobRunner
from api.models import JobRequest, RetryJobRequest, UploadJobRequest
from api.uploads import upload_router
from vidmeta.exports.builders import export_csv, export_json, export_txt
from vidmeta.service.database import Database
from vidmeta.settings import AppSettings, cors_origins


limiter = Limiter(key_func=get_remote_address, default_limits=["300/minute"])

db = Database()
runner = JobRunner(db)

app = FastAPI(
    title="VidMeta AI Service",
    version="2.0.0",
    description="Local-first AI video metadata service.",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Upload-Offset", "Upload-Length", "Tus-Resumable", "Location"],
)

app.include_router(upload_router(db))


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True, "service": "vidmeta-ai", "version": "2.0.0"}


@app.get("/api/settings")
async def get_settings() -> AppSettings:
    return db.get_settings()


@app.put("/api/settings")
@limiter.limit("30/minute")
async def put_settings(settings: AppSettings, request: Request) -> AppSettings:
    return db.save_settings(settings)


@app.post("/api/jobs/from-path")
@limiter.limit("60/minute")
async def create_job_from_path(payload: JobRequest, request: Request) -> dict:
    try:
        return runner.create_from_path(payload.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs/from-upload/{upload_id}")
@limiter.limit("60/minute")
async def create_job_from_upload(upload_id: str, payload: UploadJobRequest, request: Request) -> dict:
    try:
        return runner.create_from_upload(upload_id, payload.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/jobs")
async def list_jobs(limit: int = 50) -> list[dict]:
    return db.list_jobs(limit=limit)


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/jobs/{job_id}/events")
async def get_job_events(job_id: str) -> list[dict]:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.get("events", [])


@app.get("/api/jobs/{job_id}/stream")
async def stream_job_events(job_id: str, request: Request, since: int = 0):
    """Server-Sent Events stream for real-time job progress.

    Sends full event details (including frame thumbnails) as the job runs.
    Connect with ``EventSource('/api/jobs/{id}/stream')``.
    Emits ``data:`` messages (type=event|status) and a final ``event:done`` when the job ends.
    """
    async def generate():
        last_id = since
        last_status_key = ""
        while True:
            if await request.is_disconnected():
                break
            job = db.get_job(job_id)
            if not job:
                yield f"event: error\ndata: {json.dumps({'detail': 'Job not found'})}\n\n"
                break
            for ev in job.get("events", []):
                ev_id = ev.get("id", 0)
                if ev_id > last_id:
                    last_id = ev_id
                    payload = {k: v for k, v in ev.items() if k != "job_id"}
                    yield f"data: {json.dumps({'type': 'event', **payload})}\n\n"
            status_key = f"{job.get('status')}:{job.get('stage')}:{job.get('progress')}"
            if status_key != last_status_key:
                last_status_key = status_key
                yield f"data: {json.dumps({'type': 'status', 'status': job.get('status'), 'stage': job.get('stage'), 'progress': job.get('progress')})}\n\n"
            if job.get("status") in {"completed", "failed"}:
                yield f"event: done\ndata: {json.dumps({'status': job['status']})}\n\n"
                break
            await asyncio.sleep(0.8)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/jobs/{job_id}/result")
async def get_job_result(job_id: str) -> dict:
    result = db.get_result(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Job not found")
    if not result.get("metadata"):
        raise HTTPException(status_code=409, detail="Job result is not ready")
    return result


@app.post("/api/jobs/{job_id}/retry")
@limiter.limit("30/minute")
async def retry_job(job_id: str, payload: RetryJobRequest, request: Request) -> dict:
    """Create a new job from a failed job, optionally with a different provider/model."""
    try:
        override = payload.provider_settings.model_dump() if payload.provider_settings else None
        return runner.retry_job(job_id, override)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str) -> dict:
    """Delete a job and all associated data (transcript, metadata, events)."""
    deleted = db.delete_job(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"deleted": True, "job_id": job_id}


@app.get("/api/jobs/{job_id}/exports/{format_name}")
async def export_job(job_id: str, format_name: str) -> PlainTextResponse:
    result = db.get_result(job_id)
    if not result or not result.get("metadata"):
        raise HTTPException(status_code=404, detail="Job result not found")
    metadata = result["metadata"]
    if format_name == "json":
        return PlainTextResponse(export_json(metadata), media_type="application/json")
    if format_name == "csv":
        return PlainTextResponse(export_csv(metadata), media_type="text/csv")
    if format_name == "txt":
        return PlainTextResponse(export_txt(metadata), media_type="text/plain")
    raise HTTPException(status_code=404, detail="Unsupported export format")


@app.post("/api/admin/cleanup")
async def run_cleanup() -> dict:
    """Purge expired upload records and interrupted (orphaned) uploads."""
    settings = db.get_settings()
    expired = 0
    if settings.upload_retention_days > 0:
        expired = db.delete_expired_uploads(settings.upload_retention_days)
    orphaned = db.delete_orphaned_uploads(max_age_hours=24)
    return {
        "deleted_expired_uploads": expired,
        "deleted_orphaned_uploads": orphaned,
        "retention_days": settings.upload_retention_days,
    }
