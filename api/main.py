from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from api.jobs import JobRunner
from api.models import JobRequest, UploadJobRequest
from api.uploads import upload_router
from vidmeta.exports.builders import export_csv, export_json, export_txt
from vidmeta.service.database import Database
from vidmeta.settings import AppSettings


db = Database()
runner = JobRunner(db)

app = FastAPI(
    title="VidMeta AI Service",
    version="2.0.0",
    description="Local-first AI video metadata service replacing Streamlit.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
async def put_settings(settings: AppSettings) -> AppSettings:
    return db.save_settings(settings)


@app.post("/api/jobs/from-path")
async def create_job_from_path(payload: JobRequest) -> dict:
    try:
        return runner.create_from_path(payload.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs/from-upload/{upload_id}")
async def create_job_from_upload(upload_id: str, payload: UploadJobRequest) -> dict:
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


@app.get("/api/jobs/{job_id}/result")
async def get_job_result(job_id: str) -> dict:
    result = db.get_result(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Job not found")
    if not result.get("metadata"):
        raise HTTPException(status_code=409, detail="Job result is not ready")
    return result


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
