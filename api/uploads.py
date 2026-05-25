from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from api.models import CreateResumableUploadRequest
from vidmeta.service.database import Database
from vidmeta.settings import VIDEO_EXTENSIONS
from vidmeta.storage.backends import maybe_archive_upload, upload_path
from vidmeta.video.validation import is_valid_video_header as _is_valid_video_header


def upload_router(db: Database) -> APIRouter:
    router = APIRouter(prefix="/api/uploads", tags=["uploads"])

    @router.post("")
    async def multipart_upload(file: UploadFile = File(...)) -> dict:
        ext = Path(file.filename or "").suffix.lower()
        if ext not in VIDEO_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Unsupported video type")
        settings = db.get_settings()
        max_bytes = settings.max_upload_mb * 1024 * 1024
        upload_id = uuid.uuid4().hex
        target = upload_path(upload_id, file.filename or "upload.bin")
        size = 0
        with target.open("wb") as out:
            while chunk := await file.read(8 * 1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    target.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="Upload exceeds configured size limit")
                out.write(chunk)
        if not _is_valid_video_header(target):
            target.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="File content does not match a supported video format")
        stored_path = maybe_archive_upload(target, upload_id, file.filename or target.name, settings.storage_settings)
        data = db.create_upload(
            {
                "id": upload_id,
                "filename": file.filename or target.name,
                "content_type": file.content_type or "application/octet-stream",
                "size_bytes": size,
                "expected_size_bytes": size,
                "path": stored_path,
                "storage_backend": settings.storage_settings.backend,
                "status": "complete",
            }
        )
        return data

    @router.post("/resumable")
    async def create_resumable_upload(payload: CreateResumableUploadRequest, request: Request) -> JSONResponse:
        ext = Path(payload.filename).suffix.lower()
        if ext not in VIDEO_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Unsupported video type")
        settings = db.get_settings()
        if payload.size_bytes and payload.size_bytes > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Upload exceeds configured size limit")
        upload_id = uuid.uuid4().hex
        target = upload_path(upload_id, payload.filename)
        target.touch()
        db.create_upload(
            {
                "id": upload_id,
                "filename": payload.filename,
                "content_type": payload.content_type,
                "size_bytes": 0,
                "expected_size_bytes": payload.size_bytes,
                "path": str(target),
                "storage_backend": settings.storage_settings.backend,
                "status": "created",
            }
        )
        location = str(request.url_for("patch_resumable_upload", upload_id=upload_id))
        return JSONResponse({"id": upload_id, "upload_url": location}, headers={"Location": location})

    @router.options("/resumable")
    async def resumable_options() -> Response:
        return Response(
            status_code=204,
            headers={
                "Tus-Resumable": "1.0.0",
                "Tus-Version": "1.0.0",
                "Tus-Extension": "creation",
                "Access-Control-Expose-Headers": "Tus-Resumable,Tus-Version,Tus-Extension,Upload-Offset,Location",
            },
        )

    @router.head("/resumable/{upload_id}")
    async def head_resumable_upload(upload_id: str) -> Response:
        upload = db.get_upload(upload_id)
        if not upload:
            raise HTTPException(status_code=404, detail="Upload not found")
        size = os.path.getsize(upload["path"]) if os.path.exists(upload["path"]) else upload["size_bytes"]
        return Response(
            status_code=204,
            headers={
                "Tus-Resumable": "1.0.0",
                "Upload-Offset": str(size),
                "Upload-Length": str(upload["expected_size_bytes"] or size),
            },
        )

    @router.patch("/resumable/{upload_id}", name="patch_resumable_upload")
    async def patch_resumable_upload(
        upload_id: str,
        request: Request,
        upload_offset: int = Header(default=0, alias="Upload-Offset"),
    ) -> Response:
        upload = db.get_upload(upload_id)
        if not upload:
            raise HTTPException(status_code=404, detail="Upload not found")
        settings = db.get_settings()
        path = Path(upload["path"])
        current = path.stat().st_size if path.exists() else 0
        if current != upload_offset:
            raise HTTPException(status_code=409, detail=f"Offset mismatch. Current offset is {current}")
        with path.open("ab") as out:
            async for chunk in request.stream():
                out.write(chunk)
        size = path.stat().st_size
        if size > settings.max_upload_mb * 1024 * 1024:
            path.unlink(missing_ok=True)
            db.update_upload(upload_id, size_bytes=size, status="failed")
            raise HTTPException(status_code=413, detail="Upload exceeds configured size limit")
        is_complete = bool(upload["expected_size_bytes"] and size >= upload["expected_size_bytes"])
        if is_complete and not _is_valid_video_header(path):
            path.unlink(missing_ok=True)
            db.update_upload(upload_id, size_bytes=size, status="failed")
            raise HTTPException(status_code=400, detail="File content does not match a supported video format")
        status = "complete" if is_complete else "uploading"
        db.update_upload(upload_id, size_bytes=size, status=status)
        if status == "complete":
            stored_path = maybe_archive_upload(path, upload_id, upload["filename"], settings.storage_settings)
            db.update_upload(
                upload_id,
                path=stored_path,
                storage_backend=settings.storage_settings.backend,
                completed_at="CURRENT_TIMESTAMP",
            )
        return Response(
            status_code=204,
            headers={"Tus-Resumable": "1.0.0", "Upload-Offset": str(size)},
        )

    @router.get("/{upload_id}")
    async def get_upload(upload_id: str) -> dict:
        upload = db.get_upload(upload_id)
        if not upload:
            raise HTTPException(status_code=404, detail="Upload not found")
        return upload

    return router
