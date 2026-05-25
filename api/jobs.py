from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from vidmeta.ai.prompts import normalize_platforms
from vidmeta.service.database import Database
from vidmeta.service.pipeline import analyze_video
from vidmeta.settings import VIDEO_EXTENSIONS, AppSettings, BrandContext, ProviderSettings, VideoSettings, allowed_paths
from vidmeta.storage.backends import cleanup_processing_file, import_local_file_if_needed, materialize_for_processing
from vidmeta.video.validation import is_valid_video_header


def _validate_path(path: Path) -> None:
    """Enforce path allowlist when VIDMETA_ALLOWED_PATHS is set (hosted mode).
    In local mode (unset), no path restriction is applied — the user running
    the service already has full filesystem access.
    """
    allowed = allowed_paths()
    if allowed is None:
        return  # Local mode: unrestricted
    resolved = path.resolve()
    if not any(resolved.is_relative_to(a) for a in allowed):
        allowed_str = ", ".join(str(a) for a in allowed)
        raise ValueError(f"Path '{path}' is not within allowed directories: {allowed_str}")


class JobRunner:
    def __init__(self, db: Database, max_workers: int = 1) -> None:
        self.db = db
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vidmeta-job")
        self._lock = threading.Lock()
        self._futures: dict[str, object] = {}

    def create_from_path(self, payload: dict) -> dict:
        source_path = Path(payload["path"]).expanduser()
        _validate_path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Path does not exist: {source_path}")
        if source_path.is_file():
            if source_path.suffix.lower() not in VIDEO_EXTENSIONS:
                raise ValueError(f"Unsupported video type: {source_path.suffix}")
            if not is_valid_video_header(source_path):
                raise ValueError(f"File does not appear to be a valid video: {source_path.name}")
        elif source_path.is_dir() and not self._video_files(source_path):
            raise ValueError("Folder does not contain supported video files")

        job_id = uuid.uuid4().hex
        settings = self.db.get_settings()
        actual_source = str(source_path)
        if source_path.is_file():
            actual_source = import_local_file_if_needed(str(source_path), job_id, settings.storage_settings)

        job = self.db.create_job(
            {
                "id": job_id,
                "source_type": "path",
                "source_path": actual_source,
                "mode": "batch" if source_path.is_dir() else payload.get("mode", "single"),
                "status": "queued",
                "stage": "queued",
                "progress": 0,
                "request": payload,
            }
        )
        self.enqueue(job_id)
        return job

    def create_from_upload(self, upload_id: str, payload: dict) -> dict:
        upload = self.db.get_upload(upload_id)
        if not upload:
            raise FileNotFoundError(f"Upload not found: {upload_id}")
        if upload["status"] not in {"complete", "stored"}:
            raise ValueError("Upload is not complete")
        settings = self.db.get_settings()
        source = materialize_for_processing(
            upload["path"],
            upload_id,
            upload["filename"],
            settings.storage_settings,
        )
        job_id = uuid.uuid4().hex
        job = self.db.create_job(
            {
                "id": job_id,
                "source_type": "upload",
                "source_path": source,
                "mode": "single",
                "status": "queued",
                "stage": "queued",
                "progress": 0,
                "request": payload,
            }
        )
        self.enqueue(job_id)
        return job

    def enqueue(self, job_id: str) -> None:
        with self._lock:
            future = self.executor.submit(self._run, job_id)
            self._futures[job_id] = future
            # Remove completed future from dict to prevent unbounded memory growth.
            future.add_done_callback(lambda _f: self._remove_future(job_id))

    def _remove_future(self, job_id: str) -> None:
        with self._lock:
            self._futures.pop(job_id, None)

    def _run(self, job_id: str) -> None:
        job = self.db.get_job(job_id)
        if not job:
            return
        request = job["request"]
        processing_temp: str | None = None
        try:
            self._record_progress(job_id, "starting", 1, "Preparing job settings and source")
            settings = self.db.get_settings()
            brand = request.get("brand_context") or settings.brand_context.model_dump()
            video = request.get("video_settings") or settings.video_settings.model_dump()
            provider = request.get("provider_settings") or settings.provider_settings.model_dump()
            target_platforms = normalize_platforms(request.get("target_platforms"))

            # Track S3 temp files for cleanup on failure
            if job.get("source_type") == "upload" and settings.storage_settings.backend == "s3_compatible":
                processing_temp = job["source_path"]

            def progress(
                stage: str,
                value: int,
                message: str = "",
                details: dict[str, Any] | None = None,
            ) -> None:
                self._record_progress(job_id, stage, value, message, details)

            brand_model = BrandContext.model_validate(brand)
            video_model = VideoSettings.model_validate(video)
            provider_model = ProviderSettings.model_validate(provider)
            if Path(job["source_path"]).is_dir():
                result = self._run_batch(
                    job["source_path"],
                    brand_model,
                    video_model,
                    provider_model,
                    target_platforms,
                    progress,
                )
            else:
                result = analyze_video(
                    job["source_path"],
                    brand=brand_model,
                    video=video_model,
                    provider=provider_model,
                    target_platforms=target_platforms,
                    progress=progress,
                )
            self.db.save_result(
                job_id,
                result["transcript"],
                result["analysis"],
                result["metadata"],
                result["raw_output"],
            )
            self.db.update_job(
                job_id,
                status="completed",
                stage="completed",
                progress=100,
                completed_at="CURRENT_TIMESTAMP",
            )
            self.db.add_job_event(job_id, "completed", 100, "Job completed successfully")
        except Exception as exc:
            current = self.db.get_job(job_id) or job
            self.db.update_job(
                job_id,
                status="failed",
                stage="failed",
                error_message=str(exc),
                completed_at="CURRENT_TIMESTAMP",
            )
            self.db.add_job_event(job_id, "failed", int(current.get("progress") or 0), str(exc))
        finally:
            if processing_temp:
                cleanup_processing_file(processing_temp)

    def _record_progress(
        self,
        job_id: str,
        stage: str,
        value: int,
        message: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.db.update_job(job_id, status="running", stage=stage, progress=value)
        self.db.add_job_event(job_id, stage, value, message or stage.replace("_", " ").title(), details)

    def _run_batch(
        self,
        folder: str,
        brand: BrandContext,
        video: VideoSettings,
        provider: ProviderSettings,
        target_platforms: list[str],
        progress: Callable[[str, int, str, dict[str, Any] | None], None],
    ) -> dict[str, Any]:
        files = self._video_files(Path(folder))
        results: list[dict[str, Any]] = []
        transcripts: list[str] = []
        analyses: list[str] = []
        total = len(files)
        progress(
            "batch",
            2,
            f"Starting batch analysis for {total} video file{'s' if total != 1 else ''}",
            {"file_count": total, "folder": folder},
        )
        batch_start = time.monotonic()
        for index, path in enumerate(files, start=1):
            base_progress = int(((index - 1) / total) * 95)
            elapsed = time.monotonic() - batch_start
            avg_per_video = elapsed / index if index > 1 else 0
            eta_seconds = int(avg_per_video * (total - index)) if avg_per_video else None

            progress(
                "batch",
                base_progress,
                f"Processing {path.name} ({index} of {total})",
                {
                    "file": path.name,
                    "index": index,
                    "total": total,
                    "eta_seconds": eta_seconds,
                },
            )

            def batch_progress(
                stage: str,
                value: int,
                message: str = "",
                details: dict[str, Any] | None = None,
            ) -> None:
                scaled = base_progress + int((value / 100) * (95 / total))
                merged_details = {
                    "file": path.name,
                    "index": index,
                    "total": total,
                    **(details or {}),
                }
                progress(
                    f"batch_{stage}",
                    min(scaled, 95),
                    f"{path.name}: {message or stage}",
                    merged_details,
                )

            item = analyze_video(
                str(path),
                brand=brand,
                video=video,
                provider=provider,
                target_platforms=target_platforms,
                progress=batch_progress,
            )
            transcripts.append(f"## {path.name}\n{item['transcript']}")
            analyses.append(f"## {path.name}\n{item['analysis']}")
            results.append({"file": path.name, "metadata": item["metadata"]})
            progress(
                "batch",
                min(int((index / total) * 95), 95),
                f"Completed {path.name} ({index} of {total})",
                {"file": path.name, "index": index, "total": total},
            )
        return {
            "transcript": "\n\n".join(transcripts),
            "analysis": "\n\n".join(analyses),
            "metadata": {
                "video_summary": f"Processed {len(results)} videos",
                "content_category": "Batch",
                "batch_results": results,
            },
            "raw_output": None,
        }

    @staticmethod
    def _video_files(folder: Path) -> list[Path]:
        return sorted(
            path
            for root, _, files in os.walk(folder)
            for path in (Path(root) / name for name in files)
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )


def merged_settings(settings: AppSettings, payload: dict) -> dict:
    data = settings.model_dump()
    data.update(payload)
    return data
