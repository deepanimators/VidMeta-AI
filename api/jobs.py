from __future__ import annotations

import os
import threading
import time
import uuid
import subprocess
import json
import tempfile
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from vidmeta.ai.prompts import normalize_platforms
from vidmeta.service.database import Database
from vidmeta.service.pipeline import analyze_video
from vidmeta.settings import (
    DESKTOP_APP_MODE,
    VIDEO_EXTENSIONS,
    AppSettings,
    BrandContext,
    ProviderSettings,
    VideoSettings,
    allowed_paths,
    desktop_safe_provider_settings,
    desktop_safe_storage_settings,
)
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
        self._processes: dict[str, subprocess.Popen] = {}
        self._cancellations: set[str] = set()

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
        request = self._normalize_request(payload, settings)
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
                "request": request,
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
        request = self._normalize_request(payload, settings)
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
                "request": request,
            }
        )
        self.enqueue(job_id)
        return job

    def retry_job(self, job_id: str, provider_override: dict | None = None) -> dict:
        job = self.db.get_job(job_id)
        if not job:
            raise FileNotFoundError(f"Job not found: {job_id}")
        if job["status"] != "failed":
            raise ValueError(f"Only failed jobs can be retried (current status: {job['status']})")
        source_path = job["source_path"]
        if not Path(source_path).exists():
            raise FileNotFoundError(f"Source file no longer exists: {source_path}")
        # Merge provider override so the new job uses the chosen model.
        request = dict(job.get("request", {}))
        settings = self.db.get_settings()
        if provider_override:
            request["provider_settings"] = provider_override
        request = self._normalize_request(request, settings)
        new_id = uuid.uuid4().hex
        new_job = self.db.create_job(
            {
                "id": new_id,
                "source_type": job["source_type"],
                "source_path": source_path,
                "mode": job.get("mode", "single"),
                "status": "queued",
                "stage": "queued",
                "progress": 0,
                "request": request,
            }
        )
        self.enqueue(new_id)
        return new_job

    def _normalize_request(self, payload: dict[str, Any], settings: AppSettings) -> dict[str, Any]:
        request = dict(payload)
        if settings.app_mode != DESKTOP_APP_MODE:
            return request
        request["app_mode"] = DESKTOP_APP_MODE
        request["provider_settings"] = desktop_safe_provider_settings(request.get("provider_settings"))
        request["storage_settings"] = desktop_safe_storage_settings(request.get("storage_settings"))
        return request

    def enqueue(self, job_id: str) -> None:
        with self._lock:
            future = self.executor.submit(self._run, job_id)
            self._futures[job_id] = future
            # Remove completed future from dict to prevent unbounded memory growth.
            future.add_done_callback(lambda _f: self._remove_future(job_id))

    def stop(self, job_id: str) -> dict[str, bool]:
        """Request cancellation of a running job. Best-effort; marks job failed/stopped."""
        with self._lock:
            job = self.db.get_job(job_id)
            if not job:
                raise FileNotFoundError(f"Job not found: {job_id}")
            status = job.get("status")
            if status in {"completed", "failed"}:
                return {"stopped": False}
            # mark cancellation request
            self._cancellations.add(job_id)
            # attempt to cancel future if present
            fut = self._futures.get(job_id)
            try:
                if fut and hasattr(fut, "cancel"):
                    fut.cancel()
            except Exception:
                pass
            # attempt to terminate subprocess if running
            proc = self._processes.get(job_id)
            try:
                if proc and proc.pid:
                    proc.terminate()
            except Exception:
                pass
            # update DB to reflect requested stop
            self.db.update_job(job_id, status="failed", stage="cancelled", error_message="Job cancelled by user", completed_at="CURRENT_TIMESTAMP")
            self.db.add_job_event(job_id, "cancelled", int(job.get("progress") or 0), "Job cancelled by user")
            return {"stopped": True}

    def _remove_future(self, job_id: str) -> None:
        with self._lock:
            self._futures.pop(job_id, None)

    def _run(self, job_id: str) -> None:
        job = self.db.get_job(job_id)
        if not job:
            return
        processing_temp: str | None = None
        try:
            self._record_progress(job_id, "starting", 1, "Preparing job settings and source")
            # honour cancellation requests before heavy work
            if job_id in self._cancellations:
                raise RuntimeError("Job cancelled by user")
            settings = self.db.get_settings()
            request = self._normalize_request(job.get("request", {}), settings)
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
                    job_id,
                    job["source_path"],
                    brand_model,
                    video_model,
                    provider_model,
                    target_platforms,
                    progress,
                )
            else:
                result = self._spawn_worker_and_wait(
                    job_id,
                    job["source_path"],
                    brand_model,
                    video_model,
                    provider_model,
                    target_platforms,
                    progress,
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
            msg = str(exc)
            if isinstance(exc, RuntimeError) and "cancelled" in msg.lower():
                # cancellation is explicit
                self.db.update_job(
                    job_id,
                    status="failed",
                    stage="cancelled",
                    error_message="Job cancelled by user",
                    completed_at="CURRENT_TIMESTAMP",
                )
                self.db.add_job_event(job_id, "cancelled", int(current.get("progress") or 0), "Job cancelled by user")
            else:
                self.db.update_job(
                    job_id,
                    status="failed",
                    stage="failed",
                    error_message=msg,
                    completed_at="CURRENT_TIMESTAMP",
                )
                self.db.add_job_event(job_id, "failed", int(current.get("progress") or 0), msg)
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
        job_id: str,
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
            # check cancellation
            if job_id in self._cancellations:
                raise RuntimeError("Job cancelled by user")
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

            item = self._spawn_worker_and_wait(
                job_id,
                str(path),
                brand,
                video,
                provider,
                target_platforms,
                batch_progress,
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

    def _spawn_worker_and_wait(
        self,
        job_id: str,
        source_path: str,
        brand: BrandContext,
        video: VideoSettings,
        provider: ProviderSettings,
        target_platforms: list[str] | None,
        progress: Callable[[str, int, str, dict[str, Any] | None], None] | None,
    ) -> dict[str, Any]:
        # prepare request JSON
        req = {
            "source_path": source_path,
            "brand": brand.model_dump() if hasattr(brand, "model_dump") else brand,
            "video": video.model_dump() if hasattr(video, "model_dump") else video,
            "provider": provider.model_dump() if hasattr(provider, "model_dump") else provider,
            "target_platforms": target_platforms,
        }
        input_file = tempfile.NamedTemporaryFile(prefix=f"vidmeta_req_{job_id}_", suffix=".json", delete=False)
        output_file = tempfile.NamedTemporaryFile(prefix=f"vidmeta_res_{job_id}_", suffix=".json", delete=False)
        try:
            with open(input_file.name, "w", encoding="utf-8") as f:
                json.dump(req, f)

            script_path = Path(__file__).resolve().parents[1] / "scripts" / "worker.py"
            proc = subprocess.Popen([sys.executable, str(script_path), "--input", input_file.name, "--output", output_file.name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with self._lock:
                self._processes[job_id] = proc

            # wait loop with cancellation checks
            while True:
                if proc.poll() is not None:
                    break
                if job_id in self._cancellations:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    # give it a moment then kill
                    try:
                        proc.wait(timeout=2)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    break
                time.sleep(0.5)

            stdout, stderr = proc.communicate(timeout=1)
            # cleanup process mapping
            with self._lock:
                self._processes.pop(job_id, None)

            # read output file
            try:
                with open(output_file.name, "r", encoding="utf-8") as fo:
                    out = json.load(fo)
            except Exception as exc:
                raise RuntimeError(f"Worker failed: {stderr.decode() if stderr else exc}") from exc

            if not out.get("ok"):
                raise RuntimeError(out.get("error") or "Worker reported failure")
            return out.get("result") or {}
        finally:
            try:
                input_file.close()
            except Exception:
                pass
            try:
                output_file.close()
            except Exception:
                pass
            try:
                Path(input_file.name).unlink(missing_ok=True)
            except Exception:
                pass
            try:
                Path(output_file.name).unlink(missing_ok=True)
            except Exception:
                pass

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
