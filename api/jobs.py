from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from vidmeta.service.database import Database
from vidmeta.service.pipeline import analyze_video
from vidmeta.settings import VIDEO_EXTENSIONS, AppSettings, BrandContext, ProviderSettings, VideoSettings
from vidmeta.storage.backends import import_local_file_if_needed, materialize_for_processing


class JobRunner:
    def __init__(self, db: Database, max_workers: int = 1) -> None:
        self.db = db
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vidmeta-job")
        self._lock = threading.Lock()
        self._futures: dict[str, object] = {}

    def create_from_path(self, payload: dict) -> dict:
        source_path = Path(payload["path"]).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(f"Path does not exist: {source_path}")
        if source_path.is_file() and source_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video type: {source_path.suffix}")
        if source_path.is_dir() and not self._video_files(source_path):
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

    def _run(self, job_id: str) -> None:
        job = self.db.get_job(job_id)
        if not job:
            return
        request = job["request"]
        try:
            self.db.update_job(job_id, status="running", stage="starting", progress=1)
            settings = self.db.get_settings()
            brand = request.get("brand_context") or settings.brand_context.model_dump()
            video = request.get("video_settings") or settings.video_settings.model_dump()
            provider = request.get("provider_settings") or settings.provider_settings.model_dump()

            def progress(stage: str, value: int) -> None:
                self.db.update_job(job_id, stage=stage, progress=value)

            brand_model = BrandContext.model_validate(brand)
            video_model = VideoSettings.model_validate(video)
            provider_model = ProviderSettings.model_validate(provider)
            if Path(job["source_path"]).is_dir():
                result = self._run_batch(
                    job["source_path"],
                    brand_model,
                    video_model,
                    provider_model,
                    progress,
                )
            else:
                result = analyze_video(
                    job["source_path"],
                    brand=brand_model,
                    video=video_model,
                    provider=provider_model,
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
        except Exception as exc:
            self.db.update_job(
                job_id,
                status="failed",
                stage="failed",
                error_message=str(exc),
                completed_at="CURRENT_TIMESTAMP",
            )

    def _run_batch(
        self,
        folder: str,
        brand: BrandContext,
        video: VideoSettings,
        provider: ProviderSettings,
        progress: Callable[[str, int], None],
    ) -> dict[str, Any]:
        files = self._video_files(Path(folder))
        results: list[dict[str, Any]] = []
        transcripts: list[str] = []
        analyses: list[str] = []
        total = len(files)
        for index, path in enumerate(files, start=1):
            base_progress = int(((index - 1) / total) * 95)

            def batch_progress(stage: str, value: int) -> None:
                scaled = base_progress + int((value / 100) * (95 / total))
                progress(f"{index}/{total} {stage}", min(scaled, 95))

            item = analyze_video(
                str(path),
                brand=brand,
                video=video,
                provider=provider,
                progress=batch_progress,
            )
            transcripts.append(f"## {path.name}\n{item['transcript']}")
            analyses.append(f"## {path.name}\n{item['analysis']}")
            results.append({"file": path.name, "metadata": item["metadata"]})
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
        return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


def merged_settings(settings: AppSettings, payload: dict) -> dict:
    data = settings.model_dump()
    data.update(payload)
    return data
