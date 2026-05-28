from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

from vidmeta.service.database import Database
from vidmeta.service.pipeline import analyze_video
from vidmeta.settings import BrandContext, ProviderSettings, VideoSettings

_canceled = False


def _on_term(signum, frame):
    global _canceled
    _canceled = True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    with open(args.input, "r", encoding="utf-8") as f:
        req = json.load(f)
    source = req.get("source_path")
    job_id = req.get("job_id")
    brand = BrandContext.model_validate(req.get("brand") or {})
    video = VideoSettings.model_validate(req.get("video") or {})
    provider = ProviderSettings.model_validate(req.get("provider") or {})
    target_platforms = req.get("target_platforms")

    db = Database()

    def progress_cb(stage: str, value: int, message: str = "", details=None):
        try:
            if job_id:
                db.update_job(job_id, status="running", stage=stage, progress=value)
                db.add_job_event(job_id, stage, int(value), message or stage, details)
        except Exception:
            pass

    def should_cancel() -> bool:
        return _canceled

    try:
        if source == "__simulate_sleep__":
            for i in range(20):
                if should_cancel():
                    raise RuntimeError("Job cancelled by user")
                progress_cb("simulate", int((i / 19) * 100), f"Simulated step {i + 1}/20")
                time.sleep(0.2)
            result = {"transcript": "[simulated]", "analysis": "[simulated]", "metadata": {}, "raw_output": None}
        else:
            result = analyze_video(
                source,
                brand=brand,
                video=video,
                provider=provider,
                target_platforms=target_platforms,
                progress=progress_cb,
                should_cancel=should_cancel,
            )

        with open(args.output, "w", encoding="utf-8") as fo:
            json.dump({"ok": True, "result": result}, fo)
        return 0
    except Exception as exc:
        cancelled = _canceled or "cancelled" in str(exc).lower()
        if job_id:
            try:
                if cancelled:
                    db.update_job(job_id, status="failed", stage="cancelled", error_message="Job cancelled by user")
                    db.add_job_event(job_id, "cancelled", 0, "Job cancelled by user")
                else:
                    db.update_job(job_id, status="failed", stage="failed", error_message=str(exc))
                    db.add_job_event(job_id, "failed", 0, str(exc))
            except Exception:
                pass
        with open(args.output, "w", encoding="utf-8") as fo:
            json.dump({"ok": False, "error": "Job cancelled by user" if cancelled else str(exc)}, fo)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
