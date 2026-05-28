from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from vidmeta.service.database import Database


def test_worker_stop_integration_records_cancelled_progress():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        db_path = data_dir / "vidmeta.db"
        os.environ["VIDMETA_DATA_DIR"] = str(data_dir)
        os.environ["VIDMETA_DATABASE"] = str(db_path)

        db = Database(db_path)
        job_id = "job-stop-integration"
        db.create_job(
            {
                "id": job_id,
                "source_type": "path",
                "source_path": "__simulate_sleep__",
                "mode": "single",
                "status": "queued",
                "stage": "queued",
                "progress": 0,
                "request": {},
            }
        )

        input_file = Path(tmpdir) / "request.json"
        output_file = Path(tmpdir) / "result.json"
        input_file.write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "source_path": "__simulate_sleep__",
                    "brand": {},
                    "video": {},
                    "provider": {},
                    "target_platforms": ["youtube"],
                }
            ),
            encoding="utf-8",
        )

        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "worker.py"), "--input", str(input_file), "--output", str(output_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(1.0)
        proc.send_signal(signal.SIGTERM)
        stdout, stderr = proc.communicate(timeout=30)

        assert proc.returncode is not None
        result = json.loads(output_file.read_text(encoding="utf-8"))
        assert result["ok"] is False
        assert "cancelled" in result["error"].lower()
        job = db.get_job(job_id)
        assert job is not None
        assert job["stage"] == "cancelled"
        assert any(event["stage"] == "simulate" for event in job["events"])
        assert any(event["stage"] == "cancelled" for event in job["events"])
