import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.jobs import JobRunner


class MockDB:
    def __init__(self, job):
        self._job = job
        self.updated = None
        self.events = []

    def get_job(self, job_id):
        return self._job if self._job.get("id") == job_id else None

    def get_settings(self):
        # minimal settings stub
        class S: pass
        s = S()
        s.app_mode = "desktop"
        s.provider_settings = {}
        s.storage_settings = type("X", (), {"backend": "local"})
        s.brand_context = {}
        s.video_settings = {}
        return s

    def update_job(self, job_id, **kwargs):
        self.updated = kwargs

    def add_job_event(self, job_id, stage, progress, message, details=None):
        self.events.append((stage, progress, message))

    def create_job(self, payload):
        return payload


class FakeProc:
    def __init__(self):
        self.terminated = False
        self.pid = 1234

    def terminate(self):
        self.terminated = True


def test_stop_terminates_subprocess_and_marks_cancelled():
    job = {"id": "job123", "status": "running", "progress": 10}
    db = MockDB(job)
    runner = JobRunner(db)
    fake = FakeProc()
    runner._processes["job123"] = fake
    res = runner.stop("job123")
    assert res.get("stopped") is True
    assert fake.terminated is True
    assert db.updated is not None
    assert db.updated.get("stage") == "cancelled"
