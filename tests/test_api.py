import importlib

from fastapi.testclient import TestClient


def test_health_and_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDMETA_DATABASE", str(tmp_path / "vidmeta.db"))
    monkeypatch.setenv("VIDMETA_DATA_DIR", str(tmp_path / "data"))
    import api.main as main

    importlib.reload(main)
    client = TestClient(main.app)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["ok"] is True

    settings = client.get("/api/settings")
    assert settings.status_code == 200
    assert settings.json()["storage_settings"]["backend"] == "local_disk"


def test_from_path_rejects_missing_path(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDMETA_DATABASE", str(tmp_path / "vidmeta.db"))
    monkeypatch.setenv("VIDMETA_DATA_DIR", str(tmp_path / "data"))
    import api.main as main

    importlib.reload(main)
    client = TestClient(main.app)

    response = client.post(
        "/api/jobs/from-path",
        json={"path": str(tmp_path / "missing.mp4")},
    )
    assert response.status_code == 404


def test_resumable_upload_rejects_oversized_file(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDMETA_DATABASE", str(tmp_path / "vidmeta.db"))
    monkeypatch.setenv("VIDMETA_DATA_DIR", str(tmp_path / "data"))
    import api.main as main

    importlib.reload(main)
    client = TestClient(main.app)

    settings = client.get("/api/settings").json()
    settings["max_upload_mb"] = 1
    assert client.put("/api/settings", json=settings).status_code == 200

    response = client.post(
        "/api/uploads/resumable",
        json={"filename": "large.mp4", "content_type": "video/mp4", "size_bytes": 2 * 1024 * 1024},
    )
    assert response.status_code == 413
