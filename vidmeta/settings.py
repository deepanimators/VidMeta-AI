from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def data_dir() -> Path:
    path = Path(os.environ.get("VIDMETA_DATA_DIR", Path.home() / ".vidmeta-ai")).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def database_path() -> Path:
    return Path(os.environ.get("VIDMETA_DATABASE", data_dir() / "vidmeta.db")).expanduser()


def upload_dir() -> Path:
    path = Path(os.environ.get("VIDMETA_UPLOAD_DIR", data_dir() / "uploads")).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def processing_dir() -> Path:
    path = Path(os.environ.get("VIDMETA_PROCESSING_DIR", data_dir() / "processing")).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def cors_origins() -> list[str]:
    raw = os.environ.get("VIDMETA_CORS_ORIGINS", "")
    if not raw:
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


def allowed_paths() -> list[Path] | None:
    """Returns None when unrestricted (local mode), or a list of allowed base dirs."""
    raw = os.environ.get("VIDMETA_ALLOWED_PATHS", "")
    if not raw:
        return None
    return [Path(p.strip()).expanduser().resolve() for p in raw.split(":") if p.strip()]


class BrandContext(BaseModel):
    brand_name: str = ""
    brand_niche: str = ""
    target_audience: str = ""
    tone: str = ""
    custom_instructions: str = ""


class VideoSettings(BaseModel):
    use_whisper: bool = True
    whisper_model_size: str = "base"
    frame_interval: int = Field(default=5, ge=1, le=120)
    max_frames: int = Field(default=20, ge=1, le=60)


class ProviderSettings(BaseModel):
    provider: str = "ollama"
    model: str = "gemma3:4b"
    api_key: str = ""
    api_base: str = ""
    ollama_url: str = "http://localhost:11434"


class StorageSettings(BaseModel):
    backend: str = "local_disk"
    local_data_dir: str = Field(default_factory=lambda: str(data_dir()))
    import_local_files: bool = False
    s3_endpoint_url: str = ""
    s3_bucket: str = ""
    s3_region: str = ""
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""


class AppSettings(BaseModel):
    app_mode: str = "local"
    max_upload_mb: int = int(os.environ.get("VIDMETA_MAX_UPLOAD_MB", "2048"))
    upload_retention_days: int = int(os.environ.get("VIDMETA_UPLOAD_RETENTION_DAYS", "0"))
    brand_context: BrandContext = Field(default_factory=BrandContext)
    video_settings: VideoSettings = Field(default_factory=VideoSettings)
    provider_settings: ProviderSettings = Field(default_factory=ProviderSettings)
    storage_settings: StorageSettings = Field(default_factory=StorageSettings)
