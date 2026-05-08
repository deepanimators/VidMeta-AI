from __future__ import annotations

from pydantic import BaseModel, Field

from vidmeta.ai.schemas import PLATFORMS
from vidmeta.settings import BrandContext, ProviderSettings, VideoSettings


class JobRequest(BaseModel):
    path: str
    mode: str = "single"
    brand_context: BrandContext = Field(default_factory=BrandContext)
    video_settings: VideoSettings = Field(default_factory=VideoSettings)
    provider_settings: ProviderSettings = Field(default_factory=ProviderSettings)
    target_platforms: list[str] = Field(default_factory=lambda: list(PLATFORMS))


class UploadJobRequest(BaseModel):
    brand_context: BrandContext = Field(default_factory=BrandContext)
    video_settings: VideoSettings = Field(default_factory=VideoSettings)
    provider_settings: ProviderSettings = Field(default_factory=ProviderSettings)
    target_platforms: list[str] = Field(default_factory=lambda: list(PLATFORMS))


class CreateResumableUploadRequest(BaseModel):
    filename: str
    content_type: str = "application/octet-stream"
    size_bytes: int | None = None


class JobSummary(BaseModel):
    id: str
    source_type: str
    source_path: str
    mode: str
    status: str
    stage: str
    progress: int
    error_message: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None


class UploadSummary(BaseModel):
    id: str
    filename: str
    content_type: str = ""
    size_bytes: int = 0
    expected_size_bytes: int | None = None
    path: str
    storage_backend: str = "local_disk"
    status: str = "created"
