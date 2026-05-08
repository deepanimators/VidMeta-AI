from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


PLATFORMS = ("youtube", "instagram", "facebook", "tiktok", "linkedin")


class PlatformMetadata(BaseModel):
    title: str = ""
    description: str = ""
    hashtags: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    cta: str = ""
    posting_tip: str = ""


class MetadataResult(BaseModel):
    video_summary: str = ""
    content_category: str = ""
    youtube: PlatformMetadata = Field(default_factory=PlatformMetadata)
    instagram: PlatformMetadata = Field(default_factory=PlatformMetadata)
    facebook: PlatformMetadata = Field(default_factory=PlatformMetadata)
    tiktok: PlatformMetadata = Field(default_factory=PlatformMetadata)
    linkedin: PlatformMetadata = Field(default_factory=PlatformMetadata)
    raw: str | None = None

    @classmethod
    def from_llm_json(cls, data: dict[str, Any], raw: str | None = None) -> "MetadataResult":
        payload = dict(data)
        for platform in PLATFORMS:
            if not isinstance(payload.get(platform), dict):
                payload[platform] = {}
        payload["raw"] = raw
        return cls.model_validate(payload)
