from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field


SOCIAL_PLATFORMS = (
    {"key": "youtube", "label": "YouTube"},
    {"key": "youtube_shorts", "label": "YouTube Shorts"},
    {"key": "instagram_reels", "label": "Instagram Reels"},
    {"key": "instagram_feed", "label": "Instagram Feed"},
    {"key": "facebook", "label": "Facebook"},
    {"key": "facebook_reels", "label": "Facebook Reels"},
    {"key": "tiktok", "label": "TikTok"},
    {"key": "linkedin", "label": "LinkedIn"},
    {"key": "x", "label": "X / Twitter"},
    {"key": "threads", "label": "Threads"},
    {"key": "bluesky", "label": "Bluesky"},
    {"key": "mastodon", "label": "Mastodon"},
    {"key": "pinterest", "label": "Pinterest"},
    {"key": "snapchat_spotlight", "label": "Snapchat Spotlight"},
    {"key": "reddit", "label": "Reddit"},
    {"key": "whatsapp_channels", "label": "WhatsApp Channels"},
    {"key": "telegram_channels", "label": "Telegram Channels"},
    {"key": "discord", "label": "Discord"},
    {"key": "tumblr", "label": "Tumblr"},
    {"key": "medium", "label": "Medium"},
    {"key": "quora", "label": "Quora"},
    {"key": "substack_notes", "label": "Substack Notes"},
    {"key": "twitch", "label": "Twitch"},
    {"key": "vimeo", "label": "Vimeo"},
    {"key": "rumble", "label": "Rumble"},
    {"key": "dailymotion", "label": "Dailymotion"},
    {"key": "wechat_channels", "label": "WeChat Channels"},
    {"key": "douyin", "label": "Douyin"},
    {"key": "kuaishou", "label": "Kuaishou"},
    {"key": "bilibili", "label": "Bilibili"},
    {"key": "weibo", "label": "Weibo"},
    {"key": "vk", "label": "VK"},
    {"key": "line_voom", "label": "LINE VOOM"},
    {"key": "lemon8", "label": "Lemon8"},
    {"key": "sharechat", "label": "ShareChat"},
    {"key": "moj", "label": "Moj"},
    {"key": "josh", "label": "Josh"},
)

PLATFORMS = tuple(platform["key"] for platform in SOCIAL_PLATFORMS)
PLATFORM_LABELS = {platform["key"]: platform["label"] for platform in SOCIAL_PLATFORMS}
PLATFORM_ALIASES = {
    "youtube": ("youtube",),
    "youtube_shorts": ("youtube_shorts", "youtube shorts", "shorts", "youtube"),
    "instagram_reels": ("instagram_reels", "instagram reels", "reels", "instagram"),
    "instagram_feed": ("instagram_feed", "instagram feed", "instagram"),
    "facebook": ("facebook",),
    "facebook_reels": ("facebook_reels", "facebook reels", "facebook"),
    "tiktok": ("tiktok", "tik_tok", "tik tok"),
    "linkedin": ("linkedin", "linked_in", "linked in"),
    "x": ("x", "twitter", "x_twitter", "x / twitter"),
    "threads": ("threads",),
    "telegram_channels": ("telegram_channels", "telegram channels", "telegram"),
    "whatsapp_channels": ("whatsapp_channels", "whatsapp channels", "whatsapp"),
}


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
    platforms: dict[str, PlatformMetadata] = Field(default_factory=dict)
    raw: str | None = None

    @classmethod
    def from_llm_json(
        cls,
        data: dict[str, Any],
        raw: str | None = None,
        target_platforms: Sequence[str] | None = None,
    ) -> "MetadataResult":
        payload = dict(data)
        source_platforms = payload.get("platforms") if isinstance(payload.get("platforms"), dict) else {}
        candidates = _platform_candidates(payload, source_platforms)
        normalized: dict[str, Any] = {}
        targets = [platform for platform in (target_platforms or []) if platform in PLATFORM_LABELS]
        if targets:
            for platform in targets:
                match = _first_platform_match(platform, candidates)
                if isinstance(match, dict):
                    normalized[platform] = match
        elif source_platforms:
            for key, value in candidates.items():
                if isinstance(value, dict) and key not in {"video_summary", "content_category"}:
                    normalized[key] = value
        else:
            for platform in PLATFORMS:
                match = _first_platform_match(platform, candidates)
                if isinstance(match, dict):
                    normalized[platform] = match
        payload["platforms"] = normalized
        payload["raw"] = raw
        return cls.model_validate(payload)


def _platform_candidates(payload: dict[str, Any], source_platforms: dict[str, Any]) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    for key, value in payload.items():
        if key != "platforms" and isinstance(value, dict):
            candidates[_normalize_platform_key(key)] = value
    for key, value in source_platforms.items():
        if isinstance(value, dict):
            candidates[_normalize_platform_key(key)] = value
    return candidates


def _first_platform_match(platform: str, candidates: dict[str, Any]) -> dict[str, Any] | None:
    keys = (platform, *PLATFORM_ALIASES.get(platform, ()))
    for key in keys:
        value = candidates.get(_normalize_platform_key(key))
        if isinstance(value, dict):
            return value
    return None


def _normalize_platform_key(key: str) -> str:
    normalized = key.strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")
