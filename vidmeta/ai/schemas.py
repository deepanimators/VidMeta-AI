from __future__ import annotations

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
    def from_llm_json(cls, data: dict[str, Any], raw: str | None = None) -> "MetadataResult":
        payload = dict(data)
        source_platforms = payload.get("platforms") if isinstance(payload.get("platforms"), dict) else {}
        normalized: dict[str, Any] = {}
        for platform in PLATFORMS:
            if isinstance(source_platforms.get(platform), dict):
                normalized[platform] = source_platforms[platform]
            elif isinstance(payload.get(platform), dict):
                normalized[platform] = payload[platform]
            else:
                normalized[platform] = {}
        for key, value in source_platforms.items():
            if key not in normalized and isinstance(value, dict):
                normalized[key] = value
        payload["platforms"] = normalized
        payload["raw"] = raw
        return cls.model_validate(payload)
