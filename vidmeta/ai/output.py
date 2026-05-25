from __future__ import annotations

import json
from collections.abc import Sequence

from vidmeta.ai.schemas import MetadataResult, PlatformMetadata


# Platform field character limits enforced post-generation.
# LLMs hallucinate lengths; silent overruns break publishing APIs.
PLATFORM_CONSTRAINTS: dict[str, dict[str, int]] = {
    "youtube":           {"title": 100, "description": 5000},
    "youtube_shorts":    {"title": 100, "description": 5000},
    "instagram_reels":   {"description": 2200},
    "instagram_feed":    {"description": 2200},
    "facebook":          {"title": 255, "description": 63206},
    "facebook_reels":    {"description": 2200},
    "tiktok":            {"description": 2200},
    "linkedin":          {"title": 200, "description": 3000},
    "x":                 {"title": 280, "description": 280},
    "threads":           {"description": 500},
    "bluesky":           {"title": 300, "description": 300},
    "mastodon":          {"description": 500},
    "pinterest":         {"title": 100, "description": 500},
    "snapchat_spotlight":{"description": 250},
    "reddit":            {"title": 300, "description": 40000},
    "whatsapp_channels": {"description": 1024},
    "telegram_channels": {"description": 4096},
    "discord":           {"description": 2000},
    "tumblr":            {"title": 255, "description": 4096},
    "medium":            {"title": 100, "description": 25000},
    "quora":             {"title": 255, "description": 5000},
    "substack_notes":    {"description": 500},
    "twitch":            {"title": 140, "description": 1500},
    "vimeo":             {"title": 128, "description": 5000},
    "rumble":            {"title": 200, "description": 5000},
    "dailymotion":       {"title": 255, "description": 5000},
    "wechat_channels":   {"title": 100, "description": 600},
    "douyin":            {"description": 2200},
    "kuaishou":          {"description": 1000},
    "bilibili":          {"title": 80, "description": 2000},
    "weibo":             {"description": 2000},
    "vk":                {"title": 255, "description": 16000},
    "line_voom":         {"description": 1000},
    "lemon8":            {"title": 80, "description": 2200},
    "sharechat":         {"description": 1000},
    "moj":               {"description": 500},
    "josh":              {"description": 500},
}


def _enforce_constraints(meta: PlatformMetadata, platform: str) -> PlatformMetadata:
    constraints = PLATFORM_CONSTRAINTS.get(platform)
    if not constraints:
        return meta
    data = meta.model_dump()
    for field, max_len in constraints.items():
        if field in data and isinstance(data[field], str) and len(data[field]) > max_len:
            data[field] = data[field][:max_len]
    return PlatformMetadata.model_validate(data)


def parse_metadata(raw: str, target_platforms: Sequence[str] | None = None) -> MetadataResult:
    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        return MetadataResult(raw=raw)
    if not isinstance(parsed, dict):
        return MetadataResult(raw=raw)
    try:
        result = MetadataResult.from_llm_json(parsed, raw=raw, target_platforms=target_platforms)
        enforced = {
            platform: _enforce_constraints(pm, platform)
            for platform, pm in result.platforms.items()
        }
        return result.model_copy(update={"platforms": enforced})
    except Exception:
        return MetadataResult(raw=raw)
