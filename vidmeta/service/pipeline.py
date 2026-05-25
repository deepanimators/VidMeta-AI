from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from vidmeta.ai.output import parse_metadata
from vidmeta.ai.prompts import (
    ANALYSIS_PROMPT,
    METADATA_PROMPT,
    METADATA_REPAIR_PROMPT,
    normalize_platforms,
    platform_json_template,
    platform_requirements,
)
from vidmeta.ai.providers import ProviderConfig, call_llm
from vidmeta.ai.schemas import MetadataResult
from vidmeta.settings import BrandContext, ProviderSettings, VideoSettings
from vidmeta.video.frames import extract_frames
from vidmeta.video.transcription import transcribe_audio


def analyze_video(
    path: str,
    brand: BrandContext,
    video: VideoSettings,
    provider: ProviderSettings,
    target_platforms: list[str] | None = None,
    progress: Callable[[str, int, str, dict[str, Any] | None], None] | None = None,
) -> dict:
    file_path = str(Path(path).expanduser())
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Video file not found or is not a file: {file_path}")

    selected_platforms = normalize_platforms(target_platforms)

    _progress(progress, "frames", 10, "Extracting representative video frames")
    frames = extract_frames(file_path, video.frame_interval, video.max_frames)
    _progress(
        progress,
        "frames",
        30,
        f"Extracted {len(frames)} frame{'s' if len(frames) != 1 else ''}",
        {"frame_count": len(frames), "frame_interval": video.frame_interval, "max_frames": video.max_frames},
    )

    transcript = ""
    if video.use_whisper:
        _progress(
            progress,
            "audio",
            35,
            f"Extracting audio and transcribing with Whisper {video.whisper_model_size}",
            {"whisper_model_size": video.whisper_model_size},
        )
        transcript = transcribe_audio(file_path, video.whisper_model_size)
        _progress(
            progress,
            "audio",
            50,
            "Audio transcription completed",
            {"transcript_characters": len(transcript)},
        )
    else:
        _progress(progress, "audio", 50, "Audio transcription skipped", {"use_whisper": False})

    provider_config = ProviderConfig(
        provider=provider.provider,
        model=provider.model,
        api_key=provider.api_key,
        api_base=provider.api_base,
        ollama_url=provider.ollama_url,
    )

    _progress(
        progress,
        "analysis",
        60,
        "Running visual and transcript analysis",
        {"provider": provider.provider, "model": provider.model},
    )
    analysis_prompt = ANALYSIS_PROMPT.format(
        brand_name=brand.brand_name,
        brand_niche=brand.brand_niche,
        target_audience=brand.target_audience,
        tone=brand.tone,
        transcript=transcript or "No transcript available",
    )
    analysis = call_llm(frames, analysis_prompt, provider_config)
    _progress(
        progress,
        "analysis",
        75,
        "Video analysis completed",
        {"analysis_characters": len(analysis)},
    )

    _progress(
        progress,
        "metadata",
        82,
        f"Generating metadata for {len(selected_platforms)} selected social platform profiles",
        {"provider": provider.provider, "model": provider.model, "platforms": selected_platforms},
    )
    metadata_prompt = METADATA_PROMPT.format(
        analysis=analysis,
        brand_name=brand.brand_name,
        brand_niche=brand.brand_niche,
        target_audience=brand.target_audience,
        tone=brand.tone,
        platform_requirements=platform_requirements(selected_platforms),
        platform_json_template=platform_json_template(selected_platforms),
    )
    metadata_token_budget = _metadata_token_budget(len(selected_platforms))
    raw_metadata = call_llm([], metadata_prompt, provider_config, max_tokens=metadata_token_budget)
    metadata = parse_metadata(raw_metadata, target_platforms=selected_platforms)
    missing_platforms = _missing_platforms(metadata, selected_platforms)
    if missing_platforms:
        _progress(
            progress,
            "metadata",
            88,
            f"Repairing metadata for {len(missing_platforms)} missing selected platform profiles",
            {"missing_platforms": missing_platforms},
        )
        repair_prompt = METADATA_REPAIR_PROMPT.format(
            analysis=analysis,
            brand_name=brand.brand_name,
            brand_niche=brand.brand_niche,
            target_audience=brand.target_audience,
            tone=brand.tone,
            existing_metadata=json.dumps(metadata.model_dump(), ensure_ascii=False),
            platform_requirements=platform_requirements(missing_platforms),
            platform_json_template=platform_json_template(missing_platforms),
        )
        raw_repair = call_llm(
            [],
            repair_prompt,
            provider_config,
            max_tokens=_metadata_token_budget(len(missing_platforms)),
        )
        repair_metadata = parse_metadata(raw_repair, target_platforms=missing_platforms)
        metadata = _merge_metadata(metadata, repair_metadata, selected_platforms)
        raw_metadata = f"{raw_metadata}\n\n--- Missing platform repair ---\n{raw_repair}"
    metadata = _ensure_selected_platforms(metadata, selected_platforms)
    _progress(
        progress,
        "metadata",
        92,
        "Platform metadata parsed and validated",
        {"platform_count": len(metadata.platforms)},
    )

    _progress(progress, "export", 95, "Saving transcript, analysis, metadata, and export-ready output")
    return {
        "transcript": transcript,
        "analysis": analysis,
        "metadata": metadata.model_dump(),
        "raw_output": raw_metadata,
    }


def _progress(
    callback: Callable[[str, int, str, dict[str, Any] | None], None] | None,
    stage: str,
    progress: int,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    if callback:
        callback(stage, progress, message, details)


def _metadata_token_budget(platform_count: int) -> int:
    return min(12000, max(3200, 1400 + platform_count * 650))


def _missing_platforms(metadata: MetadataResult, selected_platforms: list[str]) -> list[str]:
    missing: list[str] = []
    for platform in selected_platforms:
        item = metadata.platforms.get(platform)
        if not item or not (item.title or item.description or item.hashtags or item.keywords):
            missing.append(platform)
    return missing


def _merge_metadata(
    metadata: MetadataResult,
    repair_metadata: MetadataResult,
    selected_platforms: list[str],
) -> MetadataResult:
    data = metadata.model_dump()
    platforms = data.setdefault("platforms", {})
    for platform in selected_platforms:
        repaired = repair_metadata.platforms.get(platform)
        if repaired and platform not in platforms:
            platforms[platform] = repaired.model_dump()
        elif repaired and not _has_platform_content(platforms.get(platform)):
            platforms[platform] = repaired.model_dump()
    return MetadataResult.model_validate(data)


def _ensure_selected_platforms(metadata: MetadataResult, selected_platforms: list[str]) -> MetadataResult:
    data = metadata.model_dump()
    platforms = data.setdefault("platforms", {})
    for platform in selected_platforms:
        platforms.setdefault(platform, {})
    return MetadataResult.model_validate(data)


def _has_platform_content(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return bool(value.get("title") or value.get("description") or value.get("hashtags") or value.get("keywords"))
