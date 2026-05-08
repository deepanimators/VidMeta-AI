from __future__ import annotations

from pathlib import Path
from typing import Callable

from vidmeta.ai.output import parse_metadata
from vidmeta.ai.prompts import ANALYSIS_PROMPT, METADATA_PROMPT
from vidmeta.ai.providers import ProviderConfig, call_llm
from vidmeta.settings import BrandContext, ProviderSettings, VideoSettings
from vidmeta.video.frames import extract_frames
from vidmeta.video.transcription import transcribe_audio


def analyze_video(
    path: str,
    brand: BrandContext,
    video: VideoSettings,
    provider: ProviderSettings,
    progress: Callable[[str, int], None] | None = None,
) -> dict:
    file_path = str(Path(path).expanduser())

    _progress(progress, "frames", 10)
    frames = extract_frames(file_path, video.frame_interval, video.max_frames)

    transcript = ""
    if video.use_whisper:
        _progress(progress, "audio", 35)
        transcript = transcribe_audio(file_path, video.whisper_model_size)

    provider_config = ProviderConfig(
        provider=provider.provider,
        model=provider.model,
        api_key=provider.api_key,
        api_base=provider.api_base,
        ollama_url=provider.ollama_url,
    )

    _progress(progress, "analysis", 60)
    analysis_prompt = ANALYSIS_PROMPT.format(
        brand_name=brand.brand_name,
        brand_niche=brand.brand_niche,
        target_audience=brand.target_audience,
        tone=brand.tone,
        transcript=transcript or "No transcript available",
    )
    analysis = call_llm(frames, analysis_prompt, provider_config)

    _progress(progress, "metadata", 82)
    metadata_prompt = METADATA_PROMPT.format(
        analysis=analysis,
        brand_name=brand.brand_name,
        brand_niche=brand.brand_niche,
        target_audience=brand.target_audience,
        tone=brand.tone,
    )
    raw_metadata = call_llm([], metadata_prompt, provider_config)
    metadata = parse_metadata(raw_metadata)

    _progress(progress, "export", 95)
    return {
        "transcript": transcript,
        "analysis": analysis,
        "metadata": metadata.model_dump(),
        "raw_output": raw_metadata,
    }


def _progress(callback: Callable[[str, int], None] | None, stage: str, progress: int) -> None:
    if callback:
        callback(stage, progress)
