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
from vidmeta.ai.providers import ProviderConfig, call_llm, ollama_likely_text_only
from vidmeta.ai.schemas import MetadataResult
from vidmeta.settings import BrandContext, ProviderSettings, VideoSettings
from vidmeta.video.frames import extract_frames, extract_thumbnails, get_video_metadata
from vidmeta.video.detection import detect_objects
from vidmeta.video.ocr import extract_text
from vidmeta.video.captioning import caption_frame
from vidmeta.video.transcription import transcribe_audio


def analyze_video(
    path: str,
    brand: BrandContext,
    video: VideoSettings,
    provider: ProviderSettings,
    target_platforms: list[str] | None = None,
    progress: Callable[[str, int, str, dict[str, Any] | None], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    file_path = str(Path(path).expanduser())
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Video file not found or is not a file: {file_path}")

    selected_platforms = normalize_platforms(target_platforms)

    # --- Video metadata (fast, no ML) ---
    _progress(progress, "frames", 5, "Reading video properties")
    video_meta_str = "Video metadata unavailable"
    video_meta_dict: dict[str, Any] = {}
    orientation_hint = "unknown orientation"
    try:
        vm = get_video_metadata(file_path)
        mins, secs = divmod(int(vm.duration_seconds), 60)
        dur_str = f"{mins}m {secs}s" if mins else f"{secs}s"
        video_meta_str = (
            f"Duration: {dur_str} | "
            f"Resolution: {vm.width}×{vm.height} | "
            f"Aspect ratio: {vm.aspect_ratio} ({vm.orientation}) | "
            f"FPS: {vm.fps} | "
            f"Total frames: {vm.frame_count}"
        )
        orientation_hint = f"{vm.aspect_ratio} {vm.orientation}"
        video_meta_dict = {
            "duration_seconds": vm.duration_seconds,
            "width": vm.width,
            "height": vm.height,
            "fps": vm.fps,
            "aspect_ratio": vm.aspect_ratio,
            "orientation": vm.orientation,
        }
    except Exception:
        pass

    # --- Audio transcription ---
    transcript = ""
    if video.use_whisper:
        _progress(
            progress,
            "audio",
            12,
            f"Extracting audio and transcribing with Whisper {video.whisper_model_size}",
            {"whisper_model_size": video.whisper_model_size},
        )
        transcript = transcribe_audio(file_path, video.whisper_model_size)
        if should_cancel and should_cancel():
            raise RuntimeError("Job cancelled by user")
        _progress(
            progress,
            "audio",
            18,
            "Audio transcription completed",
            {
                "transcript": transcript,
                "transcript_preview": transcript[:1000],
                "transcript_characters": len(transcript),
                "whisper_model_size": video.whisper_model_size,
            },
        )
    else:
        _progress(progress, "audio", 18, "Audio transcription skipped", {"use_whisper": False})

    # --- Frame extraction ---
    _progress(progress, "frames", 20, "Extracting representative video frames")
    def frame_stage_progress(start: int, span: int):
        def _report(value: int, message: str = "", details: dict[str, Any] | None = None) -> None:
            mapped = start + int((max(0, min(value, 100)) / 100) * span)
            _progress(progress, "frames", min(start + span, mapped), message, details)

        return _report

    frames_with_ts = extract_frames(
        file_path,
        video.frame_interval,
        video.max_frames,
        progress=frame_stage_progress(20, 14),
    )
    if should_cancel and should_cancel():
        raise RuntimeError("Job cancelled by user")
    frames = [b64 for _, b64 in frames_with_ts]
    frame_timestamps = [ts for ts, _ in frames_with_ts]
    thumbnails = extract_thumbnails(
        file_path,
        video.frame_interval,
        video.max_frames,
        progress=frame_stage_progress(34, 6),
    )
    _progress(
        progress,
        "frames",
        30,
        f"Extracted {len(frames)} frame{'s' if len(frames) != 1 else ''} "
        f"({len(thumbnails)} thumbnail{'s' if len(thumbnails) != 1 else ''})",
        {
            "frame_count": len(frames),
            "frame_interval": video.frame_interval,
            "max_frames": video.max_frames,
            "thumbnails": thumbnails,
            "video_metadata": video_meta_dict,
        },
    )

    # --- Frame enrichment (object detection / OCR / captioning) ---
    frame_annotations_block = ""
    any_enrichment = video.enable_object_detection or video.enable_ocr or video.enable_frame_captioning
    if any_enrichment and frames:
        _progress(progress, "frames", 42, "Running frame enrichment (detection / OCR / captioning)")
        frame_annotations_block = _build_frame_annotations(
            frames_with_ts,
            video,
            progress=frame_stage_progress(42, 8),
            should_cancel=should_cancel,
        )
        _progress(
            progress, "frames", 50,
            f"Frame enrichment complete — {len(frames_with_ts)} frames annotated",
            {"frame_annotations": frame_annotations_block[:500]},
        )

    provider_config = ProviderConfig(
        provider=provider.provider,
        model=provider.model,
        api_key=provider.api_key,
        api_base=provider.api_base,
        ollama_url=provider.ollama_url,
    )

    # Detect Ollama text-only model before sending frames
    vision_warning = ""
    if provider.provider.lower() == "ollama" and frames and ollama_likely_text_only(provider.model):
        vision_warning = (
            f"Model '{provider.model}' may not support image inputs. "
            "Analysis will be based on transcript only. "
            "Switch to a vision-capable Ollama model (llava, llama3.2-vision, moondream, "
            "minicpm-v, or gemma3) for true visual analysis."
        )

    _progress(
        progress,
        "analysis",
        60,
        "Running visual and transcript analysis",
        {
            "provider": provider.provider,
            "model": provider.model,
            **({"vision_warning": vision_warning} if vision_warning else {}),
        },
    )
    custom_block = f"\nCustom Instructions: {brand.custom_instructions}" if getattr(brand, "custom_instructions", "") else ""
    analysis_prompt = ANALYSIS_PROMPT.format(
        brand_name=brand.brand_name or "Not specified",
        brand_niche=brand.brand_niche or "Not specified",
        target_audience=brand.target_audience or "Not specified",
        tone=brand.tone or "Not specified",
        custom_instructions_block=custom_block,
        video_metadata=video_meta_str,
        frame_annotations_block=frame_annotations_block,
        orientation_hint=orientation_hint,
        transcript=transcript or "No transcript available",
    )
    # Pass video_path so Gemini can use native video API instead of JPEG frames.
    analysis = call_llm(
        frames, analysis_prompt, provider_config,
        max_tokens=120000,
        video_path=file_path, frame_timestamps=frame_timestamps,
    )
    if should_cancel and should_cancel():
        raise RuntimeError("Job cancelled by user")
    # If the provider truncated the response, attempt to continue the generation
    def _looks_truncated(text: str) -> bool:
        if not text:
            return True
        t = text.strip()
        # heuristics: ends with ellipsis or last char not a terminal punctuation
        if t.endswith("..."):
            return True
        if t[-1] not in {'.', '!', '?', '"', "'"}:
            # short responses that are incomplete are suspicious
            if len(t) < 400 or t.count('\n') < 3:
                return True
        return False

    def _call_llm_with_continuation(frames, prompt, config, max_tokens=120000, video_path="", timestamps=None, max_attempts=3):
        text = call_llm(frames, prompt, config, max_tokens=max_tokens, video_path=video_path, frame_timestamps=timestamps)
        attempts = 1
        while attempts < max_attempts and _looks_truncated(text):
            # Ask provider to continue; don't resend images to save bandwidth for continuation
            cont_prompt = (
                "The previous response was truncated. Please continue the analysis from where it left off. "
                "Continue the analysis and finish any incomplete sentences or lists.\n\nPrevious output:\n" + text
            )
            try:
                more = call_llm([], cont_prompt, config, max_tokens=max_tokens, video_path="", frame_timestamps=None)
                if not more or more.strip() == text.strip():
                    break
                text = text.rstrip() + "\n\n" + more.lstrip()
            except Exception:
                break
            attempts += 1
        return text

    analysis = _call_llm_with_continuation(frames, analysis_prompt, provider_config, max_tokens=120000, video_path=file_path, timestamps=frame_timestamps)
    _progress(
        progress,
        "analysis",
        75,
        "Video analysis completed",
        {
            "analysis_preview": analysis[:1000],
            "analysis_characters": len(analysis),
            "provider": provider.provider,
            "model": provider.model,
            **({"vision_warning": vision_warning} if vision_warning else {}),
        },
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
        brand_name=brand.brand_name or "Not specified",
        brand_niche=brand.brand_niche or "Not specified",
        target_audience=brand.target_audience or "Not specified",
        tone=brand.tone or "Not specified",
        custom_instructions_block=custom_block,
        platform_requirements=platform_requirements(selected_platforms),
        platform_json_template=platform_json_template(selected_platforms),
    )
    metadata_token_budget = _metadata_token_budget(len(selected_platforms))
    raw_metadata = _call_llm_with_continuation([], metadata_prompt, provider_config, max_tokens=metadata_token_budget)
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
            brand_name=brand.brand_name or "Not specified",
            brand_niche=brand.brand_niche or "Not specified",
            target_audience=brand.target_audience or "Not specified",
            tone=brand.tone or "Not specified",
            custom_instructions_block=custom_block,
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


def _build_frame_annotations(
    frames_with_ts: list[tuple[float, str]],
    settings: VideoSettings,
    progress: Callable[[int, str, dict[str, Any] | None], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> str:
    """Run enabled enrichment on each frame; return a prompt-ready annotation block."""

    Checks `should_cancel()` between frames to allow cooperative cancellation.
    """
    lines: list[str] = []
    total_frames = max(1, len(frames_with_ts))
    for i, (ts, b64) in enumerate(frames_with_ts):
        if should_cancel and should_cancel():
            raise RuntimeError("Job cancelled by user")
        mins, secs = divmod(int(ts), 60)
        label = f"{mins}:{secs:02d}"
        parts: list[str] = []

        if settings.enable_object_detection:
            objs = detect_objects(b64)
            if objs:
                parts.append(f"Objects: [{', '.join(objs)}]")

        if settings.enable_ocr:
            texts = extract_text(b64)
            if texts:
                parts.append(f"On-screen text: [{' | '.join(texts[:6])}]")

        if settings.enable_frame_captioning:
            caption = caption_frame(b64)
            if caption:
                parts.append(f"Caption: {caption}")

        if parts:
            lines.append(f"  Frame {i + 1} ({label}): {' | '.join(parts)}")

        if progress:
            progress(
                int(((i + 1) / total_frames) * 100),
                f"Enriched frame {i + 1}/{total_frames}",
                {"frame_index": i + 1, "frame_total": total_frames},
            )

    if not lines:
        return ""

    header = "\n--- PRE-EXTRACTED FRAME ANNOTATIONS ---\n[Automatically detected — use as additional evidence alongside the visual frames]\n"
    return header + "\n".join(lines) + "\n"


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
