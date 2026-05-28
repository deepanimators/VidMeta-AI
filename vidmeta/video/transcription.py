from __future__ import annotations

import os
import subprocess
import threading
from dataclasses import dataclass
from typing import Iterable


_model_lock = threading.Lock()
_model_cache: dict[str, object] = {}
_diarization_lock = threading.Lock()
_diarization_pipeline: object | None = None


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str


def _get_whisper_model(model_size: str):
    with _model_lock:
        if model_size not in _model_cache:
            from faster_whisper import WhisperModel
            _model_cache[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
        return _model_cache[model_size]


def transcribe_audio(video_path: str, model_size: str) -> str:
    audio_path = f"{video_path}_audio.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1", "-y", audio_path],
            capture_output=True,
            timeout=180,
            check=False,
        )
        if not os.path.exists(audio_path):
            return "[No audio track]"
        try:
            model = _get_whisper_model(model_size)
            segments, _ = model.transcribe(audio_path, beam_size=3, vad_filter=True)
            transcript_segments = [
                TranscriptSegment(
                    start=float(segment.start or 0.0),
                    end=float(segment.end or 0.0),
                    text=str(segment.text or "").strip(),
                )
                for segment in segments
                if str(segment.text or "").strip()
            ]
            if not transcript_segments:
                return "[Silent]"

            speaker_turns = _diarize_audio(audio_path)
            if speaker_turns:
                return _format_transcript_with_speakers(transcript_segments, speaker_turns)
            return _format_transcript_with_timestamps(transcript_segments)
        except ImportError:
            import whisper
            model = whisper.load_model(model_size)
            text = (model.transcribe(audio_path).get("text") or "").strip()
            return _format_plain_transcript(text)
    except FileNotFoundError:
        return "[ffmpeg not installed - audio skipped]"
    except Exception as exc:
        return f"[Transcription error: {exc}]"
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


def _diarize_audio(audio_path: str) -> list[tuple[float, float, str]]:
    pipeline = _get_diarization_pipeline()
    if not pipeline:
        return []
    try:
        diarization = pipeline(audio_path)
        turns: list[tuple[float, float, str]] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append((float(turn.start), float(turn.end), str(speaker)))
        return turns
    except Exception:
        return []


def _get_diarization_pipeline() -> object | None:
    token = (
        os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        return None
    with _diarization_lock:
        global _diarization_pipeline
        if _diarization_pipeline is None:
            try:
                from pyannote.audio import Pipeline
                _diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=token,
                )
            except Exception:
                _diarization_pipeline = False
        return _diarization_pipeline or None


def _format_transcript_with_speakers(
    segments: list[TranscriptSegment],
    speaker_turns: list[tuple[float, float, str]],
) -> str:
    speaker_names: dict[str, str] = {}
    lines: list[str] = []
    for segment in segments:
        raw_speaker = _best_speaker_for_segment(segment.start, segment.end, speaker_turns)
        if raw_speaker is None:
            speaker_label = "Speaker"
        else:
            speaker_label = speaker_names.setdefault(raw_speaker, f"Speaker {len(speaker_names) + 1}")
        lines.append(f"{speaker_label} [{_format_time(segment.start)} - {_format_time(segment.end)}]: {segment.text}")
    return "\n".join(lines).strip() or "[Silent]"


def _format_transcript_with_timestamps(segments: list[TranscriptSegment]) -> str:
    lines = [f"[{_format_time(segment.start)} - {_format_time(segment.end)}]: {segment.text}" for segment in segments]
    return "\n".join(lines).strip() or "[Silent]"


def _format_plain_transcript(text: str) -> str:
    return text or "[Silent]"


def _format_time(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, remaining = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining:02d}"
    return f"{minutes:02d}:{remaining:02d}"


def _best_speaker_for_segment(
    start: float,
    end: float,
    speaker_turns: list[tuple[float, float, str]],
) -> str | None:
    best_label: str | None = None
    best_overlap = 0.0
    for turn_start, turn_end, speaker in speaker_turns:
        overlap = min(end, turn_end) - max(start, turn_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = speaker
    return best_label if best_overlap > 0 else None
