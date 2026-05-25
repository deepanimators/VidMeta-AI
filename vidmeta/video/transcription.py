from __future__ import annotations

import os
import subprocess
import threading


_model_lock = threading.Lock()
_model_cache: dict[str, object] = {}


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
            segments, _ = model.transcribe(audio_path, beam_size=3)
            return " ".join(segment.text for segment in segments).strip() or "[Silent]"
        except ImportError:
            import whisper
            model = whisper.load_model(model_size)
            return model.transcribe(audio_path)["text"].strip() or "[Silent]"
    except FileNotFoundError:
        return "[ffmpeg not installed - audio skipped]"
    except Exception as exc:
        return f"[Transcription error: {exc}]"
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
