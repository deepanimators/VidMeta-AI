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
            speaker_turns = _diarize_audio(audio_path, transcript_segments)
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


def _diarize_audio(audio_path: str, segments: list[TranscriptSegment] | None = None) -> list[tuple[float, float, str]]:
    """Attempt diarization. Prefer pyannote Pipeline (HuggingFace) when available,
    otherwise fall back to a lightweight offline clustering-based diarizer.

    Returns a list of (start, end, speaker_label) tuples.
    """
    pipeline = _get_diarization_pipeline()
    if pipeline:
        try:
            diarization = pipeline(audio_path)
            turns: list[tuple[float, float, str]] = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                turns.append((float(turn.start), float(turn.end), str(speaker)))
            return turns
        except Exception:
            # fall through to offline attempt
            pass

    # Offline fallback: use Resemblyzer embeddings + clustering around transcript segments.
    try:
        return _diarize_offline(audio_path, segments)
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


def _diarize_offline(audio_path: str, segments: list[TranscriptSegment] | None = None) -> list[tuple[float, float, str]]:
    """Offline diarization using Resemblyzer + DBSCAN clustering of segment embeddings.

    This is a lightweight best-effort approach that works without Hugging Face tokens.
    Requires optional packages: `resemblyzer`, `librosa`, `scikit-learn`, `numpy`.
    """
    if not segments:
        return []
    try:
        import numpy as _np
        import librosa as _librosa
        from resemblyzer import VoiceEncoder as _VoiceEncoder
        from sklearn.cluster import DBSCAN as _DBSCAN
        from sklearn.preprocessing import StandardScaler as _StandardScaler
    except Exception:
        return []

    # load audio at 16k
    y, sr = _librosa.load(audio_path, sr=16000, mono=True)
    encoder = _VoiceEncoder()
    embeddings: list[_np.ndarray] = []
    seg_times: list[tuple[float, float]] = []
    for seg in segments:
        start_s = max(0.0, float(seg.start))
        end_s = max(start_s + 0.2, float(seg.end))
        start = int(start_s * sr)
        end = int(end_s * sr)
        if end > len(y):
            end = len(y)
        clip = y[start:end]
        # ensure minimum length ~0.5s for embedding quality
        min_samples = int(0.5 * sr)
        if clip.shape[0] < min_samples:
            mid = int(((start + end) / 2))
            left = max(0, mid - min_samples // 2)
            right = min(len(y), left + min_samples)
            clip = y[left:right]
            # recompute times
            start_s = left / sr
            end_s = right / sr
        if clip.shape[0] <= 0:
            continue
        try:
            emb = encoder.embed_utterance(_np.asarray(clip, dtype=_np.float32))
        except Exception:
            continue
        embeddings.append(emb)
        seg_times.append((start_s, end_s))

    if len(embeddings) < 2:
        return []

    X = _np.vstack(embeddings)
    # scale embeddings for clustering
    try:
        Xs = _StandardScaler().fit_transform(X)
    except Exception:
        Xs = X

    # DBSCAN auto-detects clusters; eps tuned conservatively, min_samples=1 allows small speaker turns
    try:
        db = _DBSCAN(eps=0.6, min_samples=1, metric="euclidean").fit(Xs)
        labels = db.labels_
    except Exception:
        # fallback single speaker
        return [(s, e, "spk_0") for (s, e) in seg_times]

    # remap labels (DBSCAN may give -1 for noise)
    label_map: dict[int, int] = {}
    next_id = 0
    turns: list[tuple[float, float, str]] = []
    for (s, e), lab in zip(seg_times, labels):
        lab_i = int(lab)
        if lab_i == -1:
            speaker_label = f"spk_{next_id}"
            next_id += 1
        else:
            if lab_i not in label_map:
                label_map[lab_i] = next_id
                next_id += 1
            speaker_label = f"spk_{label_map[lab_i]}"
        turns.append((s, e, speaker_label))

    # merge adjacent segments with same speaker
    return _merge_adjacent_turns(turns)


def _merge_adjacent_turns(turns: list[tuple[float, float, str]], gap_threshold: float = 0.5) -> list[tuple[float, float, str]]:
    """Merge consecutive turns with the same speaker if gaps are small."""
    if not turns:
        return []
    merged: list[tuple[float, float, str]] = []
    # ensure sorted by start
    turns_sorted = sorted(turns, key=lambda x: x[0])
    cur_s, cur_e, cur_spk = turns_sorted[0]
    for s, e, spk in turns_sorted[1:]:
        if spk == cur_spk and s <= cur_e + gap_threshold:
            # extend current
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e, cur_spk))
            cur_s, cur_e, cur_spk = s, e, spk
    merged.append((cur_s, cur_e, cur_spk))
    return merged


def _align_speakers_to_faces(turns: list[tuple[float, float, str]], *_args, **_kwargs) -> list[tuple[float, float, str]]:
    """Optional hook for face alignment.

    This repository does not currently bundle a face-recognition pipeline, so
    the offline path remains deterministic and local-only by default.
    """
    return turns


def _format_transcript_with_speakers(
    segments: list[TranscriptSegment],
    speaker_turns: list[tuple[float, float, str]],
) -> str:
    speaker_names: dict[str, str] = {}
    lines: list[str] = []
    for segment in segments:
        raw_speaker = _best_speaker_for_segment(segment.start, segment.end, speaker_turns)
        name_hint = _speaker_name_hint(segment.text)
        if name_hint:
            speaker_label = name_hint
        elif raw_speaker is None:
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


def _speaker_name_hint(text: str) -> str | None:
    """Best-effort offline speaker-name heuristic.

    Recognizes simple local transcript prefixes like `John:` or `Sarah -`.
    This does not query the internet and only uses transcript text.
    """
    import re

    candidate = text.strip()
    if len(candidate) < 3:
        return None
    match = re.match(r"^([A-Z][A-Za-z0-9_.-]{1,30})\s*[:\-–—]\s+", candidate)
    if not match:
        return None
    name = match.group(1).strip()
    if name.lower() in {"speaker", "interviewer", "host", "guest"}:
        return name.title()
    return name


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
