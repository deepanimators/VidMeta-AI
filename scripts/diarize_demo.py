#!/usr/bin/env python3
"""Simple CLI to run offline diarization on a video or audio file.

Usage:
    python scripts/diarize_demo.py /path/to/video.mp4 [whisper_model_size]

Notes:
    - Uses local Whisper transcription plus the offline diarizer.
    - Requires optional packages from the `offline-diarization` extra.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidmeta.video.transcription import get_transcript_segments, _diarize_offline


def main():
    if len(sys.argv) < 2:
        print("Usage: diarize_demo.py /path/to/video.mp4 [whisper_model_size]")
        raise SystemExit(2)
    path = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "small"
    print(f"Transcribing {path} (model={model})...")
    segments = get_transcript_segments(path, model)
    print(f"Got {len(segments)} segments")
    turns = _diarize_offline(f"{path}_audio.wav", segments)
    print(json.dumps({"turns": turns}, indent=2))


if __name__ == "__main__":
    main()
