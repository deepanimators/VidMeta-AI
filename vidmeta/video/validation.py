from __future__ import annotations

from pathlib import Path


def is_valid_video_header(path: Path) -> bool:
    """Pure-Python magic-byte video detection — no system library required.

    Reads the first 16 bytes and checks against known video container signatures.
    Falls back to True on any read error to avoid blocking legitimate uploads.
    """
    try:
        with path.open("rb") as f:
            header = f.read(16)
        if len(header) < 8:
            return False
        # MP4 / MOV / M4V — ftyp atom at offset 4
        if header[4:8] == b"ftyp":
            return True
        # MKV / WebM — EBML magic bytes
        if header[:4] == b"\x1aE\xdf\xa3":
            return True
        # AVI — RIFF container with AVI subtype
        if header[:4] == b"RIFF" and header[8:12] == b"AVI ":
            return True
        # Ogg container (OGV)
        if header[:4] == b"OggS":
            return True
        # MPEG-1 / MPEG-2 video
        if header[:3] in (b"\x00\x00\x01", b"\x00\x00\x00\x01"):
            return True
        return False
    except Exception:
        return True
