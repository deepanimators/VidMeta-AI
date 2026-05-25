from __future__ import annotations

import base64
import math
from dataclasses import dataclass

import cv2


@dataclass
class VideoMetadata:
    duration_seconds: float
    fps: float
    width: int
    height: int
    frame_count: int
    aspect_ratio: str
    orientation: str


def get_video_metadata(video_path: str) -> VideoMetadata:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps else 0.0
        aspect_ratio, orientation = _aspect_info(width, height)
        return VideoMetadata(
            duration_seconds=round(duration, 1),
            fps=round(fps, 2),
            width=width,
            height=height,
            frame_count=frame_count,
            aspect_ratio=aspect_ratio,
            orientation=orientation,
        )
    finally:
        cap.release()


def _aspect_info(width: int, height: int) -> tuple[str, str]:
    if not width or not height:
        return "unknown", "unknown"
    ratio = width / height
    for w, h in [(16, 9), (9, 16), (1, 1), (4, 3), (3, 4), (4, 5), (5, 4), (21, 9)]:
        if abs(ratio - w / h) < 0.05:
            orient = "portrait" if h > w else ("square" if h == w else "landscape")
            return f"{w}:{h}", orient
    g = math.gcd(width, height)
    ar = f"{width // g}:{height // g}"
    orient = "landscape" if width > height else ("portrait" if height > width else "square")
    return ar, orient


def extract_frames(video_path: str, interval_sec: int, max_frames: int) -> list[str]:
    """Extract representative frames as 800px-wide base64 JPEGs for LLM analysis.

    Uses scene-change detection to prioritise visually distinct moments, then
    fills remaining slots with uniform-interval samples.  Blurry frames are
    filtered out; if all frames are blurry the sharpest available are kept.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps if fps else 0.0
        positions = _select_positions(cap, fps, total, duration, interval_sec, max_frames)

        raw: list[tuple[float, bytes]] = []
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            if w > 800:
                frame = cv2.resize(frame, (800, int(h * 800 / w)))
            sharpness = _sharpness(frame)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                raw.append((sharpness, bytes(buf)))

        sharp = [(s, d) for s, d in raw if s > 12.0]
        if not sharp and raw:
            sharp = sorted(raw, reverse=True)
        return [base64.b64encode(d).decode("ascii") for _, d in sharp[:max_frames]]
    finally:
        cap.release()


def extract_thumbnails(video_path: str, interval_sec: int, max_frames: int) -> list[str]:
    """Extract 180px-wide thumbnails for live UI display."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps if fps else 0.0
        positions = _select_positions(cap, fps, total, duration, interval_sec, max_frames)

        thumbnails: list[str] = []
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            if w > 180:
                frame = cv2.resize(frame, (180, int(h * 180 / w)))
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            if ok:
                thumbnails.append(base64.b64encode(bytes(buf)).decode("ascii"))
        return thumbnails[:max_frames]
    finally:
        cap.release()


def _select_positions(
    cap: cv2.VideoCapture,
    fps: float,
    total: int,
    duration: float,
    interval_sec: int,
    max_frames: int,
) -> list[int]:
    scene_pos = _detect_scene_changes(cap, fps, total, max_frames * 2)
    uniform_pos: list[int] = []
    t = 0.0
    while t < duration and len(uniform_pos) < max_frames * 3:
        uniform_pos.append(int(t * fps))
        t += interval_sec
    return _merge_deduplicate(scene_pos, uniform_pos, max_frames, fps)


def _detect_scene_changes(
    cap: cv2.VideoCapture, fps: float, total: int, max_scenes: int
) -> list[int]:
    sample_step = max(1, int(fps * 2))
    positions: list[int] = []
    prev_hist = None
    pos = 0
    while pos < total and len(positions) < max_scenes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            pos += sample_step
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)
        if prev_hist is None:
            positions.append(pos)
        else:
            diff = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff > 0.3:
                positions.append(pos)
        prev_hist = hist
        pos += sample_step
    return positions


def _merge_deduplicate(
    scene: list[int], uniform: list[int], max_frames: int, fps: float
) -> list[int]:
    min_gap = max(1, int(fps))
    merged = list(scene)
    for pos in uniform:
        if len(merged) >= max_frames:
            break
        if all(abs(pos - existing) >= min_gap for existing in merged):
            merged.append(pos)
    return sorted(set(merged))[:max_frames]


def _sharpness(frame) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
