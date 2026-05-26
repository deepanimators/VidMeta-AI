from __future__ import annotations

import base64
import math
from dataclasses import dataclass

import cv2
import numpy as np


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


def extract_frames(video_path: str, interval_sec: int, max_frames: int) -> list[tuple[float, str]]:
    """Extract representative frames as (timestamp_sec, base64_jpeg) tuples.

    Frame selection: PySceneDetect ContentDetector (falls back to histogram diff).
    Frame quality: multi-signal score combining sharpness and color entropy.
    Deduplication: histogram-based — removes near-identical frames before encoding.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps if fps else 0.0
        positions = _select_positions(cap, fps, total, duration, interval_sec, max_frames, video_path)

        # Collect candidates as (timestamp_sec, quality_score, bgr_ndarray)
        candidates: list[tuple[float, float, np.ndarray]] = []
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
            if not ok:
                continue
            ts = pos / fps
            h, w = frame.shape[:2]
            if w > 800:
                frame = cv2.resize(frame, (800, int(h * 800 / w)))
            score = _frame_quality_score(frame)
            candidates.append((ts, score, frame))

        # Quality filter — fallback to all sorted by score if everything is blurry
        QUALITY_THRESHOLD = 12.0
        good = [(ts, s, f) for ts, s, f in candidates if s > QUALITY_THRESHOLD]
        if not good and candidates:
            good = sorted(candidates, key=lambda x: x[1], reverse=True)

        # Remove near-duplicate frames before handing to LLM
        selected = _dedup_similar(good, max_frames)

        result: list[tuple[float, str]] = []
        for ts, _, frame in selected:
            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok2:
                result.append((ts, base64.b64encode(bytes(buf)).decode("ascii")))
        return result
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
        positions = _select_positions(cap, fps, total, duration, interval_sec, max_frames, video_path)

        thumbnails: list[str] = []
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            if w > 180:
                frame = cv2.resize(frame, (180, int(h * 180 / w)))
            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            if ok2:
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
    video_path: str = "",
) -> list[int]:
    # PySceneDetect gives semantically correct cut points; fall back to histogram diff
    scene_pos = _detect_scenes_pyscenedetect(video_path, max_frames * 2) if video_path else []
    if not scene_pos:
        scene_pos = _detect_scene_changes_histogram(cap, fps, total, max_frames * 2)

    uniform_pos: list[int] = []
    t = 0.0
    while t < duration and len(uniform_pos) < max_frames * 3:
        uniform_pos.append(int(t * fps))
        t += interval_sec
    return _merge_deduplicate(scene_pos, uniform_pos, max_frames, fps)


def _detect_scenes_pyscenedetect(video_path: str, max_scenes: int) -> list[int]:
    """HSV-aware shot detection via PySceneDetect. Returns [] if not installed."""
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
        video = open_video(video_path)
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=27.0))
        manager.detect_scenes(video, show_progress=False)
        scenes = manager.get_scene_list()
        return [int(s[0].get_frames()) for s in scenes[:max_scenes]]
    except Exception:
        return []


def _detect_scene_changes_histogram(
    cap: cv2.VideoCapture, fps: float, total: int, max_scenes: int
) -> list[int]:
    """Fallback: grayscale histogram Bhattacharyya distance."""
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


def _frame_quality_score(frame: np.ndarray) -> float:
    """Sharpness × color richness. Blurry or monochrome frames score low."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist = hist / (hist.sum() + 1e-9)
    nonzero = hist[hist > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero))) if len(nonzero) else 0.0

    return sharpness * (1.0 + entropy / 5.0)


def _frame_hist(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _dedup_similar(
    candidates: list[tuple[float, float, np.ndarray]], max_frames: int
) -> list[tuple[float, float, np.ndarray]]:
    """Greedy: keep frames whose HSV histogram is sufficiently different from already-selected."""
    if not candidates:
        return []
    selected = [candidates[0]]
    selected_hists = [_frame_hist(candidates[0][2])]
    for item in candidates[1:]:
        if len(selected) >= max_frames:
            break
        h = _frame_hist(item[2])
        max_sim = max(
            float(cv2.compareHist(h, sh, cv2.HISTCMP_CORREL)) for sh in selected_hists
        )
        if max_sim < 0.85:
            selected.append(item)
            selected_hists.append(h)
    return selected
