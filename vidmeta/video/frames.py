from __future__ import annotations

import base64

import cv2


def extract_frames(video_path: str, interval_sec: int, max_frames: int) -> list[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps else 0

    positions: list[int] = []
    t = 0.0
    while t < duration and len(positions) < max_frames:
        positions.append(int(t * fps))
        t += interval_sec

    frames: list[str] = []
    for position in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        ok, frame = cap.read()
        if not ok:
            continue
        height, width = frame.shape[:2]
        if width > 800:
            frame = cv2.resize(frame, (800, int(height * 800 / width)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            frames.append(base64.b64encode(buf).decode("ascii"))

    cap.release()
    return frames
