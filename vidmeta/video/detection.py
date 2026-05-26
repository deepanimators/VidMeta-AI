from __future__ import annotations

_model = None


def detect_objects(frame_b64: str) -> list[str]:
    """Detect objects in a base64 JPEG frame using YOLOv8n.

    Returns deduplicated class name list. Returns [] if ultralytics not installed.
    """
    try:
        import base64

        import cv2
        import numpy as np

        data = base64.b64decode(frame_b64)
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return []

        model = _get_model()
        results = model(img, verbose=False, conf=0.4)
        seen: list[str] = []
        for r in results:
            for cls_id in r.boxes.cls.tolist():
                name = model.names[int(cls_id)]
                if name not in seen:
                    seen.append(name)
        return seen
    except Exception:
        return []


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("yolov8n.pt")
    return _model
