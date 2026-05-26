from __future__ import annotations

import sys

_easyocr_reader = None


def extract_text(frame_b64: str) -> list[str]:
    """Extract visible text from a base64 JPEG frame.

    Uses Apple Vision on macOS (zero-weight, Neural Engine accelerated).
    Falls back to EasyOCR on other platforms or if Vision is unavailable.
    Returns [] if no OCR package is available.
    """
    if sys.platform == "darwin":
        result = _apple_vision_ocr(frame_b64)
        if result is not None:
            return result
    return _easyocr(frame_b64)


def _apple_vision_ocr(frame_b64: str) -> list[str] | None:
    """macOS Vision framework OCR via pyobjc. Returns None if unavailable."""
    try:
        import base64
        import tempfile
        import os
        from Foundation import NSURL  # type: ignore[import]
        import Vision  # type: ignore[import]

        data = base64.b64decode(frame_b64)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            url = NSURL.fileURLWithPath_(tmp_path)
            handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, {})
            request = Vision.VNRecognizeTextRequest.alloc().init()
            request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
            handler.performRequests_error_([request], None)
            observations = request.results() or []
            texts = []
            for obs in observations:
                top = obs.topCandidates_(1)
                if top:
                    text = str(top[0].string()).strip()
                    if text:
                        texts.append(text)
            return texts
        finally:
            os.unlink(tmp_path)
    except Exception:
        return None


def _easyocr(frame_b64: str) -> list[str]:
    """EasyOCR cross-platform fallback. Returns [] if not installed."""
    try:
        import base64

        import cv2
        import numpy as np

        data = base64.b64decode(frame_b64)
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return []

        reader = _get_easyocr_reader()
        results = reader.readtext(img, detail=0, paragraph=True)
        return [r.strip() for r in results if isinstance(r, str) and r.strip()]
    except Exception:
        return []


def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr  # type: ignore[import]
        _easyocr_reader = easyocr.Reader(["en"], gpu=False)
    return _easyocr_reader
