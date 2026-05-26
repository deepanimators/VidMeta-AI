from __future__ import annotations

_model = None


def caption_frame(frame_b64: str) -> str:
    """Generate a natural-language caption for a frame using moondream2 int4.

    Model file (~900MB) is downloaded once to ~/.cache/moondream on first use.
    Returns "" if moondream is not installed or model download fails.
    """
    try:
        import base64
        import io
        from PIL import Image  # type: ignore[import]

        model = _get_model()
        data = base64.b64decode(frame_b64)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        result = model.caption(img)
        return str(result.get("caption", "")).strip()
    except Exception:
        return ""


def _get_model():
    global _model
    if _model is None:
        import moondream as md  # type: ignore[import]
        model_path = _ensure_model_file()
        _model = md.vl(model=model_path)
    return _model


def _ensure_model_file() -> str:
    """Download moondream-2b-int4.mf from HuggingFace if not already cached."""
    import gzip
    import os

    cache_dir = os.path.expanduser("~/.cache/moondream")
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, "moondream-2b-int4.mf")

    if os.path.exists(model_path):
        return model_path

    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import]
        gz_path = hf_hub_download(
            repo_id="vikhyatk/moondream2",
            filename="moondream-2b-int4.mf.gz",
        )
        with gzip.open(gz_path, "rb") as f_in:
            with open(model_path, "wb") as f_out:
                while chunk := f_in.read(65536):
                    f_out.write(chunk)
        return model_path
    except Exception as exc:
        raise RuntimeError(
            "moondream model not found. Download moondream-2b-int4.mf from "
            "https://huggingface.co/vikhyatk/moondream2 and place it at "
            f"{model_path}"
        ) from exc
