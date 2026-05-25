from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


@dataclass(frozen=True)
class ProviderConfig:
    provider: str = "ollama"
    model: str = "gemma3:4b"
    api_key: str = ""
    api_base: str = ""
    ollama_url: str = "http://localhost:11434"


# Keywords that identify vision-capable Ollama models.
# Models whose names do NOT contain any of these are likely text-only.
_OLLAMA_VISION_KEYWORDS: frozenset[str] = frozenset({
    "llava", "bakllava", "moondream", "vision", "minicpm",
    "qwen2-vl", "pixtral", "internvl", "cogvlm", "gemma3",
})


def ollama_likely_text_only(model: str) -> bool:
    """Return True if the Ollama model name does not indicate vision support."""
    lower = model.lower()
    return not any(kw in lower for kw in _OLLAMA_VISION_KEYWORDS)


def _is_retryable(exc: BaseException) -> bool:
    import requests
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    retryable = {
        "APIConnectionError", "APITimeoutError", "InternalServerError",
        "ServiceUnavailableError", "RateLimitError", "OverloadedError",
    }
    return type(exc).__name__ in retryable


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)
def call_llm(
    frames: list[str],
    prompt: str,
    config: ProviderConfig,
    max_tokens: int = 2200,
    video_path: str = "",
) -> str:
    """Call the configured LLM with optional visual context.

    Args:
        frames: Base64-encoded JPEG frames for vision providers.
        prompt: Text prompt.
        config: Provider / model configuration.
        max_tokens: Maximum response tokens.
        video_path: Local video file for Gemini native video analysis (optional).
    """
    provider = config.provider.lower()
    if provider == "ollama":
        return _call_ollama(frames, prompt, config.ollama_url, config.model, max_tokens)
    if provider in {"openrouter", "openai"}:
        base = config.api_base or (
            "https://openrouter.ai/api/v1" if provider == "openrouter" else "https://api.openai.com/v1"
        )
        return _call_openai_compat(frames, prompt, config.api_key, config.model, base, max_tokens)
    if provider == "anthropic":
        return _call_anthropic(frames, prompt, config.api_key, config.model, max_tokens)
    if provider == "gemini":
        return _call_gemini(frames, prompt, config.api_key, config.model, max_tokens, video_path)
    if provider == "nvidia":
        return _call_nvidia(frames, prompt, config.api_key, config.model, max_tokens)
    raise ValueError(f"Unsupported provider: {config.provider}")


def _clean_api_error(exc: BaseException) -> str:
    """Extract the human-readable message from OpenAI/Anthropic SDK status errors."""
    try:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            msg = (body.get("error") or {}).get("message") or body.get("message")
            if msg:
                return str(msg)
    except Exception:
        pass
    return getattr(exc, "message", None) or str(exc)


def _call_ollama(frames: list[str], prompt: str, url: str, model: str, max_tokens: int) -> str:
    import requests
    base_url = url.rstrip("/")
    messages = [{"role": "user", "content": prompt, "images": frames}]
    response = requests.post(
        f"{base_url}/api/chat",
        json={"model": model, "messages": messages, "stream": False, "options": {"num_predict": max_tokens}},
        timeout=180,
    )
    if response.status_code == 404:
        response = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "images": frames, "stream": False, "options": {"num_predict": max_tokens}},
            timeout=180,
        )
        response.raise_for_status()
        return response.json().get("response", "")
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "")


def _call_openai_compat(
    frames: list[str], prompt: str, api_key: str, model: str, base_url: str, max_tokens: int
) -> str:
    from openai import APIStatusError, OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    content: list[dict] = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}", "detail": "high"}}
        for frame in frames[:20]
    ]
    content.append({"type": "text", "text": prompt})
    token_param = (
        {"max_completion_tokens": max_tokens}
        if _is_openai_base_url(base_url)
        else {"max_tokens": max_tokens}
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            **token_param,
        )
        return response.choices[0].message.content or ""
    except APIStatusError as exc:
        raise RuntimeError(_clean_api_error(exc)) from exc


def _is_openai_base_url(base_url: str) -> bool:
    return base_url.rstrip("/").lower() == "https://api.openai.com/v1"


def _call_nvidia(frames: list[str], prompt: str, api_key: str, model: str, max_tokens: int) -> str:
    """Call NVIDIA NIM API — OpenAI-compatible endpoint with 5-image limit."""
    from openai import APIStatusError, OpenAI
    client = OpenAI(api_key=api_key, base_url="https://integrate.api.nvidia.com/v1")
    # NIM default max is 5 images per request.
    content: list[dict] = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}}
        for frame in frames[:5]
    ]
    content.append({"type": "text", "text": prompt})
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except APIStatusError as exc:
        raise RuntimeError(_clean_api_error(exc)) from exc


def _call_anthropic(frames: list[str], prompt: str, api_key: str, model: str, max_tokens: int) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    content: list[dict] = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": frame}}
        for frame in frames[:20]
    ]
    content.append({"type": "text", "text": prompt})
    try:
        response = client.messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text
    except anthropic.APIStatusError as exc:
        raise RuntimeError(_clean_api_error(exc)) from exc


def _call_gemini(
    frames: list[str], prompt: str, api_key: str, model: str, max_tokens: int, video_path: str = ""
) -> str:
    import google.generativeai as genai
    from PIL import Image
    genai.configure(api_key=api_key)

    # Prefer native video API — Gemini processes the full video at 1fps natively.
    if video_path and os.path.isfile(video_path):
        try:
            return _call_gemini_native_video(prompt, model, max_tokens, video_path)
        except Exception:
            pass  # Fallback to frame-based analysis

    parts: list = [Image.open(io.BytesIO(base64.b64decode(frame))) for frame in frames[:20]]
    parts.append(prompt)
    response = genai.GenerativeModel(model).generate_content(
        parts, generation_config={"max_output_tokens": max_tokens}
    )
    return response.text or ""


def _call_gemini_native_video(prompt: str, model: str, max_tokens: int, video_path: str) -> str:
    """Upload video via Gemini Files API and run native video analysis."""
    import time
    import google.generativeai as genai

    video_file = genai.upload_file(path=video_path)
    # Poll until Gemini finishes processing (up to ~120 s).
    for _ in range(60):
        video_file = genai.get_file(video_file.name)
        if video_file.state.name == "ACTIVE":
            break
        if video_file.state.name == "FAILED":
            raise ValueError("Gemini video file processing failed")
        time.sleep(2)
    else:
        raise TimeoutError("Gemini video file processing timed out after 120 s")
    try:
        response = genai.GenerativeModel(model).generate_content(
            [video_file, prompt],
            generation_config={"max_output_tokens": max_tokens},
        )
        return response.text or ""
    finally:
        try:
            genai.delete_file(video_file.name)
        except Exception:
            pass
