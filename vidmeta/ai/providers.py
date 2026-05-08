from __future__ import annotations

import base64
import io
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderConfig:
    provider: str = "ollama"
    model: str = "gemma4"
    api_key: str = ""
    api_base: str = ""
    ollama_url: str = "http://localhost:11434"


def call_llm(frames: list[str], prompt: str, config: ProviderConfig) -> str:
    provider = config.provider.lower()
    if provider == "ollama":
        return _call_ollama(frames, prompt, config.ollama_url, config.model)
    if provider in {"openrouter", "openai"}:
        base = config.api_base or (
            "https://openrouter.ai/api/v1"
            if provider == "openrouter"
            else "https://api.openai.com/v1"
        )
        return _call_openai_compat(frames, prompt, config.api_key, config.model, base)
    if provider == "anthropic":
        return _call_anthropic(frames, prompt, config.api_key, config.model)
    if provider == "gemini":
        return _call_gemini(frames, prompt, config.api_key, config.model)
    raise ValueError(f"Unsupported provider: {config.provider}")


def _call_ollama(frames: list[str], prompt: str, url: str, model: str) -> str:
    import requests

    messages = [{"role": "user", "content": prompt, "images": frames}]
    response = requests.post(
        f"{url.rstrip('/')}/api/chat",
        json={"model": model, "messages": messages, "stream": False},
        timeout=180,
    )
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "")


def _call_openai_compat(frames: list[str], prompt: str, api_key: str, model: str, base_url: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}}
        for frame in frames[:6]
    ]
    content.append({"type": "text", "text": prompt})
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=2200,
    )
    return response.choices[0].message.content or ""


def _call_anthropic(frames: list[str], prompt: str, api_key: str, model: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": frame}}
        for frame in frames[:5]
    ]
    content.append({"type": "text", "text": prompt})
    response = client.messages.create(
        model=model,
        max_tokens=2200,
        messages=[{"role": "user", "content": content}],
    )
    return response.content[0].text


def _call_gemini(frames: list[str], prompt: str, api_key: str, model: str) -> str:
    import google.generativeai as genai
    from PIL import Image

    genai.configure(api_key=api_key)
    parts = [Image.open(io.BytesIO(base64.b64decode(frame))) for frame in frames[:5]]
    parts.append(prompt)
    response = genai.GenerativeModel(model).generate_content(parts)
    return response.text or ""
