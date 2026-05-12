from __future__ import annotations

import base64
import io
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderConfig:
    provider: str = "ollama"
    model: str = "gemma3:4b"
    api_key: str = ""
    api_base: str = ""
    ollama_url: str = "http://localhost:11434"


def call_llm(frames: list[str], prompt: str, config: ProviderConfig, max_tokens: int = 2200) -> str:
    provider = config.provider.lower()
    if provider == "ollama":
        return _call_ollama(frames, prompt, config.ollama_url, config.model, max_tokens)
    if provider in {"openrouter", "openai"}:
        base = config.api_base or (
            "https://openrouter.ai/api/v1"
            if provider == "openrouter"
            else "https://api.openai.com/v1"
        )
        return _call_openai_compat(frames, prompt, config.api_key, config.model, base, max_tokens)
    if provider == "anthropic":
        return _call_anthropic(frames, prompt, config.api_key, config.model, max_tokens)
    if provider == "gemini":
        return _call_gemini(frames, prompt, config.api_key, config.model, max_tokens)
    raise ValueError(f"Unsupported provider: {config.provider}")


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
            json={
                "model": model,
                "prompt": prompt,
                "images": frames,
                "stream": False,
                "options": {"num_predict": max_tokens},
            },
            timeout=180,
        )
        response.raise_for_status()
        return response.json().get("response", "")
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "")


def _call_openai_compat(
    frames: list[str],
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    max_tokens: int,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}}
        for frame in frames[:6]
    ]
    content.append({"type": "text", "text": prompt})
    token_limit_param = (
        {"max_completion_tokens": max_tokens}
        if _is_openai_base_url(base_url)
        else {"max_tokens": max_tokens}
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        **token_limit_param,
    )
    return response.choices[0].message.content or ""


def _is_openai_base_url(base_url: str) -> bool:
    return base_url.rstrip("/").lower() == "https://api.openai.com/v1"


def _call_anthropic(frames: list[str], prompt: str, api_key: str, model: str, max_tokens: int) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": frame}}
        for frame in frames[:5]
    ]
    content.append({"type": "text", "text": prompt})
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": content}],
    )
    return response.content[0].text


def _call_gemini(frames: list[str], prompt: str, api_key: str, model: str, max_tokens: int) -> str:
    import google.generativeai as genai
    from PIL import Image

    genai.configure(api_key=api_key)
    parts = [Image.open(io.BytesIO(base64.b64decode(frame))) for frame in frames[:5]]
    parts.append(prompt)
    response = genai.GenerativeModel(model).generate_content(
        parts,
        generation_config={"max_output_tokens": max_tokens},
    )
    return response.text or ""
