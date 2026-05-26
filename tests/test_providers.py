from __future__ import annotations

import sys
import types


def test_openai_provider_uses_max_completion_tokens_for_openai_base(monkeypatch):
    from vidmeta.ai.providers import _call_openai_compat

    captured: dict = {}

    class FakeCompletions:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="done"))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(
        OpenAI=FakeOpenAI, APIStatusError=Exception
    ))

    result = _call_openai_compat([], "prompt", "key", "gpt-5.5", "https://api.openai.com/v1", 1234, [])

    assert result == "done"
    assert captured["max_completion_tokens"] == 1234
    assert "max_tokens" not in captured


def test_openai_compatible_provider_keeps_max_tokens_for_non_openai_base(monkeypatch):
    from vidmeta.ai.providers import _call_openai_compat

    captured: dict = {}

    class FakeCompletions:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="done"))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(
        OpenAI=FakeOpenAI, APIStatusError=Exception
    ))

    result = _call_openai_compat([], "prompt", "key", "openrouter/auto", "https://openrouter.ai/api/v1", 1234, [])

    assert result == "done"
    assert captured["max_tokens"] == 1234
    assert "max_completion_tokens" not in captured


def test_ollama_falls_back_to_generate_when_chat_endpoint_is_missing(monkeypatch):
    from vidmeta.ai.providers import _call_ollama

    requests_seen: list[dict] = []

    class FakeResponse:
        def __init__(self, status_code: int, payload: dict):
            self.status_code = status_code
            self._payload = payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise AssertionError(f"unexpected status: {self.status_code}")

        def json(self):
            return self._payload

    def fake_post(url, json, timeout):
        requests_seen.append({"url": url, "json": json, "timeout": timeout})
        if url.endswith("/api/chat"):
            return FakeResponse(404, {})
        return FakeResponse(200, {"response": "generated"})

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=fake_post))

    result = _call_ollama(["frame"], "prompt", "http://localhost:11434/", "gemma3:4b", 321, [])

    assert result == "generated"
    assert [request["url"] for request in requests_seen] == [
        "http://localhost:11434/api/chat",
        "http://localhost:11434/api/generate",
    ]
    assert requests_seen[1]["json"]["images"] == ["frame"]
    assert requests_seen[1]["json"]["options"]["num_predict"] == 321
