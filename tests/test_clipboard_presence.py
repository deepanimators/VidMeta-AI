from pathlib import Path


def test_clipboard_usage_present():
    src = Path("web/src/main.tsx").read_text(encoding="utf-8")
    assert "navigator.clipboard.writeText" in src
