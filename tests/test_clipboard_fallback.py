from pathlib import Path


def test_clipboard_fallback_helper_present():
    src = Path("web/src/main.tsx").read_text(encoding="utf-8")
    assert "document.execCommand(\"copy\")" in src
    assert "navigator.clipboard?.writeText" in src
