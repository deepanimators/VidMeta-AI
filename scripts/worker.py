#!/usr/bin/env python3
"""Worker process to run `analyze_video` for a single file and write JSON result.
Usage: python scripts/worker.py --input request.json --output result.json
The request JSON should contain keys: source_path, brand, video, provider, target_platforms
"""
import argparse
import json
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vidmeta.settings import BrandContext, VideoSettings, ProviderSettings
from vidmeta.service.pipeline import analyze_video

_canceled = False


def _on_term(signum, frame):
    global _canceled
    _canceled = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    with open(args.input, "r", encoding="utf-8") as f:
        req = json.load(f)
    source = req.get("source_path")
    brand = BrandContext.model_validate(req.get("brand") or {})
    video = VideoSettings.model_validate(req.get("video") or {})
    provider = ProviderSettings.model_validate(req.get("provider") or {})
    target_platforms = req.get("target_platforms")

    def should_cancel():
        return _canceled

    try:
        result = analyze_video(
            source,
            brand=brand,
            video=video,
            provider=provider,
            target_platforms=target_platforms,
            progress=None,
            should_cancel=should_cancel,
        )
        out = {"ok": True, "result": result}
        with open(args.output, "w", encoding="utf-8") as fo:
            json.dump(out, fo)
        return 0
    except Exception as exc:
        with open(args.output, "w", encoding="utf-8") as fo:
            json.dump({"ok": False, "error": str(exc)}, fo)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
