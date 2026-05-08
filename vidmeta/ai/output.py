from __future__ import annotations

import json
from collections.abc import Sequence

from vidmeta.ai.schemas import MetadataResult


def parse_metadata(raw: str, target_platforms: Sequence[str] | None = None) -> MetadataResult:
    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        return MetadataResult(raw=raw)
    if not isinstance(parsed, dict):
        return MetadataResult(raw=raw)
    try:
        return MetadataResult.from_llm_json(parsed, raw=raw, target_platforms=target_platforms)
    except Exception:
        return MetadataResult(raw=raw)
