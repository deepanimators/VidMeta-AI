from __future__ import annotations

import csv
import io
import json
from typing import Any

from vidmeta.ai.schemas import PLATFORM_LABELS, PLATFORMS


def export_json(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, indent=2, ensure_ascii=False)


def export_csv(metadata: dict[str, Any]) -> str:
    if isinstance(metadata.get("batch_results"), list):
        return _export_batch_csv(metadata["batch_results"])
    rows: list[dict[str, str]] = []
    for platform, data in _platform_entries(metadata):
        tags = data.get("hashtags", [])
        keywords = data.get("keywords", [])
        rows.append(
            {
                "Platform": _platform_label(platform),
                "Title": str(data.get("title", "")),
                "Description": str(data.get("description", "")),
                "Hashtags": " ".join(tags) if isinstance(tags, list) else str(tags),
                "Keywords": ", ".join(keywords) if isinstance(keywords, list) else str(keywords),
                "CTA": str(data.get("cta", "")),
                "Posting Tip": str(data.get("posting_tip", "")),
            }
        )
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def export_txt(metadata: dict[str, Any]) -> str:
    legend = "LEGEND: Speaker N is the voice cluster; Face N is the sampled face cluster."
    if isinstance(metadata.get("batch_results"), list):
        return f"{legend}\n\n{_export_batch_txt(metadata['batch_results'])}"
    sections: list[str] = []
    for platform, data in _platform_entries(metadata):
        tags = data.get("hashtags", [])
        keywords = data.get("keywords", [])
        sections.append(
            f"{'=' * 40}\n{_platform_label(platform)}\n{'=' * 40}\n"
            f"TITLE:\n{data.get('title', '')}\n\n"
            f"DESCRIPTION:\n{data.get('description', '')}\n\n"
            f"HASHTAGS:\n{' '.join(tags) if isinstance(tags, list) else tags}\n\n"
            f"KEYWORDS:\n{', '.join(keywords) if isinstance(keywords, list) else keywords}\n\n"
            f"CTA: {data.get('cta', '')}\n"
            f"POSTING TIP: {data.get('posting_tip', '')}\n"
        )
    if not sections:
        return legend
    return f"{legend}\n\n" + "\n\n".join(sections)


def _export_batch_csv(results: list[dict[str, Any]]) -> str:
    rows: list[dict[str, str]] = []
    for result in results:
        metadata = result.get("metadata", {})
        filename = str(result.get("file", ""))
        if not isinstance(metadata, dict):
            continue
        for platform, data in _platform_entries(metadata):
            tags = data.get("hashtags", [])
            keywords = data.get("keywords", [])
            rows.append(
                {
                    "File": filename,
                    "Platform": _platform_label(platform),
                    "Title": str(data.get("title", "")),
                    "Description": str(data.get("description", "")),
                    "Hashtags": " ".join(tags) if isinstance(tags, list) else str(tags),
                    "Keywords": ", ".join(keywords) if isinstance(keywords, list) else str(keywords),
                    "CTA": str(data.get("cta", "")),
                    "Posting Tip": str(data.get("posting_tip", "")),
                }
            )
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _export_batch_txt(results: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for result in results:
        parts.append(f"{'#' * 48}\n{result.get('file', '')}\n{'#' * 48}\n")
        metadata = result.get("metadata", {})
        if isinstance(metadata, dict):
            parts.append(export_txt(metadata))
    return "\n\n".join(parts)


def _platform_entries(metadata: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    source = metadata.get("platforms") if isinstance(metadata.get("platforms"), dict) else metadata
    entries: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for platform in PLATFORMS:
        data = source.get(platform) if isinstance(source, dict) else None
        if isinstance(data, dict):
            entries.append((platform, data))
            seen.add(platform)
    if isinstance(source, dict):
        for platform, data in source.items():
            if platform not in seen and isinstance(data, dict):
                entries.append((platform, data))
    return entries


def _platform_label(platform: str) -> str:
    return PLATFORM_LABELS.get(platform, platform.replace("_", " ").title())
