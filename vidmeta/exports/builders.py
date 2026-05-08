from __future__ import annotations

import csv
import io
import json
from typing import Any

from vidmeta.ai.schemas import PLATFORMS


PLATFORM_LABELS = {
    "youtube": "YouTube",
    "instagram": "Instagram",
    "facebook": "Facebook",
    "tiktok": "TikTok",
    "linkedin": "LinkedIn",
}


def export_json(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, indent=2, ensure_ascii=False)


def export_csv(metadata: dict[str, Any]) -> str:
    if isinstance(metadata.get("batch_results"), list):
        return _export_batch_csv(metadata["batch_results"])
    rows: list[dict[str, str]] = []
    for platform in PLATFORMS:
        data = metadata.get(platform, {})
        if not isinstance(data, dict):
            continue
        tags = data.get("hashtags", [])
        keywords = data.get("keywords", [])
        rows.append(
            {
                "Platform": PLATFORM_LABELS[platform],
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
    if isinstance(metadata.get("batch_results"), list):
        return _export_batch_txt(metadata["batch_results"])
    sections: list[str] = []
    for platform in PLATFORMS:
        data = metadata.get(platform, {})
        if not isinstance(data, dict):
            continue
        tags = data.get("hashtags", [])
        keywords = data.get("keywords", [])
        sections.append(
            f"{'=' * 40}\n{PLATFORM_LABELS[platform]}\n{'=' * 40}\n"
            f"TITLE:\n{data.get('title', '')}\n\n"
            f"DESCRIPTION:\n{data.get('description', '')}\n\n"
            f"HASHTAGS:\n{' '.join(tags) if isinstance(tags, list) else tags}\n\n"
            f"KEYWORDS:\n{', '.join(keywords) if isinstance(keywords, list) else keywords}\n\n"
            f"CTA: {data.get('cta', '')}\n"
            f"POSTING TIP: {data.get('posting_tip', '')}\n"
        )
    return "\n\n".join(sections)


def _export_batch_csv(results: list[dict[str, Any]]) -> str:
    rows: list[dict[str, str]] = []
    for result in results:
        metadata = result.get("metadata", {})
        filename = str(result.get("file", ""))
        if not isinstance(metadata, dict):
            continue
        for platform in PLATFORMS:
            data = metadata.get(platform, {})
            if not isinstance(data, dict):
                continue
            tags = data.get("hashtags", [])
            keywords = data.get("keywords", [])
            rows.append(
                {
                    "File": filename,
                    "Platform": PLATFORM_LABELS[platform],
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
