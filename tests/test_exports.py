from vidmeta.exports.builders import export_csv, export_json, export_txt
from vidmeta.ai.output import parse_metadata


def sample_metadata():
    return {
        "video_summary": "A product video.",
        "content_category": "Product Showcase",
        "youtube": {
            "title": "Great title",
            "description": "Useful description",
            "hashtags": ["#one", "#two"],
            "keywords": ["one", "two"],
            "cta": "Subscribe",
            "posting_tip": "Post tomorrow",
        },
        "instagram": {},
        "facebook": {},
        "tiktok": {},
        "linkedin": {},
    }


def test_export_json_contains_platforms():
    exported = export_json(sample_metadata())
    assert '"youtube"' in exported
    assert "Great title" in exported


def test_export_csv_contains_rows():
    exported = export_csv(sample_metadata())
    assert "Platform,Title,Description" in exported
    assert "YouTube,Great title" in exported


def test_export_txt_contains_sections():
    exported = export_txt(sample_metadata())
    assert "YouTube" in exported
    assert "TITLE:" in exported


def test_batch_csv_contains_file_column():
    exported = export_csv({"batch_results": [{"file": "a.mp4", "metadata": sample_metadata()}]})
    assert "File,Platform,Title" in exported
    assert "a.mp4,YouTube,Great title" in exported


def test_export_csv_contains_new_platform_map():
    exported = export_csv(
        {
            "platforms": {
                "youtube_shorts": {"title": "Short hook", "description": "Short caption"},
                "whatsapp_channels": {"title": "Channel hook", "description": "Channel caption"},
            }
        }
    )
    assert "YouTube Shorts,Short hook" in exported
    assert "WhatsApp Channels,Channel hook" in exported


def test_parse_metadata_normalizes_platforms_map():
    result = parse_metadata(
        """
        {
          "video_summary": "A product video.",
          "platforms": {
            "reddit": {"title": "Reddit discussion"},
            "threads": {"title": "Thread starter"}
          }
        }
        """
    )
    assert result.platforms["reddit"].title == "Reddit discussion"
    assert result.platforms["threads"].title == "Thread starter"
