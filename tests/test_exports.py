from vidmeta.exports.builders import export_csv, export_json, export_txt


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
