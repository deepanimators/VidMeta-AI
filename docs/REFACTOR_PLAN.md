# Full Code Refactor Plan

## Goal

Keep the same local Streamlit experience while making the app easier to test and publish.

## Step 1 - Settings

Create `vidmeta/settings.py` for:

- Upload/message limits.
- Provider defaults.
- Supported video extensions.
- Cookie password lookup.

## Step 2 - Video module

Move:

- `extract_frames` to `vidmeta/video/frames.py`.
- `transcribe_audio` to `vidmeta/video/transcription.py`.
- Uploaded temp-file handling to `vidmeta/video/files.py`.

## Step 3 - AI module

Move:

- Prompt strings to `vidmeta/ai/prompts.py`.
- Provider calls to `vidmeta/ai/providers.py`.
- JSON parsing and repair to `vidmeta/ai/output.py`.

## Step 4 - Export module

Move JSON, CSV, and TXT generation into pure functions under `vidmeta/exports/`.

## Step 5 - Tests

Add tests for:

- CLI upload limit flags.
- Frame extraction with a fixture video.
- Metadata JSON parsing.
- CSV/TXT export formatting.
- Prompt injection guard text.

## Step 6 - Optional persistence

Add SQLite only after the core modules are split.
