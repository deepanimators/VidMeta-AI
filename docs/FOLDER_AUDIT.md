# Folder Audit

VidMeta AI is now a FastAPI + React/Vite + Tauri project.

```text
api/                 FastAPI app, routes, uploads, jobs
web/                 React/Vite dashboard
desktop/             Tauri desktop shell
vidmeta/             Reusable Python service modules
  ai/                prompts, provider adapters, metadata parsing
  video/             frame extraction and transcription
  exports/           JSON/CSV/TXT export builders
  service/           SQLite database and pipeline orchestration
  storage/           local disk and S3-compatible storage helpers
legacy/              old Streamlit app retained for migration reference
tests/               API and export tests
```

## Strengths

- Large local videos can be processed by path without browser upload.
- Browser uploads use chunked resumable endpoints.
- SQLite stores settings, jobs, transcripts, and metadata outputs.
- Storage can be local disk or S3-compatible.
- Tauri shell provides native file/folder picking.

## Remaining Gaps

- Tauri currently expects the FastAPI service to be started separately.
- Hosted mode has no auth and must remain private/self-hosted.
- More tests are needed for real video fixtures, provider mocks, resumable resume, and desktop packaging.
