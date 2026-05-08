# Roadmap

## Phase 1 - Public MIT local release

- Keep the app simple and local-first.
- Document installation, upload limits, Docker, and provider setup.
- Add CI compile checks and dependency audit.
- Move Streamlit config to `.streamlit/config.toml`.
- Keep local path and batch folder mode prominent for large files.

## Phase 2 - Code quality

- Split `app.py` into video, AI, export, settings, and UI modules.
- Add metadata schema validation.
- Add tests for prompt formatting, JSON cleanup, export generation, and CLI flags.
- Add structured logging for processing failures.

## Phase 3 - Local history

- Add optional SQLite history.
- Save previous runs and exports.
- Add re-run from previous settings.

## Phase 4 - Hosted/SaaS option

- Add auth, user workspaces, billing, job queue, object storage, and Postgres.
- Add upload antivirus scanning and abuse controls.
- Add async workers for long videos.
