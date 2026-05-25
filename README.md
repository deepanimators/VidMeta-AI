# VidMeta AI

VidMeta AI is a local-first AI video metadata service. It analyzes local videos and generates platform-ready titles, descriptions, hashtags, keywords, CTAs, and posting tips for 37 social platforms across major video, social, community, publishing, and regional networks.

## Screenshot

<img width="2442" height="10234" alt="image" src="https://github.com/user-attachments/assets/4a88bafc-d607-48ae-b0c1-1c0b8493c1e1" />

## Features

- **37 platform profiles** generated from a single video in one pass
- **Local-first** — Tauri desktop mode passes real filesystem paths; no upload required
- **Resumable uploads** — TUS protocol for large files with automatic resume on reconnect
- **5 LLM providers** — Ollama (local), OpenAI, Anthropic, Gemini, OpenRouter; switch without code changes
- **Audio transcription** — Faster-Whisper with thread-safe model caching; warm subsequent jobs
- **Batch processing** — point at a folder to analyze all video files in one job with ETA
- **Encrypted secrets** — API keys and S3 credentials encrypted at rest (AES-256 Fernet)
- **Platform constraints** — titles and descriptions truncated to platform character limits post-LLM
- **Retry resilience** — exponential backoff on transient LLM errors (rate limits, timeouts, overload)
- **S3-compatible storage** — local disk or any S3-compatible backend (AWS, Cloudflare R2, MinIO)
- **Export** — JSON, CSV, plain-text export per job and per platform
- **Rate limiting** — 300 req/min global, 60 req/min job creation, 30 req/min settings writes
- **Adaptive UI polling** — 2 s while jobs are active, 15 s when idle
- **Job history** — SQLite WAL, cascading deletes, N+1-free batch event loading

## Supported Platforms

| Group | Platforms |
|---|---|
| Core video networks | YouTube, YouTube Shorts, Instagram Reels, Instagram Feed, Facebook, Facebook Reels, TikTok, LinkedIn |
| Social conversation | X / Twitter, Threads, Bluesky, Mastodon, Reddit, Quora |
| Messaging and communities | WhatsApp Channels, Telegram Channels, Discord, LINE VOOM |
| Visual discovery | Pinterest, Snapchat Spotlight, Lemon8, Tumblr |
| Publishing and long-form | Medium, Substack Notes, Twitch, Vimeo, Rumble, Dailymotion |
| Regional high-scale | WeChat Channels, Douyin, Kuaishou, Bilibili, Weibo, VK, ShareChat, Moj, Josh |

## Architecture

The supported runtime:

- **FastAPI service** in `api/` — REST API and job runner
- **React/Vite dashboard** in `web/` — single-page UI
- **Tauri desktop shell** in `desktop/` — native file picker, no upload size limit
- **SQLite database** at `~/.vidmeta-ai/vidmeta.db` (WAL mode, FK cascade on)

The old Streamlit app is retained as a migration reference under `legacy/`.

## Prerequisites

| Tool | Required | Purpose |
|---|---|---|
| Python 3.10+ | Yes | FastAPI service |
| ffmpeg | Yes for audio | Audio extraction and transcription |
| Node.js 20+ | Yes for UI | React/Vite dashboard |
| Rust + Tauri prerequisites | Desktop only | Native desktop shell |
| Ollama | Optional | Local/private LLM inference |

## Backend Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install .
vidmeta serve
```

The API runs at `http://127.0.0.1:8000`.

Copy `.env.example` to `.env` and configure before first run:

```bash
cp .env.example .env
```

## Web Dashboard

```bash
cd web
npm install
npm run dev
```

Open `http://localhost:5173`. The Vite dev server proxies `/api` to `http://127.0.0.1:8000`.

## Desktop App

```bash
vidmeta serve

cd web && npm install
cd ../desktop && npm install && npm run dev
```

The desktop shell enables native file/folder picking — the preferred workflow for large local videos with no upload size limit.

Installer builds:

```bash
npm run build:mac
npm run build:windows
npm run build:linux
```

## Docker

```bash
docker compose up --build
```

The API is exposed on `http://localhost:8000`. Mount videos into `./videos` and use paths like `/videos/example.mp4` for local-path jobs inside the container.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Service health check |
| `GET` | `/api/settings` | Load current settings |
| `PUT` | `/api/settings` | Save settings |
| `POST` | `/api/uploads` | Multipart video upload |
| `POST` | `/api/uploads/resumable` | Create TUS resumable upload session |
| `OPTIONS` | `/api/uploads/resumable` | TUS capabilities |
| `HEAD` | `/api/uploads/resumable/{id}` | Get current upload offset |
| `PATCH` | `/api/uploads/resumable/{id}` | Append chunk to upload |
| `GET` | `/api/uploads/{id}` | Get upload record |
| `POST` | `/api/jobs/from-path` | Create job from local filesystem path |
| `POST` | `/api/jobs/from-upload/{upload_id}` | Create job from completed upload |
| `GET` | `/api/jobs` | List jobs (last 50) |
| `GET` | `/api/jobs/{id}` | Get job with events |
| `DELETE` | `/api/jobs/{id}` | Delete job and all associated data |
| `GET` | `/api/jobs/{id}/events` | Get job events only |
| `GET` | `/api/jobs/{id}/result` | Get job result (409 if not ready) |
| `GET` | `/api/jobs/{id}/exports/{format}` | Export result as `json`, `csv`, or `txt` |
| `POST` | `/api/admin/cleanup` | Purge expired and orphaned upload records |

## Configuration

All configuration via environment variables. Copy `.env.example` to `.env` to get started.

| Variable | Default | Description |
|---|---|---|
| `VIDMETA_API_HOST` | `127.0.0.1` | Bind host for the API server |
| `VIDMETA_API_PORT` | `8000` | Bind port |
| `VIDMETA_DATA_DIR` | `~/.vidmeta-ai` | Base directory for database, uploads, processing temp |
| `VIDMETA_DATABASE` | `$DATA_DIR/vidmeta.db` | SQLite database path |
| `VIDMETA_UPLOAD_DIR` | `$DATA_DIR/uploads` | Uploaded video storage |
| `VIDMETA_PROCESSING_DIR` | `$DATA_DIR/processing` | Temp files during analysis |
| `VIDMETA_MAX_UPLOAD_MB` | `2048` | Maximum upload size in MB |
| `VIDMETA_SECRET_KEY` | auto-generated | Fernet encryption key for secrets at rest |
| `VIDMETA_ALLOWED_PATHS` | unrestricted | Colon-separated allowlist for local-path jobs (hosted mode) |
| `VIDMETA_CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins |
| `VIDMETA_UPLOAD_RETENTION_DAYS` | `0` | Days to keep upload records (0 = never purge) |
| `VITE_API_BASE` | `` | Frontend API base URL override |

## Security

**Secrets at rest**: API keys and S3 credentials are encrypted with AES-256 (Fernet) before writing to SQLite. A `.secret_key` file (mode 600) is auto-generated in the data directory on first run. Set `VIDMETA_SECRET_KEY` explicitly to make encryption portable when migrating data between hosts.

**Path allowlist**: In hosted deployments, set `VIDMETA_ALLOWED_PATHS` to a colon-separated list of allowed directories. Local-path jobs that resolve outside these directories are rejected with HTTP 400. In local/desktop mode the env var is unset and no restriction is applied.

**CORS**: Defaults to `*`. Tighten in hosted deployments with `VIDMETA_CORS_ORIGINS=https://your-domain.com`.

**Rate limiting**: 300 req/min global (slowapi), 60 req/min on job creation, 30 req/min on settings writes. Responds with HTTP 429 on breach.

**Upload validation**: Both multipart and TUS resumable uploads validate magic bytes (MP4/MOV ftyp, MKV/WebM EBML, AVI RIFF, OGG, MPEG) before the upload record is accepted. File extension and content must match.

**No authentication**: By design for local/open-source use. Do not expose this service publicly without adding an auth layer in front.

## Storage Modes

Storage is configured from the UI or `PUT /api/settings`:

- `local_disk` — uploaded videos stored under the VidMeta data directory
- `s3_compatible` — uploaded videos archived to S3-compatible storage (AWS S3, Cloudflare R2, MinIO)

Local path jobs do not copy files unless `import_local_files` is enabled in settings.

## Provider Modes

| Provider | Key required | Notes |
|---|---|---|
| `ollama` | No | Local inference via `http://localhost:11434` |
| `openrouter` | Yes | Access to many hosted models |
| `openai` | Yes | GPT-4o and variants |
| `anthropic` | Yes | Claude models |
| `gemini` | Yes | Gemini models |

Provider and model are configurable per-job or as persistent settings. Transient errors (rate limits, timeouts, overload) are retried up to 3 times with exponential backoff.

## Running Tests

**Backend** (pytest):

```bash
pip install pytest httpx
pytest tests/
```

**Frontend E2E** (Playwright, requires dev server running):

```bash
cd web
npx playwright install chromium
npm run test:e2e
```

## Hosted Mode Warning

Hosted mode has no authentication. Treat it as private/self-hosted only. Do not expose the API publicly without an auth layer, tightened CORS origins, path allowlist, and secret management. See [Production checklist](docs/PRODUCTION_CHECKLIST.md).

## Legacy Streamlit

The old app can be run only if Streamlit is installed separately:

```bash
vidmeta legacy-streamlit
```

This is not the supported runtime.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [API contract](docs/API_CONTRACT.md)
- [Database schema](docs/schema.sql)
- [Environment variables](docs/ENVIRONMENT.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Production checklist](docs/PRODUCTION_CHECKLIST.md)
- [Roadmap](docs/ROADMAP.md)
- [Dependency security audit](docs/DEPENDENCY_SECURITY_AUDIT.md)
- [Windows trusted installation](docs/WINDOWS_TRUSTED_INSTALLATION.md)

## License

MIT. See [LICENSE](LICENSE).
