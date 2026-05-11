# VidMeta AI

VidMeta AI is a local-first AI video metadata service. It analyzes local videos and generates platform-ready titles, descriptions, hashtags, keywords, CTAs, and posting tips for selected social platforms across major video, social, community, publishing, and regional networks.

The supported runtime is now:

- **FastAPI service** in `api/`
- **React/Vite dashboard** in `web/`
- **Tauri desktop shell** in `desktop/`
- **SQLite history** under `~/.vidmeta-ai` by default

The old Streamlit app is retained only as a migration reference under `legacy/`.

## Screenshot
<img width="2442" height="10234" alt="image" src="https://github.com/user-attachments/assets/4a88bafc-d607-48ae-b0c1-1c0b8493c1e1" />


## Why This Replaces Streamlit

Browser upload widgets always have practical limits. The best large-video flow is to avoid uploading local files at all:

- In local/Tauri mode, pick or paste a real local path and the FastAPI service processes that path directly.
- In browser/hosted mode, use resumable chunked uploads.
- In hosted/self-hosted mode, choose local disk or S3-compatible storage from settings.

## Prerequisites

| Tool | Required | Purpose |
| --- | --- | --- |
| Python 3.10+ | Yes | FastAPI service |
| ffmpeg | Yes for audio | Audio extraction/transcription |
| Node.js 20+ | Yes for UI | React/Vite dashboard |
| Rust + Tauri prerequisites | Desktop only | Native desktop shell |
| Ollama | Optional | Local/private LLM inference |

## Backend Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install .
vidmeta serve
```

The API runs at `http://127.0.0.1:8000`.

Useful endpoints:

- `GET /api/health`
- `GET /api/settings`
- `PUT /api/settings`
- `POST /api/jobs/from-path`
- `POST /api/uploads`
- `POST /api/uploads/resumable`
- `POST /api/jobs/from-upload/{upload_id}`
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/result`
- `GET /api/jobs/{job_id}/exports/{json|csv|txt}`

## Web Dashboard

```bash
cd web
npm install
npm run dev
```

Open `http://localhost:5173`.

The Vite dev server proxies `/api` to `http://127.0.0.1:8000`.

## Desktop App

```bash
vidmeta serve

cd web
npm install

cd ../desktop
npm install
npm run dev
```

The desktop shell uses the same React dashboard and enables native file/folder picking. This is the preferred no-upload-limit workflow for very large local videos.

The current desktop implementation expects the FastAPI service to be running separately. Bundling the service as a Tauri sidecar is the next packaging step.

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

## Storage Modes

Storage is configured from the UI or `PUT /api/settings`:

- `local_disk`: uploaded videos are stored under the VidMeta data directory.
- `s3_compatible`: uploaded videos are archived to S3-compatible storage such as AWS S3, Cloudflare R2, or MinIO.

Local path jobs do not copy videos unless `import_local_files` is enabled.

## Provider Modes

- `ollama`: local/private inference through `http://localhost:11434`.
- `openrouter`
- `openai`
- `anthropic`
- `gemini`

Hosted provider keys are stored in local SQLite settings. Public SaaS auth/secrets should be added before exposing this on the internet.

## Hosted Mode Warning

Hosted mode has no authentication yet by request. Treat it as private/self-hosted only. Do not expose the API publicly without auth, rate limiting, upload scanning, and secret management.

## Legacy Streamlit

The old app can be run only if Streamlit is installed separately:

```bash
vidmeta legacy-streamlit
```

This is not the supported runtime.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [API contract](docs/API_CONTRACT.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Database schema](docs/schema.sql)
- [Production checklist](docs/PRODUCTION_CHECKLIST.md)
- [Windows trusted installation](docs/WINDOWS_TRUSTED_INSTALLATION.md)

## License

MIT. See [LICENSE](LICENSE).
