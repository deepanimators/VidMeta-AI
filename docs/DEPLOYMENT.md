# Deployment

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install .
vidmeta serve
```

In another terminal:

```bash
cd web
npm install
npm run dev
```

Open `http://localhost:5173`.

## Desktop Development

```bash
vidmeta serve
cd web && npm install
cd ../desktop && npm install && npm run dev
```

## Desktop Installer Builds

Run installer builds from `desktop/` on the target operating system:

```bash
npm run build:mac
npm run build:windows
npm run build:linux
```

Windows signing and trusted private installation steps are documented in `docs/WINDOWS_TRUSTED_INSTALLATION.md`.

## Docker

```bash
docker compose up --build
```

API: `http://localhost:8000`.

To confirm the backend container has Python installed:

```bash
docker compose exec vidmeta python --version
```

For a background run:

```bash
docker compose up --build -d
```

For local video jobs, put files in `./videos` and use paths like `/videos/example.mp4` from the app. The compose file mounts that folder into the container as a read-only volume.

To stop the stack:

```bash
docker compose down
```

## Hosted Warning

Hosted mode currently has no auth. Use only for private/self-hosted deployments until auth, rate limiting, upload scanning, and secret isolation are added.
