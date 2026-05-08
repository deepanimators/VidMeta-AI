# Deployment Guide

## Local install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install .
vidmeta run
```

Open `http://localhost:8501`.

## Large local files

For files larger than the browser upload limit, use the app's `Local file path` mode or `Batch - Folder` mode. This avoids sending the entire file through the browser upload channel.

To raise browser upload limits:

```bash
VIDMETA_MAX_UPLOAD_MB=4096 VIDMETA_MAX_MESSAGE_MB=4096 vidmeta run
```

## Docker

```bash
docker compose up --build
```

Open `http://localhost:8501`.

Place videos in `./videos` and use paths like `/videos/example.mp4` inside the container.

## Ollama with Docker

If Ollama is running on the host, use:

```text
http://host.docker.internal:11434
```

in the app sidebar.

## Hosted warning

This app is designed for local use. Hosted deployments need stronger secret management, authentication, upload scanning, rate limiting, and storage controls.
