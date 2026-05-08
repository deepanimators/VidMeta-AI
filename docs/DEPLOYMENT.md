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

## Docker

```bash
docker compose up --build
```

API: `http://localhost:8000`.

## Hosted Warning

Hosted mode currently has no auth. Use only for private/self-hosted deployments until auth, rate limiting, upload scanning, and secret isolation are added.
