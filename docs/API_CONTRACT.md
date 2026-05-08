# API Contract

## Service

Base URL in local development: `http://127.0.0.1:8000`.

## Settings

```text
GET /api/settings
PUT /api/settings
```

Settings include app mode, upload limit, brand context, video settings, provider settings, and storage settings.

## Jobs

```text
POST /api/jobs/from-path
POST /api/jobs/from-upload/{upload_id}
GET  /api/jobs
GET  /api/jobs/{job_id}
GET  /api/jobs/{job_id}/result
GET  /api/jobs/{job_id}/exports/{format}
```

`from-path` accepts a file or folder path. Folder paths run as batch jobs.

Job creation accepts `target_platforms`, an array of platform keys. The AI prompt and saved result are limited to those selected platforms. If omitted, the service defaults to all supported platform keys.

Example:

```json
{
  "path": "/Users/you/Videos/product-demo.mp4",
  "mode": "single",
  "target_platforms": ["youtube", "instagram_reels", "tiktok"]
}
```

Job statuses:

- `queued`
- `running`
- `completed`
- `failed`
- `cancelled`

Job stages:

- `queued`
- `starting`
- `frames`
- `audio`
- `analysis`
- `metadata`
- `export`
- `completed`
- `failed`

## Uploads

Small upload fallback:

```text
POST /api/uploads
```

Resumable upload:

```text
OPTIONS /api/uploads/resumable
POST    /api/uploads/resumable
HEAD    /api/uploads/resumable/{upload_id}
PATCH   /api/uploads/resumable/{upload_id}
GET     /api/uploads/{upload_id}
```

The resumable flow follows the core tus model: create an upload, append chunks with `PATCH` and `Upload-Offset`, and resume by reading the current offset with `HEAD`.

## Exports

Supported formats:

- `json`
- `csv`
- `txt`
