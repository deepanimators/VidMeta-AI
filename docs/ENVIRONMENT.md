# Environment Variables

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `VIDMETA_API_HOST` | No | `127.0.0.1` | FastAPI bind host used by `vidmeta serve`. |
| `VIDMETA_API_PORT` | No | `8000` | FastAPI port used by `vidmeta serve`. |
| `VIDMETA_DATA_DIR` | No | `~/.vidmeta-ai` | Local app data directory. |
| `VIDMETA_DATABASE` | No | `$VIDMETA_DATA_DIR/vidmeta.db` | SQLite database path. |
| `VIDMETA_UPLOAD_DIR` | No | `$VIDMETA_DATA_DIR/uploads` | Local upload directory. |
| `VIDMETA_PROCESSING_DIR` | No | `$VIDMETA_DATA_DIR/processing` | Temporary processing directory. |
| `VIDMETA_MAX_UPLOAD_MB` | No | `2048` | UI/configured upload policy value. |

Provider and S3 settings are stored through `/api/settings`.
