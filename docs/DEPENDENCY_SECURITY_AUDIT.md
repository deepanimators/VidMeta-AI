# Dependency Security Audit

Runtime dependencies are now the FastAPI service stack, video processing libraries, LLM provider SDKs, and optional S3 storage support.

## Main dependencies

- `fastapi`
- `uvicorn`
- `python-multipart`
- `opencv-python-headless`
- `Pillow`
- `faster-whisper`
- `openai`
- `anthropic`
- `google-generativeai`
- `requests`
- `boto3`
- `filelock`

## Audit command

```bash
pip-audit -r requirements.txt
```

CI runs this audit on Python 3.12.

## Notes

- Keep `Pillow`, `requests`, `FastAPI`, and `python-multipart` current because they handle untrusted content.
- Hosted deployments need auth, rate limiting, upload scanning, and secret isolation before public exposure.
- S3 credentials should be treated as private local/self-hosted settings until a real secrets system is added.
