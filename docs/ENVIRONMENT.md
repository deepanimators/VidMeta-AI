# Environment Variable Extraction

## Current variables

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `VIDMETA_COOKIE_PASSWORD` | No | `vidmeta-local-dev-cookie-password` | Encrypts Streamlit cookie settings. Set a unique value for any shared/hosted deployment. |
| `COOKIES_PASSWORD` | No | none | Backward-compatible fallback for cookie encryption. |
| `VIDMETA_MAX_UPLOAD_MB` | No | `2048` | Browser upload limit used by `vidmeta run` and displayed in the app. |
| `VIDMETA_MAX_MESSAGE_MB` | No | same as upload limit | Streamlit websocket message limit used by `vidmeta run`. |

## Optional provider keys

The app currently collects provider keys in the sidebar. For local use this is convenient. For hosted use, prefer environment variables and avoid cookie persistence for secrets.

Suggested future variables:

| Variable | Provider |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GEMINI_API_KEY` | Google Gemini |
| `OPENROUTER_API_KEY` | OpenRouter |
| `OLLAMA_BASE_URL` | Ollama |

## Files

- Use `.env.example` as the publishable template.
- Do not commit a real `.env`.
- `.gitignore` already excludes `.env`.
