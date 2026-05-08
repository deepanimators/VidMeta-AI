# Deep Folder-by-Folder Audit

## Current repository type

VidMeta AI is currently a Python Streamlit desktop/local web app, not a React/Vite frontend plus Node/Express backend.

## Current structure

```text
.
├── app.py                    # Main Streamlit UI, video processing, prompts, provider calls
├── vidmeta/
│   ├── __init__.py
│   └── cli.py                # `vidmeta run` launcher
├── .streamlit/config.toml    # Streamlit runtime config, including upload size
├── config.toml               # Legacy/root copy for reference
├── requirements.txt          # Runtime dependencies
├── pyproject.toml            # Package metadata and console entry point
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
├── docs/
└── README.md
```

## What is strong

- Local-first workflow: users can run the app on their own machine and process local video paths.
- Simple setup: one Streamlit app and one CLI entry point.
- Multi-provider LLM support: Ollama, OpenRouter, OpenAI, Anthropic, and Gemini.
- Batch mode already exists for local folders.
- Large upload support is now configured through `.streamlit/config.toml` and `vidmeta run`.

## Main risks

- `app.py` owns UI, video processing, prompts, provider clients, JSON parsing, and exports. This is fine for an MVP but hard to test.
- No automated tests yet.
- No typed request/response schema for generated metadata.
- Provider model names and prompt contracts are not centrally versioned.
- API keys can be typed into the sidebar and stored in browser cookies. This is acceptable for local use, but hosted deployments need stronger secret handling.

## Recommended next structure

```text
vidmeta/
├── cli.py
├── settings.py
├── video/
│   ├── frames.py
│   └── transcription.py
├── ai/
│   ├── prompts.py
│   ├── providers.py
│   └── schemas.py
├── exports/
│   ├── csv_export.py
│   └── text_export.py
└── ui/
    └── streamlit_app.py
tests/
```

Keep this as a refactor target, not a prerequisite for releasing the MIT local app.
