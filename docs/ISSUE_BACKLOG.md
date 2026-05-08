# Issue Backlog

## High priority

- Add tests for `vidmeta run` upload-limit flags.
- Extract LLM provider calls from `app.py` into `vidmeta/ai/providers.py`.
- Add schema validation for LLM metadata JSON.
- Stop storing hosted-provider API keys in cookies for non-local deployments.
- Add better JSON repair/error recovery for LLM output.

## Medium priority

- Add optional SQLite run history.
- Add progress details for long transcriptions.
- Add provider-specific model configuration in one place.
- Add sample fixture video that is intentionally licensed for tests.
- Add import/export of saved brand presets.

## Low priority

- Improve UI polish and copy consistency.
- Add screenshots to README.
- Add platform-specific export templates.
- Add pre-commit formatting once the module split is done.
