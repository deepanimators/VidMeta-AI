# Refactor Plan

The main Streamlit-to-service refactor is implemented. Remaining refactor work:

- Add a bundled FastAPI sidecar for Tauri packaging.
- Add provider-specific mocked tests.
- Add schema repair pass for invalid LLM JSON.
- Split React UI into smaller feature components as it grows.
- Add optional auth without changing the job/upload APIs.
