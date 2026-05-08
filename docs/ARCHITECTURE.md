# Enterprise Architecture

## Correct current architecture

```mermaid
flowchart LR
    U["Local user"] --> UI["Streamlit app"]
    UI --> V["Video processing"]
    V --> F["Frame extraction - OpenCV"]
    V --> T["Audio transcription - ffmpeg + Whisper"]
    UI --> P["Prompt templates"]
    P --> L["LLM provider"]
    L --> M["Metadata JSON"]
    M --> E["Exports - JSON CSV TXT"]
```

## Local-first principles

- Videos remain on the user's machine when using local file path or batch folder mode.
- Browser upload mode copies the selected file to a temporary local file and deletes it after processing.
- Ollama can keep inference local. Hosted LLM providers receive extracted frames and prompts.

## Target modular architecture

```mermaid
flowchart LR
    UI["Streamlit UI"] --> ORCH["Analysis orchestrator"]
    ORCH --> VIDEO["Video module"]
    ORCH --> AI["AI module"]
    ORCH --> EXPORT["Export module"]
    VIDEO --> FRAMES["Frame extraction"]
    VIDEO --> AUDIO["Transcription"]
    AI --> PROMPTS["Prompt registry"]
    AI --> PROVIDERS["Provider adapters"]
    AI --> SCHEMA["Metadata schema"]
```

## Hosted/SaaS architecture option

Only use this if the project changes from local MIT app to hosted SaaS:

```mermaid
flowchart LR
    WEB["Web UI"] --> API["API service"]
    API --> Q["Job queue"]
    Q --> W["Worker"]
    W --> AI["AI providers"]
    W --> DB["Postgres"]
    W --> OBJ["Object storage"]
    API --> AUTH["Auth and billing"]
```

For the stated GitHub MIT local goal, the hosted architecture is a future blueprint, not required now.
