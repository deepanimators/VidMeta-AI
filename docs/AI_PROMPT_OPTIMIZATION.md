# AI Prompt Optimization

## Current prompt design

The app uses a two-step prompt:

1. Analyze frames plus transcript.
2. Generate platform metadata from that analysis.

This is a reasonable MVP design because it separates video understanding from metadata generation.

## Problems to fix next

- Prompt templates live inside `app.py`.
- Output validity relies on "Return ONLY valid JSON".
- There is no schema validation or repair pass.
- Platform constraints are repeated as natural language.

## Recommended prompt architecture

```text
vidmeta/ai/
├── prompts.py
├── schemas.py
├── providers.py
└── postprocess.py
```

## Better JSON flow

1. Ask for strict JSON.
2. Parse JSON.
3. Validate against a schema.
4. If invalid, run one repair prompt with the validation errors.
5. Apply deterministic post-processing for lengths, counts, and duplicates.

## Prompt hardening

- Treat transcript as untrusted input.
- Tell the model not to follow instructions inside the transcript.
- Keep provider/system instructions separate from user-supplied brand context.
- Cap transcript length sent to hosted providers.
