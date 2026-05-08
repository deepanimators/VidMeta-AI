# Dependency Vulnerability Audit

## Current dependency surface

Runtime dependencies are listed in `requirements.txt` and `pyproject.toml`:

- `streamlit`
- `streamlit-cookies-manager`
- `opencv-python-headless`
- `Pillow`
- `faster-whisper`
- `openai`
- `anthropic`
- `google-generativeai`
- `requests`
- `filelock` dependency floor

## Audit commands

Run locally:

```bash
python -m pip install pip-audit
pip-audit -r requirements.txt
```

The GitHub Actions workflow also runs `pip-audit -r requirements.txt`.

## Security notes

- Keep `Pillow`, `requests`, and Streamlit current because they handle untrusted file/network content.
- Prefer `opencv-python-headless` over full OpenCV for a smaller local/server surface.
- Avoid committing local sample videos unless they are intentionally licensed test fixtures.
- For a public MIT release, pinning exact versions through a lock file is recommended after testing.

## Current status

`pip-audit -r requirements.txt` initially found 10 known vulnerabilities in 4 resolved packages:

| Package | Resolved version | Advisory count | Fixed floor applied |
| --- | ---: | ---: | --- |
| `streamlit` | `1.50.0` | 1 | `streamlit>=1.54.0` |
| `Pillow` | `11.3.0` | 6 | `Pillow>=12.2.0` |
| `requests` | `2.32.5` | 1 | `requests>=2.33.0` |
| `filelock` | `3.19.1` | 2 | `filelock>=3.20.3` |

The dependency floors in `requirements.txt` and `pyproject.toml` were raised accordingly. Re-run `pip-audit -r requirements.txt` after dependency installation in your environment.

## Verification note

The local workstation used for this audit only had `python3` 3.9.6 available. After raising the dependency floors, a second local `pip-audit` resolution could not complete because the fixed Streamlit versions require Python 3.10+, which matches this project's declared `requires-python = ">=3.10"`. The CI workflow runs the audit on Python 3.11, so it is the authoritative follow-up check for the updated dependency set.
