from __future__ import annotations

import os
import subprocess
import sys


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"Invalid {name}={raw!r}; using {default}")
        return default


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in {"--help", "-h"}:
        print(
            """
VidMeta AI - Local video metadata service

Usage:
  vidmeta serve [uvicorn flags...]      Start FastAPI service
  vidmeta legacy-streamlit [file]       Run old Streamlit app if installed
  vidmeta --help                        Show this message

Examples:
  vidmeta serve
  vidmeta serve --host 0.0.0.0 --port 8000
            """.strip()
        )
        return

    command = args[0]
    if command == "serve":
        port = _env_int("VIDMETA_API_PORT", 8000)
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "api.main:app",
            "--host",
            os.environ.get("VIDMETA_API_HOST", "127.0.0.1"),
            "--port",
            str(port),
        ] + args[1:]
        try:
            subprocess.run(cmd, check=False)
        except KeyboardInterrupt:
            return
        except FileNotFoundError:
            print("Uvicorn is not installed. Run: pip install -r requirements.txt")
            sys.exit(1)
        return

    if command == "legacy-streamlit":
        target = args[1] if len(args) > 1 else "legacy/app_streamlit.py"
        max_upload_mb = _env_int("VIDMETA_MAX_UPLOAD_MB", 2048)
        try:
            subprocess.run(
                [
                    "streamlit",
                    "run",
                    target,
                    "--browser.gatherUsageStats",
                    "false",
                    "--server.maxUploadSize",
                    str(max_upload_mb),
                    "--server.maxMessageSize",
                    str(max_upload_mb),
                ]
                + args[2:],
                check=False,
            )
        except KeyboardInterrupt:
            return
        except FileNotFoundError:
            print("Streamlit is not installed. Install it separately to run the legacy app.")
            sys.exit(1)
        return

    print(f"Unknown command: {command}. Run `vidmeta --help`")
    sys.exit(1)


if __name__ == "__main__":
    main()
