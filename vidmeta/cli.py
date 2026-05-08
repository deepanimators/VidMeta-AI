import subprocess
import sys
import os
from pathlib import Path


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"Invalid {name}={raw!r}; using {default}")
        return default


def main():
    args = sys.argv[1:]

    if not args or args[0] in ("--help", "-h"):
        print("""
VidMeta AI — Video Metadata Generator

Usage:
  vidmeta run [app.py] [streamlit flags...]     Launch the UI
  vidmeta --help                                Show this message

Examples:
  vidmeta run
  vidmeta run app.py
  VIDMETA_MAX_UPLOAD_MB=4096 vidmeta run
        """)
        return

    if args[0] == "run":
        has_target = len(args) > 1 and not args[1].startswith("-")
        target = args[1] if has_target else "app.py"
        passthrough_args = args[2:] if has_target else args[1:]
        target_path = Path(target).resolve()
        max_upload_mb = _env_int("VIDMETA_MAX_UPLOAD_MB", 2048)
        max_message_mb = _env_int("VIDMETA_MAX_MESSAGE_MB", max_upload_mb)

        if not target_path.exists():
            print(f"❌ File not found: {target_path}")
            sys.exit(1)

        print(f"🎬 Starting VidMeta AI → {target_path.name}")
        print(f"Upload limit: {max_upload_mb} MB")
        try:
            subprocess.run([
                "streamlit", "run", str(target_path),
                "--browser.gatherUsageStats", "false",
                "--server.maxUploadSize", str(max_upload_mb),
                "--server.maxMessageSize", str(max_message_mb),
            ] + passthrough_args)
        except FileNotFoundError:
            print("Streamlit is not installed. Run: pip install -r requirements.txt")
            sys.exit(1)
    else:
        print(f"❌ Unknown command: {args[0]}. Run `vidmeta --help`")
        sys.exit(1)


if __name__ == "__main__":
    main()
