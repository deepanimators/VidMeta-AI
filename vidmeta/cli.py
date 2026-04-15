import subprocess
import sys
import os
from pathlib import Path


def main():
    args = sys.argv[1:]

    if not args or args[0] in ("--help", "-h"):
        print("""
VidMeta AI — Video Metadata Generator

Usage:
  vidmeta run [app.py]     Launch the UI (default: app.py)
  vidmeta --help           Show this message

Examples:
  vidmeta run
  vidmeta run app.py
        """)
        return

    if args[0] == "run":
        target = args[1] if len(args) > 1 else "app.py"
        target_path = Path(target).resolve()

        if not target_path.exists():
            print(f"❌ File not found: {target_path}")
            sys.exit(1)

        print(f"🎬 Starting VidMeta AI → {target_path.name}")
        subprocess.run([
            "streamlit", "run", str(target_path),
            "--browser.gatherUsageStats", "false",
        ] + args[2:])
    else:
        print(f"❌ Unknown command: {args[0]}. Run `vidmeta --help`")
        sys.exit(1)


if __name__ == "__main__":
    main()
