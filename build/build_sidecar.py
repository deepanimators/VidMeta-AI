"""Build the vidmeta-server sidecar binary using PyInstaller."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SPEC = ROOT / "vidmeta-server.spec"


def main() -> None:
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        str(SPEC),
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
