"""PyInstaller entry point for the vidmeta-server sidecar binary."""
import multiprocessing
import sys
import os

# Required for PyInstaller multiprocessing support on Windows/macOS
multiprocessing.freeze_support()

# Ensure the bundled packages are importable
if getattr(sys, "frozen", False):
    sys.path.insert(0, sys._MEIPASS)

from vidmeta.cli import main

if __name__ == "__main__":
    main()
