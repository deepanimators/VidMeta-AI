#!/usr/bin/env python3
"""Thin CLI wrapper for the worker module.
Usage: python scripts/worker.py --input request.json --output result.json
"""
from vidmeta.worker import main


if __name__ == "__main__":
    raise SystemExit(main())
