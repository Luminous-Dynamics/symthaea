#!/usr/bin/env python3
"""
Entry point for `python -m mycelix_cli`

This allows the CLI to be run as a module:
    python -m mycelix_cli --help
    python -m mycelix_cli demo --scenario healthcare
"""

from .cli import app

if __name__ == "__main__":
    app()
