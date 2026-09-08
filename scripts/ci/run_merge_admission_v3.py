#!/usr/bin/env python3
"""Import-safe CLI launcher for the pure merge-admission v3 policy core."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ci.evaluate_merge_admission_v3 import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
