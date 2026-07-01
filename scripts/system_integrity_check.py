#!/usr/bin/env python3
"""Run lightweight integrity checks for the Broca logic bridge scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_FILES = (
    "examples/harvest_distillation.rs",
    "docs/planning/BROCA_ENHANCEMENT_PLAN.md",
    "scripts/generate_rust_distillation.py",
    "scripts/verify_gating_logic.py",
)

SUPPORTED_SUBSTRATES = {"rust", "python"}
REQUIRED_PATTERNS = {
    "fibonacci_iterative": "fibonacci",
    "binary_search": "binary_search",
    "even_filter": "evens",
}


def check_required_files(root: Path) -> list[str]:
    missing = [path for path in REQUIRED_FILES if not (root / path).exists()]
    return [f"missing required file: {path}" for path in missing]


def check_generator_patterns(root: Path) -> list[str]:
    generator = root / "scripts/generate_rust_distillation.py"
    text = generator.read_text(encoding="utf-8")
    errors: list[str] = []
    for name, needle in REQUIRED_PATTERNS.items():
        if needle not in text:
            errors.append(f"generator missing pattern {name!r} ({needle!r})")
    for substrate in SUPPORTED_SUBSTRATES:
        if f'"{substrate}"' not in text:
            errors.append(f"generator missing substrate {substrate!r}")
    return errors


def run(root: Path) -> int:
    errors = check_required_files(root)
    if not errors:
        errors.extend(check_generator_patterns(root))

    if errors:
        print("Symthaea logic bridge integrity: FAIL")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("Symthaea logic bridge integrity: PASS")
    print(f"  root: {root}")
    print(f"  substrates: {', '.join(sorted(SUPPORTED_SUBSTRATES))}")
    print(f"  patterns: {', '.join(sorted(REQUIRED_PATTERNS))}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Symthaea repository root.",
    )
    args = parser.parse_args()
    return run(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
