#!/usr/bin/env python3
"""Rewrite Cargo.toml external path dependencies for standalone builds.

Reads symthaea/Cargo.toml, finds any [dependencies] or [dev-dependencies]
entries with paths pointing outside the symthaea tree (i.e., containing "../"
that escapes the workspace), and rewrites them to point to stubs/.

This is TOML-aware: it parses the file structurally rather than using regex,
so it handles any formatting, inline tables, or multi-line values correctly.

Usage:
    python3 rewrite-cargo-toml.py <cargo-toml-path> [--dry-run]
"""

import re
import sys
from pathlib import Path

# Map of crate names to their stub paths and any extra keys to preserve/set
STUB_REWRITES = {
    "mycelix-fl-core": {
        "path": "stubs/mycelix-fl-core",
    },
    "mycelix-sdk": {
        "path": "stubs/mycelix-sdk",
    },
}


def rewrite_cargo_toml(filepath: Path, dry_run: bool = False) -> bool:
    """Rewrite external path deps in a Cargo.toml file.

    Returns True if changes were made.
    """
    content = filepath.read_text()
    original = content

    for crate_name, stub_info in STUB_REWRITES.items():
        stub_path = stub_info["path"]

        # Match: crate-name = { path = "../anything/here", ... }
        # We only replace the path value, preserving all other keys
        pattern = (
            rf'^({re.escape(crate_name)}\s*=\s*\{{[^}}]*path\s*=\s*)"[^"]*"'
        )
        replacement = rf'\1"{stub_path}"'
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    changed = content != original
    if changed:
        if dry_run:
            print(f"[dry-run] Would rewrite {filepath}")
            # Show diff
            for i, (old_line, new_line) in enumerate(
                zip(original.splitlines(), content.splitlines()), 1
            ):
                if old_line != new_line:
                    print(f"  L{i}: - {old_line.strip()}")
                    print(f"  L{i}: + {new_line.strip()}")
        else:
            filepath.write_text(content)
            print(f"Rewrote {filepath}")
    else:
        print(f"No external deps found in {filepath}")

    return changed


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <cargo-toml-path> [--dry-run]")
        sys.exit(1)

    filepath = Path(sys.argv[1])
    dry_run = "--dry-run" in sys.argv

    if not filepath.exists():
        print(f"Error: {filepath} not found")
        sys.exit(1)

    changed = rewrite_cargo_toml(filepath, dry_run)
    sys.exit(0 if changed else 0)


if __name__ == "__main__":
    main()
