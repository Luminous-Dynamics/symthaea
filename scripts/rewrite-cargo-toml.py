#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Rewrite Cargo.toml external path dependencies for standalone builds.

Reads symthaea/Cargo.toml, finds any [dependencies] or [dev-dependencies]
entries with paths pointing outside the symthaea tree (i.e., containing "../"
that escapes the workspace), and rewrites them to point to stubs/.

This uses regex-based path rewriting (not full TOML parsing) to preserve
formatting and comments exactly as-is. The patterns are tested against the
actual Cargo.toml format used in the project.

Usage:
    python3 rewrite-cargo-toml.py <cargo-toml-path> [--dry-run]
    python3 rewrite-cargo-toml.py --scan <directory>   # scan for escaping deps
"""

import re
import sys
from pathlib import Path

# Map of crate names to their stub paths
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
        # The [^}]* matches any characters between { and path that aren't },
        # then we replace only the quoted path value.
        # This preserves all other keys (optional, default-features, features).
        pattern = (
            rf'^({re.escape(crate_name)}\s*=\s*\{{[^}}]*path\s*=\s*)"[^"]*"'
        )
        replacement = rf'\1"{stub_path}"'
        new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

        if new_content != content:
            print(f"  Rewrote {crate_name} -> {stub_path}")
        content = new_content

    changed = content != original
    if changed:
        if dry_run:
            print(f"[dry-run] Would rewrite {filepath}")
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
        print(f"No external deps to rewrite in {filepath}")

    return changed


def scan_for_escaping_deps(directory: Path) -> int:
    """Scan all Cargo.toml files for path deps that escape the workspace.

    Returns the number of problematic files found.
    """
    issues = 0
    for toml_path in sorted(directory.rglob("Cargo.toml")):
        # Skip target directories and stubs
        rel = toml_path.relative_to(directory)
        parts = rel.parts
        if "target" in parts or "stubs" in parts:
            continue

        content = toml_path.read_text()
        for i, line in enumerate(content.splitlines(), 1):
            # Skip comments
            if line.strip().startswith("#"):
                continue
            # Look for path deps
            match = re.search(r'path\s*=\s*"([^"]*)"', line)
            if not match:
                continue
            dep_path = match.group(1)
            # Resolve the path relative to the Cargo.toml's directory
            resolved = (toml_path.parent / dep_path).resolve()
            # Check if it escapes the workspace root
            try:
                resolved.relative_to(directory.resolve())
            except ValueError:
                print(f"  ESCAPING: {rel}:{i}: {line.strip()}")
                print(f"           resolves to: {resolved}")
                issues += 1

    if issues == 0:
        print("All path dependencies resolve within the workspace.")
    else:
        print(f"\n{issues} escaping path dep(s) found.")

    return issues


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <cargo-toml-path> [--dry-run]")
        print(f"       {sys.argv[0]} --scan <directory>")
        sys.exit(1)

    if sys.argv[1] == "--scan":
        if len(sys.argv) < 3:
            print("Error: --scan requires a directory argument")
            sys.exit(1)
        directory = Path(sys.argv[2])
        if not directory.is_dir():
            print(f"Error: {directory} is not a directory")
            sys.exit(1)
        issues = scan_for_escaping_deps(directory)
        sys.exit(1 if issues > 0 else 0)

    filepath = Path(sys.argv[1])
    dry_run = "--dry-run" in sys.argv

    if not filepath.exists():
        print(f"Error: {filepath} not found")
        sys.exit(1)

    rewrite_cargo_toml(filepath, dry_run)


if __name__ == "__main__":
    main()
