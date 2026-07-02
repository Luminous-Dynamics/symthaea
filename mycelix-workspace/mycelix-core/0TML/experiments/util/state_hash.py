# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Comprehensive state hashing for cache invalidation.

This module computes a deterministic hash of all relevant source files,
ensuring cache invalidation when experiments, datasets, or configs change.
"""

import hashlib
from pathlib import Path
from typing import List


# Comprehensive file patterns to include in state hash
GLOBS = [
    "src/gen5/**/*.py",                  # Core implementation
    "experiments/**/*.py",               # Experiment definitions
    "experiments/configs/**/*.yaml",      # Configuration files
    "experiments/datasets/**/*.py",       # Dataset loaders
    "pyproject.toml",                    # Dependencies
    "requirements.txt",                  # Legacy dependencies
    "nix/**/*",                          # Nix environment
]


def compute_state_hash(root: str = ".") -> str:
    """Compute deterministic hash of all relevant source files.

    Args:
        root: Root directory for file search (default: current directory)

    Returns:
        16-character hex string (SHA256 prefix)

    Example:
        >>> state_hash = compute_state_hash()
        >>> print(f"State: {state_hash}")
        State: 3f7a9b2e1c4d8f6a
    """
    h = hashlib.sha256()
    base = Path(root).resolve()

    # Collect all matching files
    files: List[Path] = []
    for pattern in GLOBS:
        files.extend(base.glob(pattern))

    # Sort for deterministic ordering
    files = sorted(set(files), key=lambda p: str(p.relative_to(base)))

    # Hash each file (path + content)
    for p in files:
        if p.is_file():
            # Hash relative path
            h.update(str(p.relative_to(base)).encode())
            h.update(b"\0")

            # Hash file content
            h.update(p.read_bytes())
            h.update(b"\0")

    # Return 16-char prefix for readability
    return h.hexdigest()[:16]


if __name__ == "__main__":
    # Test state hash computation
    state = compute_state_hash()
    print(f"Current state hash: {state}")

    # Show which files are included
    base = Path(".").resolve()
    files = []
    for pattern in GLOBS:
        matched = list(base.glob(pattern))
        files.extend(matched)
        print(f"\n{pattern}: {len(matched)} files")

    files = sorted(set(files), key=lambda p: str(p.relative_to(base)))
    print(f"\nTotal files hashed: {len(files)}")
