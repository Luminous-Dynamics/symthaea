#!/usr/bin/env python3
"""Cross-check the concrete PostgreSQL driver against crate-level safety limits."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = (ROOT / "src" / "postgres_sync.rs").read_text(encoding="utf-8")
SQL = (ROOT / "src" / "postgresql.rs").read_text(encoding="utf-8")
BACKEND = (ROOT / "src" / "production_backend.rs").read_text(encoding="utf-8")

errors: list[str] = []

def integer_constant(text: str, name: str) -> int:
    match = re.search(rf"pub const {re.escape(name)}: usize = ([0-9_]+)", text)
    if match is None:
        errors.append(f"missing usize constant {name}")
        return -1
    return int(match.group(1).replace("_", ""))

rotation_limit = integer_constant(BACKEND, "MAX_PRODUCTION_ROTATION_BUNDLES")
match = re.search(
    r'pub const POSTGRES_LOAD_ROTATIONS_SQL: &str =\s*"([^"]+)";', SQL
)
if match is None:
    errors.append("POSTGRES_LOAD_ROTATIONS_SQL is missing or no longer a normal string literal")
elif rotation_limit >= 0:
    expected = rotation_limit + 1
    limit_match = re.search(r"\bLIMIT\s+([0-9]+)\b", match.group(1), re.IGNORECASE)
    if limit_match is None or int(limit_match.group(1)) != expected:
        errors.append(
            f"rotation query must use LIMIT {expected} to detect one row beyond the {rotation_limit}-bundle cap"
        )

for fragment in ("current_database()", "current_user", "server_version_num"):
    if fragment not in DRIVER:
        errors.append(f"PostgreSQL server identity query is missing {fragment!r}")

required_driver_fragments = (
    "verify_server_identity(",
    ".map_err(SyncPostgresError::Commit)",
    "commit_outcome_may_be_unknown",
    "MAX_PRODUCTION_SNAPSHOT_BYTES",
    "MAX_PRODUCTION_ROTATION_BUNDLES",
)
for fragment in required_driver_fragments:
    if fragment not in DRIVER:
        errors.append(f"concrete PostgreSQL driver is missing contract fragment {fragment!r}")

if "Self::Database(error) | Self::Commit(error) => Some(error)" not in DRIVER:
    errors.append("PostgreSQL errors do not preserve both statement and commit sources")

if errors:
    for error in errors:
        print(f"postgres-driver error: {error}", file=sys.stderr)
    raise SystemExit(1)

print(
    "validated PostgreSQL driver contract: "
    f"rotation cap {rotation_limit}, identity revalidation, commit reconciliation"
)
