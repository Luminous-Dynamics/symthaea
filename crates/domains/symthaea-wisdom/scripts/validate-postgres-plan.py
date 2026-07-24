#!/usr/bin/env python3
"""Static invariants for the fixed PostgreSQL statement plan."""

from __future__ import annotations

import ast
import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src" / "postgresql.rs"
TEXT = SOURCE.read_text(encoding="utf-8")
CONSTANT = re.compile(
    r'pub const (?P<name>POSTGRES_[A-Z0-9_]+): &str = r#"(?P<body>.*?)"#;',
    re.DOTALL,
)
SQL = {match.group("name"): match.group("body") for match in CONSTANT.finditer(TEXT)}
NORMAL_CONSTANT = re.compile(
    r'pub const (?P<name>POSTGRES_[A-Z0-9_]+): &str =\s*(?P<body>"(?:\\.|[^"\\])*");',
    re.DOTALL,
)
for match in NORMAL_CONSTANT.finditer(TEXT):
    SQL.setdefault(match.group("name"), ast.literal_eval(match.group("body")))

EXPECTED_PLACEHOLDERS = {
    "POSTGRES_LOAD_SCHEMA_VERSION_SQL": set(),
    "POSTGRES_INITIALIZE_SQL": set(range(1, 8)),
    "POSTGRES_STARTUP_CORE_SQL": set(),
    "POSTGRES_ACQUIRE_FENCE_SQL": {1},
    "POSTGRES_LOAD_LEDGER_SQL": set(),
    "POSTGRES_COMPARE_EXCHANGE_LEDGER_SQL": set(range(1, 12)),
    "POSTGRES_LOAD_ROTATIONS_SQL": set(),
    "POSTGRES_APPEND_ROTATION_SQL": set(range(1, 5)),
    "POSTGRES_CLAIM_STARTUP_ATTEMPT_SQL": {1, 2, 3},
    "POSTGRES_COMPLETE_STARTUP_ATTEMPT_SQL": {1, 2, 3, 4},
    "POSTGRES_FAIL_STARTUP_ATTEMPT_SQL": {1, 2, 3, 4},
    "POSTGRES_LOAD_STARTUP_ATTEMPT_SQL": {1},
}

for name, expected in EXPECTED_PLACEHOLDERS.items():
    body = SQL.get(name)
    if body is None:
        raise SystemExit(f"missing SQL constant {name}")
    actual = {int(value) for value in re.findall(r"\$(\d+)", body)}
    if actual != expected:
        raise SystemExit(f"{name} placeholders {sorted(actual)} != {sorted(expected)}")

for name, body in SQL.items():
    upper = body.upper()
    if " BIGINT" in upper or " DOUBLE PRECISION" in upper or " REAL" in upper:
        raise SystemExit(f"{name} contains a forbidden lossy/signed numeric type")
    if "::TEXT AS" in body and "NUMERIC(20,0)" in body:
        # This combination is valid across separate clauses, but not as a
        # malformed column declaration produced by a broad text replacement.
        for line in body.splitlines():
            if "::TEXT AS" in line and "NUMERIC(20,0)" in line:
                raise SystemExit(f"{name} has malformed cast/declaration line: {line!r}")

required_fragments = {
    "POSTGRES_MIGRATION_BOOTSTRAP_SQL": "symthaea_wisdom_schema_migration",
    "POSTGRES_STARTUP_CORE_SQL": "transaction_isolation",
    "POSTGRES_COMPARE_EXCHANGE_LEDGER_SQL": "FOR UPDATE",
    "POSTGRES_APPEND_ROTATION_SQL": "snapshot_generation = snapshot_generation + 1",
}
for name, fragment in required_fragments.items():
    if fragment not in SQL.get(name, ""):
        raise SystemExit(f"{name} is missing required fragment {fragment!r}")

print(f"validated {len(SQL)} PostgreSQL SQL constants")
