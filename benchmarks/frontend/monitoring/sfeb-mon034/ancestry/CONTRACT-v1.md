# MON-034 — PostgreSQL Timeline Ancestry Contract v1

**Profile:** `mon034-postgres-timeline-ancestry-input-v1`  
**Status:** offline parser/evaluator and adversarial tests implemented; runtime archive/restore adapter pending.

## Evidence basis

PostgreSQL's source parser defines each timeline-history record as parent TLI, WAL switchpoint and reason. It constructs parent timeline intervals from the previous switchpoint to the current one, then adds the requested target timeline beginning at the final switchpoint. Parent TLI identifiers increase and precede the target TLI. Timeline 1 has no history file.

References:
- https://doxygen.postgresql.org/backend_2access_2transam_2timeline_8c_source.html
- https://www.postgresql.org/docs/current/continuous-archiving.html

## Invariant

A recovery claim names a base-backup timeline and LSN, plus a recovered target timeline and replay LSN. The proof establishes that the base TLI is an ancestor of the target TLI and that each LSN lies in the valid interval of its own timeline.

The verifier never compares numeric LSNs across different timeline IDs. Cross-TLI recovery is established by the parent chain and forkpoints in the target timeline history file.

## Input and parsing rules

- System identifiers are canonical, non-zero decimal uint64 values.
- TLI 1 has no history file: both `history_text` and `history_sha256` are empty.
- TLI greater than 1 requires the exact history text and its SHA-256 digest. The history TLI must equal the recovered TLI.
- Each non-comment history row has a decimal parent TLI, a HIGH/LOW hexadecimal switchpoint, and a non-empty reason. Blank lines and lines whose first non-space character is `#` are ignored.
- Parent TLI IDs strictly increase, begin at timeline 1 and precede the target TLI.
- Each LSN half fits within 32 bits.
- Each parent TLI interval is `[previous switchpoint, current switchpoint)`; equal consecutive forkpoints preserve an empty intermediate interval rather than silently dropping a timeline.
- The target TLI begins at the final switchpoint.

## Verdict semantics

- `qualified`: identities match; source TLI is an ancestor; base LSN is inside its source interval; and target LSN is strictly after the target TLI forkpoint (or is at/after the base LSN when both are on the same TLI).
- `blocked`: invalid or mismatched identity; unrelated branch; malformed history or digest; base LSN before its source interval or at/after its end fork; target before its target timeline start; or same-TLI target before the base backup.
- `indeterminate`: target history is unavailable, or a required LSN is exactly at a timeline-start boundary where ownership needs explicit runtime evidence.

Blocked takes precedence over indeterminate if independent checks produce both.

## Limits

This module validates an offline, digest-bound ancestry assertion. It does not fetch timeline history files, validate WAL bytes, inspect `pg_control`, perform PITR, or establish that a recovered server can serve workloads. Runtime adapters must capture system identifiers, checkpoint/replay LSNs, timeline IDs and required history files from the actual backup/archive/restore environment.

The raw observations and their custody proofs should enter MON-026/027 before MON-024 composes a production recovery claim. A qualified result from this parser alone is not a completed recovery test.
