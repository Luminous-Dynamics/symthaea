#!/usr/bin/env python3
"""Fail when security-relevant monotonic identities can wrap or saturate."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

errors: list[str] = []

prohibited_fragments = (
    ".wrapping_add(1)",
    "lease_epoch.saturating_add(1)",
    "snapshot_generation.saturating_add(1)",
    "next_sequence.saturating_add(1)",
    "next_nonce.saturating_add(1)",
    "next_prediction_id.saturating_add(1)",
    "previous.generation.saturating_add(1)",
    "expected.saturating_add(1)",
)

for path in sorted(SRC.rglob("*.rs")):
    text = path.read_text(encoding="utf-8")
    for fragment in prohibited_fragments:
        offset = text.find(fragment)
        if offset >= 0:
            line = text.count("\n", 0, offset) + 1
            errors.append(
                f"{path.relative_to(ROOT)}:{line}: prohibited monotonic counter expression "
                f"{fragment!r}"
            )

required_fragments = {
    "src/evidence.rs": [
        "pub fn try_append(",
        "EvidenceAppendError::SequenceExhausted",
        "EvidenceDecodeError::SequenceMismatch",
        "EvidenceDecodeError::CounterMismatch",
    ],
    "src/coordination.rs": ["InMemoryStoreError::LeaseEpochExhausted"],
    "src/storage.rs": ["InMemoryAtomicBackendError::LeaseEpochExhausted"],
    "src/production_backend.rs": ["CounterExhausted"],
    "src/provisioning.rs": ["ProvisioningSuccessorError::GenerationExhausted"],
    "src/production_network.rs": ["ProductionNetworkError::SequenceExhausted"],
    "src/ethics.rs": [
        "PermitIssueError::ExpiryOverflow",
        "PermitIssueError::NonceSpaceExhausted",
    ],
    "src/meta_cognition.rs": ["PredictionIssueError::TicketIdExhausted"],
    "src/replay.rs": ["OperationalReplayError::SequenceSpaceExhausted"],
    "src/archive_replay.rs": ["ArchiveOperationalRestoreError::SequenceSpaceExhausted"],
}

for relative, fragments in required_fragments.items():
    text = (ROOT / relative).read_text(encoding="utf-8")
    for fragment in fragments:
        if fragment not in text:
            errors.append(f"{relative}: missing monotonic-identity guard {fragment!r}")

if errors:
    print("monotonic identity validation failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)

print("validated fail-closed monotonic identities across wisdom runtime state")
