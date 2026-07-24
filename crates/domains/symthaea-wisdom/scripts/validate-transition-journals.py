#!/usr/bin/env python3
"""Dependency-free guard for Series XXII transition journals and commit ambiguity."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
errors: list[str] = []

required = {
    "src/evidence.rs": [
        "pub(crate) struct EvidenceAppendJournal",
        "pub(crate) fn try_append_batch_with_journal(",
        "pub(crate) fn rollback_append(",
        "appended_fingerprints",
    ],
    "src/lib.rs": [
        "pub(crate) struct WisdomObservationJournal",
        "try_update_from_observation_with_evidence_journaled(",
        "rollback_observation_internal(",
        "snapshot_for_transition()",
    ],
    "src/runtime.rs": [
        "pub(crate) struct RuntimeTransitionJournal",
        "pub(crate) fn apply_journaled(",
        "pub(crate) fn rollback_transition(",
        "RollbackInvariantViolated",
    ],
    "src/execution.rs": [
        "pub(crate) struct ActionPreparationJournal",
        "pub(crate) struct ActionCompletionJournal",
        "pub(crate) fn prepare_execution_journaled(",
        "pub(crate) fn rollback_preparation(",
        "pub(crate) fn complete_execution_journaled(",
        "pub(crate) fn rollback_completion(",
    ],
    "src/coordination.rs": [
        "pub const fn commit_outcome_may_be_unknown(&self) -> bool",
        "matches!(self, Self::StoreFailure(_))",
    ],
    "src/service.rs": [
        "PersistenceOutcomeUnknown",
        "CompletionPersistenceOutcomeUnknown",
        "error.commit_outcome_may_be_unknown()",
        "rollback_observation_internal(journal)",
        "rollback_transition(&mut self.state, journal)",
        "rollback_preparation(&mut self.state, preparation_journal)",
        "rollback_completion(&mut self.state, completion_journal)",
    ],
    "src/authority_checkpoint.rs": [
        "pub fn attest_authority_checkpoint_from_cursors",
    ],
}

for relative, fragments in required.items():
    text = (ROOT / relative).read_text(encoding="utf-8")
    for fragment in fragments:
        if fragment not in text:
            errors.append(f"{relative}: missing transition contract {fragment!r}")

prohibited = {
    "src/lib.rs": ["fn clone_for_transition", "let mut candidate = self.clone"],
    "src/execution.rs": ["fn clone_for_transition", "candidate_wisdom"],
    "src/runtime.rs": ["fn clone_for_transition"],
    "src/service.rs": [
        "clone_for_transition",
        "let prior_state",
        "let prior_execution",
        "let durable_started_state",
        "let durable_started_execution",
        "self.runtime_integrations.values().cloned()",
    ],
    "src/authority_recovery.rs": ["clone_for_transition", "let mut candidate = state"],
}
for relative, fragments in prohibited.items():
    text = (ROOT / relative).read_text(encoding="utf-8")
    for fragment in fragments:
        if fragment in text:
            errors.append(f"{relative}: prohibited full-state rollback pattern {fragment!r}")

service = (ROOT / "src/service.rs").read_text(encoding="utf-8")
for rollback in (
    "rollback_observation_internal(journal)",
    "rollback_transition(&mut self.state, journal)",
    "rollback_preparation(&mut self.state, preparation_journal)",
    "rollback_completion(&mut self.state, completion_journal)",
):
    pos = service.find(rollback)
    if pos < 0:
        continue
    window = service[max(0, pos - 500):pos]
    if "commit_outcome_may_be_unknown()" not in window:
        errors.append(f"src/service.rs: rollback {rollback!r} is not guarded by ambiguity classification")

if errors:
    print("transition journal validation failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)
print("validated narrow transition journals and ambiguous-commit handling")
