# VLEB shift-event evidence audit — 2026-10-29

## Purpose

The clean/shifted analyzer groups result records and measures transition rates. It deliberately does not prove that a declared intervention occurred. This contract adds a second boundary: validate the detailed runner-event and revalidation records that a trusted runner is expected to place inside its evidence bundle.

## Contract

`tools/vleb/audit_shift_event_evidence.py` accepts an evaluator-pinned `vleb-shift-evidence-v1` JSON manifest containing a frozen scenario inventory, seed set, and exactly one clean and shifted receipt per scenario × seed cell.

For each scenario, the manifest pins:
- task/change family and intervention position;
- before/after task contract hashes and expected intervention payload hash;
- initial workspace/checkpoint, source revision, immutable runner image;
- exact required validation-check IDs and the stale-artifact inventory that should be invalidated by the shift.

Each receipt binds the scenario and protocol, a canonical receipt payload digest, run ID/attempt, initial execution identity, final workspace hash, audit hash and required revalidation outputs. Each shifted receipt must also contain a runner-event record with a matching event payload hash, pre-change workspace hash, post-change workspace hash, transition diff hash, log hash, event sequence, and monotonic observation timestamp. The clean receipt may not claim that a shift was applied.

A shifted receipt must account for every predeclared stale artifact exactly once and link each invalidation/regeneration to the observed event ID. Every required check must be represented exactly once and bind to the final workspace hash. A receipt may not claim `pass` if any required check failed or remained unresolved.

## Integrity versus authenticity

This auditor requires external digest pins for the complete manifest, protocol and auditor source. It rejects duplicate JSON keys, non-finite numbers, missing/extra fields, incomplete cohort cells, duplicate run/receipt/audit identities, retry substitution, mismatched scenario/task/protocol/state pins, stale-evidence omissions, and partial/incorrect revalidation coverage.

**It does not authenticate event-log, diff, receipt or audit bytes and cannot prove the runner really applied the shift.** `event_source="trusted_runner"`, `applied=true`, and hashes are declarations until a trusted attestation policy authenticates the exact bundle. Confirmatory results must therefore remain ineligible until the caller verifies signed provenance for the exact workflow run, attempt, repository, source revision, workflow identity, and evidence bundle. This validator emits `runner_attestation_required=true` and `score_eligible=false` unconditionally.

## Proposed acceptance criteria

1. 100% receipt coverage for the frozen scenario × seed × clean/shifted inventory; no retry substitution.
2. Zero mismatches among scenario, protocol, source revision, image, checkpoint, model/scaffold and initial execution profile within each pair.
3. Every shift event content hash agrees with the preregistered shift payload.
4. 100% coverage of the predeclared stale-artifact inventory and required post-change checks.
5. Zero cases where a receipt marked passing has a failed or unresolved required check.
6. No score eligibility until independent signed-runner attestation is verified and the referenced bundle bytes are audited.
7. Negative controls demonstrate rejection of fabricated event IDs, changed payloads, state drift, stale-evidence omissions, old test reuse, incomplete validation, and receipt tampering.

This is an integrity gate, not a measured robustness score. Synthetic tests validate schema/binding behavior only.
