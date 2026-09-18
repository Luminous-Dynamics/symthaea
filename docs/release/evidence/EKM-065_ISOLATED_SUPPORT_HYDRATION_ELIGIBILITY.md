# EKM-065 — Isolated Support Hydration Eligibility

## Purpose

EKM-064 can re-run every persisted belief mutation through the real EKM-026 firewall against protected mutation-time evidence and reproduce the exact support transition. It intentionally does not claim that rejected or unapplied EKM-025 revision receipts were historically re-evaluated; private placeholders are used only to preserve their receipt IDs when needed.

EKM-065 binds that replay proof to the exact EKM-053 read-only quarantine and EKM-054 protected trust-context checkpoint, while keeping **support-store hydration review** separate from **complete revision-history hydration review**.

The central invariant is:

**reproducible support mutation history does not imply reproducible complete epistemic audit history.**

## Inputs

EKM-065 requires the exact chain:

1. restart-v2 snapshot
2. EKM-057 mutation-seal sidecar
3. EKM-044 restart validation receipt
4. EKM-059 protected mutation-seal checkpoint
5. EKM-060 protected-source admission
6. EKM-061 current-head evidence
7. EKM-062 historical replay eligibility
8. EKM-063 historical evidence projection
9. EKM-064 isolated firewall replay report
10. EKM-053 read-only quarantine
11. EKM-054 protected joint trust-context checkpoint

The EKM-064 replay report is re-derived from the exact source inputs before any eligibility decision is made.

## Binding checks

EKM-065 requires:

- exact restart outer checksum and V2 digest equality with the read-only quarantine
- exact restart capture cycle equality
- EKM-054 trust-context digest equality with the quarantine trust-context digest
- valid, unexpired EKM-054 protected checkpoint
- unexpired EKM-061 current-head evidence
- review time not before EKM-064 replay or EKM-054 checkpoint verification
- zero unexpected authority on quarantine, trust checkpoint, and replay report
- exact persisted mutation receipt reproduction from EKM-064
- exact final support-state equivalence
- exact consumed-authorization-count equivalence

## Split eligibility result

If the above checks pass, EKM-065 reports:

- `historical_support_mutations_reproduced = true`
- `support_store_hydration_review_eligible = true`

But complete audit-state review depends on EKM-064's placeholder count.

When `non_mutation_revision_placeholder_count == 0`:

- `full_revision_history_hydration_review_eligible = true`
- `complete_epistemic_hydration_review_eligible = true`
- `immutable_revision_restore_path_required = false`

When the placeholder count is non-zero:

- `full_revision_history_hydration_review_eligible = false`
- `complete_epistemic_hydration_review_eligible = false`
- `immutable_revision_restore_path_required = true`

This prevents support-state reproducibility from being silently promoted into a claim that all rejected/unapplied revision audit records were historically reconstructed.

## Why support review may proceed independently

Persisted support state is determined by the mutation receipts that actually reached EKM-026. EKM-064 independently reproduces those mutation receipts and the resulting final support store.

Rejected/unapplied EKM-025 receipts remain important immutable audit history, but they do not participate in the support-state transition. Their restoration should therefore have its own typed, non-authoritative contract rather than being fabricated merely to satisfy a hydration API.

## Authority boundary

EKM-065 is review evidence only. It constructs no writable support store or revision history.

Even when support-store hydration review is eligible, it reports:

- `writable_hydration_authorized = false`
- `writable_state_export_authorized = false`
- `activation_authorized = false`

It does not:

- expose an EKM-064 temporary ledger/store/firewall
- construct EKM-055 hydration state
- create a mutation authorization
- restore non-mutation revision history
- advance any trust checkpoint
- alter legacy confidence
- mutate causal/world-model/action state
- perform file/network I/O or key custody

## Recommended next boundary

When placeholders exist, the next safe tranche is an **immutable revision-audit restoration contract**. It should reconstruct typed, read-only EKM-025 receipt semantics from the already persisted EKM-031/EKM-038 data without creating mutation authority or pretending each rejected decision was historically re-run against unavailable full ledger chronology.

Only after that audit-history contract and EKM-064 receive executable qualification should EKM-055 be revised to consume these evidence layers and construct a sealed writable sandbox.

## Qualification status

This tranche is stacked on EKM-064. GitHub Actions remains the executable authority. Parent EKM-064 exact-head CI #7364 was still queued when this note was prepared.

Static/API review is not rustfmt, compilation, Clippy, unit-test, integration-test, runtime, or deployment evidence.
