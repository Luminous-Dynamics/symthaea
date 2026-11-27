# REPAIR-Bench synthetic event fixtures v0

Status: synthetic benchmark-contract fixture only. No agent was run and no performance result is claimed.

## Purpose

This first slice operationalizes the two-phase evaluation in issue #13926: freeze an evidence graph and conclusions, then introduce a source, parser, method or scope event and require dependency-aware reassessment.

## Files

- REPAIR_BENCH_V0_SYNTHETIC_EVENT_FIXTURES.json — one baseline graph with byte-pinned input artifacts, 14 dependency nodes, and 12 independent change events.
- scripts/validate_repair_bench_fixtures.py — standard-library-only consistency validator, run by .github/workflows/repair-bench-fixture-contract.yml.

## Fixture properties

- Two raw artifacts: a CSV dataset and a JSON protocol.
- Six located observations, two analysis nodes, four claims and two decisions.
- A deliberate parser rounding defect and comparator defect in the baseline to exercise parser/method correction events.
- Twelve event families: raw-data correction, non-material update, retraction, threshold correction, parser correction, method correction, scope correction, contradictory source, duplicate-origin mirror, semantically equivalent raw-byte version, unavailable correction, and untrusted notice.
- Every event has expected changed, unchanged, blocked, unresolved, non-dependent and/or ignored-untrusted outcomes.

The artifact byte strings and SHA-256 values are included so a later harness can check exact input identity. The expected transitions are proposed gold, not yet independently adjudicated. In particular, graph dependencies, event ordering and the deliberately defective baseline method/parser need domain and engineering review before any confirmatory use.

## Evaluation rules

- A source event never directly declares a dependent claim globally false; it changes evidence status and triggers reassessment.
- A documented unchanged decision counts toward correction reach.
- An unavailable or untrusted update must not be silently interpreted as no change.
- Same-origin mirrors do not count as independent corroboration.
- Semantically equivalent byte changes should trigger artifact-version reconciliation but not collateral invalidation.
- Every trial, including blocked and unresolved outcomes, remains in the denominator.

## Claim boundary

This file does not implement update propagation, graph invalidation, parser execution, replay, or a benchmark scorer. No scientific accuracy, correction-reach, SOTA or production claim is made.
