# VLEB requirement-to-oracle traceability — 2026-10-28

## Decision

A scored task must not contain a behavioral requirement that has no executable test, and its test suite must not contain an unaccounted-for logical test. Schema validity by itself cannot express equality and coverage constraints across nested arrays. VLEB now makes traceability a scored-task admission requirement and checks it in the independent receipt auditor.

For tasks classified as `private_holdout` or `private_confirmatory`, the task contract must include `behavior_contract.requirement_traceability`. Each entry pins:

- a stable requirement ID;
- the source kind and exact source text from the frozen task statement, invariant, allowed behavior change, or observable-effect inventory;
- one or more logical test IDs required to demonstrate that requirement;
- the observable effects used to evaluate the requirement.

The independent auditor requires every declared source item to appear exactly once in the trace inventory, rejects duplicate requirement IDs or duplicate source mappings, rejects test IDs absent from the frozen oracle inventory, and requires the union of trace-mapped test IDs to cover the complete `oracle_contract.required_test_ids` set. Existing execution-map validation then requires those logical test IDs to map to concrete tests and the receipt to demonstrate that they passed.

## Why it matters

Benchmark scores can be misleading in both directions: tests can enforce implementation details not stated in the task, or tests can fail to cover behavior that the task actually requires. SWE-bench Pro V2's September 2026 refresh described extensive task auditing, blind re-implementation of revised instructions, and a two-sided release gate requiring reference success and empty-patch failure (https://labs.scale.com/leaderboard/swe_bench_pro_public_v2?tab=full).

This traceability layer addresses a narrower, auditable part of that problem: it prevents requirement/test inventory drift and silent omission. It does **not** establish that requirements are complete, that mapped tests are semantically adequate, or that a reviewer is independent. Those still require blind human review, the baseline/reference two-sided gate, mutation review, and the independent runtime evidence gates.

## Admission behavior

- Public-development fixtures may remain incomplete while the harness is under development; they remain ineligible for headline scoring.
- Private holdout/confirmatory contracts fail closed if the traceability inventory is absent or inconsistent.
- All declared requirements and all observable effects must be represented.
- Each required logical test must be traced to at least one declared requirement and must continue to map to concrete executable test IDs.
- A trace entry may not invent requirements or observables outside the frozen contract and may not reference tests outside the task's required-test inventory.
- Confirmatory scoring still requires the trusted manifest, attestation, receipt audit, claim-evidence checks, and semantic verifier obligations. Passing this traceability check alone does not grant score eligibility.

## Files

- `tools/vleb/task-contract.schema.json` defines the requirement-traceability structure and requires it for scored task classes.
- `tools/vleb/audit_receipt.py` performs cross-field traceability and coverage checks in the independent task/receipt audit.
- `tools/vleb/test_task_contract_schema.py` tests scored-task schema admission.
- `tools/vleb/test_receipt_auditor.py` includes negative controls for missing requirements, unknown tests, uncovered test IDs, invented source text, undeclared effects, and duplicate requirement IDs.
