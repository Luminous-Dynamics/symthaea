# Subterranean Capability Campaign IX — Certifiable Autonomy

## Objective

Campaign IX closes the gap between possessing many safety mechanisms and being able to construct a bounded, reproducible, rejectable release argument around them.

## Patch sets 77–90

1. Canonical release-blocking requirement registry with stable identifiers.
2. Independent final-command runtime invariant monitor.
3. Bounded fault-tree and minimal-cut-set analysis.
4. Reproducible, physically validated scenario manifests.
5. Requirement-to-test, invariant, analysis, and evidence traceability.
6. Structured claim–argument–evidence safety case.
7. Same-frame invariant enforcement and operational evidence integration.
8. Traceability and evidence-summary compile closure found by warning-denied checking.
9. Deterministic scenario execution and acceptance reporting.
10. Role-separated release approval and bounded waiver policy.
11. Deterministic self-validating certification bundle export.
12. Preregistered certifiable-autonomy acceptance suite.
13. Evidence verification-method alignment found by the full test campaign.
14. Certifiable Autonomy Protocol and explicit non-claims.

## Causal improvements

The campaign is not documentation around unchanged behavior. The final physical command now passes through an independent invariant authority after every learned, operator, recovery, power, maintenance, and isolation decision. A violation is corrected before the plant advances and is visible in evidence.

Release evidence is also now internally referential:

- requirements identify obligations;
- manifests identify deterministic tests;
- traceability links obligations to artifacts;
- fault trees identify modeled combinations of failure;
- safety claims consume those artifacts;
- signoff evaluates the completed case;
- the bundle verifies that manifests and reports still correspond.

## Verification result

Offline API-compatible validation on Rust 1.85:

- warning-denied `cargo check --all-targets`: passed;
- library tests: 205 passed, 0 failed, 1 intentionally ignored hardware timing benchmark;
- deterministic certification acceptance suite: passed;
- Rust formatting and `git diff --check`: passed.

The offline workspace uses stand-ins for omitted workspace dependencies and therefore does not replace the authoritative Rust 1.94 full-workspace build, Clippy, HDC/FEP integration, hardware testing, or independent assurance.
