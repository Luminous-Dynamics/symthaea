# Subterranean Capability Campaign XIX

**Theme:** Resource-conflict and multi-objective assurance
**Date:** 2026-07-20
**Baseline:** Hardened v18 / patch 287

## Objective

Campaign XIX closes a remaining authority gap between individually safe subsystems. Before this campaign, physical safety, return reserve, environmental stewardship, maintenance, restoration, peer assistance, communications, and productive work could each request action, but the crate had no single explicit account of whether their combined demands fit within the remaining battery, thermal, time, and recovery envelope.

The campaign introduces safety-monotonic arbitration. Protected objectives are funded first, discretionary work cannot consume reserved return or recovery capacity, persistent deferral becomes visible starvation debt, service imbalance remains attributable, and unavoidable conflicts become throttled work, protected return, or accountable hold rather than opaque controller behavior.

## Patch sequence

1. Explicit normalized multi-objective resource budgets.
2. Bounded objective-starvation debt.
3. Bounded stakeholder service-fairness ledger.
4. Safety-monotonic resource-conflict supervisor.
5. Protected reserve preservation during soft arbitration.
6. Runtime derivation of live objective demands.
7. Public resource-conflict API exports.
8. Operational checkpoint schema v12.
9. Accountability decision-stage integration.
10. Embodiment persistence and public assessment access.
11. Same-frame runtime command enforcement.
12. Resource-conflict fallback identities and reset semantics.
13. Public operational evidence exports.
14. Bounded starvation and fairness evidence.
15. Live evidence emission.
16. Seven preregistered release contracts.
17. Public validation API exports.
18. Five canonical `SUB-RES-*` requirements.
19. Requirement-to-evidence traceability.
20. Throttled-work semantic correction and counterfactual explanation.
21. Top-level certifiable-autonomy gate integration.
22. Self-consistent resource-conflict assurance bundle.
23. Validation import cleanup.
24. Resource-conflict counterfactual contract.
25. Full checkpoint-continuity release contract.
26. Resource-conflict and multi-objective assurance protocol.
27. Campaign record.
28. Verification record.

## Resulting authority behavior

- Invalid runtime resource inputs fail closed as physical-safety demand.
- Physical safety, protected return, and environmental containment cannot be displaced by mission urgency.
- Obligatory and discretionary work use only capacity remaining after the protected reserve.
- Reduced work may continue under `Throttled`; it is not falsely reported as a total work ban.
- `ReturnOnly` removes cutter and auger authority and permits only protected withdrawal and recovery.
- `HoldForReview` removes movement while preserving safety-required recovery actuators.
- Starvation and fairness can narrow authority but cannot create motion.
- Resource-conflict decisions appear in command traces, counterfactual explanations, operational evidence, checkpoints, traceability, and the top-level release gate.

## Important hardening correction

Initial integration treated every `Throttled` assessment as if productive work were completely forbidden. That contradicted both the command envelope—which deliberately caps work at a reduced level—and the authority label. The correction allows bounded productive work under `Nominal` or `Throttled`, while `ReturnOnly` and `HoldForReview` remain strict work prohibitions.

The correction also adds a dedicated resource-conflict counterfactual blocker so operators can distinguish reserve-driven throttling from policy, temporal, epistemic, or physical-hazard restrictions.

## Validation boundary

The current sandbox contains no Rust compiler, Cargo, Rustfmt, Nix toolchain, or network path from which one could be retrieved. Campaign XIX therefore performs source-level and patch-level validation only in this environment:

- delimiter and lexical-balance scans over changed Rust files;
- exhaustive struct-literal checks for newly required persisted and evidence fields;
- match-coverage audits for new authority enums;
- production panic-marker scans before test modules;
- `git diff --check`;
- incremental and complete clean-room patch application;
- exact Git-tree and recursive tracked-file comparison;
- artifact checksum generation and verification.

The added Rust tests and release contracts are included but were not executed here. Authoritative acceptance requires the real Rust 1.94 workspace, Clippy, the complete test suite, calibrated resource models, HIL conflict and exhaustion studies, and independent resource-governance review.
