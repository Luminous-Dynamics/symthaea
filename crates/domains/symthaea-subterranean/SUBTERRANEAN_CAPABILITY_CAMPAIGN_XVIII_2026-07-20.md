# Subterranean Capability Campaign XVIII

**Theme:** Temporal and causal robustness  
**Date:** 2026-07-20  
**Baseline:** Hardened v17 / patch 260

## Objective

Campaign XVIII closes a remaining field-authority gap: a safe controller must distinguish current evidence from delayed history, valid causal order from contradictory event order, and a still-valid plan from one invalidated by changes in hazards, topology, calibration, state, or mission.

The campaign deliberately avoids turning timestamps or model correlation into unwarranted certainty. Temporal uncertainty may narrow authority to probing, protected return, or review hold. It cannot weaken physical safety.

## Patch sequence

1. Replay-resistant multi-domain clock discipline.
2. Bounded causal event ledger.
3. Purpose-aware delayed-observation authority.
4. Revision-bound plan freshness.
5. Conservative command-response attribution.
6. Composite temporal assurance supervisor.
7. Deterministic nominal runtime timing evidence.
8. Public temporal and causal API exports.
9. Compile-audit corrections.
10. Preregistered temporal release contracts.
11. Operational checkpoint schema v11.
12. Accountability decision-stage integration.
13. Live same-frame temporal command enforcement.
14. Temporal probe, return, and hold fallback identities.
15. Counterfactual temporal blocker explanations.
16. Bounded operational temporal evidence.
17. Runtime publication of temporal evidence.
18. Accountability fixture compatibility correction.
19. Live restriction and restart-continuity tests.
20. Canonical temporal requirements and traceability.
21. Self-consistent temporal assurance bundle.
22. Causal runtime-sequence checkpoint correction.
23. Temporal and causal assurance protocol.
24. Campaign record.
25. Mechanical formatting normalization.
26. Verification record.

## Important defect discovered

The complete test suite exposed a restart interaction between the new temporal hold and the pre-existing formal replay monitor. The operational checkpoint preserved formal replay history but not the embodiment's current runtime step. After restart, the next frame reused step zero; the formal monitor correctly treated it as replay and replaced the more precise temporal fallback label with a formal hold.

The platform remained motionless, so physical safety was preserved, but evidence attribution and causal sequence continuity were wrong. The correction persists the causal runtime step in checkpoint schema v11 and restores it only after all checkpoint domains validate.

## Resulting authority behavior

- Replayed or impossible clock state causes `HoldForReview`.
- Stale immediate-control evidence removes productive authority.
- Historical evidence remains available without steering the current machine.
- Causally impossible dependencies latch review.
- Ambiguous overlapping command effects cannot be claimed as unique causes.
- Hazard, topology, calibration, state, mission, or age changes invalidate stale plans.
- Same-frame command restriction preserves emergency recovery actuators.
- Clean service-location dwell is required before a temporal review hold clears.
- Restart cannot reset causal sequence identity or silently restore authority.

## Validation boundary

Offline validation uses Rust 1.85 and API-compatible stand-ins for workspace dependencies omitted from the uploaded archive. It exercises Rust types, deterministic runtime behavior, serialization structures, exhaustive matches, authority ordering, tests, and clean-room patch reconstruction.

Authoritative production acceptance still requires the complete Rust 1.94 workspace, Clippy, calibrated clocks, authenticated timestamps, HIL delay/reordering tests, power-loss and rollover studies, controlled real-time measurements, and independent temporal/causal review.
