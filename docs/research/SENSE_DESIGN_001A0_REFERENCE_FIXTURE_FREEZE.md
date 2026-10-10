# SENSE-DESIGN-001A0 reference fixture freeze

Date: 2026-09-25

Issues: #5827, #5833, #5823, #5826, #5839.

Base source: `main@2dfddf6027d8eaf62221f8a71be2bb6d1d7bd9a9`.

## Scope

This tranche freezes a language-neutral known-answer corpus for sensor-system comparison and records the #2030 decision-diagnosability reuse audit.

It intentionally adds no production Rust, no sensor physics, no calibration implementation, no optimizer, no actuation authority, and no physical PASS.

Core claim ceiling:

`the reference corpus has deterministic expected dispositions for synthetic sensor-system comparisons`

This is strictly weaker than:

`a production reducer is correct`
`a real sensor is better`
`a physical mechanism is established`
`a sensor is calibrated`
`a sensor is safe or deployment-ready`

## Comparison ownership

Generic multidimensional trade-study / Pareto semantics remain owned by SE-OPT #3686.

SENSE adds only domain constraints needed to keep sensor comparisons honest:

- P1 physical sensor characterization;
- P2 task evidence;
- P3 system/resource evidence;
- protected-axis behavior;
- missing/invalid/out-of-profile states;
- fair-baseline validity;
- evidence lineage / independence constraints;
- held-out and physical-article lineage;
- system-level improvement vs component causal attribution.

The fixture corpus must therefore remain usable as an independent oracle input even if the eventual production implementation is a thin SE-OPT specialization.

## Deterministic corpus

The sibling JSON fixture file freezes 28 known-answer cases from #5827.

Canonicalization rule for the recorded digest:

1. remove `known_answer_digest_sha256`;
2. serialize the remaining JSON object with lexicographically sorted object keys;
3. use no insignificant whitespace;
4. encode as UTF-8;
5. SHA-256 the resulting bytes.

Frozen digest:

`007ea7cb38a62dc49fdb81f83485bb420491000611e47a1359ae1f133da4b759`

The fixture corpus covers:

- ordinary tradeoffs;
- Pareto improvement;
- explicit baseline dominance when the candidate is worse on every comparable axis;
- declared equivalence tolerance;
- protected-axis regression;
- required missing evidence;
- held-out leakage;
- fair-baseline invalidity;
- system-vs-component causal attribution;
- simulation-only evidence;
- article replication;
- common physical evidence roots;
- resource-boundary mistakes;
- explicit null/negative outcomes.

No fixture requires a scalar global sensor score.

The explicit `DOMINATED` disposition is intentional: a candidate that is worse on every comparable axis is not a `TRADEOFF`. This prevents the comparison vocabulary from hiding a complete non-improvement behind neutral language.

## #2030 / PIE-009N reuse audit

The source audit found that #2030's current executable artifact is PR #2032:

- PR: `PIE-009N: independent diagnosability and sensor-cut oracle`
- head: `fc12a156eda988b02b0a76500525c0fd87a3a4f7`
- status at audit: open + draft
- contents:
  - `scripts/pie-009n-diagnosability-oracle.py`
  - `docs/release/evidence/PIE_009N_DIAGNOSABILITY_SENSOR_CUTS.md`

The Python oracle is implementation-independent and uses generic local dataclasses (`World`, `Sensor`, `DiagnosabilityResult`), but it is still a standalone reference oracle, not a production library interface. It is also not on `main`.

Therefore the current result of #5833 is:

`do not extract production code yet`

Preferred near-term relationship:

`#2032 independent oracle semantics`
`+ SENSE cross-domain fixtures`
`-> adapter / parity qualification later`

Venue-neutral extraction becomes justified only when all of the following are true:

1. at least two executable production consumers need the exact same theorem;
2. one current qualified source implementation exists to extract;
3. dependency direction remains clean;
4. extraction preserves original #2030 fixture behavior exactly;
5. fresh exact-head qualification is performed after code movement.

Until then, SENSE should not fork the diagnosability algorithm and should not convert #2032's oracle into a shared crate merely to remove PIE naming.

## Cross-domain theorem retained

The theorem SENSE consumes is:

`full hidden-state identification is unnecessary when all surviving hidden worlds imply the same decision`

A sensing set is sufficient only if every pair of hidden worlds requiring different decisions is separated by at least one informative active sensing path.

SENSE later projects physical sensor slots, placements, modalities, calibration eligibility and #4918 common-mode dependencies into that theorem.

This remains distinct from:

`structurally diagnosable != calibrated physical evidence`
`minimum sensor cut != failure probability`
`derived representation != independent sensor`
`diagnosable != safe to act`

## Promotion sequence

1. freeze this fixture corpus;
2. independently validate the JSON schema/digest in a later oracle tranche;
3. add a tiny tactile cross-domain diagnosability fixture against #2032 semantics;
4. converge on SE-OPT and canonical observation/calibration owners;
5. only then add a production SENSE adapter if a real gap remains.

## Nonclaims

This freeze does not establish physical sensor accuracy, calibration, robustness, event-camera superiority, tactile superiority, causal attribution, manufacturing readiness, SE-OPT qualification, #2032 qualification, or physical authority.
