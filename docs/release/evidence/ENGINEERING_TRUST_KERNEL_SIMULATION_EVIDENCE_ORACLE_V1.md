# Engineering Trust Kernel — Simulation Evidence Admission Oracle V1

**Status:** independent reference oracle; structural admission semantics only  
**Branch:** `engineering/etk-2a-simulation-evidence-oracle`  
**Base:** `main` at `3afeee3d40af0bae0b85e70869571e024c28f07b`

## Theorem boundary

```text
simulation converged
!= engineering evidence
!= admissible simulation evidence
!= discharged proof obligation
!= qualified design
!= manufacturing approval
!= physical actuation authority
```

This oracle decides only whether one candidate simulation artifact is structurally eligible to be admitted as `Simulation` evidence for one exact proof obligation. `Admit` is not a discharge receipt and grants no verification, qualification, certification, manufacturing, deployment, or actuation authority.

## Independence

`scripts/etk-simulation-evidence-admission-oracle.py` is standard-library Python and imports no Symthaea code. It is based directly on `main`, not on the ETK-1 Rust implementation branch. A production Rust implementation must reproduce this contract independently rather than invoking this script as its authority source.

## Existing simulation semantics preserved

The oracle deliberately reuses the existing `symthaea-sim-bridge` vocabulary:

- only `external_solver` can admit;
- `dry_run` and unknown execution cannot admit;
- convergence is necessary but insufficient;
- normalized metrics are required;
- backend, solver version, rendered-input digest, raw-output digest, and parser version are required;
- run-level and metric-level epistemic/aleatoric uncertainty are preserved;
- metric-level uncertainty overrides run-level uncertainty when present.

## Exact V1 bindings

Admission requires exact binding of:

- obligation ID and obligation revision;
- engineering subject;
- design/twin revision;
- accepted requirement revision;
- evidence-policy ID;
- simulation request ID;
- validity-domain ID;
- currentness state plus currentness-proof reference;
- candidate artifact identity and source-lineage identity;
- expected rendered-input digest;
- one required metric, unit, scalar inequality predicate, and uncertainty budget.

The schema is closed-world. Unknown/shadow fields deny admission.

## Conservative uncertainty theorem

V1 supports only `<`, `<=`, `>`, and `>=` predicates. If the effective uncertainty includes an interval, `<`/`<=` are checked at the interval upper bound and `>`/`>=` at the interval lower bound.

Therefore:

```text
passing point estimate != passing uncertainty-bounded evidence
```

The metric point must itself lie inside its declared interval, and epistemic/aleatoric uncertainty must stay within the obligation's declared budget.

Every normalized metric is validated before identity construction, including metrics not used by the acceptance predicate. A malformed secondary metric therefore fails closed rather than reaching serialization or identity minting.

## Canonical semantic identity

Before hashing, an admitted candidate is projected into a normalized semantic V1 identity with all authority-relevant fields explicit. Optional uncertainty intervals normalize to explicit `null` when absent. Semantically equivalent JSON spellings therefore cannot fork the admitted-evidence identity.

The admitted-evidence ID is SHA-256 over a domain-separated, key-sorted canonical JSON preimage. It is an audit/content identity only; possession of it grants no authority.

## Synthetic positive fixture

The positive fixture binds:

- obligation `O-structural-stress-42:r3`;
- policy `ETK-SIM-ADMISSION-V1`;
- subject `bracket-alpha`;
- design revision `design:G17`;
- requirement revision `REQ-STRESS:r5`;
- request `sim-static-G17-LC9`;
- validity domain `VD-static-G17-LC9`;
- stress predicate `max_stress_mpa <= 250 MPa`;
- uncertainty budget `epistemic <= 0.2`, `aleatoric <= 0.1`;
- observed stress `181.2 MPa` with interval `[175, 190] MPa`;
- external CalculiX `2.22` provenance;
- exact input/output/parser identities;
- currentness-proof and source-lineage references.

The frozen expected admitted-evidence vector remains:

```text
sha256:7066c8509f0563484acc3a2d16d5b9b689606a2250389da56ad66560dbc83ff8
```

The latest validation-only hardening does not change the normalized positive-fixture preimage.

## Adversarial suite

The built-in self-test requires fail-closed denial for dry-run execution, stale evidence, missing currentness reference, rendered-input substitution, design/twin substitution, wrong evidence kind, validity-domain substitution, incomplete provenance, absent required metrics, threshold failure, interval-crossing threshold failure, excessive uncertainty, malformed intervals, non-convergence, request substitution, unknown/shadow execution fields, non-finite confidence, malformed secondary metrics, invalid secondary uncertainty, malformed required binding fields, and deterministic ordering of simultaneous independent faults.

It also checks canonical-ID equivalence when an optional `interval: null` is omitted from an otherwise identical uncertainty object.

## Evidence status

The immediately preceding canonical-identity candidate was locally executed under Python 3.13.5 and passed its built-in self-test and `py_compile`, producing the same frozen expected vector above.

The current head additionally hardens all-metric and missing-binding fail-closed behavior. Its checked-in script blob is:

```text
66f6d1cd70575a0d93950733f90ff68d502c5938
```

No claim of exact-head runtime qualification is made for that newest blob until repository CI or another exact-byte execution records it. This document intentionally distinguishes prior candidate execution from current-head qualification.

## Deliberate non-goals

V1 does not establish solver correctness, physical truth, multi-source evidence independence, authenticated currentness, complete/calibrated uncertainty, contradiction handling, supersession, semantic-staleness propagation, formal-proof/test/telemetry admission, discharge receipts, safety-case closure, design qualification, certification, manufacturing approval, deployment approval, or actuation authority.

`source_lineage_id` and `currentness_proof_id` are binding/reference slots. Their presence does not prove independence or the trustworthiness of the referenced currentness mechanism.

## Production follow-up

The production path remains:

```text
SimulationResult
-> candidate simulation evidence
-> ETK admission decision
-> AdmittedSimulationEvidenceV1
-> separately constructed ObligationDischargeReceiptV1
```

The current `converged simulation -> discharged Simulation obligation` shortcut must be removed only after typed admission and discharge-receipt primitives exist. Admission and discharge remain different propositions.
