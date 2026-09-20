# HUM-OBS-003D — Execution-Receipt-Gated High-Assurance Admission

Status: source-design candidate
Issue: #5028
Parent: HUM-OBS-003C / #5022
Authority: benchmark evidence admission only; **no physical, safety, or deployment authority**

## Purpose

HUM-OBS-003B proves that imported result metadata matches a predeclared matrix. HUM-OBS-003C proves execution provenance for an individual planned episode. HUM-OBS-003D composes those contracts into a separate stronger assessor without silently redefining either one.

The core theorem is:

```text
003B metadata-valid case
!=
003D execution-provenance-valid case
```

## Input tuple

Each high-assurance evidence item carries:

```text
HumanoidBenchMatrixCaseResultV1
+ optional HumanoidBenchRunnerSubjectV1
+ optional HumanoidBenchExecutionReceiptV1
```

The optional shape is deliberate: missing provenance is represented as missing evidence instead of forcing callers to fabricate a placeholder receipt.

## Admission order

003D first runs the unchanged 003B assessor over the imported results. An episode can only proceed to provenance evaluation if it is a planned case with no episode-level 003B violation and is not missing from 003B admission.

Then 003D requires:

- a runner subject that validates against the same matrix;
- an execution receipt whose commitment has not already been used;
- the execution receipt to validate against that exact subject, matrix, and adapter input/result.

A duplicate, substituted, malformed, or cross-episode receipt cannot increase high-assurance coverage.

## Missing vs invalid provenance

The receipt distinguishes:

```text
missing runner subject / missing execution receipt
-> PartialHighAssuranceMatrix
```

from:

```text
metadata substitution
subject substitution
invalid execution receipt
duplicate receipt
-> InvalidHighAssuranceEvidence
```

This avoids turning "not yet proven" into "proven false" while still failing closed on contradictory evidence.

## Infrastructure failure

A valid HUM-OBS-003C infrastructure-failure receipt can establish that the governed runner reached a recorded execution phase for a planned case.

Therefore:

```text
valid execution-attempt provenance
!= completed benchmark performance
```

003D reports completed-provenance cases separately from infrastructure-indeterminate provenance cases. Infrastructure-indeterminate cases never create performance returns here.

## Receipt contents

`HumanoidBenchHighAssuranceReceiptV1` binds:

- exact matrix commitment;
- exact 003B coverage receipt commitment/status;
- planned/supplied/metadata-admitted counts;
- metadata-eligible-for-provenance count;
- provenance-valid/completed/infrastructure-indeterminate counts;
- missing provenance episode IDs;
- sorted high-assurance violation events;
- admitted execution receipt commitments in planned-case order;
- deterministic domain-separated receipt commitment.

No aggregate performance score is created.

## Claim separation

```text
predeclared case
!= metadata-valid result

metadata-valid result
!= execution-provenance-valid result

execution-provenance-valid result
!= good benchmark performance

good simulation performance
!= physical-world capability

physical-world capability
!= deployment authority
```

## Tests

Source tests cover:

- complete valid provenance admission;
- missing receipt remaining partial rather than fabricated failure;
- subject substitution becoming invalid evidence;
- valid infrastructure failure proving attempted execution without performance;
- duplicate receipt replay not increasing provenance coverage;
- deterministic high-assurance receipt identity.

## Qualification boundary

No format/compile/test/Clippy PASS is established until an exact-head qualifier executes against the frozen HUM-OBS-003D subject.
