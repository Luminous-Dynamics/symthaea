# HAK-016 — Cardinality-Aware Selector Coverage v1

Status: candidate audit/evidence tooling. No runtime authority changes.

## Purpose

HAK-014 normalization receipts preserve optional-selector `matches` and `missing` counts, but their v1 summary status is intentionally coarse:

```text
Present
Absent
Failed
```

A wildcard result can therefore be historically valid as:

```text
status = Present
matches = 7
missing = 3
```

while the word `Present` alone can be misread as full coverage.

```text
PresentSomewhere
!=
PresentEverywhere
```

HAK-016 does **not** rewrite HAK-014 receipts. It derives a separate, digest-bound coverage record from a validated HAK-014 receipt.

```text
HAK-014 receipt v1
        ↓
HAK-016 selector coverage v1
```

## Historical semantics remain historical

HAK-016 preserves the source HAK-014 optional status beside the new coverage state:

```text
source_status
coverage_state
```

For example:

```text
source_status  = Present
coverage_state = PartiallyPresent
```

is valid when HAK-014 observed both matches and missing contexts.

```text
OldReceiptHistoricallyValid
!=
OldReceiptCarriesNewCoverageSemantics
```

The HAK-014 receipt digest and bytes remain unchanged.

## Coverage states

HAK-016 optional coverage states are:

```text
Present
PartiallyPresent
Absent
NotApplicable
Failed
```

`VacuouslySatisfied` is not used for optional evidence because there is no requirement to satisfy.

### Present

At least one applicable context exists and every applicable context contains the selected value:

```text
applicable > 0
matches == applicable
missing == 0
```

### PartiallyPresent

At least one applicable context contains the value and at least one does not:

```text
matches > 0
missing > 0
applicable == matches + missing
```

### Absent

Applicable contexts exist but none contain the selected value:

```text
applicable > 0
matches == 0
missing == applicable
```

### NotApplicable

Traversal produced zero applicable contexts, for example an empty wildcard array:

```text
applicable == 0
matches == 0
missing == 0
```

```text
NotApplicable
!=
Absent
```

### Failed

Selector evaluation was structurally invalid, such as a wildcard applied to a non-array value. HAK-014 v1 does not establish a trustworthy applicability count for this state, so HAK-016 records:

```text
coverage_state = Failed
applicable = null
```

```text
FailedEvaluation
!=
ZeroApplicableContexts
```

## Applicability model

HAK-016 v1 deliberately reuses the exact `matches` and `missing` counts already emitted by HAK-014 rather than introducing a second selector traversal implementation.

For non-failed optional results:

```text
applicable = matches + missing
```

Under `hak.selector-path` v1:

- a selected value contributes one match;
- a missing optional field on an existing traversal branch contributes one missing context;
- each wildcard element recursively contributes its descendant contexts;
- an empty wildcard array contributes zero contexts;
- structural selector/type failure is not converted into a numeric applicability claim.

The model identity is:

```text
HAKSelectorPathTerminalContextV1
```

The term `applicable` is therefore a selector-evaluation count under this exact model, not a universal claim about domain applicability.

## Deterministic derivation

The implementation:

```text
scripts/hak_selector_coverage.py
```

first validates HAK-014 receipt self-consistency and then deterministically maps every `optional_selector_results` entry into a HAK-016 result.

Strong validation additionally invokes HAK-014's deterministic input/replay join against the exact:

- raw response bytes;
- normalization policy;
- HAK-014 interpreter source snapshot;
- policy artifact reference;
- interpreter artifact reference;
- raw source reference;
- resource kind.

Only after that replay succeeds is the HAK-016 record accepted.

```text
CoverageRecordSelfConsistent
!=
CoverageRecordBoundToRealNormalizationInputs
```

and:

```text
ValidatedHAK014Replay
+
DeterministicCoverageDerivation
->
InputBoundCoverageRecord
```

## Coverage digest

HAK-016 is an early consumer of HAK-015 canonical evidence encoding.

The record names:

```text
canonicalization_profile = hak.canonical-json.v1
digest_domain = hak.selector-coverage.v1
```

and `coverage_digest` is computed through the HAK-015 profile over the record excluding its digest field.

```text
CoverageMeaning
!=
CoverageDigestIntegrity
```

HAK-015 canonicalization does not itself establish that the counts or coverage state are correct; HAK-016 deterministic derivation and the HAK-014 replay join establish that correspondence for the tested path.

## Source bindings

A coverage record binds:

- HAK-014 receipt schema;
- exact HAK-014 receipt digest;
- HAK-014 execution status;
- resource kind;
- normalization policy id/digest;
- exact optional selector result order and paths;
- HAK-016 coverage semantics/version;
- HAK-015 canonicalization profile;
- HAK-016 digest domain.

A redigested coverage record that changes `PartiallyPresent` to `Present`, `NotApplicable` to `Absent`, applicability counts, source receipt digest, or source status must still fail deterministic validation.

## Migration and future receipt versions

HAK-016 is intentionally adjacent to HAK-014 rather than silently changing `hak.normalization-execution-receipt.v1`.

A future normalization receipt v2 may embed these coverage semantics only through an explicit schema/version transition.

```text
AdjacentDerivedRecord
!=
SilentHistoricalReinterpretation
```

This lets existing HAK-014 evidence remain understandable forever while providing more precise semantics for new consumers.

## Qualification boundary

HAK-016 qualification is limited to:

- deterministic coverage-state derivation from validated HAK-014 optional counts;
- the stated cardinality/conservation invariants;
- historical source-status preservation;
- HAK-014 input/replay binding before strong acceptance;
- HAK-015 profile-bound coverage digest integrity;
- schema/linter consistency for the committed test cases.

It does not prove that GitHub/provider data is authentic, that a selected field is semantically meaningful, that an absent value has domain significance, or that any human/machine action is authorized.

## Non-claims

HAK-016 does not:

- modify HAK-014 receipts;
- turn optional absence into negative evidence outside the selector policy;
- authenticate provider observations;
- establish scientific truth;
- establish legal/governance legitimacy;
- grant runtime authority.

```text
CoveragePrecision
!=
SemanticTruth
!=
Authority
```
