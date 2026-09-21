# RES-OSINT-SEC-001T — Investigation diagnostics redaction corpus v0.1

## Purpose

Freeze the diagnostic-safety theorem discovered during pre-qualification review of RES-OSINT-001A / PR #5421 before modifying product Rust.

Exact parent product:

`4765fd64a56fc9e824ba91b8a4bfdcc07ea9ae9b`

This subject is documentation/test-vector only. It does not repair the product by itself.

## Core theorem

```text
valid investigation reference
!= safe ordinary diagnostic payload

exact value required for binding/comparison
!= exact value should appear in Debug/Display/logging
```

Investigation metadata can reveal target interest, purpose, hypotheses, search/tool choices, policy state, or evidence-frontier identities even when no captured source content is present.

## Required product behavior

Every `role_ref!` family must eventually provide secret-safe ordinary diagnostics:

```text
Debug -> RoleName(<redacted>)
explicit accessor -> exact value
```

The exact value remains semantically load-bearing for equality, ordering, hashing, canonical binding, bridge conversion, and explicit access. Redaction is a presentation rule only.

### Error diagnostics

Identifier-bearing error variants must not echo exact payload strings through either `Debug` or `Display`:

- `DuplicateHypothesisId`;
- `DuplicateDiscriminatorHypothesis`;
- `UnknownHypothesis`;
- `DuplicateAssumptionAssessmentId`.

`InvalidReference` may expose only the reference role and structural rejection reason; it must never echo the rejected raw value.

`ZeroSearchBound` may name the non-sensitive structural field (`max_results` / `max_pages`).

## Nested diagnostic propagation

Redacting leaf reference types is necessary but not sufficient unless derived container diagnostics remain safe.

The frozen corpus therefore requires ordinary `Debug` for nested structures such as:

- `InvestigationManifestV1`;
- `AssumptionLedgerV1`;
- `DiscriminatingObservationCandidateV1`;
- `SearchPlanCandidateV1`;
- `HypothesisAssessmentCandidateV1`;
- `InvestigationCandidateBundleV1`;

not to reveal the exact secret fixture through their child fields.

```text
leaf redaction
+ derived container Debug
-> exact role values remain hidden
```

## Exact fixture

Schema:

`symthaea:res-osint-sec-001-redaction-corpus:v0.1`

Profile:

`symthaea:investigation-diagnostics-redaction:v0.1`

Authority:

`DiagnosticSafetyTestOnly`

Secret fixture:

`quiet-investigation-target-7f3c91`

Exact compact UTF-8 JSON SHA-256:

`d171e9fec621cd4667bd3a5a591e3ade398ffe5f84a9893b6344b6dc04299e96`

The digest identifies authored fixture bytes only.

## Metamorphic requirements

1. Replacing the secret fixture with another valid value must not change the redacted diagnostic shape or expose either value.
2. Using the same exact string in two different role types must preserve the role distinction in diagnostics while hiding the shared value.
3. Embedding sensitive refs inside manifest/search-plan/bundle containers must not make the exact value reappear through derived `Debug`.
4. Calling the explicit exact-value accessor must still return exact bytes.
5. Equality, ordering and hashing must continue to operate on exact values, not redacted renderings.

## Authority and OPSEC ceiling

Passing this corpus may establish only the ordinary `Debug`/`Display` redaction behavior exercised by the exact product/profile.

It does **not** establish:

- safe telemetry transport;
- safe persistent logging;
- safe crash dumps or panic payloads;
- side-channel resistance;
- encrypted storage;
- anonymity;
- privacy/legal compliance;
- source truth;
- collection or action authority.

```text
redacted Debug
!= encrypted
!= anonymous
!= safe to publish
```

## Product-refreeze discipline

The initial hand-authored patch construction was intentionally discarded after exact diff review showed it had been represented as patch text rather than a full replacement source file. The invalid construction is not a review subject and transfers no evidence.

The eventual Rust repair must be rebuilt from the full exact 001A source, independently execute against this corpus, and then be refrozen as a fresh product lineage before 001A final qualification.

Because RES-OSINT-001B depends on 001A, 001B must likewise be mechanically rebuilt/refrozen after the repaired 001A parent is established.

## State

**FROZEN DIAGNOSTIC-SAFETY CORPUS / PRODUCT REPAIR NOT YET QUALIFIED / NOT PASS.**
