# PIE-002I Evidence-to-Study Applicability Oracle

Status: independent reference semantics; synthetic fixtures only.

## Purpose

PIE-002F proves that all nine external utility-context facts carry explicit validated provenance. PIE-002I freezes the next distinct proposition: whether one exact evidence item is declared applicable to one exact study scope under one explicit applicability profile.

This theorem is deliberately per-evidence rather than a scalar field-level trust score.

```text
provenance exists
    != evidence applicable

one applicable source
    != every attached source applicable

strong evidence class
    != scope match
```

## Inputs

The oracle separates three authorities:

1. `UtilityStudyScope` — the exact context being studied;
2. `EvidenceScope` — what one evidence item explicitly claims to cover;
3. `ApplicabilityProfile` — the dimensions that are material for this particular applicability decision.

There are no hidden material dimensions. The caller must state the profile explicitly.

V1 understands these scope dimensions:

- exact PIE-002H-A process-content SHA-256 when process-specific;
- celestial body;
- site;
- plant configuration;
- supply configuration;
- storage configuration;
- recovery configuration;
- operating mode;
- environment profile;
- temporal basis.

The dimension set is intentionally finite and explicit. It is not a claim that all industrial applicability dimensions are already modeled.

## Evidence scope claims

For each dimension, evidence can state exactly one of:

- `Exact(value)` — evidence declares support only for that exact scope value;
- `General` — evidence explicitly declares generality across that dimension;
- `Unknown` — the dimension is unresolved.

Omitting a dimension is equivalent to `Unknown`; omission is never interpreted as a wildcard.

`General` must be explicit. It cannot carry a hidden exact value.

## Resolution rule

For every material profile dimension:

- missing study value -> unresolved;
- missing/unknown evidence claim -> unresolved;
- explicit `General` -> dimension satisfied as declared generalization;
- exact value equality -> exact match;
- exact value mismatch -> hard mismatch.

Final status is:

```text
any hard mismatch            -> Inapplicable
else any unresolved dimension -> Indeterminate
else                           -> Applicable
```

A hard mismatch remains decisive even if other material dimensions are unresolved.

The receipt preserves exact matches, explicit generalizations, mismatches, and unresolved dimensions separately.

## Process-content binding

When process content is material, the study/evidence values are lowercase 64-hex SHA-256 commitments compatible with the qualified PIE-002H-A content-identity theorem.

A friendly `process_id`, revision label, timestamp, or caller assertion cannot substitute for that content identity.

```text
same process label
+ changed ProcessDefinition content
!= inherited applicability
```

Process-independent infrastructure evidence is not forced to invent a process identity. A site-level applicability profile can omit `process_content_sha256` entirely when process content is genuinely not material.

## Per-evidence preservation

`resolve_field_evidence` returns one `EvidenceApplicabilityReceipt` per input evidence item. It does not collapse the list into a scalar trusted/untrusted result.

Therefore the same `evidence_id` reused across two PIE-002F fields can resolve differently when its declared scope differs by fact. One applicable occurrence does not launder an inapplicable occurrence into applicability or into independent corroboration.

## Evidence class boundary

Evidence class is preserved exactly in the receipt but does not alter scope matching.

- `Qualified` evidence with a hard scope mismatch remains `Inapplicable`.
- `Hypothesis` evidence with matching declared scope may be `Applicable`, but remains `Hypothesis`.

Applicability is about declared scope relationship, not epistemic strength or truth.

## Numerical boundary

The applicability resolver intentionally accepts no numerical value/range argument. Numerical similarity, conservatism, or overlap cannot override a scope mismatch.

PIE-002F/002D remain responsible for provenance-bearing numerical validity/binding. PIE-002I does not re-evaluate their arithmetic.

## Temporal boundary

`temporal_basis` is an exact declared scope token, not a trusted clock.

A historical evidence basis that differs from the study basis is a scope mismatch when temporal basis is material. A matching declared temporal basis still does not establish that the evidence is fresh/current *now*.

```text
matching declared temporal basis
!= trusted current time
!= freshness
```

## Lexical-identity boundary

This oracle does not mint the general durable lexical-ID profile. Evidence IDs and non-process scope tokens are treated as exact strings supplied by the fixture. Canonical lexical admission belongs to PIE-ID-001 (#2870 / PR #2956) and should be reused by the production mirror once qualified.

The process-content dimension is stricter here because its representation is already frozen by PIE-002H-A: lowercase 64-hex SHA-256.

## Adversarial fixtures

The self-test covers:

1. exact matching process/body/site/configuration/mode/environment/time -> `Applicable`;
2. changed process-content commitment -> `Inapplicable` even with stronger evidence class;
3. same body but different site -> `Inapplicable`;
4. same site but different storage configuration -> `Inapplicable`;
5. missing required site dimension -> `Indeterminate`;
6. explicit `General` body/site claims satisfy those material dimensions;
7. omitted body/site does not become wildcard and remains `Indeterminate`;
8. `Qualified` scope mismatch stays `Inapplicable`;
9. matching `Hypothesis` remains `Hypothesis` while resolving `Applicable`;
10. no numerical input exists that could override scope mismatch;
11. shared evidence identity across two facts receives separate applicability receipts;
12. historical temporal-basis mismatch remains a mismatch rather than currentness;
13. process-independent site infrastructure can use a profile with no fake process identity;
14. process-specific evidence cannot be rebound to changed process content by caller assertion;
15. a hard mismatch dominates simultaneous unresolved dimensions;
16. duplicate material dimensions fail closed;
17. an explicit `General` claim cannot hide an exact value;
18. malformed process-content digests fail closed.

## Relationship to the PIE stack

Preferred future path:

```text
qualified process content                 PIE-002H-A
+ evidence-bearing external facts         PIE-002F
+ explicit study/evidence/profile scopes
        -> per-evidence applicability      PIE-002I
        -> numerical binding               PIE-002D
        -> opaque authoritative witness    PIE-002G
```

PIE-002E supplies fresh invocation-bound process projection on the demand side. PIE-002H-B authority-currentness remains separate from applicability.

## Deliberate non-claims

PIE-002I does not establish:

- truth or factual correctness;
- source authenticity;
- source independence/common-cause diversity;
- calibration validity;
- numerical uncertainty adequacy;
- freshness or trusted current time;
- persistence/process authority-currentness;
- global uniqueness or lexical canonicality of general scope IDs;
- electrical utility feasibility;
- storage dispatch/load profile;
- thermodynamic closure;
- equipment qualification;
- economics;
- deployment or execution authority.

Tracks #2935, #2785, #2867, #2870, #2826, #2764, #2782, #1610, #1647, and master #1604.
