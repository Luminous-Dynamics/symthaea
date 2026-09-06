# RCA Relation Declarer Qualification v1

RCA-003b.1 separates **relation provenance** from **permission to use that relation in shadow disposition**.

A `BoundEvidenceRelationDeclarationV1` proves the identity of a declaration and its declarer/provenance. It does not prove that the declarer is reliable or qualified for any epistemic use.

## Qualification record

`RelationDeclarerQualificationV1` is intentionally narrow. V1 binds:

- exact subject declarer id;
- optional subject declarer version;
- exact declaration method;
- qualifier id/version;
- evaluator id/version;
- one exact proposition id;
- a canonical unique set of allowed proposition-relation kinds;
- exact permitted use = `ShadowRuntimeDisposition`;
- qualification-policy digest;
- qualification-artifact digest;
- qualification start time;
- finite validity end time.

`Supersedes` is not eligible for this v1 use because it is an evidence-targeting relation, while shadow runtime disposition operates over proposition-targeting declarations.

Validation sorts the allowed relation-kind set into a stable semantic order. Caller ordering therefore does not create different qualification identities.

## Structural anti-self-signing rule

V1 requires:

```text
qualifier_id != subject_declarer_id
```

This prevents the most direct self-qualification representation.

It is **not proof of organizational, causal, or model independence**. Two identifiers may still refer to correlated or controlled processes. Interpretation-lineage and stronger qualifier-independence policy remain RCA-003b.2/003b.3 concerns.

## Qualification identity

A registered qualification receives a domain-separated BLAKE3 identity over the complete normalized record:

```text
qualification_id = H(
    qualification profile contract,
    schema,
    subject declarer/version/method,
    qualifier id/version,
    evaluator id/version,
    exact proposition id,
    canonical allowed relation-kind set,
    permitted use,
    qualification policy digest,
    qualification artifact digest,
    qualified-at time,
    valid-until time
)
```

JSON bytes, debug formatting, Rust `Hash`, enum discriminants, and caller set ordering do not define identity.

`RegisteredRelationDeclarerQualificationV1` is persistable because deserialization revalidates and canonicalizes the record and recomputes the derived identity.

## Eligibility is a separate issued boundary

A registered qualification still does not make a relation declaration eligible.

Eligibility requires an exact join:

```text
BoundEvidenceRelationDeclarationV1
        +
RegisteredRelationDeclarerQualificationV1
        +
ValidatedRelationDeclarationEligibilityContextV1
        ↓
exact declarer/version/method match
exact proposition match
exact allowed relation-kind match
exact permitted-use match
qualified_at <= now <= valid_until
        ↓
DispositionEligibleRelationDeclarationV1
```

The eligibility context explicitly commits:

- exact proposition id;
- exact use;
- explicit evaluation time.

The resulting `eligibility_id` binds the declaration id, qualification id, eligibility profile, and exact context commitment.

## Persistence boundary

`DispositionEligibleRelationDeclarationV1` is private-field and deliberately has no `Deserialize` implementation.

It may serialize for audit. Archived bytes do not recreate current eligibility; callers must reload/revalidate the declaration and registered qualification and rerun the exact eligibility join against a current explicit context.

## Core theorem

```text
relation provenance
        !=
registered declarer qualification
        !=
current use eligibility
        !=
shadow disposition
        !=
canonical belief
        !=
action authority
        !=
self-improvement promotion
```

## Non-claims

RCA-003b.1 does not establish:

- independence between qualifier and declarer beyond the direct id inequality check;
- interpretation-root independence;
- universal proposition semantics;
- calibrated relation-strength semantics;
- a shadow disposition policy;
- canonical epistemic admission;
- action or promotion authority.

Those remain separate later boundaries.
