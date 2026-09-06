# RCA Interpretation Lineage v1

Status: **shadow-only epistemic-governance contract**

RCA-003b.2 distinguishes two different causal questions:

```text
evidence-root independence
    !=
interpretation-root independence
```

The evidence-lineage layer answers whether observations share evidence roots. This layer answers whether the **evidence -> proposition judgments** share an interpretation root.

## Core theorem

```text
independent observations
        !=
independent interpretations
```

Ten independent observations interpreted by the same rule/model/version are ten evidence roots but one interpretation root.

## Inputs

`assemble_interpretation_lineage_v1(...)` accepts only:

```text
DispositionEligibleRelationDeclarationV1[]
        +
one exact ValidatedRelationDeclarationEligibilityContextV1
        +
zero or more RegisteredInterpretationIndependenceQualificationV1
```

Every eligible declaration must have been issued for the exact same eligibility-context commitment. Mixed proposition/use/time contexts fail closed.

## Interpretation-root identity

V1 derives an interpretation root from:

```text
declarer_id
+ optional declarer_version
+ declaration_method
+ interpretation-lineage profile identity
```

This is deliberately conservative.

The following declarations share one interpretation root:

```text
same declarer id
same declarer version
same method
```

even when they interpret different independent evidence events.

If a declarer omits a version, internal changes do **not** automatically create a new root. That prevents unversioned producer drift from manufacturing apparent interpretive independence.

Changing declarer id, version, or method creates a distinct root identity, but:

```text
distinct root identity
        !=
independent interpretation
```

## Fail-closed independence

For two distinct interpretation roots, the default status is:

`DistinctRootsIndependenceUnknown`

not `Independent`.

Only an exact current `RegisteredInterpretationIndependenceQualificationV1` for that unordered root pair may produce:

`IndependenceQualified`

The qualification binds:

- exact canonical root pair;
- exact proposition id;
- exact permitted use;
- qualifier id/version;
- evaluator id/version;
- qualification-policy digest;
- qualification-artifact digest;
- qualification start and finite expiry.

Root-pair input order is canonicalized before identity is derived.

## Anti-self-signing boundary

During lineage assembly, the independence qualifier id may not equal either interpretation-root declarer id.

This blocks the direct representation:

```text
root A owner
    -> declares A independent from B
```

as sufficient independence authority.

This inequality is only a structural anti-self-signing rule. It does **not** prove organizational, causal, ownership, model-training, data, or evaluator independence. Stronger independence claims belong in the qualification policy/artifact and later adversarial qualification.

## Currentness

A persistable interpretation-independence qualification does not establish current independence by itself.

Lineage assembly checks:

```text
qualified_at <= eligibility_context.now <= valid_until
```

A future or expired qualification fails the exact join. Omitting a qualification leaves distinct roots `IndependenceUnknown`.

## Issued lineage

`InterpretationLineageV1` binds:

- exact proposition id;
- exact eligibility-context commitment;
- every declaration id;
- every current eligibility id;
- every derived interpretation-root id;
- declarer id/version/method;
- every pair status;
- exact independence-qualification id where one was accepted.

The complete report receives a domain-separated BLAKE3 `lineage_id`.

Caller input order does not define identity; entries are ordered by declaration id and root-pair qualifications are canonicalized.

`InterpretationLineageV1` is Serialize-only with private fields. Archived bytes are audit material and cannot be deserialized into a trusted current lineage report.

## What the statuses do not mean

`SameInterpretationRoot` does not mean the relation declaration is false.

`DistinctRootsIndependenceUnknown` does not mean the roots are dependent; it means independence has not been qualified.

`IndependenceQualified` does not mean either interpretation is correct.

None of these statuses is:

- evidence-root independence;
- proposition truth;
- relation strength calibration;
- confidence/posterior probability;
- canonical evidence admission;
- canonical belief;
- workspace/GWT authority;
- action authority;
- recursive-improvement promotion authority.

## No count voting

The number of declarations, roots, or qualified pairs has **no disposition meaning by itself**.

A later preregistered disposition policy may require particular root structures, but it may not silently convert module/candidate/pair counts into truth or confidence.

## Required adversarial cases

Qualification must prove at least:

- two declarations from the same declarer/version/method share one root;
- different declarer names default to `IndependenceUnknown`;
- exact current pair qualification is required for `IndependenceQualified`;
- root-pair order does not change qualification identity;
- the same root cannot be qualified independent from itself;
- either root owner cannot directly self-qualify pair independence;
- mixed eligibility contexts fail closed;
- expired/future independence qualification fails closed;
- persisted registered qualifications revalidate;
- issued lineage cannot deserialize;
- lineage identity is input-order independent and changes with accepted qualification state;
- no path enters belief, workspace, action, or promotion authority.
