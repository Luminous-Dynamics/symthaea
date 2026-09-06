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

This is deliberately conservative. Declarations with the same declarer id, version, and method share one interpretation root even when they interpret different independent evidence events.

If a declarer omits a version, internal changes do **not** automatically create a new root. That prevents unversioned producer drift from manufacturing apparent interpretive independence.

Changing declarer id, version, or method creates a distinct root identity, but:

```text
distinct root identity
        !=
independent interpretation
```

## Root-normalized graph

The lineage graph is normalized around unique interpretation roots rather than declaration pairs.

Conceptually:

```text
declaration A1 ─┐
declaration A2 ─┼── interpretation root A
                │
declaration A3 ─┘


declaration B1 ─┐
declaration B2 ─┼── interpretation root B
                └── ...

root A ───────── root B
       one pair assessment
```

If root A owns three declarations and root B owns four, the graph contains:

```text
7 declaration -> root mappings
2 unique interpretation roots
1 A <-> B root-pair assessment
```

not twelve repeated declaration-pair edges.

Same-root declarations are represented by the shared root identity. They do not create a synthetic `SameInterpretationRoot` pair edge.

Every distinct unordered interpretation-root pair appears exactly once. This is the topology later pairwise-independent-set reasoning is allowed to consume.

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
- every declaration id and current eligibility id;
- every declaration -> root mapping;
- the canonical unique root table;
- declarer id/version/method for each root;
- every distinct root-pair status exactly once;
- exact independence-qualification id where one was accepted.

The complete report receives a domain-separated BLAKE3 `lineage_id`.

Caller input order does not define identity: declaration mappings are ordered by declaration id, roots are ordered by root id, and pair qualifications are canonicalized as unordered root pairs.

`InterpretationLineageV1` is Serialize-only with private fields. Archived bytes are audit material and cannot be deserialized into a trusted current lineage report.

## What the statuses do not mean

`DistinctRootsIndependenceUnknown` does not mean the roots are dependent; it means independence has not been qualified.

`IndependenceQualified` does not mean either interpretation is correct.

Neither status is:

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

The number of declarations, roots, or qualified pair edges has **no disposition meaning by itself**.

The public lineage API intentionally exposes:

```text
roots()
root_pair_assessments()
```

but no `qualified_independent_pair_count()` helper.

A qualified edge count is not an independent-root-set witness. A later preregistered policy may require a set of roots where every distinct pair satisfies the required independence relation, but it may not silently convert candidate/module/root/edge counts into truth or confidence.

## Required adversarial cases

Qualification must prove at least:

- two declarations from the same declarer/version/method share one root;
- same-root declarations create no synthetic pair edge;
- multiple declarations on two roots create one unique root-pair assessment, not a Cartesian product of declaration pairs;
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
- no pair-count convenience API is exposed;
- no path enters belief, workspace, action, or promotion authority.
