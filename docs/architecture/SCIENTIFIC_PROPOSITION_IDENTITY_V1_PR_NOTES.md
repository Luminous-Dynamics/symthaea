# Scientific Proposition Identity v1 — review notes

This companion note narrows review of `SCIENTIFIC_PROPOSITION_IDENTITY_V1.md`.

## Exact review question

Does the contract define an immutable scientific target strongly enough that evidence can be attached to one exact proposition, while avoiding the opposite error of baking every measurement, falsifier, estimator, or scoring choice into proposition identity?

## Primary non-equivalence theorem

```text
proposition family
    != proposition semantic identity
    != measurement binding
    != assessment / falsifier contract
    != estimator / model implementation
    != evidence contribution
    != disposition
```

## Critical correction

The preceding disposition-graph document used an intentionally conservative illustrative revision rule. This tranche refines it:

- semantic claim changes create a new proposition identity;
- assessment-method changes normally create new assessment/evidence lineage while retaining the same proposition target;
- a domain may opt to make an operational definition constitutive of proposition semantics, but the shared kernel cannot assume that universally.

This correction is required so multiple methods can test one exact proposition without semantic laundering.

## Suggested implementation theorem

A future implementation should make the shared envelope small:

```text
profile identity
+ domain identity
+ domain semantic-schema identity
+ domain canonical semantic payload digest
    -> ScientificPropositionIdentityV1
```

The domain, not the shared kernel, owns the semantic payload vocabulary.

## Do not approve if

- human prose or title defines identity;
- JSON/serde/debug bytes define identity;
- measurement source is universally baked into proposition identity;
- falsifier or estimator identity is universally baked into proposition identity;
- family membership transfers evidence;
- proposition revision rewrites the old identity;
- compatibility is inferred from textual similarity;
- a truth/disposition field appears on the proposition object;
- legacy opaque IDs are automatically promoted to semantic proposition IDs.

## Deliberate non-scope

No Rust implementation, registry, compatibility engine, argument graph, database projection, evidence migration, or authority transfer is included here.
