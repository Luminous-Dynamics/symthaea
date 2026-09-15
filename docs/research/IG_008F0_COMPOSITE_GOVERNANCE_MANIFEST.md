# IG-008F0 — Composite governance manifest with explicit coverage ceiling

Issue: #3287

## Purpose

IG-008F0 composes already-independent Mycelix mechanism observations without promoting their authority.

The first composite binds two source-observed slices:

1. voting / tally / delegation / voting-admission semantics;
2. timelock / readiness / signature-fallback / execution-dispatch semantics.

Those slices refer to the same frozen production subject, but they do **not** cover every authority transition required to describe Mycelix governance end-to-end.

Therefore the manifest authority is:

`ObservedCompositeSlice`

and its claim ceiling is:

`ObservedCompositeSliceOnly`.

## Frozen composite

Manifest:

`docs/research/manifests/mycelix-observed-composite-fca2c107-v1.json`

Identity:

```text
manifest_id    mycelix-observed-composite-fca2c107-v1
revision       1
production     fca2c107a1ea5108823ce617ba4111b6f7f77230
authority      ObservedCompositeSlice
claim ceiling  ObservedCompositeSliceOnly
SHA-256        366b87944e74fe01369b78a8a7ea1c34f6dee05237583a47d86e5999b93b150c
```

The SHA-256 is deterministic content identity, not a governance signature or proof of correctness.

## Component references

### Voting component

```text
profile        mycelix-voting-observed-fca2c107-v2
profile SHA    680af4668889c299b7e0d74531f44894a64a384be71778bca54d3f21ca80ac01
corpus SHA     ee5e7649a773f564b443320689f465080d4641a0f4f09f13c0c49a7087d6dc10
Mycelix head   4b4e27910a98ad393e5a508e54e0f73c48c19107
Symthaea PR    #3233
```

### Execution component

```text
profile        mycelix-execution-observed-fca2c107-v1
profile SHA    c977bdcef9e5faac83351050999451432b618d5cc523bece804eba5dd1ae81f6
corpus SHA     0c6669e44d6d18396ede43324f5cf3abbb25ddd3a2a9f59abb2c8a3699ba5fd4
Mycelix head   197714209c60503f0fba4143409da383bc9cbf83
Symthaea PR    #3254
```

Both observed components bind production subject:

`fca2c107a1ea5108823ce617ba4111b6f7f77230`.

Evidence heads are intentionally allowed to differ: they are immutable research lineages observing the same production subject at different levels.

## Coverage model

The manifest explicitly records what is covered and what is not.

Covered stages currently include the observed semantics of voting weighting/tally, delegation, voting admission, timelock construction, readiness transition, execution signature fallback, and executable dispatch.

The following remain outside this composite as independently qualified mechanism stages:

- proposal creation/lifecycle;
- threshold-signing authority as its own mechanism profile;
- downstream constitutional parameter authorization;
- downstream treasury/credit authorization;
- deployment currentness.

The uncovered set is part of manifest identity. Removing an uncovered stage without adding evidence changes the content commitment and fails the frozen validator.

## Why this is not `ObservedEndToEnd`

A composed model can be internally consistent while still omit a decisive authority boundary.

For example, voting and execution can both be represented faithfully while the model still lacks an independent account of who may create/activate proposals or what exact threshold-signing authority is valid.

IG-008F0 therefore makes the following inference invalid:

```text
voting conformance + execution conformance
=> observed end-to-end governance
```

The only justified inference is:

```text
voting conformance + execution conformance
=> observed composite slice with explicit missing coverage
```

## Validator

`scripts/ig008f0_validate_composite_manifest.py` fails closed on:

- component profile or corpus commitment drift;
- component evidence-head drift;
- mixed production subjects;
- component authority promotion;
- conformance-class drift;
- removal of required uncovered stages;
- empty uncovered-stage lists for a slice claim;
- promotion to `ObservedEndToEnd`;
- generic safety/fairness/legitimacy verdict fields;
- manifest content-commitment mismatch.

Its self-test also proves that semantic mutations change manifest identity.

## Exact-head composition qualification

The qualification workflow does more than parse the manifest.

It checks out:

- the exact Symthaea PR subject;
- exact Mycelix voting evidence head `4b4e27910...`;
- exact Mycelix execution evidence head `197714209...`.

It then:

1. verifies exact Git subjects and source blobs;
2. reruns the Mycelix voting profile/counterexample validators;
3. reruns the independent Symthaea voting oracle and requires byte-identical corpus output;
4. reruns the Mycelix execution profile/counterexample validators;
5. reruns the independent Symthaea execution oracle and requires byte-identical corpus output;
6. validates the composite manifest;
7. verifies the manifest references exactly those qualified component commitments and the same production subject;
8. verifies all checkouts remain immutable.

A hosted PASS would establish only that the composite references independently reproduced frozen observations consistently.

## Experiment contract

Future institutional-lab experiment identities should bind a single composite manifest reference:

```text
CompositeMechanismManifestRef {
    manifest_id,
    revision,
    manifest_content_sha256,
}
```

rather than an unstructured list of mechanism labels.

Experiments may deliberately substitute successor or idealized components, but such a manifest must use an explicit hybrid/successor authority class and cannot inherit `ObservedCompositeSlice` merely because one component is observational.

## Path toward genuine end-to-end coverage

`ObservedEndToEnd` should become available only after a separately versioned required-stage registry has no unresolved required stages and every stage is bound to appropriate evidence.

That future transition must create a new manifest revision/content commitment. It must not edit this historical manifest in place.

## Non-claims

IG-008F0 does not establish deployment currentness, complete Mycelix governance coverage, live exploitability, fairness, constitutional legitimacy, or governance safety.
