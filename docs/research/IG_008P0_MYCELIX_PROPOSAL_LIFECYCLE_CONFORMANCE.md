# IG-008P0 — Mycelix legacy Proposal lifecycle cross-repository conformance

Issue: #3315

Parent: IG-008F1 / draft #3302

## Purpose

IG-008P0 independently reconstructs the frozen legacy Mycelix Proposal lifecycle counterexamples before proposal lifecycle can become a represented component of the composite governance manifest.

It targets **current legacy production semantics**, not the stronger draft successor authority architecture in Mycelix #44/#59/#63+.

## Frozen Mycelix evidence

Exact evidence head:

`6ddca81103c52408421e2e31da4b4dee0c0b2762`

Observed profile:

```text
id             mycelix-proposal-lifecycle-observed-fca2c107-v1
profile SHA    7f42e2a8df25df94112d23f261d1f3ffe299d46d37cb3a5a6fe02aca0aa6c108
authority      ObservedSourceBound
```

Counterexample corpus:

`13eaaa988c73d29d67bccf7381f6f72cabd4eb7be090f36f0b444978cc708324`

Source blobs:

```text
proposals coordinator eb8358353ee259ef9c3b46617a61d3439f1c714c
proposals integrity   986bc0526aec8d37436efbe5ba798bc41705e3cf
```

Semantic production subject:

`fca2c107a1ea5108823ce617ba4111b6f7f77230`

Same-tree current main used to author the evidence:

`31ede2365b81365bb119cd9351b2739119974130`.

The two history identities are retained separately. Source-tree equivalence does not erase lineage identity.

## Independent oracle

`scripts/ig008p0_mycelix_proposal_lifecycle_oracle.py` is stdlib-only and imports none of Mycelix's P0/P1 Python or Proposal Rust implementation.

It independently validates the source-bound profile facts necessary to reconstruct exactly:

- CE-PROP-01 — primary ProposalById read can remain linked to Draft while an update child exists; authoritative currentness remains unestablished;
- CE-PROP-02 — Draft→Active content mutation is not rejected by the observed pure update check;
- CE-PROP-03 — update action author is absent from the integrity theorem;
- CE-PROP-04 — voting/creation/update temporal fields are not bound at update validation;
- CE-PROP-05 — sibling lifecycle children have no explicit deterministic authoritative-child projection.

The emitted canonical corpus must be byte-identical to Mycelix P1.

## Qualification

The exact-head workflow:

1. checks out the exact Symthaea product head;
2. checks out Mycelix at exact P1 evidence head `6ddca811...`;
3. binds the exact Proposal coordinator/integrity source blobs;
4. syntax-compiles Mycelix P0/P1 and independent Symthaea oracle;
5. revalidates Mycelix P0;
6. runs Mycelix P1 twice deterministically and emits its canonical corpus;
7. runs the Symthaea oracle twice deterministically and emits its corpus;
8. requires byte-identical Mycelix/Symthaea corpus bytes;
9. asserts exact profile/corpus commitments, issue #66, production subject, same-tree evidence head and authority ceilings;
10. verifies both checkouts remain immutable.

## Migration boundary

This evidence must not absorb the draft successor stack.

A later experiment can compare:

```text
LegacyObservedProposalLifecycle
vs
SuccessorQualifiedProposalAuthority
```

but that is a mechanism migration experiment, not a reinterpretation of legacy production.

## Composite consequence

After this cross-implementation conformance is qualified, a new composite manifest revision may add `ProposalLifecycle` and remove only:

`proposal_creation_and_lifecycle_as_independent_profile`

from the uncovered set.

The component remains gap-bearing: coverage means represented, not authoritative-current, safe, or correct.

## Non-claims

IG-008P0 establishes no live unauthorized Proposal update, deployment exploit, authoritative current lifecycle, successor-stack deployment, fairness, constitutional legitimacy, or governance safety.
