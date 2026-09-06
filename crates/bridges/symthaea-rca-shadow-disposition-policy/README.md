# Symthaea RCA Shadow Disposition Policy

RCA-003b.3 freezes **policy before interpretation**.

This crate registers the complete policy surface a future pure shadow-disposition engine may consume. It does not evaluate a case and it does not emit `Supported`, `Contested`, `Defeated`, or any other disposition.

## Boundary

```text
BoundShadowEvidenceCaseV1
        +
current eligible relation declarations
        +
InterpretationLineageV1
        +
RegisteredShadowDispositionPolicyV1
        ↓
[future pure shadow engine]
```

This crate implements only the final input above.

```text
RegisteredShadowDispositionPolicyV1
        != case evaluation
        != ShadowDispositionV1
        != canonical belief
        != GWT/workspace authority
        != action authority
        != self-improvement promotion
```

## Exact scope

V1 policy is registered for one exact opaque proposition id. A digest is identity, not semantic class. No policy can silently generalize to another proposition.

Policy registration also binds the exact semantic profile digests of:

- `BoundShadowEvidenceCaseV1`;
- current relation-declaration eligibility;
- `InterpretationLineageV1`.

If a lower-layer contract changes, the old policy no longer registers against the new stack. A new policy/qualification lineage is required.

## Pairwise-independent root sets

A central rule is:

```text
minimum N roots
    means
there exists a set S with |S| >= N
and every distinct pair in S is established independent
```

This applies independently to evidence roots and interpretation roots.

Therefore none of these satisfy a root threshold by themselves:

- N candidate items;
- N modules;
- N different ids;
- N pair edges distributed across a larger graph;
- N relation declarations;
- N high-strength declarations.

For interpretation roots, a required set of four pairwise-independent roots needs all six pair relationships to be qualified independent.

V1 freezes this as `RootSetSemanticsV1::PairwiseIndependentSet`.

## Relation strength is diagnostic only

V1 exposes only:

`RelationStrengthTreatmentV1::DiagnosticOnly`

A later engine therefore may not derive policy behavior by:

- summing strengths;
- averaging strengths;
- normalizing strengths;
- multiplying strengths;
- majority voting;
- Bayesian updating;
- treating `strength_ppm` as calibrated confidence or probability.

Any future calibrated arithmetic semantics require a new registered policy contract and qualification evidence.

## Outcome root requirements

The policy separately preregisters root-set requirements for:

- tentative support;
- support;
- tentative opposition;
- opposition;
- defeaters;
- each surviving side of a contested result.

`Supported` requirements may not be weaker than `TentativelySupported`; `Opposed` may not be weaker than `TentativelyOpposed`.

All V1 outcome requirements must require at least one evidence root and one interpretation root. This intentionally keeps a bare declaration from satisfying an outcome.

## Defeaters and unknown independence

V1 exposes only:

- `DefeaterModeV1::QualifiedCurrentBlocker`;
- `UnknownInterpretationIndependenceModeV1::ForceUnderdetermined`.

A stale/unqualified defeater cannot veto a support case. Distinct interpretation roots whose independence is not qualified cannot be silently counted as independent.

## Contestation

V1 requires `contested_requires_qualified_support_and_opposition = true`.

Contested cases preserve surviving qualified disagreement rather than collapsing it into one scalar score.

## Preregistration/evaluation binding

Policy identity binds:

- preregistration-contract digest;
- evaluation-corpus digest;
- seed-plan digest;
- metric-contract digest;
- evaluator id/version.

Changing any of these creates a new policy identity.

This is designed to prevent:

```text
inspect result
→ move threshold
→ reinterpret result
→ keep same policy identity
```

## Resource feasibility

The policy also binds:

- `max_case_items`;
- `max_interpretation_pairs`.

Registration rejects an evidence-root requirement larger than `max_case_items`.

For the largest required interpretation-root set N, registration requires:

```text
max_interpretation_pairs >= N * (N - 1) / 2
```

so the declared resource ceiling can actually witness every pair required by the pairwise-independent-set semantics.

## Persistence

`RegisteredShadowDispositionPolicyV1` is persistable only because deserialization:

1. revalidates the raw policy;
2. rechecks current lower-layer profile identities;
3. recomputes the policy contract digest;
4. recomputes the complete BLAKE3 `policy_id`.

Tampering fails closed.

## Non-scope

This crate intentionally has no:

- case input parameter;
- `evaluate`/`decide`/`dispose` method;
- disposition enum;
- reason-trace generator;
- belief admission;
- workspace integration;
- action path;
- self-improvement promotion path.

Those belong to later tranches only after this policy contract and all prerequisites qualify.