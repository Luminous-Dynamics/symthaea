# Symthaea RCA Shadow Disposition Policy

RCA-003b.3 freezes **policy before interpretation**.

This crate registers the policy surface a future pure shadow-disposition engine may consume. It does not evaluate a case and does not emit `Supported`, `Contested`, `Defeated`, or any other disposition.

## Boundary

```text
BoundShadowEvidenceCaseV1
        +
IndependentEvidenceSetWitnessV1
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

## Exact scope/profile binding

V1 policy is registered for one exact opaque proposition id. A digest is identity, not semantic class.

Policy registration binds the exact semantic profile digests of:

- `BoundShadowEvidenceCaseV1`;
- `IndependentEvidenceSetWitnessV1`;
- current relation-declaration eligibility;
- `InterpretationLineageV1`.

If a lower-layer contract changes, the old policy no longer registers against the new stack.

## Evidence items are not ancestry roots

The evidence-side requirement is now explicitly:

```text
minimum N pairwise-independent EVIDENCE ITEMS
```

not:

```text
minimum N ancestry roots
```

A single derived evidence item may inherit several roots. Those roots preserve provenance but do not create several confirmations.

V1 freezes:

`EvidenceSetSemanticsV1::IssuedPairwiseIndependentItems`

A future engine may satisfy an evidence threshold only using the selected item set in an issued `IndependentEvidenceSetWitnessV1` bound to the current witness profile.

Therefore these do not satisfy an evidence-item threshold by themselves:

- N ancestry roots inside one derived item;
- N candidate/module ids;
- N relation declarations;
- N pair edges;
- N high-strength relations.

## Interpretation requirements remain root sets

Interpretation independence is structurally different. Each relation declaration maps to one canonical interpretation root, and #527 exposes a normalized unique-root graph.

V1 freezes:

`InterpretationRootSetSemanticsV1::PairwiseIndependentRoots`

A threshold N means there exists a set of at least N interpretation roots where every distinct root pair is `IndependenceQualified` in the exact current `InterpretationLineageV1`.

For four interpretation roots, all six root-pair relationships must be qualified independent.

So the policy intentionally distinguishes:

```text
evidence:       independent ITEM set witness
interpretation: independent ROOT set witness/topology
```

## Outcome topology requirements

Each outcome preregisters:

```text
min_pairwise_independent_evidence_items
min_pairwise_independent_interpretation_roots
```

for tentative support, support, tentative opposition, opposition, defeaters, and each surviving side of a contested result.

`Supported` may not be weaker than `TentativelySupported`; `Opposed` may not be weaker than `TentativelyOpposed`. Every V1 outcome requires at least one evidence item and one interpretation root.

## Relation strength is diagnostic only

V1 exposes only `RelationStrengthTreatmentV1::DiagnosticOnly`.

A later engine may not sum, average, normalize, multiply, vote on, Bayesian-update, or treat `strength_ppm` as calibrated probability/confidence.

## Defeaters and unknown independence

V1 exposes only:

- `DefeaterModeV1::QualifiedCurrentBlocker`;
- `UnknownInterpretationIndependenceModeV1::ForceUnderdetermined`.

A stale/unqualified defeater cannot veto support. Unknown interpretation independence never becomes a weak vote.

## Contestation

V1 requires `contested_requires_qualified_support_and_opposition = true`.

Qualified disagreement remains disagreement instead of collapsing into one scalar score.

## Canonical preregistration binding

RCA already has `RegisteredExperimentContractV1`. Its `contract_digest()` transitively commits hypothesis, baseline/candidate identity, development/held-out corpora, evaluator, metrics, thresholds, seed plan, falsification criteria, allowed outcomes, and experiment resource ceilings.

The policy therefore binds only:

```text
EXPERIMENT_CONTRACT_SCHEMA_VERSION
+
RegisteredExperimentContractV1.contract_digest()
```

and does not duplicate those fields.

## Resource feasibility

Policy identity binds:

- `max_case_items`;
- `max_interpretation_pairs`.

Registration requires:

```text
max_case_items >= largest independent-evidence-item requirement
```

and, for the largest interpretation-root requirement N:

```text
max_interpretation_pairs >= N * (N - 1) / 2
```

so the policy budget can structurally witness its own thresholds.

## Persistence

`RegisteredShadowDispositionPolicyV1` is persistable only because deserialization revalidates the raw policy, all exact lower-layer profiles, the RCA experiment-contract schema/reference, resource feasibility, and the complete BLAKE3 policy identity.

Tampering fails closed.

## Non-scope

This crate intentionally has no:

- bound-case or evidence-witness evaluation input;
- `evaluate` / `decide` / `dispose` method;
- disposition enum;
- reason-trace generator;
- belief admission;
- workspace integration;
- action path;
- self-improvement promotion path.
