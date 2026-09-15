# IG-008F2 — Composite governance manifest v3

Issue: #3327

Parent: IG-008P0 / draft #3326

## Purpose

IG-008F2 advances the observed Mycelix governance composite from three independently represented stages to four by adding the frozen **legacy ProposalLifecycle** component.

It does not rewrite v1 or v2 and does not reinterpret the stronger draft successor governance architecture as current production.

## Manifest identity

```text
id             mycelix-observed-composite-fca2c107-v3
revision       3
production     fca2c107a1ea5108823ce617ba4111b6f7f77230
authority      ObservedCompositeSlice
claim ceiling  ObservedCompositeSliceOnly
SHA-256        3acdbd29d0fd631b863fa10a10eed0877af2336536578453acdc83f095632baf
```

Exact predecessor:

```text
mycelix-observed-composite-fca2c107-v2
e3b42dbd7dacaa2f44de8bc330e85ff7d624f291d27c4a303fbaccf57091e541
```

The v3 validator revalidates v2 against exact v1 before accepting v3, producing a checked lineage:

```text
v1 366b8794...
  -> v2 e3b42dbd...
      -> v3 3acdbd29...
```

## Coverage definition

The frozen rule remains:

`SourceObservedMechanismRepresentedNotPropertySatisfied`.

Coverage is epistemic representation, not a correctness theorem.

A gap-bearing component can therefore become covered precisely because its current behavior and counterexamples have been bounded accurately.

## New component — ProposalLifecycle

```text
profile            mycelix-proposal-lifecycle-observed-fca2c107-v1
profile SHA        7f42e2a8df25df94112d23f261d1f3ffe299d46d37cb3a5a6fe02aca0aa6c108
corpus SHA         13eaaa988c73d29d67bccf7381f6f72cabd4eb7be090f36f0b444978cc708324
Mycelix evidence   6ddca81103c52408421e2e31da4b4dee0c0b2762
semantic subject   fca2c107a1ea5108823ce617ba4111b6f7f77230
same-tree authoring head
                   31ede2365b81365bb119cd9351b2739119974130
source authority   ObservedSourceBound
Symthaea conformance CrossImplementationConformance
Symthaea PR        #3326
```

The separate same-tree authoring head is preserved as provenance and does not replace the semantic production subject.

## Proposal semantics now represented

v3 adds exactly four coverage entries:

- proposal creation semantics;
- ProposalById/read-projection semantics;
- proposal update-integrity semantics;
- coordinator status-update semantics.

The component explicitly preserves Mycelix #66 and CE-PROP-01..05:

- linked reads do not establish authoritative current lifecycle state;
- Draft -> Active content mutation is not structurally rejected by the observed pure update checker;
- update author authority is not established by integrity validation;
- temporal fields/order are not update-bound;
- competing sibling updates have no observed deterministic authoritative-child projector.

Therefore:

```text
ProposalLifecycle covered
!= authoritative-current Proposal lifecycle
!= deterministic fork resolution
!= repaired Proposal lifecycle
```

## Retained components

Voting, ThresholdSigning, and Execution must remain semantically identical to v2.

All four components bind the same semantic production subject and remain:

```text
source authority        ObservedSourceBound
Symthaea conformance    CrossImplementationConformance
```

No component authority promotion is permitted.

## Monotonic coverage delta

Relative to v2, v3 may add only:

```text
proposal_creation_observed_semantics
proposal_lookup_projection_observed_semantics
proposal_update_integrity_observed_semantics
proposal_status_update_observed_semantics
```

and may remove only:

`proposal_creation_and_lifecycle_as_independent_profile`

from the uncovered set.

Still uncovered:

```text
constitution_parameter_authorization_downstream
treasury_credit_authorization_downstream
deployment_currentness
```

The end-to-end required-stage registry remains unchanged.

## Qualification

The exact-head workflow independently replays all four represented components in one run:

1. ProposalLifecycle — Mycelix P0/P1 vs Symthaea P0;
2. Voting — Mycelix A3/A4 vs Symthaea A0;
3. ThresholdSigning — Mycelix S0/S1 vs Symthaea S0;
4. Execution — Mycelix E0/E1 vs Symthaea E0.

For each component, the Mycelix and Symthaea canonical corpus outputs must be byte-identical.

The qualifier then:

- validates v1;
- validates v2 against v1;
- validates v3 twice byte-identically against v2+v1;
- checks the exact four-component set;
- checks retained-component immutability;
- checks exact Proposal provenance including the same-tree authoring head;
- checks exact coverage/uncovered deltas;
- verifies every checkout remains immutable.

A queued or unexecuted job is not a PASS.

## Migration boundary

The stronger Mycelix proposal/constitutional/execution successor work (#44/#59/#63+) remains a distinct lineage.

Future Institutional Lab work may compare:

```text
LegacyObservedGovernance
vs
SuccessorQualifiedGovernance
```

but successor design evidence cannot retroactively strengthen this legacy production manifest.

## Non-claims

IG-008F2 does not establish:

- corrected Proposal lifecycle;
- authoritative Proposal currentness;
- deterministic Proposal fork resolution;
- deployed successor governance;
- secure threshold signing;
- cryptographic signature validity;
- downstream Constitution authority;
- downstream Treasury/Credit authority;
- deployment currentness;
- end-to-end governance safety;
- fairness;
- constitutional legitimacy.
