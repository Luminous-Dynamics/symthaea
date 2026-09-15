# IG-008F3 — Composite governance manifest v4

Issue: #3359

Parent: IG-008CP0 / draft #3349

## Purpose

Advance the immutable observed Mycelix governance composite from four independently represented components to five by adding the frozen `ConstitutionParameter` downstream boundary.

Historical v1/v2/v3 evidence remains unchanged.

## Manifest identity

```text
id             mycelix-observed-composite-fca2c107-v4
revision       4
production     fca2c107a1ea5108823ce617ba4111b6f7f77230
authority      ObservedCompositeSlice
claim ceiling  ObservedCompositeSliceOnly
SHA-256        993a17b4ab1e876beb7815f8e90fc137b8776c5dcce3b0d9e30da971b13a797f
```

Exact predecessor:

```text
mycelix-observed-composite-fca2c107-v3
3acdbd29d0fd631b863fa10a10eed0877af2336536578453acdc83f095632baf
```

The validator mechanically revalidates the entire lineage:

```text
v1 366b8794...
 -> v2 e3b42dbd...
    -> v3 3acdbd29...
       -> v4 993a17b4...
```

## Coverage definition

The frozen rule remains:

`SourceObservedMechanismRepresentedNotPropertySatisfied`.

A stage becomes covered when its observed mechanism semantics are source-bound and independently represented. Coverage never means the desired safety/authorization theorem is satisfied.

## New component — ConstitutionParameter

```text
profile          mycelix-constitution-parameter-observed-fca2c107-v1
profile SHA      770552d12489df1d2cdf8b0af676b01ea9a3da21940f70ed8a71910deaa35009
corpus SHA       b37be9d2e3fd0cbec3696a19327c26dc4ba7062ec089ad92ad99bc28a11fea8e
Mycelix evidence 3232d611d8833b03eba9f5412f5d7cb0cf89d4e1
production       fca2c107a1ea5108823ce617ba4111b6f7f77230
same-tree authoring
                 31ede2365b81365bb119cd9351b2739119974130
source authority ObservedSourceBound
Symthaea conformance CrossImplementationConformance
Symthaea PR      #3349
```

## Gap-preserving representation

The component preserves #1002 / CE-CP-01..05, including both containment and weaknesses:

- existing parameter + execution dispatch without proposal ID is rejected by the observed presence gate;
- a previously absent parameter can pass the observed coordinator gate with `proposal_id=None`;
- `Some(proposal_id)` is presence, not independently reconstructed proposal authority;
- parameter integrity shape validity does not establish mutation authority;
- maximum-link-timestamp read projection does not establish authoritative fork resolution.

Therefore:

```text
ConstitutionParameter covered
!= authorized parameter mutation
!= authoritative parameter currentness
!= corrected downstream governance authority
```

## Monotonic coverage delta

v4 adds only:

```text
constitution_parameter_execution_dispatch_observed_semantics
constitution_parameter_coordinator_gate_observed_semantics
constitution_parameter_integrity_observed_semantics
constitution_parameter_projection_observed_semantics
```

and removes only:

`constitution_parameter_authorization_downstream`

from the uncovered set.

Still uncovered:

```text
treasury_credit_authorization_downstream
deployment_currentness
```

The required end-to-end registry itself does not change.

## Retained components

The exact v3 representations of:

- ProposalLifecycle;
- Voting;
- ThresholdSigning;
- Execution;

must remain semantically identical.

All five components bind the same semantic production subject and remain `ObservedSourceBound / CrossImplementationConformance`.

## Qualification

The exact-head workflow independently replays all five represented component corpora in one run:

1. ProposalLifecycle;
2. Voting;
3. ThresholdSigning;
4. Execution;
5. ConstitutionParameter.

For every component, the Mycelix and Symthaea canonical corpus outputs must be byte-identical.

The qualifier then validates v1, v2, v3 and v4; runs v4 twice byte-identically; verifies retained-component equality; verifies exact CP profile/corpus/history provenance; verifies only the declared coverage/uncovered delta; and requires every checkout to remain immutable.

A queued or unexecuted workflow is not a PASS.

## Next boundary

After v4, only two composite gaps remain:

1. `treasury_credit_authorization_downstream`;
2. `deployment_currentness`.

Treasury/Credit must be audited against the exact legacy `TransferCredits -> governance_bridge` path rather than assuming newer Finance architecture covers it.

Deployment currentness should remain a distinct final evidence class concerning the actually packaged/deployed mechanism lineage, not a semantic component inferred from source alone.

## Non-claims

IG-008F3 establishes no authorized ConstitutionParameter mutation, authoritative parameter currentness, secure threshold signing, complete Treasury/Credit authority, deployment currentness, governance safety, fairness, or constitutional legitimacy.
