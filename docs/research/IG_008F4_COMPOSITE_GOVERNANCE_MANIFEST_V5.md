# IG-008F4 — Composite governance manifest v5

Issue: #3393

Parent: IG-008CD0 / draft #3392.

## Purpose

Advance the immutable observed-governance composite from v4 to v5 by consuming the content-bound `ConstitutionRuntimePolicyDownstreamV1` submanifest and refining v4's downstream stage taxonomy without rewriting v4 history.

## V5 identity

```text
manifest_id    mycelix-observed-composite-fca2c107-v5
revision       5
production     fca2c107a1ea5108823ce617ba4111b6f7f77230
authority      ObservedCompositeSlice
claim ceiling  ObservedCompositeSliceOnly
SHA-256        a63f9cd8a72824708925d6938ff6cffa2efda614794259cfe86189c27a4b44fe
```

Exact predecessor:

```text
mycelix-observed-composite-fca2c107-v4
993a17b4ab1e876beb7815f8e90fc137b8776c5dcce3b0d9e30da971b13a797f
```

## Monotonic direct evidence

V5 retains v4's five direct components exactly:

1. ProposalLifecycle;
2. Voting;
3. ThresholdSigning;
4. Execution;
5. ConstitutionParameter.

No direct component is replaced, rebased, or flattened.

## Added composed evidence

V5 adds exactly one nested content-bound submanifest:

```text
role       ConstitutionRuntimePolicyDownstream
manifest   mycelix-constitution-runtime-policy-downstream-observed-fca2c107-v1
SHA-256    157402e76abffd3849dc6d2001a7c0bbde296e80ab6a7f5c77c8f3e288302183
subject    fca2c107a1ea5108823ce617ba4111b6f7f77230
authority  ObservedCompositeSubslice
ceiling    ObservedConstitutionRuntimePolicySliceOnly
Symthaea   #3392 / af73f6f0601c5ea3a2935fc62573078ce27a712f
issues     Mycelix #943/#944/#1002
```

The nested structure matters: the submanifest owns the composition theorem for ConstitutionParameter + ConstitutionBridgeSync + GovernanceConfig. V5 consumes that theorem by content address instead of copying its internals into the top-level component list.

## Taxonomy correction

V4 used:

`constitution_parameter_authorization_downstream`

as the end-to-end stage name. That label was underspecified because ConstitutionParameter representation alone does not represent the later synchronization and runtime-config mutation mechanisms.

V5 replaces the canonical stage name with:

`constitution_runtime_policy_downstream`.

The correction is explicitly classified:

`ScopeRefinementNotHistoricalInvalidation`.

V4's hashes and observations remain historical evidence; v5 improves the scope vocabulary around them.

## Coverage delta

V5 preserves every v4 covered stage and adds exactly:

`constitution_runtime_policy_downstream_observed_slice`.

The remaining top-level uncovered stages are still:

```text
treasury_credit_authorization_downstream
deployment_currentness
```

Coverage continues to mean **source-observed mechanism represented**, not that authorization, correctness or safety properties are satisfied.

## Represented but unestablished properties

V5 promotes the constitutional submanifest's unresolved properties into an explicit top-level set:

```text
AuthorizedConstitutionParameterMutation
AtomicConstitutionRuntimePolicyMutation
ConstitutionRuntimeConfigSynchronization
ContentBoundReconciliation
AuthorizedGovernanceConfigMutation
RuntimeGovernanceConfigCurrentness
DeploymentCurrentnessQualified
GovernanceSafety
```

A future top-level reader therefore cannot infer these properties merely because the corresponding mechanism plane is represented.

## Scope exclusions

V5 explicitly does not claim coverage of:

```text
charter_creation_and_currentness
constitutional_amendment_lifecycle_and_application
enhanced_immutable_core_amendment_requirements
```

Those require separate source-bound assurance.

## Validation

The v5 validator binds exact v4 and constitutional-submanifest commitments, requires the five direct components to remain identical to v4, permits exactly one new covered-stage slice, requires the exact taxonomy correction, preserves the unresolved-property and scope-exclusion sets, and rejects authority/claim-ceiling promotion.

The exact-head workflow additionally executes the #3392 submanifest qualifier before validating v5, so a stale or merely listed submanifest hash cannot enter the top-level composite without replay.

## Next boundary

After v5, the only top-level semantic mechanism gap remains:

`treasury_credit_authorization_downstream`.

That audit should follow the actual legacy `TransferCredits -> governance_bridge` path on the same semantic production subject and should not infer authority from the newer Finance stack.

`deployment_currentness` remains a separate evidence class after semantic mechanism representation.

## Non-claims

No ObservedEndToEnd, complete Constitution coverage, charter/amendment coverage, authorized parameter/config mutation, successful synchronization, Treasury/Credit authority, deployment currentness, governance safety, fairness, or constitutional legitimacy is established.
