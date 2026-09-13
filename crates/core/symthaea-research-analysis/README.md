# symthaea-research-analysis

Content-addressed contracts for proving **which frozen external analysis invocation executed over which exact inputs and produced which exact output**.

## Why this exists

The protocol/result stack can establish:

```text
frozen external-rule identity
+ endpoint-bound Analysis artifact
```

but that is not yet evidence that the frozen rule was faithfully executed.

This crate closes that narrow relationship:

```text
FrozenExternalAnalysisPlanV1
  + exact canonical V2 run binding
  + exact concrete input artifacts
  + exact input schemas/source plans
  + execution witness
  + verifier evidence
  + exact output binding
  -> FrozenAnalysisExecutionReceiptV1
```

`ExternalAnalysisQualifiedResultV1` then requires an exact plan/receipt census for every `ExternalFrozenAnalysisRule` endpoint in an endpoint-complete V2 result.

## Frozen invocation identity

A plan commits:

- endpoint + canonical confirmatory-protocol identity;
- exact frozen rule id/artifact digest;
- executable artifact digest;
- invocation digest (argv/config/entrypoint semantics);
- toolchain digest;
- execution-environment digest;
- minimum verifier class;
- exact ordered input roles;
- input schema digests;
- baseline/auxiliary source-plan digests.

Executable bytes alone are not an invocation.

## Baseline observations

The following remain distinct:

```text
baseline_id
!= baseline source/execution plan
!= baseline observation artifact
!= concrete analysis input
```

Baseline and auxiliary inputs therefore require a frozen `source_plan_digest` and a concrete result artifact.

## Verification classes

V1 keeps verification strength explicit:

- `BindingOnly`;
- `ExecutionWitnessBindingValidated`;
- `DeterministicReexecutionExact`;
- `IndependentReexecutionExact`.

The minimum acceptable class is frozen in the invocation plan. `BindingOnly` can never support a positive/negative/null scientific conclusion.

For either exact re-execution class, a Passed verifier must record `reexecuted_output_digest`, and it must equal the exact result-bound analysis output digest. Independent exact re-execution additionally requires an independent-verifier declaration.

## Terminal failure evidence

Execution failures are first-class states rather than missing records:

- `InputMissing`;
- `InputSchemaMismatch`;
- `RuleArtifactMismatch`;
- `ExecutionFailed`;
- `VerifierFailed`;
- `InfrastructureFailure`;
- `NotRunWithReason`.

A non-Completed execution cannot carry a Passed verifier. Failed/inconclusive analysis may support an inconclusive/not-evaluated terminal result, but cannot be laundered into positive, negative, or null support.

## Result composition

The strong wrapper requires the set of plan IDs and receipt IDs to equal exactly the set of frozen external-rule endpoint IDs. Missing, duplicate, and orphan evidence all fail closed.

Concrete bindings are rechecked against #2190 result evidence:

- endpoint metric input artifacts must be referenced by the exact `MetricResult`;
- baseline observations must resolve to concrete `Metrics` artifacts;
- input/output artifact digests must match result artifacts;
- all inputs bind the exact canonical V2 run;
- any produced output must match the endpoint's exact claim-bound `Analysis` artifact, even for an inconclusive terminal result;
- positive/negative/null conclusions require Completed execution plus verification stronger than `BindingOnly` and at least the frozen minimum class.

## Custody and chronology hooks

A receipt may carry canonicalized external custody-access receipt digests. These are identity hooks only. #193 owns custody/access policy and principal/action/phase semantics.

The invocation-plan digest itself does **not** prove when the plan was frozen. #1946 remains responsible for recording plan/source-plan identities in the append-only research lineage before the declared unblinding boundary.

```text
plan digest valid
!= plan frozen before outcomes
```

## Non-claims

This crate does not establish:

- statistical or scientific correctness of the frozen rule;
- authenticity of the verifier identity or execution witness;
- operating-system/process isolation;
- custody policy correctness;
- chronology/current-head completeness;
- independent scientific replication;
- deployment validity;
- authority to act.

A software-contract PASS is also not a scientific PASS.

## Required focused gates

The exact workspace lock graph must include this crate and its protocol/result dependencies before `--locked` qualification.

```bash
cargo fmt --package symthaea-research-analysis -- --check
cargo check --locked -p symthaea-research-analysis --all-targets
cargo test --locked -p symthaea-research-analysis
cargo clippy --locked -p symthaea-research-analysis --all-targets -- -D warnings
```

The focused lane in #2033 now includes this package; admission/backpressure remains governed separately by #986.
