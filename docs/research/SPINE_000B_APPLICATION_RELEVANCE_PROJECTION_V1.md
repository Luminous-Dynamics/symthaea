# SPINE-000B — Application Relevance Projection v1

**Status:** preregistered measurement-only projection contract

**Authority:** measurement-only

**Issue:** #3311

R2 answers a narrower question than causal attribution:

> Given qualified `I_all` and `I_withoutS`, does removing subject S change the deterministic Phase-C source condition, projected operation reachability, or lowered scalar operand?

It does **not** answer whether the counterfactual application executed, whether destination state changed, or whether S was load-bearing.

## 1. Required upstream evidence

R2 consumes one P1/C1 subject entry:

```text
integrated_all
integrated_without_subject
changed_channel_mask
uniquely_contributed_flags
integration_changed
```

and an explicit active-feature profile.

Both integrated objects must pass N2. R2 independently recomputes the changed-channel mask, unique-flag mask and `integration_changed` from canonical bits. Any mismatch fails closed.

## 2. Projection, not counterfactual execution

`I_withoutS` is a leave-one-out integration result. Before SPINE-000D actually intervenes on the running system, it has no runtime `StateApplicationReceipt`.

Therefore R2 may construct only:

```text
I_all       -> actual-side Phase-C projection
I_withoutS  -> leave-one-out Phase-C projection
```

The words `actual-side` and `counterfactual-side` name which integrated object is being lowered. They do not claim either projected operation executed.

Every R2 record must state:

```text
actual_execution_claimed=false
counterfactual_execution_claimed=false
causal_load_claimed=false
```

## 3. Scalar projection law

For each P1-changed scalar channel, R2 uses N2's exact production trigger semantics and exact post-lowering f32 operand bits.

For the single v1 registry operation associated with that scalar:

- neither side triggered -> no application relevance; this includes +0.0/-0.0 bit differences that are numerically identity;
- one side triggered -> operation reachability differs, even if the triggered side lowers to a zero/identity f32 operand;
- both sides triggered and lowered operand bits differ -> projected operand changes;
- both sides triggered and lowered operand bits are identical -> P1 integration influence exists, but R2 application projection is unchanged (`prelowering_only_scalar_mask`).

This explicitly captures f64 differences that collapse under the production f64->f32 cast.

## 4. Flag projection law

Only P1 `uniquely_contributed_flags` are attributable to the subject at this rung.

For each unique bit:

- no registry applications -> `unconsumed_unique_flag_mask`; required negative control: `HAS_TELEMETRY`;
- applications exist but their feature gates are inactive -> `feature_gated_inactive_unique_flag_mask`;
- active `FLAG_SET` application -> projected on the `I_all` side;
- active `FLAG_CLEAR` application -> projected on the `I_withoutS` side.

`REQUEST_GEODESIC` is the required set/clear control: unique presence may project set-path operations under `I_all` and the explicit clear operation under `I_withoutS`.

R2 does not invent application indices or global execution order.

## 5. Source reachability is not execution

The Phase-C registry binds the source condition and feature gate. Some operations may have additional runtime guards or complex state-dependent behavior.

Therefore a projected operation means only:

```text
this source/feature configuration reaches the registered application path
```

not:

```text
this operation executed
```

Actual runtime `StateApplicationReceipt`s are a later observational rung and can confirm actual-cycle execution only.

## 6. Relevance classes

Subject-level `relevance_kind` is compositional:

```text
NO_INTEGRATION_INFLUENCE
NONE
SCALAR
FLAG
MIXED
```

`NONE` is a valid and important result: P1 may detect integration-bit influence that disappears at the Phase-C source/lowering boundary.

Examples include:

- unique `HAS_TELEMETRY`;
- +0.0 vs -0.0 scalar bit changes;
- distinct f64 values that lower to identical f32 operand bits;
- unique feature-gated flags when the gate is inactive.

## 7. Required R2 witness surface

At minimum:

```text
integration_changed
changed_channel_mask
uniquely_contributed_flags
relevance_kind
projection_changed
actual_projected_operations
counterfactual_projected_operations
actual_only_projected_operation_ids
counterfactual_only_projected_operation_ids
shared_projected_operand_changed_operation_ids
shared_projected_operand_equal_operation_ids
relevant_scalar_operation_ids
relevant_flag_operation_ids
prelowering_only_scalar_mask
unconsumed_unique_flag_mask
feature_gated_inactive_unique_flag_mask
feature_gated_inactive_operation_ids
i_all_n2
i_without_subject_n2
```

Projections are sorted for deterministic evidence presentation, not claimed runtime order.

## 8. Required controls

The executable R2 oracle must prove at least:

1. no P1 integration influence -> no R2 relevance;
2. unique `HAS_TELEMETRY` -> P1 influence but R2 `NONE`;
3. scalar identity -> nonidentity -> actual-side projected operation;
4. two changed f64 values lowering to identical f32 bits -> R2 `NONE`;
5. +0.0/-0.0 exact-bit change -> R2 `NONE`;
6. both scalar sides triggered with different lowered bits -> shared operation, changed projected operand;
7. tiny nonzero f64 that lowers to `0.0f32` -> operation reachability changes even though operand is zero;
8. unique consumed flag -> flag relevance;
9. unique `REQUEST_GEODESIC` with feature active -> set projection on `I_all`, clear projection on `I_withoutS`;
10. same geodesic bit with feature inactive -> R2 `NONE`;
11. mixed scalar+flag relevance -> `MIXED`;
12. any N2-invalid integrated object -> R2 refuses classification;
13. tampered P1 summary fields -> fail closed.

## 9. Epistemic ladder after R2

The claim hierarchy is now:

```text
emitted
-> admitted
-> P1 integration influence
-> N2 qualified lowering
-> R2 projected application reachability
-> actual cycle application observed
-> actual cycle state change observed
-> SPINE-000D matched intervention
-> causal/load-bearing claim
```

No rung may be skipped by terminology.

## 10. Qualification boundary

A green R2 workflow establishes deterministic agreement between qualified P1/N2 integrated objects and the frozen Phase-C source/lowering registry on the synthetic controls.

It establishes no actual counterfactual execution, no destination-state effect, no subsystem causality, no benefit, and no epistemic/action authority.