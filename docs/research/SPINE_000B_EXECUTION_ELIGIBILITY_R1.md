# SPINE-000B — Execution Eligibility Semantics R1

**Status:** preregistered measurement contract; runtime capture pending

**Authority:** measurement-only

**Issue:** #3391

This contract freezes how `SubsystemExecutionReceiptV2` scheduling and execution fields are interpreted before any live observer is wired into `CognitiveLoopService`.

## 1. Core separation

```text
manager exists
!=
manager scheduled this cycle
!=
manager health-enabled
!=
manager process() executed
!=
manager emitted a proposal
!=
collector admitted that proposal
```

Absence of an invocation is never sufficient evidence for `SKIPPED_SCHEDULE`.

## 2. Scheduling authority

`CognitiveSubsystem::should_run(cycle, urgency)` is the production scheduling authority.

The trait provides a default interval/urgency implementation, but the method is explicitly overrideable. Qualified runtime evidence therefore binds the **actual boolean returned by the production call**.

The observer must not:

- reconstruct eligibility later from `interval()` and urgency;
- infer `eligible=false` from absence of a `run_subsystem!` invocation;
- call `should_run` a second time for evidence;
- replace a future manager-specific scheduling override with the trait default.

At the live seam, production must evaluate `should_run` once, store that exact boolean in bounded observer state, and use the same boolean for control flow.

## 3. Frozen eligibility / outcome truth table

```text
Outcome                    eligible  emitted  admitted  proposal
---------------------------------------------------------------------
SKIPPED_SCHEDULE            false     false    false     None
SKIPPED_HEALTH_DISABLED     true      false    false     None
PANICKED_CAUGHT             true      false    false     None
FAILED_OTHER                true      false    false     None
EXECUTED_NEUTRAL            true      true     false     Some(exact)
EXECUTED_NON_NEUTRAL        true      true     true      Some(exact)
```

Any other combination is invalid evidence and fails closed.

`FAILED_OTHER` is reserved for a future explicitly observed post-scheduling failure mode. It must never be used as a generic fallback for missing evidence.

## 4. Production neutrality is part of receipt validity

The execution label must agree with the actual production `SubsystemOutput::is_neutral()` semantics.

Current production neutrality is numeric equality over:

```text
confidence_delta == 0.0
lr_modulation == 1.0
exploration_delta == 0.0
arousal_delta == 0.0
valence_delta == 0.0
flags == 0
```

`_reserved` is intentionally **not** part of current production neutrality.

Therefore:

- `EXECUTED_NEUTRAL` requires the exact committed proposal to be production-neutral;
- `EXECUTED_NON_NEUTRAL` requires the exact committed proposal to be production-non-neutral;
- a reserved-only nonzero output remains `EXECUTED_NEUTRAL` under production semantics while N1 separately classifies it `RESERVED_NONZERO` and makes the cycle ineligible for qualified SPINE proposal evidence.

This preserves observed production behavior without allowing N1 evidence qualification to rewrite it.

## 5. IEEE comparison semantics

Production neutrality uses Rust numeric equality, not bit equality:

- `+0.0 == -0.0` is true;
- any NaN comparison to the neutral scalar is false;
- different NaN payloads remain distinct in C2 canonical bytes;
- a NaN-containing output is production-non-neutral and N1-invalid.

R1 must reproduce those semantics exactly from committed proposal bits.

## 6. Runtime capture consequence

The current code calls `run_subsystem!` only inside outer `if manager.should_run(...)` blocks. Therefore a future Stage-A execution observer must span both seams:

```text
actual should_run result
        ↓
eligible=false ──> finalize SKIPPED_SCHEDULE event
        │
        └ eligible=true
             ↓
health check
             ↓
process / catch_unwind
             ↓
neutrality + collector admission
             ↓
finalize exactly one execution event
```

For qualified runtime evidence, every in-scope registered manager must finalize exactly one execution event per cycle, unless the manager observer buffer overflowed. Missing finalization is evidence incompleteness, not an inferred outcome.

## 7. Health and panic ordering

`SKIPPED_HEALTH_DISABLED`, `PANICKED_CAUGHT`, `EXECUTED_NEUTRAL`, and `EXECUTED_NON_NEUTRAL` are all post-scheduling outcomes and therefore require `eligible_to_run=true`.

A health-disabled manager never emits a proposal. A caught panic never emits or admits a proposal.

## 8. Required controls

R1 qualification must prove at least:

1. valid `SKIPPED_SCHEDULE` row accepted;
2. `SKIPPED_SCHEDULE + eligible=true` rejected;
3. health-disabled/panic/executed outcome with `eligible=false` rejected;
4. executed outcome without proposal rejected;
5. non-executed outcome with proposal rejected;
6. `EXECUTED_NEUTRAL` with non-neutral proposal rejected;
7. `EXECUTED_NON_NEUTRAL` with neutral proposal rejected;
8. exact neutral proposal accepted as `EXECUTED_NEUTRAL`;
9. additive signed-zero proposal remains production-neutral;
10. reserved-only nonzero proposal remains production-neutral and therefore valid only under `EXECUTED_NEUTRAL` while remaining N1-invalid;
11. NaN-containing proposal is production-non-neutral and N1-invalid;
12. source preflight confirms `should_run` remains overrideable and live manager calls remain guarded by production `should_run` decisions.

## 9. Qualification boundary

A green R1 exact-head run establishes only deterministic interpretation of scheduling/execution fields on synthetic controls and source-bound current semantics.

It does **not** establish that a runtime observer captures every manager correctly, observer non-interference, application execution, state change, subsystem causality, load-bearing benefit, or authority.
