# SPINE-000B — Proposal Influence Receipt Contract

**Status:** preregistered instrumentation contract; no production cognition rewiring

**Authority:** measurement-only

**Purpose:** define exactly what SPINE-000B must measure before any `CognitiveSubsystem` manager can be promoted from merely reachable to causally load-bearing.

## 1. Core distinction

SPINE-000B keeps five facts separate:

```text
scheduled / eligible
    -> executed
    -> emitted a proposal
    -> proposal admitted to OutputCollector
    -> admitted proposal changed the integrated result
    -> changed integrated result was consumed downstream
```

None of those facts, individually or together, establishes causal load.

`LOAD_BEARING` still requires a later matched intervention / ablation showing that the component changes a preregistered observable relative to a control.

## 2. Why this receipt is needed

Symthaea currently contains a migration-era dual-write architecture. Proposal-based managers can run alongside legacy inline logic that owns similar responsibilities.

Therefore all of the following can occur:

- a manager executes but returns `SubsystemOutput::NEUTRAL`;
- a manager emits a non-neutral proposal that is admitted but averaged away;
- a manager uniquely sets a flag, or sets a flag already supplied by another manager;
- a manager changes the collector result but an inline path later overwrites the destination state;
- a manager changes a state field that no later decision actually reads;
- a manager and inline path both influence the same observable, preventing attribution.

SPINE-000B exists to make those cases mechanically distinguishable.

## 3. Receipt schema

One dynamic receipt is emitted per subsystem execution attempt.

```text
ProposalInfluenceReceipt
  schema
  authority
  evidence_lineage

  cycle_number
  subsystem_name
  subsystem_type
  source_path
  schedule_interval
  urgency

  eligible_to_run
  executed
  execution_outcome
  duration_ns

  proposal
    confidence_delta_bits
    lr_modulation_bits
    exploration_delta_bits
    arousal_delta_bits
    valence_delta_bits
    flags
    is_neutral

  collector
    admitted
    contributor_count_before
    contributor_count_after

  integration
    integrated_all
    integrated_without_subject
    changed_channels[]
    uniquely_contributed_flags
    integration_changed

  consumption[]
    destination
    before_bits
    after_bits
    consumer_stage
    applied

  duplicate_path_context[]

  causal_load_claimed
```

Required constants:

```text
authority = measurement-only
causal_load_claimed = false
```

The receipt MUST NOT contain a field that upgrades a subsystem to `LOAD_BEARING`.

## 4. Execution outcomes

`execution_outcome` is one of:

- `SKIPPED_SCHEDULE` — `should_run()` returned false;
- `SKIPPED_HEALTH_DISABLED` — health policy suppressed execution;
- `EXECUTED_NEUTRAL` — execution completed and output was neutral;
- `EXECUTED_NON_NEUTRAL` — execution completed and output was non-neutral;
- `PANICKED_CAUGHT` — current panic isolation caught a panic;
- `FAILED_OTHER` — explicit non-panic instrumentation failure.

A skipped or failed manager must not be represented as having emitted or admitted a proposal.

## 5. Exact proposal representation

SPINE-000B must preserve the exact IEEE-754 payload bits of scalar proposal fields rather than only decimal rendering.

For each `SubsystemOutput`:

```text
confidence_delta  -> f64::to_bits()
lr_modulation     -> f64::to_bits()
exploration_delta -> f64::to_bits()
arousal_delta     -> f32::to_bits()
valence_delta     -> f32::to_bits()
flags             -> exact u32
```

This avoids inventing an epsilon after evidence is observed.

Human-readable decimal values may be emitted in addition to the exact bit representation, but the bit representation is authoritative for equality inside this measurement artifact.

## 6. Admission

`admitted = true` means the exact attributed proposal was recorded by the current `OutputCollector` for that cycle.

Execution does not imply admission.

The receipt must bind the subsystem name used for attribution to the exact name inserted into the collector.

## 7. Integration influence

A non-neutral admitted proposal does not necessarily change the integrated result.

For each admitted subsystem `S`, SPINE-000B computes two results using the **same integration implementation**:

```text
I_all      = integrate(all admitted proposals)
I_withoutS = integrate(all admitted proposals except S)
```

Then:

```text
integration_changed = canonical_bits(I_all) != canonical_bits(I_withoutS)
```

`changed_channels` lists only channels whose exact integrated bit representation changes.

For flags:

```text
uniquely_contributed_flags = flags(I_all) & !flags(I_withoutS)
```

A subsystem may emit a flag while having zero unique flag influence if another admitted proposal supplied the same bit.

The leave-one-out comparison is a measurement of influence under the current integration rule. It is **not** a causal-ablation result and MUST NOT be reported as one.

## 8. Downstream consumption

SPINE-000B must instrument the exact state-application boundary where integrated proposal fields are applied to `CognitiveLoopService` / feedback state.

A consumption witness records:

```text
destination
before_bits
after_bits
consumer_stage
applied
```

`applied = true` means the integrated value participated in the destination update at that boundary.

It does not mean the update survived to final output.

Later SPINE tranches may add a second-order witness proving that a subsequent decision actually read the changed destination. Until then, use the wording **state-application consumed**, not **behaviorally consumed**.

## 9. Duplicate-path context

Every proposal receipt SHOULD carry static overlap context from SPINE-000C when an explicit source witness exists.

For example:

```text
manager: reasoning_manager
explicit inline witness: cycle_phase_dynamics.rs:...
```

This context prevents an admitted manager proposal from being interpreted as unique behavioral ownership while a parallel inline path still exists.

Absence of an explicit static witness does not prove absence of overlap.

## 10. Non-authority boundary

SPINE telemetry is observational.

It MUST NOT:

- affect scheduling;
- affect proposal values;
- affect integration weights;
- affect panic handling;
- affect downstream state application;
- become an HDC similarity input;
- become a CfC feature;
- alter Broca authorization;
- alter epistemic confidence;
- grant action authority.

Instrumentation failure should fail or omit the evidence run, not silently alter cognition to make telemetry complete.

## 11. Determinism and ordering

Receipt order must be deterministic for a fixed subject, seed, input sequence, feature set, and scheduler state.

The canonical sort key for persisted receipts is:

```text
(cycle_number, subsystem_name)
```

If two manager implementations expose the same non-empty `subsystem_name`, SPINE qualification fails before dynamic evidence is interpreted.

## 12. Required negative controls

Before SPINE-000B evidence may be used, tests must establish at least:

1. neutral proposal -> executed, admitted, `integration_changed = false` when it has no flag effect;
2. unique scalar proposal -> expected changed channel only;
3. duplicated equal scalar proposals -> leave-one-out effect reflects the actual averaging semantics;
4. unique flag -> appears in `uniquely_contributed_flags`;
5. duplicated flag -> emitted but not unique when another manager supplies it;
6. skipped manager -> no emitted/admitted claim;
7. caught panic -> no admitted proposal;
8. instrumentation disabled -> existing cognition output remains unchanged;
9. telemetry serialization failure -> no mutation of cognitive state;
10. exact subject replay -> byte-identical receipt content except explicitly excluded host/runtime metadata.

## 13. Qualification boundary

SPINE-000B can establish:

> Under one exact execution subject, subsystem S executed, emitted proposal P, P was admitted, and removing P from the same collected proposal set would / would not change the integrated proposal; the integrated proposal was / was not applied to named downstream state destinations.

SPINE-000B cannot establish:

> S is necessary, beneficial, intelligent, conscious, correct, or load-bearing.

Those are later intervention claims.

## 14. Next theorem

After SPINE-000B is executable and sealed, SPINE-000D should compare matched arms such as:

```text
control: current dual-write behavior
candidate: manager contribution disabled or substituted
```

with identical seeds / inputs and preregistered observables.

Only that class of intervention can upgrade a component to `LOAD_BEARING`.
