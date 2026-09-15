# SPINE-000B Runtime Receipt Contract R2

Status: preregistered measurement-only successor to the original SPINE-000B monolithic receipt draft.

Issue: #3228

This contract is intentionally frozen before runtime instrumentation. It corrects ambiguities discovered by comparing the original receipt draft against current `OutputCollector::record()` and `safe_process()` behavior.

## 1. Core evidence model

Do not encode every fact into one per-subsystem monolithic receipt.

SPINE-000B R2 uses four normalized record families:

```text
SubsystemExecutionReceipt
        │
        ├──────────────┐
        ▼              ▼
CycleIntegrationReceipt      RuntimeTelemetryEnvelope
        │                     (noncanonical)
        ▼
StateApplicationReceipt
```

A derived analysis view may join these records by `(cycle_number, subsystem_name)` and cycle identifiers, but the canonical evidence store remains normalized.

This prevents a cycle-level integrated result or downstream state application from being misrepresented as uniquely owned by one subsystem.

## 2. Canonical SubsystemExecutionReceipt

One canonical receipt exists for every subsystem scheduling decision.

```text
SubsystemExecutionReceipt
  schema = symthaea.spine.000b.execution.v2
  authority = measurement-only
  causal_load_claimed = false

  evidence_lineage
  cycle_number
  subsystem_name
  subsystem_type
  source_path
  schedule_interval
  urgency

  eligible_to_run
  execution_outcome
  emitted

  proposal: Option<ExactProposal>

  collector
    admitted
    contributor_count_before
    contributor_count_after
```

`subsystem_name` MUST be non-empty and unique among registered subsystem identities for the qualified subject.

### ExecutionOutcome

The canonical outcome enum is:

```text
SKIPPED_SCHEDULE
SKIPPED_HEALTH_DISABLED
EXECUTED_NEUTRAL
EXECUTED_NON_NEUTRAL
PANICKED_CAUGHT
```

Instrumentation/serialization failure is not a subsystem outcome. It fails or invalidates the measurement run at the evidence layer instead of being encoded as `FAILED_OTHER`.

## 3. Exact outcome truth table

```text
Outcome                  executed  emitted  proposal   admitted
-----------------------------------------------------------------
SKIPPED_SCHEDULE         false     false    None       false
SKIPPED_HEALTH_DISABLED  false     false    None       false
PANICKED_CAUGHT          false*    false    None       false
EXECUTED_NEUTRAL         true      true     Some(P)    false
EXECUTED_NON_NEUTRAL     true      true     Some(P)    true
```

`PANICKED_CAUGHT` means execution was attempted but did not successfully return a proposal. The optional convenience field `executed` therefore means **successfully returned from `process()`**, not merely entered the call boundary.

If a later implementation needs both concepts, use two explicit booleans:

```text
execution_attempted
execution_completed
```

rather than overloading one field.

## 4. Neutral output semantics

Current `OutputCollector::record()` filters `SubsystemOutput::NEUTRAL`.

Therefore:

```text
EXECUTED_NEUTRAL
emitted = true
admitted = false
```

A neutral returned output is still captured exactly in the execution receipt, but it is not a collector contributor and does not receive an integration-influence measurement.

Do not create a fake zero-valued integration record for a non-admitted subsystem.

## 5. Exact proposal representation

`ExactProposal` preserves canonical bits:

```text
confidence_delta_bits  = f64::to_bits()
lr_modulation_bits     = f64::to_bits()
exploration_delta_bits = f64::to_bits()
arousal_delta_bits     = f32::to_bits()
valence_delta_bits     = f32::to_bits()
flags                   = exact u32
is_neutral              = exact current production predicate
```

Decimal renderings may be included only as non-authoritative convenience fields.

## 6. Panic observability requirement

Current `safe_process()` returns `Some(SubsystemOutput::NEUTRAL)` after a caught panic. An observer placed only after that function cannot distinguish a real neutral result from a caught panic.

SPINE-000B runtime instrumentation MUST therefore observe the panic-isolation boundary directly.

The preferred implementation is behavior-preserving:

```text
MeasuredProcessResult
  outcome: ExecutionOutcome
  output: Option<SubsystemOutput>
```

Production cognition continues to receive the same effective value it receives today. Measurement receives the additional outcome discriminator.

Do not infer panic state from logs or panic counters after the fact.

## 7. Canonical CycleIntegrationReceipt

Exactly one integration receipt exists per cycle with at least one admitted contributor. An optional explicit empty-cycle integration receipt may be emitted for qualification/replay, but it must not fabricate subsystem influence entries.

```text
CycleIntegrationReceipt
  schema = symthaea.spine.000b.integration.v2
  authority = measurement-only
  causal_load_claimed = false

  evidence_lineage
  cycle_number
  admitted_subsystems[]  // deterministic canonical ordering
  contributor_count
  integrated_all
  leave_one_out[]
```

Each admitted subsystem receives one leave-one-out entry:

```text
LeaveOneOutInfluence
  subsystem_name
  integrated_without_subject
  changed_channels[]
  uniquely_contributed_flags
  integration_changed
```

Both `integrated_all` and `integrated_without_subject` preserve exact scalar bits, exact flags, and contributor count.

## 8. Integration influence definition

For admitted subsystem S:

```text
I_all      = production_integrate(all admitted proposals)
I_withoutS = production_integrate(all admitted proposals except S)
```

`changed_channels` includes only proposal-value channels:

```text
confidence_delta
lr_modulation
exploration_delta
arousal_delta
valence_delta
```

Flags use:

```text
uniquely_contributed_flags = flags(I_all) & !flags(I_withoutS)
```

`n_contributors` is canonical metadata and MUST be compared for exact cross-implementation equivalence, but it is **not** itself an influence channel. Otherwise every removal would trivially make `integration_changed=true`.

Therefore:

```text
integration_changed =
    any proposal-value channel changed
    OR uniquely_contributed_flags != 0
```

not merely `canonical_bits(I_all) != canonical_bits(I_withoutS)` over metadata-inclusive structures.

## 9. Non-admitted integration applicability

Only admitted contributors receive leave-one-out influence entries.

For a per-subsystem derived view, a non-admitted subsystem is represented as:

```text
integration = NotApplicable(NotAdmitted)
```

Never encode unmeasured influence as an all-zero/default structure.

## 10. Canonical StateApplicationReceipt

Downstream application belongs to the cycle-level integrated result, not to one subsystem.

One receipt is emitted per named destination/application boundary:

```text
StateApplicationReceipt
  schema = symthaea.spine.000b.application.v2
  authority = measurement-only

  evidence_lineage
  cycle_number
  integration_receipt_id
  destination
  consumer_stage
  source_channel

  before_bits
  after_bits
  applied
  state_changed
```

Definitions:

- `applied=true`: the integrated channel participated in the production update at this boundary.
- `state_changed=true`: `before_bits != after_bits` after the production update.

These facts are intentionally separate. Clamping, saturation, or an exact no-op may produce `applied=true` with `state_changed=false`.

This receipt does **not** prove later behavioral consumption.

Use the phrase **state-application consumed** only when `applied=true`.

## 11. No per-subsystem downstream attribution yet

A subsystem with `integration_changed=true` may be joined analytically to a cycle whose integrated result was applied downstream, but SPINE-000B MUST NOT conclude:

```text
subsystem S caused destination D to change
```

from those two facts alone.

The integrated result may contain interactions among multiple contributors and legacy inline paths may still overlap.

Per-subsystem behavioral causality remains a SPINE-000D matched-intervention question.

## 12. RuntimeTelemetryEnvelope is noncanonical

Wall-clock timing is intentionally excluded from canonical semantic replay.

```text
RuntimeTelemetryEnvelope
  execution_receipt_id
  duration_ns
  serialization_duration_ns
  host/runtime diagnostics
```

These fields may vary across exact semantic replays and MUST NOT participate in canonical receipt hashes or byte-identical replay requirements.

Canonical replay compares only deterministic semantic receipts.

## 13. Evidence-run failure semantics

Instrumentation failure MUST NOT mutate cognition to preserve telemetry completeness.

If canonical evidence cannot be recorded correctly:

```text
cognition continues according to production semantics
measurement run = INVALID / INCOMPLETE
```

The evidence harness must fail or mark the run unusable. It must not synthesize missing receipts.

## 14. Duplicate identity gate

Before dynamic qualification begins, registered subsystem identities must satisfy:

```text
all names non-empty
all names unique
```

Duplicate names make leave-one-out attribution ambiguous and fail qualification before receipts are interpreted.

## 15. Canonical ordering

Canonical persisted ordering:

```text
SubsystemExecutionReceipt:
  (cycle_number, subsystem_name)

CycleIntegrationReceipt:
  cycle_number

LeaveOneOutInfluence:
  subsystem_name

StateApplicationReceipt:
  (cycle_number, consumer_stage, destination, source_channel)
```

Runtime scheduling order may be recorded separately when useful, but semantic persistence must be deterministic.

## 16. Duplicate-path context

Static SPINE-000C overlap evidence is linked by stable references rather than duplicated into every receipt payload.

A derived report may show:

```text
subsystem_name -> explicit static overlap witness(es)
```

Absence of an explicit witness never proves absence of overlap.

## 17. Required negative controls

Before runtime evidence is interpretable, qualification MUST establish:

1. schedule-skipped => no output, no admission, no influence;
2. health-disabled => no output, no admission, no influence;
3. caught panic => distinguishable from real neutral, no proposal, no admission;
4. real neutral => emitted exact neutral proposal, not admitted;
5. unique scalar contributor => only expected proposal-value channel changes;
6. identical duplicated scalar contributors => influence matches current averaging semantics;
7. unique flag => unique flag attribution present;
8. duplicated flag => emitted/admitted but not uniquely attributed;
9. non-admitted subsystem => no fabricated leave-one-out structure;
10. applied-but-clamped/no-op destination => `applied=true`, `state_changed=false`;
11. instrumentation disabled => preregistered cognitive observables unchanged;
12. telemetry serialization failure => cognition unchanged, evidence run invalid;
13. duplicate subsystem name => qualification fails closed;
14. exact semantic replay => canonical receipts byte-identical;
15. timing metadata may differ without invalidating canonical semantic replay.

## 18. Claim boundary

SPINE-000B R2 may establish:

> Under one exact qualified execution subject, named subsystem scheduling/execution outcomes were observed; exact returned proposals were or were not admitted according to production collector semantics; for admitted contributors, production leave-one-out integration did or did not change named proposal-value channels or uniquely contributed flags; and the cycle-level integrated result was or was not applied at named state boundaries.

It does not establish:

- that a subsystem is behaviorally necessary;
- that removing the subsystem changes final behavior;
- that a proposal is beneficial;
- intelligence;
- consciousness;
- epistemic authority;
- semantic truth.

`LOAD_BEARING` remains reserved for SPINE-000D matched interventions.
