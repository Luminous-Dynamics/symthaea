# SPINE-000B Runtime Receipt Contract R2

Status: preregistered measurement-only successor to the original SPINE-000B monolithic receipt draft.

Issue: #3228

This contract is frozen before runtime instrumentation. It is bound to the **live Phase B manager execution seam**, not to an unused/helper-only abstraction.

Current production Phase B establishes the relevant ordering:

```text
manager.should_run(cycle, urgency)
        ↓
run_subsystem! macro
        ↓
health.is_faulted(name)
        ↓
catch_unwind(manager.process(snapshot))
        ↓
Ok(output) ──→ health.record_success(name)
        │       collector.record(name, output)
        │
        └ Err ─→ health.record_panic(name)
```

`OutputCollector::record()` filters neutral outputs.

This live structure is the observational seam SPINE-000B R2 instruments.

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

A derived analysis view may join these records by stable IDs, but canonical evidence remains normalized.

This prevents cycle-level integration or downstream state application from being misrepresented as uniquely owned by one subsystem.

## 2. Stable canonical identifiers

Every canonical record has a deterministic semantic identifier derived from its lineage and natural key.

The implementation may choose the repository's qualified canonical commitment utility, but the semantic key must be equivalent to:

```text
ExecutionReceiptId = H(
  schema,
  evidence_lineage,
  cycle_number,
  subsystem_name
)

IntegrationReceiptId = H(
  schema,
  evidence_lineage,
  cycle_number
)

ApplicationReceiptId = H(
  schema,
  evidence_lineage,
  cycle_number,
  consumer_stage,
  destination,
  source_channel
)
```

Do not use wall-clock time or process-local pointer identity in canonical IDs.

## 3. Canonical SubsystemExecutionReceipt

One canonical execution receipt exists for every registered subsystem scheduling decision in the qualified manager set.

```text
SubsystemExecutionReceipt
  schema = symthaea.spine.000b.execution.v2
  authority = measurement-only
  causal_load_claimed = false

  receipt_id
  evidence_lineage
  cycle_number
  subsystem_name
  subsystem_type
  source_path
  schedule_interval
  urgency

  eligible_to_run
  execution_attempted
  execution_completed
  execution_outcome
  emitted

  proposal: Option<ExactProposal>

  collector
    admitted
    contributor_count_before
    contributor_count_after
```

`subsystem_name` MUST be non-empty and unique among registered subsystem identities for the qualified subject.

## 4. ExecutionOutcome truth table

Canonical outcomes:

```text
SKIPPED_SCHEDULE
SKIPPED_HEALTH_DISABLED
EXECUTED_NEUTRAL
EXECUTED_NON_NEUTRAL
PANICKED_CAUGHT
```

Exact semantics:

```text
Outcome                  eligible  attempted  completed  emitted  proposal   admitted
----------------------------------------------------------------------------------------
SKIPPED_SCHEDULE         false     false      false      false    None       false
SKIPPED_HEALTH_DISABLED  true      false      false      false    None       false
PANICKED_CAUGHT          true      true       false      false    None       false
EXECUTED_NEUTRAL         true      true       true       true     Some(P)    false
EXECUTED_NON_NEUTRAL     true      true       true       true     Some(P)    true
```

`eligible_to_run` means `should_run()` returned true.

`execution_attempted` means control entered the panic-isolated `process()` call.

`execution_completed` means `process()` returned normally with a `SubsystemOutput`.

`emitted` means a real subsystem output was returned. A panic's effective absence of proposal must not be represented as a real neutral emission.

Instrumentation/serialization failure is not a subsystem outcome. It invalidates or truncates the evidence run at the measurement layer.

## 5. Live Phase B observation points

Instrumentation MUST observe the existing production branches without changing them:

1. immediately after `should_run()` determines eligibility;
2. at the `run_subsystem!` health-disabled check;
3. at the existing `catch_unwind` `Ok(output)` / `Err(payload)` split;
4. immediately before/after `subsystem_collector.record(name, output)`.

Do not refactor the macro merely to make instrumentation aesthetically cleaner unless that refactor receives a separate behavioral-equivalence lineage.

In particular, SPINE-000B does **not** require routing live execution through `safe_process()`.

## 6. Neutral output semantics

Current `OutputCollector::record()` admits only non-neutral outputs.

Therefore:

```text
EXECUTED_NEUTRAL
emitted = true
admitted = false
```

The exact returned neutral output is captured in the execution receipt, but it is not a collector contributor and receives no leave-one-out influence entry.

Never create a fake zero/default integration result for a non-admitted subsystem.

## 7. Exact proposal representation

`ExactProposal` preserves canonical bits:

```text
confidence_delta_bits  = f64::to_bits()
lr_modulation_bits     = f64::to_bits()
exploration_delta_bits = f64::to_bits()
arousal_delta_bits     = f32::to_bits()
valence_delta_bits     = f32::to_bits()
flags                   = exact u32
is_neutral              = current production predicate
```

Decimal renderings may exist only as convenience metadata.

## 8. Canonical CycleIntegrationReceipt

Emit exactly one cycle integration receipt for every qualified cognitive cycle, including zero-contributor cycles.

```text
CycleIntegrationReceipt
  schema = symthaea.spine.000b.integration.v2
  authority = measurement-only
  causal_load_claimed = false

  receipt_id
  evidence_lineage
  cycle_number
  admitted_subsystems[]
  contributor_count
  integrated_all
  leave_one_out[]
```

For an empty cycle:

```text
admitted_subsystems = []
contributor_count = 0
integrated_all = qualified neutral IntegratedOutput
leave_one_out = []
```

This preserves cycle completeness without fabricating subsystem influence.

## 9. Leave-one-out influence

For each admitted subsystem S:

```text
I_all      = production_integrate(all admitted proposals)
I_withoutS = production_integrate(all admitted proposals except S)
```

One entry is emitted:

```text
LeaveOneOutInfluence
  subsystem_name
  integrated_without_subject
  changed_channels[]
  uniquely_contributed_flags
  integration_changed
```

Both `I_all` and `I_withoutS` preserve exact scalar bits, exact flags, and contributor count.

`changed_channels` contains only proposal-value channels:

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

`n_contributors` is canonical metadata and MUST be compared for cross-implementation equivalence, but it is **not** itself an influence channel. Otherwise every removal would trivially make `integration_changed=true`.

Therefore:

```text
integration_changed =
    any proposal-value channel changed
    OR uniquely_contributed_flags != 0
```

## 10. Non-admitted integration applicability

Only admitted contributors receive leave-one-out entries.

A derived per-subsystem view represents non-admission explicitly:

```text
integration = NotApplicable(NotAdmitted)
```

Never encode unmeasured influence as an all-zero/default structure.

## 11. Canonical StateApplicationReceipt

Downstream application belongs to the cycle-level integrated result, not to one subsystem.

Emit one receipt per monitored destination/application boundary:

```text
StateApplicationReceipt
  schema = symthaea.spine.000b.application.v2
  authority = measurement-only

  receipt_id
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
- `state_changed=true`: the canonical before/after value changed.

These facts are separate. Clamping, saturation, or an exact no-op may produce:

```text
applied=true
state_changed=false
```

If Phase C skips proposal application because there are no contributors, monitored destinations may emit `applied=false` receipts when complete cycle coverage is desired.

This receipt does **not** prove later behavioral consumption.

Use the phrase **state-application consumed** only when `applied=true`.

## 12. No per-subsystem downstream causality claim

A subsystem with `integration_changed=true` may be joined to a cycle whose integrated result was applied downstream, but SPINE-000B MUST NOT conclude:

```text
subsystem S caused destination D to change
```

The integrated result may contain interactions among contributors, and legacy inline paths may still overlap.

Per-subsystem behavioral causality remains a SPINE-000D matched-intervention question.

## 13. RuntimeTelemetryEnvelope is noncanonical

Wall-clock timing and host diagnostics are intentionally excluded from semantic replay:

```text
RuntimeTelemetryEnvelope
  execution_receipt_id
  duration_ns
  serialization_duration_ns
  host/runtime diagnostics
```

These fields may vary across exact semantic replays and MUST NOT participate in canonical receipt hashes or canonical replay equality.

## 14. Evidence-run failure semantics

Instrumentation failure MUST NOT mutate cognition to preserve telemetry completeness.

If canonical evidence cannot be recorded correctly:

```text
cognition continues according to production semantics
measurement run = INVALID / INCOMPLETE
```

The harness must fail or mark the run unusable. It must not synthesize missing canonical receipts.

## 15. Duplicate identity gate

Before dynamic qualification begins:

```text
all subsystem names non-empty
all subsystem names unique
```

Duplicate names make leave-one-out attribution ambiguous and fail qualification before receipt interpretation.

## 16. Canonical ordering

Persist canonical records in deterministic order:

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

Runtime execution order may be captured separately if needed, but semantic persistence uses the canonical ordering above.

## 17. Duplicate-path context

Static SPINE-000C overlap evidence is linked by stable evidence references rather than duplicated into every receipt payload.

Absence of an explicit overlap witness never proves absence of overlap.

## 18. Canonical replay

Canonical replay equality excludes `RuntimeTelemetryEnvelope`.

The persistence implementation must freeze one canonical semantic serialization or canonical digest procedure before first runtime evidence. Exact subject replay then compares that canonical semantic representation.

Do not claim byte-identical replay across languages/formats without a frozen language-neutral canonicalization rule.

## 19. Required negative controls

Before runtime evidence is interpretable, qualification MUST establish:

1. schedule-skipped => eligible false, no execution/output/admission;
2. health-disabled after eligibility => no process attempt/output/admission;
3. caught panic => attempted true, completed false, no real proposal/admission;
4. real neutral => completed/emitted true, admitted false;
5. unique scalar contributor => only expected proposal-value channel changes;
6. identical duplicated scalar contributors => influence matches current averaging semantics;
7. unique flag => unique flag attribution present;
8. duplicated flag => emitted/admitted but not uniquely attributed;
9. non-admitted subsystem => no fabricated leave-one-out structure;
10. empty contributor cycle => neutral integration receipt with zero leave-one-out entries;
11. applied-but-clamped/no-op destination => `applied=true`, `state_changed=false`;
12. instrumentation disabled => preregistered cognitive observables unchanged;
13. telemetry serialization failure => cognition unchanged, evidence run invalid;
14. duplicate subsystem name => qualification fails closed;
15. exact semantic replay => canonical semantic receipts/digests equal;
16. timing metadata may differ without invalidating canonical semantic replay.

## 20. Claim boundary

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
