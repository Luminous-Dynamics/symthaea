# SPINE-000B — Application Observer Completeness v1

**Status:** preregistered architecture; runtime observer implementation pending

**Authority:** measurement-only

**Issue:** #3356

A1 freezes the theorem required before actual `StateApplicationReceipt` evidence can be trusted:

```text
registered production operation executes
<=>
exactly one corresponding application event is captured
```

This is a completeness/non-fabrication theorem for actual-cycle observation. It is not a subsystem-causality theorem.

## 1. Source-order safety boundary

The subsystem integration/application block lives in `src/cognitive_loop/cycle_phase_output/mod.rs`.

The cognition-affecting wall-clock planning budget is evaluated earlier in `phase_dynamics`. Within `phase_output`, the first `metadata.cycle_duration_us` sample occurs before subsystem integration; the later `cycle_start.elapsed()` sample populates final `CycleResult.cycle_time_us` telemetry.

A1 therefore permits bounded capture at the real application seam, but expensive encoding/hashing/export remains deferred until the full application sequence is complete.

Any future source-order drift that moves a cognition-affecting wall-clock decision after the application seam invalidates this preregistration.

## 2. Observer-local operation IDs

The live seam must not allocate/copy canonical operation strings per event.

`docs/research/SPINE_000B_OPERATION_OBSERVER_IDS_V1.json` freezes append-only `u16` observer IDs for every v1 Phase-C registry `operation_id`.

Rules:

- ID 0 is invalid/reserved;
- existing IDs never change meaning;
- retired operations keep their historical IDs;
- IDs are never reused;
- new operations append new IDs greater than the current maximum;
- canonical C1 operation strings are resolved only in deferred evidence derivation.

Changing any existing ID mapping starts a new observer-ID lineage.

## 3. Hot-seam event

The implementation lineage may capture only bounded data needed to reconstruct C1 application evidence:

```text
application_index u32
observer_operation_id u16
source_tag u8
source_flag u32
source_condition u8
applied bool
applied_argument_kind/tag + exact bits when stable
observation kind/tag
before exact value when observed
after exact value when observed
```

The concrete POD layout may be optimized, but it must preserve this semantic surface exactly.

The hot seam MUST NOT perform SHA-256, JSON serialization, filesystem/network I/O, RNG consumption, unbounded allocation, manager callbacks, or leave-one-out integration.

## 4. Event capacity

A1 derives the theoretical maximum actual application fan-out from the frozen Phase-C registry.

For scalar sources, count all applications.

For each flag source, `FLAG_SET` and `FLAG_CLEAR` are mutually exclusive in one integrated result, so use the maximum application count across those source-condition groups. Feature-gated operations are included when computing the theoretical maximum, because the bound must remain valid under the maximal supported feature profile.

For the current v1 registry:

```text
operation identities in registry = 18
maximum applications in one cycle = 17
v1 observer buffer capacity       = 32
```

The 32-event capacity is qualification infrastructure, not a cognitive limit.

If a future registry can exceed 32 events in one cycle, A1 must fail before runtime qualification.

## 5. Overflow law

An implementation must never overwrite old events or reallocate unbounded memory at the application seam.

On event 33:

```text
application_observer_overflow = true
application evidence for cycle = UNQUALIFIED
production operation still executes normally
```

No runtime behavior may be changed merely to keep evidence within the observer capacity.

## 6. Completeness and non-fabrication

A1-DYNAMIC must eventually prove both directions for every registry operation under the relevant feature/condition profile:

```text
operation executed -> one event
one event -> corresponding operation executed
```

No event may be synthesized later from metadata or from an R2 projection.

Multi-operation flag fan-out must preserve actual execution order through contiguous `application_index` values.

## 7. Before/after integrity

For observable boundaries:

```text
before = exact value immediately before the real operation
after  = exact value immediately after the real operation
```

`StateChangeStatus` is derived later from the captured values. It is never guessed from the applied argument.

For complex/external boundaries the registry may require `NOT_OBSERVED_AT_BOUNDARY`; no scalar proxy may be invented.

## 8. Static preregistration gate

A1-STATIC must fail if:

- the application registry schema drifts;
- any registry operation lacks exactly one observer ID;
- an observer ID is duplicated/zero/out of u16 range;
- current mappings are not append-only/contiguous for v1;
- theoretical maximum per-cycle fan-out exceeds 32;
- the output-phase ordering anchors cannot be found;
- subsystem integration no longer lies after the early metadata duration sample and before the final cycle-time telemetry sample;
- a later output-phase wall-clock control gate appears between integration and cycle completion;
- runtime application-observer implementation symbols appear in this preregistration lineage.

A1-STATIC establishes architecture consistency only.

## 9. Dynamic qualification requirements

A later implementation lineage must add source-bound hooks and mutation controls proving:

- every scalar application path;
- every consumed flag set path;
- `REQUEST_GEODESIC` clear path;
- feature-off absence;
- duplicate/drop detection;
- contiguous application indices;
- exact applied-argument bits;
- exact before/after ordering;
- explicit complex `NOT_OBSERVED_AT_BOUNDARY` handling;
- overflow qualification failure without cognitive behavior change.

## 10. Claim boundary

A1 may eventually establish:

```text
actual Phase-C application observation is complete and non-fabricated
```

It may not establish:

```text
subsystem S caused destination D to change
I_withoutS would execute its projected operation
state change was beneficial
subsystem is load-bearing
```

Those require later A2/SPINE-000D evidence.