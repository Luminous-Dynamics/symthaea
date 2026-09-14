# GEOM-003D0A3 — Canonical deterministic cycle-state receipt

Status: implementation candidate. No target GWT lesion outcome was used to choose receipt fields.

Parent programs: #3146, #3157
Implementation issue: #3163
Exact predecessor: `bba5e615d67f39b2d8bde90518980cb77c7922fd`

## Purpose

D0 Gate 0 requires stronger evidence than equality of the 32-dimensional thought-vector observation alone. Two matched replicas could emit the same observation while strategy, GWT, FEP, neuromodulatory, or control state has already diverged.

D0A3 defines a curated scientific-state receipt from public `CycleResult`/`CycleMetadata` values. It is measurement-only and has no feedback path into the cognitive loop.

## Exact representation

All included `f32` and `f64` values are stored as raw IEEE bit patterns. No epsilon, rounding, string formatting, or numeric tolerance is used.

The typed receipt is serialized with the pinned workspace bincode implementation and committed with BLAKE3 under the domain:

`SYMTHAEA_GEOM_SCIENTIFIC_CYCLE_RECEIPT_V1\0`

The typed receipt remains authoritative. The digest is a compact evidence index, not a replacement for field-level diagnostics.

## Included state

The receipt includes:

- native output, prediction error, peak attention, detected primitives, learning/training/compression/recall state, and complete thought vector;
- a BLAKE3 commitment to the full 16,384-bit `wisdom_hv` bytes;
- selected strategy, reasoning confidence, and actual effective learning rate;
- GWT broadcast, coalition size, attention-budget status, built-in memory-handler firing, and perception-handler count;
- consciousness-level and EquationV2 outputs;
- FEP action, surprise, TD error, exploration modulation, coherence velocity, and metacognitive accuracy;
- circadian phase/plasticity/hour/effective-hour/timezone and effective DA/NE/5-HT/ACh signals.

These families overlap existing longitudinal instrumentation rather than inventing a new GEOM-only cognitive score.

## Explicit exclusions

The projection does not read:

- `CycleResult.cycle_time_us`;
- `CycleMetadata.cycle_duration_us`;
- `module_timings_us`;
- wall-clock/snapshot timestamps;
- language text/source;
- canvas/render/audio/UI outputs;
- identity signatures or network transport metadata.

Those values may be preserved separately as operational evidence, but they cannot fail exact scientific-state equality merely because the scheduler, renderer, or host clock differed.

## D0 Gate 0

For every corresponding intact/sham cycle:

1. require exact native `thought_vector` equality;
2. construct a D0A3 receipt for each arm;
3. require typed receipt equality and equal domain-separated digest;
4. on mismatch, record `differing_fields()` and fail the reproducibility gate;
5. never estimate a tolerance from observed sham mismatch.

A receipt mismatch is evidence of uncontrolled state divergence, not measurement noise.

## Current controls

The candidate unit-controls:

- exact receipt/digest equality;
- single-bit floating-point divergence;
- wisdom-HV commitment divergence;
- strategy/GWT/control participation;
- circadian/neuromod participation.

The source projection itself is intentionally the proof that timing/presentation fields are excluded: it never reads them. A future real-loop D0 harness must additionally mutate/observe operational timing independently while showing receipt equality.

This tranche does not authorize lesion execution and does not define a consciousness score, causal-emergence metric, or gravity analogue.
