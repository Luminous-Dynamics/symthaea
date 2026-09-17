# FIELD Physical I/O Architecture

Status: Draft
Issue: #3589 (`FIELD-000`)

## Purpose

FIELD is Symthaea's canonical architecture for calibrated physical observations,
multimodal physical-state fusion, model-prediction residuals, and — only in later,
separately qualified tranches — authority-mediated physical actuation.

FIELD does **not** replace existing physics domains. It binds them to live or
simulated evidence without collapsing measurement, inference, prediction, and
control into one interface.

## Existing ownership boundaries

The following existing modules remain authoritative for their current scopes:

- `symthaea-acoustics`: applied physical acoustics.
- `symthaea-optics`: applied geometric optics.
- `symthaea-core::physics::acoustics`: exploratory/HDC acoustic concepts.
- `symthaea-core::physics::optics`: exploratory/HDC optical, nonlinear,
  quantum-optical, fiber, holographic, and laser concepts.
- `symthaea-continuum-physics`: continuum solvers, including acoustics and
  plasma MHD.
- `src/perception/sensor_fusion.rs`: real/simulated sensor-to-`ContinuousHV`
  perception seam.
- `symthaea-physics-bridge`: physics catalog/search knowledge into cognition.
- `symthaea-hal`: physical authority, admission, safety-case, audit, watchdog,
  and output-gating boundary.

FIELD must bridge these layers rather than fork their physics equations or their
authority semantics.

## Fundamental semantic separation

The following are distinct semantic classes and MUST NOT be interchangeable:

1. **Observation** — evidence emitted by a physical, simulated, or replay
   source under an explicit calibration/clock/frame context.
2. **Inference** — a derived estimate based on one or more observations.
3. **Prediction** — a model output describing an expected observation/state.
4. **Residual** — an evidence-bearing comparison between compatible prediction
   and observation records.
5. **Intent** — a requested experimental effect or measurement goal.
6. **Command** — an authority-scoped device/runtime instruction produced only
   after a later physical-control boundary admits an intent.

No API may silently promote one class into another.

## Primary data path

```text
physical / simulated / replay source
                |
                v
       FieldObservationV1
                |
                v
 validation + calibration +
 clock/frame normalization
                |
                v
          FieldFrameV1
          /          \
         v            v
 raw provenance      field -> HDC
 / evidence ref          |
                         v
                 perception / cognition
                         ^
                         |
              physics prediction
                         |
                         v
          prediction <-> observation
                    residual
```

The raw observation/provenance path remains addressable even after an HDC
representation has entered cognition. HDC is a cognitive representation, not a
replacement for evidence.

## Physical-control boundary

FIELD-000 through FIELD-008 are observation/prediction/cognition work and MUST
NOT create physical actuation.

FIELD-009 defines simulation-only intent semantics. Simulated results return
through the same observation path and remain explicitly marked as simulated.

Any future physical actuation is permitted only through a separately qualified
adapter into `symthaea-hal`:

```text
cognition
   |
   v
FieldIntent
   |
   v
HAL policy / authority / admission / safety case
   |
   v
independent physical output gate
   |
   v
physical device
   |
   v
independent observation -> FIELD
```

Cognition never directly energizes hardware and never receives an API for
arbitrary low-level voltage/current/power writes.

Command success is not evidence that a requested physical effect occurred.
Independent observations must establish the resulting physical state.

## Observation invariants

A trusted physical observation must make the following semantics explicit:

- modality / field family;
- measured quantity;
- unit and scale;
- source class (`Physical`, `Simulated`, or `Replay`);
- source identity;
- coordinate-frame identity;
- capture clock domain and timestamp;
- local receive/ingress monotonic time where applicable;
- calibration identity and validity state;
- uncertainty semantics;
- provenance/evidence identity;
- data-quality/validity flags.

Missing information is not physical zero. Unknown, unavailable, clipped,
saturated, stale, invalid-calibration, and out-of-range states must remain
representable.

## Clock invariant

Clock domain is part of observation semantics.

A wall-clock timestamp must never be treated as monotonic. A monotonic timestamp
must never be interpreted across a process/boot epoch without explicit epoch or
synchronization evidence.

The current IMU sensor type documents its timestamp both as monotonic capture
time and as microseconds since epoch. FIELD-001 must remove this ambiguity in the
new canonical observation contract without silently reinterpreting historical
IMU evidence.

## Units and frames

A numeric value is insufficient physical evidence without quantity, unit, and
frame semantics.

- Unit normalization must be deterministic and explicit.
- Incompatible quantity/unit pairs fail closed.
- Coordinate transforms must carry identity/version/provenance.
- A transform is evidence, not an implicit convenience conversion.
- Cross-frame fusion without an admitted transform is invalid.

## Calibration and uncertainty

Calibration and uncertainty are first-class evidence.

- Calibration identity is bound to the observation.
- Expired/invalid calibration cannot silently become trusted data.
- Sensor uncertainty and model uncertainty remain separate.
- Fusion must not erase individual source uncertainties.
- A low residual is not automatically model validation.
- A large residual is not automatically sensor failure.

## Cross-modal fusion

Fusion must preserve disagreement.

When optical, acoustic, plasma, thermal, electromagnetic, chemical, or other
sources disagree, FIELD must retain enough constituent evidence to expose the
conflict. An averaging operation must not hide source disagreement or erase its
provenance.

Temporal lag/correlation may be represented, but correlation alone is not a
causal claim.

## PhysicsBridge separation

`symthaea-physics-bridge` currently connects physics catalog/search results to
HDC/cognition. That is a knowledge/reasoning bridge.

FIELD is the live/simulated evidence bridge.

The two may interact through explicit typed interfaces, but a catalog result,
textbook equation, or model hypothesis must never be mislabeled as a sensor
observation.

## Plasma regime separation

The existing continuum plasma module contains ideal/MHD quantities appropriate
to fusion/space/plasma-physics contexts. Cold atmospheric plasma diagnostics,
reactive chemistry, and treatment contexts are not automatically equivalent to
that regime.

FIELD-005 must preserve explicit regime semantics and distinguish measured,
inferred, and solver-derived quantities.

## HDC/LTC role

FIELD should use HDC as a deterministic multimodal representation layer and LTC
as a temporal-dynamics layer only after physical semantics have been validated.

The preferred relation is:

```text
canonical observation
      |
      +----> raw evidence / provenance
      |
      v
role/filler HDC encoding
      |
      v
HDC/LTC cognition
```

The encoder must preserve, either in the HDC roles or attached metadata,
modality, quantity, normalized value, uncertainty, calibration/source class,
and evidence identity linkage.

## Prediction-residual architecture

A central FIELD primitive is the comparison between expected and observed
physical state.

For a scalar compatible quantity:

```text
residual = observation - prediction
```

For multimodal/vector/HDC observations, the residual may use an explicitly named
metric over compatible canonical representations.

A residual record must bind:

- prediction/model identity;
- observation/frame identity;
- temporal-alignment evidence;
- unit/frame compatibility;
- predicted uncertainty;
- observed uncertainty;
- residual metric/value/structure;
- any candidate explanations explicitly as hypotheses.

For prospective validation, the prediction must be evidence-bound before the
held-out observation is examined.

## Safety invariant

Software authorization is not a substitute for physical interlocks,
output-enable/e-stop wiring, watchdogs, containment, or independent shutdown
hardware.

Any later laser, ultrasound/acoustic, plasma, or related physical-control path
must preserve HAL's existing separation of operator identity, signed authority,
command ingress, time integrity, safety-case evidence, audit continuity, and
independent physical output gating.

## FIELD issue sequence

- #3589 — FIELD-000: architecture and invariants
- #3590 — FIELD-001: canonical observation / clock / provenance
- #3591 — FIELD-002: deterministic field-to-HDC encoding
- #3592 — FIELD-003: acoustic observation plane
- #3593 — FIELD-004: optical/laser observation plane
- #3594 — FIELD-005: plasma diagnostics observation plane
- #3595 — FIELD-006: cross-modal synchronization and fusion
- #3596 — FIELD-007: digital twin / observation residual engine
- #3597 — FIELD-008: cognitive-loop multiphysics perception
- #3598 — FIELD-009: simulation-only field intents
- #3599 — FIELD-010: HAL-mediated physical actuation authority
- #3600 — FIELD-011: bounded multiphysics experiment benchmark suite

## FIELD-000 exit gate

FIELD-000 is complete when review establishes that:

1. ownership boundaries are explicit;
2. observation/inference/prediction/residual/intent/command cannot be confused;
3. raw evidence remains addressable after HDC fusion;
4. clock/unit/frame/calibration/uncertainty semantics are fail-closed;
5. `PhysicsBridge` and physical evidence have separate roles;
6. FIELD-000 through FIELD-008 expose no physical actuation path;
7. future actuation is structurally constrained to HAL and independent physical
   safety mechanisms;
8. no existing evidence bytes are reinterpreted by this architecture-only
   tranche.
