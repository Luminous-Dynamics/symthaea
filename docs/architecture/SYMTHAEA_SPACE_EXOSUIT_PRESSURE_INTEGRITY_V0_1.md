# Symthaea Space Exosuit — Pressure Integrity & Damage Containment v0.1

**Status:** simulation/research architecture only  
**Scope:** SX-021  
**Authority:** observation, prediction, and planning only; no oxygen-regulator, purge, pressure-valve, relief-valve, mobility, or hatch authority.

## Goal

Turn the coarse `pressure_integrity` observation already consumed by the powered-assist safety kernel into an evidence-bearing causal model:

`damage -> effective leak area -> gas loss -> pressure-loss rate -> time-to-threshold -> localization -> patch evidence -> safe-haven planning`.

The model must never turn a software prediction into survival-hardware authority.

## External engineering context

This architecture is motivated by, but does not claim equivalence with, NASA EVA hardware or qualification evidence.

- NASA xEMU MMOD risk work treats pressure-garment penetration as a life-critical hazard because a sufficiently large leak can prevent safe return to an airlock.  
  https://ntrs.nasa.gov/citations/20240005191
- NASA-STD-3001 Volume 2 requires spacesuits to maintain declared pressure behavior and treats pressure suits as protection during high-risk depressurization/MMOD operations.  
  https://www.nasa.gov/reference/11-0-spacesuits-vol-2/
- NASA-supported EVA material research includes damage-tolerant glove palms and self-healing pressure-bladder/composite concepts. These are candidate technologies, not assumed performance in this model.  
  https://techport.nasa.gov/projects/9430  
  https://techport.nasa.gov/projects/89616  
  https://techport.nasa.gov/projects/33505
- NASA operational history and standards emphasize explicit leak checks, anomaly assessment, safe-haven posture, and repair evidence rather than treating temporary pressure stability as proof that a leak is gone.

## Core model

### Pressure zones

The simulation distinguishes:

- torso;
- helmet;
- left/right arm;
- left/right leg;
- left/right glove;
- PLSS interface;
- suitport interface.

This is localization metadata, not a claim that these are independently pressure-isolatable compartments.

### Damage sources

Typed sources include:

- MMOD impact;
- secondary lunar-regolith ejecta;
- abrasion puncture;
- cut/tear;
- seal leak;
- joint leak;
- connector damage;
- bladder damage;
- unknown.

A damage site carries an effective leak-orifice area, optional growth rate, patch state, patch residual fraction, and evidence level.

### Leak physics

The v0.1 reference model uses an idealized compressible-orifice mass-flow calculation with explicit configurable:

- free volume;
- specific gas constant;
- heat-capacity ratio;
- discharge coefficient;
- ambient pressure;
- gas temperature.

These values are trade-study inputs. They are not xEMU/AxEMU/EMU specifications and are not suitable for human-rating decisions without calibration against hardware.

### Makeup flow

Observed makeup flow is modeled separately from leak flow.

A regulator can therefore temporarily maintain pressure while gas is still being lost. The architecture explicitly preserves:

`stable pressure != repaired pressure boundary`.

Makeup flow is an observation only. SX-021 never commands the regulator.

## Sensor fusion

Pressure readings carry confidence and age. Only fresh/high-confidence readings enter the weighted pressure estimate.

Excess disagreement fails closed to `ContradictoryPressureSensors`; the system does not average mutually incompatible pressure evidence into a reassuring number.

Leak-localization observations similarly carry:

- zone;
- anomaly score;
- confidence;
- age.

Low-confidence or stale localization cannot authorize a patch recommendation.

## Patch model

A patch is represented only by a residual effective-leak fraction plus evidence state:

- none;
- temporary applied;
- temporary verified;
- permanent repair.

`TemporaryApplied` is not equivalent to `TemporaryVerified` in the evidence model, even if a trade-study caller supplies the same numerical residual fraction.

No self-healing or patch technology receives a default guaranteed effectiveness.

## Safe-haven coupling

The pressure clock can dominate every other endurance estimate.

If PLSS consumables imply 120 minutes of independent endurance but pressure integrity predicts critical pressure in 9 minutes, the safe-haven planner receives 9 minutes.

The declared reserve is never silently relaxed. If pressure-limited endurance is already below the required reserve, the response becomes `NoFeasibleSafeHaven` rather than changing the reserve until a route appears feasible.

## Powered-assist coupling

`PressureIntegrityAssessment.integrity_fraction` may only reduce the existing `SuitSafetyState.pressure_integrity` observation.

It cannot increase a previously lower integrity estimate and cannot command exoskeleton motors.

The existing deterministic assist supervisor retains authority over whether powered assistance is permitted.

## Deterministic fault campaign

SX-021 exercises:

1. intact nominal suit;
2. MMOD puncture;
3. propagating/growing tear;
4. contradictory pressure sensors;
5. stale leak localization;
6. makeup flow masking net pressure decay;
7. verified patch reducing modeled leak;
8. leak clock shorter than every reachable safe haven.

The campaign checks containment properties rather than requiring every scenario to recover successfully.

## Explicit non-claims

SX-021 does **not** establish:

- flight readiness;
- human rating;
- actual AxEMU/xEMU/EMU leak performance;
- actual suit free volume or gas composition;
- MMOD ballistic-limit performance;
- validated hole-size inference from pressure decay;
- actual emergency patch effectiveness;
- self-healing material reliability;
- physiological critical-pressure thresholds;
- certification or superiority over any existing suit.

All current numerical thresholds and coefficients remain `Simulation` evidence.

## Evidence ladder

`software simulation`
→ `calibrated pressure-vessel/bladder bench rig`
→ `instrumented puncture/leak tests`
→ `patch/self-healing coupon tests`
→ `pressurized articulated garment rig`
→ `vacuum/thermal + abrasion/MMOD-representative testing`
→ `human-in-loop suited tests under approved protocols`
→ `integrated EVA-system testing`
→ `program qualification`.

Simulation evidence cannot self-promote to a higher rung.

## Next fidelity targets

1. Calibrate leak-area inference against measured pressure-decay and makeup-flow data.
2. Replace generic localization scores with declared sensor modalities and confusion/error models.
3. Add MMOD/secondary-ejecta exposure probability as a separate risk model; do not conflate impact probability with post-impact leak physics.
4. Close radiator-dust -> PLSS thermal coupling in the integrated mission harness.
5. Fold powered-glove hand work into whole-body metabolic accounting without double counting.
6. Add human-state estimates only as evidence inputs to deterministic safety limits.
