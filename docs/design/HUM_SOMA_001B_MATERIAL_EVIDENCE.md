# HUM-SOMA-001B — Material Compliance, Hysteresis, and Fatigue Evidence

Status: source-design candidate
Issue: #4736
Authority: mechanical evidence only; **no human-contact or motor authority**

## Purpose

Replace vague `softness` assumptions with provenance-bearing evidence about the current mechanical response of an exact somatic layer/material/specimen state.

## Evidence identity

Each evidence snapshot binds:

- exact evidence identity;
- somatic profile identity;
- layer identity;
- material lot identity;
- specimen identity;
- test/method identity;
- evidence origin;
- capture + validity window;
- test environment;
- fatigue/cycle context;
- explicit mechanical-response claims.

## Evidence origins

V1 distinguishes:

- bench-measured evidence;
- simulation/model evidence;
- supplier-declared evidence.

These origins are deliberately not interchangeable. A downstream physical-evidence join can require bench measurement without allowing simulation or declaration to masquerade as measurement.

Even bench evidence does not automatically establish human-contact qualification; it remains one input into later qualification.

## Mechanical response

The contract can represent sampled compression and shear stress/strain response plus optional:

- hysteresis/energy-loss fraction;
- permanent-set fraction;
- creep fraction.

There is intentionally no single scalar `softness` field.

Future material models may add rate dependence, relaxation curves, anisotropy, tear/puncture/delamination evidence, and richer constitutive models, but those stronger claims should be independently evidenced.

## Environmental context

Evidence records test temperature, optional relative humidity, and wet/dry condition. A response established under one environment cannot silently be generalized to another without a separate qualification rule.

## Fatigue / aging

Evidence includes prior representative cycle count and campaign cycle count. Long-duration interface state is later composed with HUM-SOMA-001M rather than assuming a newly qualified material remains equivalent forever.

Aged/fatigued material cannot inherit fresh-material qualification merely because the material name or profile ID is unchanged.

## Freshness

Evidence is time-bounded. Stale measurement is not current evidence.

Later systems should intersect:

```text
exact interface profile
    ∩ exact layer/material lot
    ∩ current maintenance/fatigue state
    ∩ fresh mechanical evidence
    ∩ tactile/calibration evidence
    ∩ consent
    ∩ contact safety envelope
```

before any comfort-related physical adaptation becomes eligible.

## Epistemic rules

- missing evidence stays missing;
- simulation stays simulation;
- supplier declaration stays declaration;
- bench measurement stays tied to exact specimen/test conditions;
- preference cannot widen the demonstrated mechanical envelope;
- comfort optimization cannot reinterpret stale evidence as current;
- model predictions and measurements remain independently inspectable.

## Nonclaims

This tranche establishes no production material choice, human-tissue safety, biocompatibility, hygiene, safe pressure/force/temperature threshold, actuator command, motor authority, human-trial authorization, medical claim, or product certification.
