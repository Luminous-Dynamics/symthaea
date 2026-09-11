# Symthaea Space Exosuit — SX-020 Integrated EVA Campaign v0.1

## Status

Simulation campaign only. This document does **not** establish flight readiness,
human rating, clinical/physiological validity, spacesuit qualification, or
superiority over AxEMU/xEVAS/EMU.

## Goal

Move from isolated subsystem tests to replayable whole-EVA scenarios where
multiple evidence-bearing subsystems can degrade at once while their authority
boundaries remain independent.

The campaign composes existing models for:

- whole-EVA mission resource accounting;
- dust/durability effects;
- powered-glove workload and fail-passive behavior;
- PLSS primary/secondary paths;
- protected multi-bus power;
- radiation preflight/return logic;
- local safe-haven selection;
- rover/habitat service availability.

It intentionally does **not** create a new monolithic physics engine.

## Initial scenarios

1. `Nominal`
2. `DustAndDexterityDegradation`
3. `MobilityPowerShortfall`
4. `PrimaryOxygenPathLoss`
5. `SolarParticleEventCommBlackout`
6. `GlovePowerLoss`
7. `ReturnNeededServiceAndShelterUnavailable`

Each scenario produces an explicit integrated disposition:

```text
Continue
 -> Degrade
 -> ReturnToSafeHaven
 -> ImmediateShelter
 -> NoFeasibleSafeHaven
 -> Abort
```

Subsystem results are retained alongside the integrated disposition so an
escalation can be traced back to the actual source rather than hidden inside one
opaque health score.

## Containment invariants

- mobility-power shortfall must push work back to the human / degrade the
  mission rather than consume protected survival power;
- primary oxygen-path failure may use the independently modelled secondary path
  but may not invent oxygen;
- an energetic-particle alert must preflight-veto nominal work before mission
  oxygen/power/time are spent;
- glove power loss must remove active glove force and preserve a backdrivable
  mechanical state;
- dust/service/safe-haven failures must remain distinguishable rather than
  collapsing into a generic `fault` flag;
- unavailable service or shelter may not be invented simply to make a mission
  recoverable;
- all integrated outputs remain recommendations/evidence, not unrestricted
  actuator authority.

## Explicit fidelity gaps

The campaign records approximation notes in every report. Two important gaps are
currently deliberate:

1. Dust-derived radiator heat-rejection fraction is represented in the
   durability decision layer, but the existing `IntegratedEvaMission` harness
   does not yet inject that fraction into its internal PLSS step.
2. Powered-glove hand work/fatigue is modelled by the glove twin but is not yet
   folded into whole-body metabolic workload.

Those gaps are future integration targets, not hidden assumptions.

## Why preserve separate solvers

The campaign is intended to answer system questions without destroying the
architecture that makes later verification possible:

```text
high-fidelity / subsystem model
        |
        v
explicit evidence-bearing output
        |
        v
campaign composition
        |
        v
integrated recommendation
```

A future validated thermal model can therefore replace today's radiator
assumption without rewriting glove control, radiation policy, or suitport
servicing.

## Next evidence steps

```text
software replay
 -> calibrated subsystem models
 -> hardware-in-loop power/PLSS/glove rigs
 -> dusty thermal-vacuum integration
 -> rover/suitport analog
 -> instrumented human-in-loop mission analog
 -> reduced-gravity / vacuum testing
 -> program qualification
```

Simulation success at this stage is evidence that software containment and
accounting behave as intended. It is not evidence that the physical suit is safe
for a person.
