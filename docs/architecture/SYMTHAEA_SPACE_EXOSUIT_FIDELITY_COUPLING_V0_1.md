# Symthaea Space Exosuit — EVA Fidelity Coupling v0.1

**Status:** simulation/research architecture only  
**Scope:** SX-022  
**Authority:** composition/accounting only; no PLSS, exoskeleton, glove, or thermal-controller actuation authority.

## Goal

Close two explicitly documented approximation gaps in the integrated Space Exosuit mission model:

1. `radiator dust -> actual PLSS heat rejection`, not only a durability recommendation;
2. `powered-glove hand effort -> whole-body mission workload`, without allowing body-assist accounting to claim work performed by the fingers/hands.

## External engineering context

- NASA's 2026 radiator/EDS testing evaluates dust and EDS performance using thermal effectiveness/effective emittance, reinforcing that contamination must affect thermal accounting rather than only optical cleanliness:  
  https://ntrs.nasa.gov/citations/20250010063
- NASA's 2026 dust-mitigation catalog treats thermal-radiator EDS as its own technology and lunar dust as a cross-system hazard:  
  https://www.nasa.gov/dust-mitigation/
- NASA-STD-3001 Volume 2 requires muscle endurance/fatigue to be factored into system design and specifically notes forearm fatigue from repetitive force while manipulating items in pressurized EVA gloves:  
  https://www.nasa.gov/reference/4-0-physical-characteristics-and-capabilities-vol-2/
- NASA pressure-garment testing continues to evaluate elevated pressure against glove strength, fatigue, dexterity, and tactility. The existing human data are not reduced here to a universal coefficient:  
  https://ntrs.nasa.gov/citations/20250003995

These references motivate the coupling dimensions only. They do not validate the current simulation coefficients.

## PLSS environmental heat-rejection state

`PlssReferenceTwin` now carries an explicit environment-established heat-rejection fraction in `[0,1]`.

- default: `1.0`;
- ordinary `step()` consumes the current fraction;
- explicit one-off `step_with_heat_rejection_fraction()` remains available;
- invalid fractions fail closed;
- changing the fraction does not command a physical radiator, pump, sublimator, evaporator, valve, or PLSS controller.

The durability/dust layer remains responsible for producing the fraction. PLSS does not infer dust physics itself.

## Hand workload composition

The powered-glove twin already produces per-digit:

- pressure resistance;
- exercise resistance;
- powered assist;
- human-required force;
- electrical power;
- tactile/fatigue state.

SX-022 combines these with the declared tendon speed.

For moving digits:

`hand mechanical work rate = human-required force * tendon speed`.

For near-isometric grip, v0.1 uses an explicit simulation-only force-to-equivalent-workload coefficient so static grip is not incorrectly treated as zero human cost.

That coefficient is **not** a validated physiological law and must be calibrated/replaced with human data.

## Preventing body-assist double counting

The body-exoskeleton requested mechanical assistance is frozen before hand workload is added.

After hand workload is added to total positive human task load, the requested body-assist fraction is recomputed so the absolute requested body-assist mechanical power remains unchanged.

Therefore:

`adding hand work != giving the leg/body exoskeleton credit for finger assistance`.

## Glove power and heat

Powered-glove electrical demand is assigned to the mobility/augmentation bus, preserving the existing rule that optional robotic loads cannot consume the protected survival reserve.

Estimated glove actuator mechanical power is derived from active force and tendon speed.

`glove waste heat = electrical power - actuator mechanical power`, clamped non-negative in the current model.

The waste heat enters the suit equipment-heat load seen by PLSS.

Exercise/resistive modes remain research approximations; future regenerative behavior should be modeled explicitly rather than inferred from this v0.1 term.

## Segment-scoped thermal derating

`step_fidelity_segment()` applies the declared radiator heat-rejection fraction to the actual PLSS mission step and restores the previous PLSS environment value immediately afterward.

This prevents one dusty segment from silently contaminating later clean segments through hidden mutable state.

## Evidence and non-claims

All new coefficients remain `Simulation` evidence, including:

- isometric hand workload equivalence;
- speed threshold separating dynamic/isometric hand work;
- glove motor/thermal accounting assumptions;
- radiator heat-rejection fraction supplied by the dust model.

SX-022 does not establish:

- human metabolic validity;
- glove physiological equivalence;
- AxEMU/xEMU/EMU performance;
- calibrated thermal-emittance degradation;
- qualified EDS effectiveness;
- human-rating or flight readiness;
- superiority over another suit.

## Evidence ladder

`software composition tests`
→ `instrumented powered-glove benchtop rig`
→ `human grip/workload protocol at ambient pressure`
→ `pressurized glovebox human-in-loop testing`
→ `radiator dust/EDS coupon calibration`
→ `PLSS/thermal hardware-in-loop`
→ `integrated suit chamber test`
→ `program qualification`.

## Next fidelity targets

1. Replace isometric hand-equivalent coefficient with protocol-calibrated physiological data and uncertainty.
2. Distinguish upper-body/hand effort from locomotion/postural effort without double counting.
3. Feed measured radiator effective-emittance data into heat-rejection fractions.
4. Add explicit EDS electrical/thermal trade across cleaning schedule, radiator margin, and battery state.
5. Couple pressure-integrity response (SX-021) into whole-mission adversarial campaigns.
6. Add human-state estimates (heart rate, respiratory load, core/skin temperature, hydration, fatigue) as evidence inputs only; deterministic safety limits retain authority.
