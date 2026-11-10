# Mission thermal derating overlay

**Module:** `src/mission_thermal.rs`  
**Version:** `electric-aircraft-mission-thermal-overlay-v0.1.0`  
**Evidence class:** reduced-order software model with synthetic unit tests only.  
**Purpose:** connect the existing mission report to the multi-node thermal RC network and check temperature-dependent continuous-power limits along a mission.

## What is implemented

The overlay consumes a `MissionInput`, its matching `MissionReport`, a `ThermalNetwork`, an initial `ThermalState`, and segment-aligned ambient / external-heat profiles.

It reconstructs three component loss sources from the mission report:

- Battery pack: reported I²R energy divided by segment duration.
- Motor: shaft power divided by motor efficiency, minus shaft output.
- Inverter: motor electrical input divided by inverter efficiency, minus motor electrical input.

It checks three power boundaries against explicit temperature-dependent derating curves:

- Battery terminal power, including the mission's auxiliary load.
- Traction-inverter DC input power, excluding auxiliaries.
- Motor shaft output power.

Each curve is tied to a thermal-network node. The network is advanced with backward-Euler RC steps no larger than the configured maximum step. Component limits are checked before and after every step. A modelled node ceiling, derating violation, uncharacterized temperature, malformed curve, or mission/report mismatch rejects the overlay.

The overlay verifies segment IDs, flight phases, all-up mass, drag, propulsion-chain power, terminal energy and resistive heat against the supplied mission input. It does not silently apply a thermal profile to a report from a different segment sequence or duration.

## Why this is a meaningful increment

The prior energy kernel and multi-node thermal network were separate. The overlay now carries mission-derived battery, motor and inverter loss estimates into a component thermal network and asks a more useful design question:

> Can the modeled mission power demand be sustained over time while the component temperatures evolve and the available power derates?

That makes it possible to examine high-power climb followed by cruise, hot ambient operation, repeated sorties, hot turnaround, degraded heat rejection and different component thermal masses/conductances using the same segment structure.

It is a bridge, not a fully coupled electro-thermal solver. Temperatures do **not** yet feed back into battery OCV/resistance maps or propulsion-efficiency maps, and the mission energy result is not re-solved after thermal derating. The code rejects a demand that exceeds the available power; it does not invent a lower achieved thrust or silently stretch the mission. A future coupled solver must iterate the operating point and energy state, preserve reserve, and converge or fail closed.

## Required evidence for component inputs

A derating curve is not a datasheet decoration. Each point should eventually be linked to a configuration-controlled test or qualified supplier dataset with:

- component serial/configuration and cooling installation;
- continuous-power definition, voltage/current or shaft boundary, and duty duration;
- initial temperature, ambient, altitude/air density and cooling boundary;
- instrument calibration and uncertainty;
- steady-state criteria and thermal protection behaviour;
- raw time-series digest, processing code, and independent review status.

The present API accepts numerical points but does not authenticate a source, fetch its URI, establish supplier authority, or approve an aircraft configuration. The caller is responsible for provenance until the evidence-authority pipeline is implemented.

## Numerical / software properties

- Derating points are sorted by strictly increasing temperature.
- Available power must be non-increasing with temperature.
- Linear interpolation is used only between measured points.
- Temperatures outside the characterized curve domain are rejected, never extrapolated. Power curves may be non-monotonic so battery cold-side limits and hot-side derating can both be represented from measured points.
- Mission durations are subdivided to a declared maximum integration step.
- Thermal state is returned only if all modeled node ceilings and power limits pass.
- Each thermal step reports a network energy-balance residual; the maximum absolute residual is retained.
- The dense RC solver remains a small-network model and is not a substitute for CFD or thermal-fluid analysis.

## Explicit limitations

This increment does not model:

- temperature feedback into OCV, internal resistance, SOC, or available battery energy;
- thermal feedback into motor/inverter efficiency maps;
- electrical control response, power allocation, thrust reduction or mission re-planning;
- variable coolant flow, pump maps, heat-exchanger effectiveness or airspeed-dependent convection;
- within-component temperature gradients, battery-cell imbalance, cell hot spots or thermal propagation;
- thermal runaway, venting, combustion, containment or abuse testing;
- transient aerodynamics, control laws, aeroelasticity, or certification evidence.

In particular, a lumped pack node is not a cell-hot-spot model. A passing overlay is not a battery safety assessment or an aircraft airworthiness finding.

## Benchmark plan

All values below are internal research targets, not external standards, achieved results, or safety acceptance thresholds.

### Tier A — deterministic software verification

- 100% pass rate on preregistered deterministic fixtures at the exact source revision.
- Exact reproducibility of the thermal trajectory for a frozen manifest and input.
- Zero accepted out-of-domain derating lookups.
- Zero accepted duplicate or mismatched mission segment profiles.
- Energy-balance residual reported for every substep; threshold set from numerical conditioning and an independent analytical oracle.
- Mutation tests for loss terms, temperature ordering, power monotonicity, node identity, mission duration, and derating boundaries.

### Tier B — component / rig correlation

- Component-temperature RMSE below 2 K on an independent, preregistered holdout set, with empirical 95% prediction-interval coverage reported.
- Peak-temperature absolute error below 5 K and peak-time error below 5% of the tested segment duration.
- Conservatively predicted continuous-power availability: no false-safe observations on the held-out test matrix under the explicitly declared test conditions.
- Report false derates separately; an overly conservative model that constantly rejects valid operating points is not a useful design tool.
- Power-loss reconstruction error below 3% against separately instrumented motor / inverter / pack measurements, after agreeing on the electrical and mechanical measurement boundaries.

### Tier C — integrated thermal mission

- Mission-level energy bias within ±3% and RMSE below 5% after independent aircraft or representative integrated-rig validation.
- Calibrated temperature prediction intervals with at least 95% empirical coverage on an independent holdout set.
- Peak component temperatures and power capability correlated across hot/cold ambient, SOC, SOH, altitude, climb, cruise, descent, hot soak, repeated sorties and degraded cooling.
- Demonstrate that the coupled model never reports a feasible mission when an independently measured component limit is exceeded in the frozen test matrix.
- Report runtime and numerical error together; a useful design-space model should target at least 100× faster-than-real-time execution for a defined small-aircraft mission on a documented commodity CPU, without reducing the test matrix or hiding failed cases.

A passing Tier A result is only software correctness. Tier B requires physical subsystem tests. Tier C requires integrated data. None replaces hazard analysis, battery qualification, flight testing, or certification.

## Industry / research comparison

NASA's NEAT supports integrated MW-scale powertrain ground tests and simulated altitudes up to 60,000 ft. NASA's HEMM is a 1.4 MW partially superconducting motor programme with public targets of 16 kW/kg and 99% efficiency. Those are research and test capabilities; they are not the performance of a complete Luminous aircraft installation. NASA's 2026 liquid-cooled battery-module work measures a module and its coolant loop across 0.5C–2.5C, including component temperatures, flow and pressure drop. Those measurements illustrate why the network must eventually be correlated against physical heat-source and coolant data.

Commercial comparison requires care. BETA publishes H500B peak and continuous power, power density, efficiency and installed mass, but those numbers need a common reconciled denominator before a valid specific-power comparison. Pipistrel's Velis Electro is an EASA type-certified, two-seat trainer with a 57.6 kW certified motor and 24.8 kWh nominal dual-pack battery system; it is a certification / operating reference, not a payload-range peer for larger utility aircraft. The correct near-term Luminous goal is not to claim higher power density. It is to build a traceable thermal mission model that can predict component limits and failure envelopes with independently measured uncertainty.

## Sources

- NASA NEAT: https://www.nasa.gov/eap-labs-and-testbeds/neat/
- NASA HEMM: https://www.nasa.gov/eap-technology/electric-machines/hemm/
- NASA liquid-cooled battery module thermal-fluid analysis (2026): https://ntrs.nasa.gov/citations/20260008248
- NASA electrified-aircraft powertrain health digital twin (2026): https://ntrs.nasa.gov/citations/20260004592
- BETA H500B published specifications: https://beta.team/motor
- Pipistrel Velis Electro technical data: https://www.pipistrel.si/products/general-aviation/velis-electro/
- FAA AC 20-184A (issued 2026-09-18): https://www.faa.gov/airports/resources/advisory_circulars/index.cfm/go/document.information/documentID/1045700
