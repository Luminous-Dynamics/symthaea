# MARINE-CAT-001 — mission-optimised catamaran design and validation

Status: concept hypotheses, schema and validation protocol started. No catamaran hull optimisation, CFD run, resistance prediction, seakeeping result or sea-trial result is claimed.

Research issue: [MARINE-CAT-001 #7719](https://github.com/Luminous-Dynamics/symthaea/issues/7719)

## Core paradigm shift: design an operating envelope, not a speed-point winner

A catamaran is a coupled system: two free-surface hulls, the tunnel and bridge deck, transverse structure, appendages/foils, propulsion, control and the route's waves interact. The goal must be a robust family of designs whose trade-offs are understood over speed, loading, heading and sea state—not one drag minimum at one speed.

Three separate concept families are started here:

- `LD-SAIL-CAT-1500-V0`: 15 m bluewater sailing catamaran. Primary criteria are passage performance, reefed/heavy-weather handling, bridge-deck impact, crew workload, payload, serviceability and safe offshore degraded modes.
- `LD-EFOIL-CAT-024-V0`: 24 m electric hydrofoil passenger catamaran. Primary criteria are total route energy, foiling/hullborne regime transitions, charging closure, ride comfort, foil/control failure and the declared refuge/weather envelope.
- `LD-EXPLORER-POWER-CAT-045-V0`: 45 m expedition power catamaran. Primary criteria are range, low-noise/port operation, energy and hotel-load efficiency, cross-deck impact, redundancy, payload, maintainability and route access.

These are concept hypotheses. Dimensionless geometry ratios and unknown values stay explicit. None is yet a chosen hull form or a seaworthy/approved vessel.

## Why hull spacing cannot be guessed

The CNR/INSEAN Delft 372 research programme measured resistance, trim and sinkage for monohull/catamaran configurations with several hull separations over Froude numbers 0.1–0.8, and investigated wave cuts and seakeeping. This gives a physically relevant foundation for a spacing/design-space study, but not a universal optimum for a sailing yacht, hydrofoil ferry or 45 m expedition cat.

The Delft 372 static-drift CFD-validation work compared hydrodynamic loads, sinkage/trim, velocity fields from PIV, vortex structures and wave patterns across multiple solvers and model choices. It reports the importance of grid resolution and turbulence treatment and also notes substantial mismatch when numerical and experimental conditions or scales are not aligned. Use that as a reason to bind exact test conditions and refuse cross-condition comparisons, not as a generic 'CFD accuracy' number.

The 2020 experimental catamaran head-wave study covered regular and irregular waves and quantified statistical uncertainty using autocovariance and block-bootstrap methods. Our benchmark should therefore report response distributions and uncertainty, rather than a single deterministic pitch/acceleration value.

References:

- [CNR/INSEAN: calm-water and seakeeping investigation for a fast catamaran](https://iris.cnr.it/handle/20.500.14243/173703)
- [CNR/INSEAN: CFD validation of Delft 372 in static drift](https://iris.cnr.it/handle/20.500.14243/351603)
- [Durante et al. (2020), experimental catamaran head-wave benchmark and uncertainty quantification](https://doi.org/10.1016/j.oceaneng.2019.106685)
- [ITTC benchmark database procedure for CFD resistance and propulsion validation](https://www.ittc.info/media/9771/75-03-02-02.pdf)

## Main design variables

Keep these as explicit variables, ranges and evidence—not tacit designer choices:

1. Demi-hull slenderness, fore/aft volume distribution, bow entrance, rocker, stern shape and propulsor integration.
2. Hull-centreline spacing divided by LWL; individual demi-hull beam and displacement/loading; compare across Froude number, trim and sinkage.
3. Wet-deck/bridge-deck clearance relative to draft, loading and dynamic wave elevation; local impact pressure and resulting cross-deck bending/torsion/fatigue.
4. Overall beam versus port/mooring constraints, windage, transverse stability, global loads and usable deck volume.
5. Appendage/foil geometry, retraction, added drag, cavitation/ventilation, impact/fouling, control coupling and fail-safe modes.
6. Propulsion architecture and duty-cycle power—including takeoff/transition demand, hullborne return and battery aging for the hydrofoil variant.
7. Passenger/crew comfort, noise/vibration, accessibility, wet-deck exposure and emergency access by operating mode.

No single variable can be optimised without all the coupled hard constraints. A geometry that reduces calm-water resistance but has unacceptable slamming, structural loads, low reserve buoyancy or poor degraded-mode behaviour must be rejected, not rescued by a weighted score.

## Research lessons from real concepts

The EF-24 electric hydrofoil ferry is reported in a 2022 public project report as a 24 m vessel for up to 150 passengers, 100 nmi range and top speed of 38 kn, with an 'up to 90% less energy' claim made by the project. Use it only as a project-reported comparator. Do not transfer the stated range/energy saving to our different design; wait for suitable independent data and matched route/load conditions. Source: [ITV coverage of Condor/Artemis EF-24](https://www.itv.com/news/channel/2022-09-26/look-inside-condors-new-100-electric-ferry).

Regulations are design inputs. IMO's HSC Code applies to qualifying high-speed craft on international voyages, with vessel and route-specific applicability limits and requirements for passengers and operation. Do not assume that the HSC Code automatically applies to every catamaran—or that it is the only requirement. Determine vessel category, speed, voyage/refuge conditions, flag and class requirements first. Sources: [IMO High-speed craft](https://www.imo.org/en/ourwork/safety/pages/hsc.aspx) and [IMO ship design and stability](https://www.imo.org/en/ourwork/safety/pages/shipdesignandstability-default.aspx).

## Multi-fidelity benchmark design

### Phase 0 — input and evidence contracts

- Validate every geometry, operating point and result with units, coordinate/sign convention, scale, displacement/loading, Reynolds/Froude regime, uncertainty and source revision.
- Record public metadata separately from raw data. The 2020 head-wave source record identifies a restricted-access manuscript; only use exact data if lawful access and rights are confirmed.
- Treat benchmark records as case-specific. A Delft 372 calm-water drift case cannot validate a large cruising catamaran, a foilborne ferry or an irregular-wave slamming envelope by itself.

### Phase 1 — wide design-space screening

Use low-cost arithmetic/analytical filters across at least 10,000 concept variants per archetype after defensible parameter bounds are chosen. Sweep hull-spacing ratio, slenderness, wet-deck clearance, loading, speed/regime, propulsor assumptions and route/wave cases. Freeze train/tuning and evaluation splits before policy or parameter tuning. Screening identifies informative candidates; it must not be reported as validated resistance/comfort output.

### Phase 2 — benchmark-calibrated resistance/flow

First reproduce one compatible, rights-reviewed experimental reference condition, including exact hull geometry, scale, speed, loading, sinkage/trim, and measurement uncertainty. Assess convergence/discretisation uncertainty, boundary conditions, turbulence/free-surface choices and condition mismatch. Keep a per-condition error table; do not average away a bad operating point.

Proposed target: on named validation points only, aim for resistance error within 5% once measurement/numerical uncertainty and comparability have been considered. The target is ambitious and must be reported with actual uncertainty; it does not legitimise transfer outside the tested envelope. Compare trim, sinkage, lateral force/yaw moment and flow/wave fields where reference data support it.

### Phase 3 — seakeeping and wet-deck evidence

Use regular waves to understand mechanism and irregular waves to estimate response distributions. Report heave, pitch, roll, vertical acceleration, added resistance, pressure/impact, slamming/green-water indicators and passenger comfort proxies over wave heading, encounter period, wave height, speed and loading. Bind block-bootstrap or another prespecified uncertainty method. Do not claim that a calm-water pass establishes seakeeping.

### Phase 4 — propulsion and mission energy

Close mode-resolved energy: hullborne, sail propulsion, takeoff/foiling, cruising, dock/charge, hotel services, manoeuvring, one-drive/one-generator degraded cases and safe return. Include charger power/availability, conversion efficiency, battery degradation/reserve, payload/mass effect and well-to-wake fuel/electricity factors. Report kWh/passenger-nautical-mile or kWh/tonne-nautical-mile against a matched monohull and incumbent-catamaran baseline.

### Phase 5 — Pareto set and independent review

Publish nondominated candidates across resistance/energy, seakeeping/comfort, cross-deck structural mass, payload, range, cost, maintenance, noise, emissions, shallow-water access, beam/port fit and safety margins. Stability, structure, slamming, fire, emergency operation, collision/grounding, and applicable flag/class gates remain non-compensable.

## Proposed benchmark gates — targets, not measured results

| Gate | Initial target |
|---|---|
| Input integrity | 100% of critical inputs have units, source/assumption state, conditions and explicit unknown handling; zero silent defaults |
| Search breadth | ≥10,000 lower-fidelity variants per archetype with frozen candidate and held-out sets |
| Calm-water resistance | Aim for ≤5% error on explicitly named, compatible validation conditions only, with uncertainty; no blanket extrapolation |
| Seakeeping calibration | Report held-out error and 90/95% interval coverage separately by Froude, spacing, loading, wave heading/encounter and response quantity |
| Wet-deck/structure | Zero false-safe classifications in prespecified impact, pressure, fatigue and degraded-loading controls |
| Hydrofoil failures | Zero unsafe outcomes across control/actuator/sensor/power faults, asymmetric lift, impact, out-of-envelope transition and degraded return controls |
| Route-energy model | Report paired matched-baseline effects, tail cases, energy reserve and upstream emissions; do not claim a fixed percent saving before data |
| Multi-objective quality | Publish a reproducible Pareto frontier with assumptions/sensitivity and separate hard-gate dispositions |

## Implementation order

1. Freeze the catamaran-specific schema and provisional concept fixtures.
2. Add schema validation and negative controls for contradictory units, invalid ratios, unsupported/out-of-domain reference cases and hidden unknown-to-favorable conversions.
3. Build dimensionless design-variant generation and reproducible manifest/receipt in Rust without pretending to be a CFD solver.
4. Select exact reference cases from ITTC/CNR records after investigating access, rights and conditions; start metadata-only where necessary.
5. Add lower-fidelity resistance/interference estimators only with declared domains and interval outputs, then calibrate against suitable test data.
6. Add seakeeping, wet-deck, structural and electric/foil operating-envelope evaluation as independent gates.
7. Execute the held-out suite; only then report the best candidates or actual comparative improvements.

## Claim ceiling

A concept/schema pass is not a resistance result. A CFD result is not a seakeeping certificate. A calm-water result is not an offshore operating limit. A design benchmark cannot establish safe passenger-carrying, full-scale performance, seaworthiness, or class/flag approval. Those require qualified multihull naval architects, hydrodynamicists, structural and electrical engineers, appropriate experiments and competent authority review.