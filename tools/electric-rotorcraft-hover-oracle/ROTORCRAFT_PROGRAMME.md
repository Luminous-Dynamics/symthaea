# Electric Rotorcraft Programme — Architecture, Benchmarks and Competitive Baselines

**Programme:** AIRROAD-ROTOR-001 (#11490)  
**Reference implementation:** electric-rotorcraft-hover-oracle v0.1.0  
**Status:** research and analytical benchmark contract. This document is not an aircraft design release, validated performance prediction, safety case, airworthiness finding or certification basis.

## 1. Product scope

The initial target family is one-to-three **total occupants**, including the pilot where applicable. Passenger count, pilot count, baggage, mission payload and optional equipment must remain separate fields. A pilotless vehicle has different autonomy, operational and certification assumptions from a piloted vehicle; do not compare the two by seat count alone.

The programme should study three product groups:

1. **Single-seat / one-person personal or utility aircraft:** low acquisition and operating complexity, training, inspection, survey and short-range utility missions.
2. **Two-occupant aircraft:** either pilot + one passenger or two passengers in a pilotless architecture; distinguish them explicitly.
3. **Three total occupants:** pilot + two passengers, or an explicitly defined pilotless configuration. This is a separate mass, egress, energy and operational class—not merely a stretched two-seat airframe.

No fixed payload, range, endurance, cruise speed or MTOW target should be adopted until a mass/energy/thermal/noise tradespace shows feasibility with reserve and a defensible certification/operations boundary.

## 2. Vehicle architectures

### Conventional electric helicopter

Retain main-rotor and anti-torque architecture semantics, rotor inertia and transient response, drivetrain and rotor-drive loads, tail-rotor/anti-torque power, autorotation only where the specific design physically supports it, and helicopter-specific structural and handling requirements.

A battery-electric helicopter must not be modelled by merely replacing the existing fuel tank with a battery. Electric motor torque response, inverter limits, DC bus, battery power/voltage/current limits, fault isolation, cooling, mass distribution and rotor-drive transients require explicit interfaces.

### Multicopter

Keep fixed-pitch or variable-pitch electric propulsors, independent motor/ESC channels, thrust allocation and control authority, propulsor-out cases, wake interaction, thermal and electrical common-cause failures, and ground/approach effects distinct from helicopter physics.

Do not assume helicopter-like autorotation or safe landing after a motor failure. Those claims require architecture-specific aerodynamic and control evidence.

### Powered-lift and hybrid vertical/cruise aircraft

Lift-plus-cruise, tiltrotor and tilt-propeller concepts require their own conversion and transition models: wing/rotor interaction, mode-transition corridor, conversion-mechanism loads, transient power, control authority and failure states. They may use the same mission/evidence vocabulary as other aircraft, but not a copied hover model as their complete flight model.

## 3. Paradigm shifts to investigate

### A. Treat rotor disk area as a system-level design variable

For the ideal steady-hover actuator disk, induced power scales as:

`P_i = T^{3/2} / sqrt(2 rho A)`

At fixed total thrust and air density, doubling effective non-overlapping disk area reduces this ideal induced-power lower bound by a factor of `1/sqrt(2)`. This is a useful physical design lever, not a guaranteed installed-aircraft improvement. Larger disks can increase blade, hub, structure, landing footprint, mass, profile power, drag and packaging cost; coaxial and overlapping rotor disks also have interference that this oracle does not model.

The correct design variable is therefore not rotor count alone. Compare effective disk area, disk loading, blade/rotor profile losses, RPM/tip speed, mass and installation geometry together.

### B. Optimize the mission, not the hover specification

Report takeoff/vertical-lift energy, climb, transition, forward cruise, approach/landing, go-around/diversion, reserve and ground-to-ground energy separately. A vehicle with excellent hover efficiency may still be poor over a mission if forward-flight drag, conversion losses, thermal loads or reserve requirements dominate.

The mission objective should include delivered payload-km or passenger-km, but the energy denominator, occupant definition, reserve and operating environment must be frozen.

### C. Make redundancy a common-cause question

More motors do not automatically mean more aircraft-level redundancy. Evaluate failures across propulsion channels, common battery strings, contactors, buses, cooling loops, sensors, control computers, software, wiring routes, structural attachments, fire zones and charging infrastructure. Report which failures allow continued controlled flight, which require an emergency landing, and which are unrecoverable in each architecture. Do not convert motor-count into a reliability score.

### D. Co-design thermal and acoustic systems with propulsion

Battery, motor, inverter/ESC and cabin loads share a finite mass, cooling and heat-rejection budget. Likewise, rotor noise, tonal content, vibration, blade-vortex interactions and downwash influence the design envelope and landing-site acceptability. Cooling ducts/heat exchangers and rotor/airframe interactions can also change mass, drag, vibration and noise.

Treat thermal energy, cooling mass/volume, noise, vibration and downwash as first-class Pareto dimensions rather than post-hoc subsystem checks.

### E. Optimize the operating day

Repeated short sorties can make turnaround, charging, thermal reset, preconditioning, maintenance and dispatch availability as important as single-flight range. Compare flight-only battery-terminal energy separately from ground battery energy and shore/grid energy. Do not hide charging, thermal regeneration or preconditioning outside the objective denominator.

### F. Design for inspectability and repair from the start

A manufacturable aircraft needs configuration-controlled drawings/models, traceable materials and batches, serial-level build records, calibrated test instrumentation, inspectable load paths, replaceable line-replaceable units, defined inspection limits, maintenance/repair instructions, fatigue life evidence and controlled return-to-service. “Modular” is not a repairability metric; measure diagnosis time, repair time, post-repair test time, parts availability, first-pass yield and rework.

## 4. Current competitive reference set

These are public manufacturer or regulator statements. They are not independent Luminous test data and must not be compared without matching payload, mission, reserve, environmental conditions, configuration, aircraft boundary and regulatory category.

| Reference | Publicly stated configuration/status | How to use it |
|---|---|---|
| Volocopter VoloXPro | One pilot + one passenger; 18 electric propulsion units; 70 km/h cruise; up to 40 km depending on battery variant. Volocopter describes the standard configuration as following initial DULV type certification under the Ultralight Multicopter Technical Standards. | Closest piloted two-occupant distributed-multicopter reference. Confirm payload, battery variant, reserve, mission and approval scope from the current technical dossier before numerical comparison. |
| EHang EH216-S | Pilotless human-carrying eVTOL; coaxial dual-rotor configuration; manufacturer lists 620 kg MTOW, 130 km/h maximum design speed and 30 km range. EHang reports CAAC type certification in 2023, production certification in 2024 and operator certificates in 2025. | Important two-passenger pilotless reference, but not directly equivalent to a piloted helicopter or a different jurisdiction's approval basis. Manufacturer statements need independent context. |
| Joby aircraft | Larger category: one pilot + four passengers, six electric tilt-propeller units and four battery packs in company-published specifications. Joby's 4 November 2026 Q3 update reported 17 aircraft in production and continued FAA type-certification testing. | Larger powered-lift comparator for manufacturing, redundancy, certification and transition architecture; outside the 1–3 total-occupant core product class. |
| Archer Midnight | Larger piloted powered-lift aircraft. Archer announced an FAA Market Survey Special Airworthiness Certificate on 9 November 2026 for piloted operations under the market-survey / eIPP pathway, ahead of type certification. | Useful operational and certification-progress reference; a market-survey authorization must not be represented as type certification. |

### Why this comparison changes our opportunity

The closest direct reference products already establish that passenger-carrying electric multicopters are beyond concept renderings. The field is also progressing from prototype operations toward production and regulatory approvals. The opportunity is not to claim novelty for having multiple electric rotors. It is to offer measurably better mission utility, lifecycle performance, maintainability, evidence quality and operational economics within a clearly stated aircraft class.

## 5. External research frontier

NASA's full-scale eVTOL aerodynamic/acoustic research identifies rotor-rotor and rotor-airframe interactions, flight mechanics, vibration and noise as important challenges, and notes that high-quality full-scale validation data are needed for model validation. This is a direct reason not to treat ideal actuator-disk hover equations as the whole model.

Use progressively higher fidelity:
- ideal actuator-disk lower bounds for independent arithmetic checks;
- momentum/Blade Element Momentum or other suitable rotor models with explicit domains;
- validated component maps for rotor, motor, inverter/ESC and cooling;
- aeroacoustic and structural/vibration models for configuration-specific interactions;
- instrumented subscale/full-scale test data on held-out configurations;
- integrated aircraft mission correlation only after boundary and uncertainty qualification.

Never tune the final model on the same holdout data used to claim validation.

## 6. Regulatory and approval map

The correct category depends on the design and intended operation:

- Conventional normal-category rotorcraft: review FAA Part 27 / Part 29 as applicable and EASA CS-27 / CS-29 as applicable.
- Powered-lift / VTOL-capable aircraft: identify the aircraft-specific airworthiness criteria and applicable VTOL special condition / means of compliance. EASA updated its VTOL page on 5 November 2026 with the final Fifth Publication of MOC-5 SC-VTOL Issue 2.
- FAA powered-lift operational/pilot rules are not the same thing as an aircraft type certificate. A special airworthiness certificate for market surveys is also not a type certificate.
- Pilotless passenger-carrying aircraft, ultralights, experimental aircraft and piloted commercial aircraft may have materially different certification, operator and operational pathways.
- For South African activity, consult SACAA on applicable aircraft classification, type certification/type acceptance, design and manufacturing approvals, experimental flight permissions and operational rules. Do not assume a US, EU or Chinese approval automatically transfers to South Africa.

No certification plan or compliance claim should be authored from a generic eVTOL checklist. A qualified aviation certification specialist and the applicable authority must establish the aircraft-specific basis.

## 7. Benchmark ladder

All targets are internal research goals, not regulatory limits or achieved performance.

### Tier 0 — analytic/reference correctness

- 100% agreement with the independent ideal-hover oracle on frozen deterministic cases within preregistered tolerances.
- Dimensional consistency and deterministic replay for exact manifests.
- Zero false-feasible outputs in frozen invalid-input and model-boundary negative controls.
- Explicit UNKNOWN/OOD/non-PASS outcomes for unsupported domains, missing evidence, invalid geometry, overlap ambiguity and numerical failure.

### Tier 1 — component characterization

Instrument and characterize thrust, torque, RPM, shaft power, motor/inverter efficiency, electrical input, thermal state and acoustic/vibration outputs over density, inflow, advance ratio, RPM and temperature ranges. Publish model discrepancy by configuration and regime. Set numeric correlation thresholds only after the measurement uncertainty and baseline capability are known.

### Tier 2 — integrated electric powertrain and thermal rig

Independently meter battery/DC-bus voltage and current, propulsion channels, rotor/shaft torque and RPM, component temperatures, cooling loads, flow/pressure and acoustic/vibration boundaries. Target bus-energy closure within 0.5% when the instrumentation supports it. Temperature RMSE below 2 K and peak absolute error below 5 K are stretch targets for a configuration-held-out dataset, not targets to force on inadequate sensors.

### Tier 3 — aircraft mission correlation

Target mission energy bias within ±3% and RMSE below 5% on a preregistered independent holdout if measurement uncertainty supports it. Report energy by mission phase, reserve, payload, thermal margin, noise, vibration, vertical-lift energy share, turnaround and dispatch separately. Target at least 100x real-time offline simulation on a named CPU without skipping cases or weakening accuracy.

### Tier 4 — architecture/policy/lifecycle value

Compare Pareto vectors for energy per passenger-km/payload-km, payload, reserve, installed mass/volume, noise, vibration, turnaround, availability, maintenance and lifecycle cost. A 10% improvement in a preregistered primary metric is a stretch target only when the matched baseline and hard constraints are frozen. Report false-safe observations as 0/N with confidence bounds and domain coverage, and report false-derate cost alongside them.

### Tier 5 — physical qualification and approval

Use separately approved component, integrated-rig, environmental, structural/fatigue, rotor-containment, electrical fault, thermal-abuse and flight-test procedures as applicable. No software or simulation PASS authorizes hardware actuation, passenger carriage, flight test, airworthiness or certification.

## 8. Sources

- Volocopter VoloXPro current product configuration: https://www.volocopter.com/en/solutions/voloxpro
- Volocopter VoloXPro announcement and stated 600 kg MTOM / 154 kg payload figures: https://www.volocopter.com/en/newsroom/volocopter-launches-new-product-voloxpro
- EHang EH216-S product configuration and manufacturer-stated performance: https://www.ehang.com/ehang216s/
- EHang EH216-S airworthiness milestones: https://www.ehang.com/ehang216s/airworthiness/
- NASA full-scale eVTOL aerodynamic/acoustic flight-test, modeling and simulation project: https://techport.nasa.gov/projects/154389
- Joby Q3 2026 certification/manufacturing update (published 4 November 2026): https://www.jobyaviation.com/news/joby-reports-third-quarter-2026-financial-results
- Archer Q3 2026 Market Survey certificate update (published 9 November 2026): https://investors.archer.com/news/news-details/2026/Archer-Announces-Q3-2026-Results-Receives-Market-Survey-Certificate-Unlocking-LA-Operations-and-Boeing-Deal-Close-Imminent/default.aspx
- FAA powered-lift pilot certification and operations final rule: https://www.faa.gov/newsroom/integration-powered-lift-pilot-certification-and-operations-miscellaneous-amendments
- EASA VTOL special condition / latest MOC publication: https://www.easa.europa.eu/en/document-library/product-certification-consultations/special-condition-vtol
- EASA CS-27 / CS-29 specification listing: https://www.easa.europa.eu/en/document-library/certification-specifications/reg/initial-airworthiness-and-environmental-protection
- FAA rotorcraft regulations and guidance: https://www.faa.gov/aircraft/air_cert/design_approvals/rotorcraft/rot_regs
- SACAA airworthiness and engineering approval information: https://www.caa.co.za/industry-information/airworthiness/

The benchmark's contribution is evidence-qualified and matched comparison across rotorcraft families—not a claim that an ideal hover oracle, generic multicopter simulation or AI-generated design is a passenger-aircraft engineering result.
