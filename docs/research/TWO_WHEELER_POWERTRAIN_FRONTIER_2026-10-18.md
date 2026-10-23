# TWO-WHEELER-POWERTRAIN-FRONTIER-001 — Powertrain redesign without energy-accounting illusions

**Status:** Research protocol and architecture screen, v0.1  
**Tracker:** https://github.com/Luminous-Dynamics/symthaea/issues/8056  
**Parent program:** https://github.com/Luminous-Dynamics/symthaea/issues/8044  
**Review date:** 2026-10-18  
**Evidence status:** Literature/source review plus proposed benchmark design. No powertrain has been modelled, built, or tested by this document. The machine-readable candidate register is [`two_wheeler_powertrain_candidates_v0_1.json`](data/two_wheeler_powertrain_candidates_v0_1.json); it defines screening roles and required evidence but contains no measured performance ranking.

## Executive decision

**First target: beat a combustion motorcycle on the complete mobility mission with a repairable battery-electric powertrain—not by inventing a more exotic engine, but by optimizing the entire energy-storage-to-wheel system.** In parallel, screen series hybrid architectures where charging access, range, or fast replenishment justifies the additional engine/generator/battery/motor complexity.

Plasma deserves two distinct research branches:

1. **Plasma-assisted combustion** is a plausible research subject because plasma ignition can affect ignition stability, lean-burn limits and emissions. It still burns fuel; its net benefit must exceed electrical parasitics and added hardware/maintenance. It is an ICE improvement hypothesis, not a free-energy or combustion-free engine.
2. **Plasma/ion propulsion** is a real class of spacecraft thruster. NASA describes electric power ionizing propellant and electromagnetic fields accelerating it, with applications in spacecraft. It optimizes propellant efficiency and sustained acceleration, not motorcycle launch acceleration or wheel power. High specific impulse is not a substitute for thrust-to-power, thrust density, system mass and atmospheric operation.

Spark Engine is the **LCF scientific anomaly-investigation toolkit**, not a propulsion engine. Its README says it is a research tool, not a power source, and its standard room-temperature Gamow baseline yields Q far below one. NASA's 2020 LCF publications report neutron signatures consistent with nuclear reactions in deuterated metal under energetic irradiation; such a result is not evidence of net useful energy or a self-sustaining vehicle power source. Spark Engine's own report says the credible application, if the signal is confirmed and made controllable, is a compact neutron source. The correct next step is falsifiable, instrumented replication and improved model/evidence semantics—not an LCF motorcycle concept.

## 1. Language and system boundaries

The word “engine” is too ambiguous for this comparison. Every candidate must declare which element it proposes to replace:

- **Energy source/storage:** liquid fuel, battery, hydrogen, or a future verified source.
- **Prime mover/converter:** combustion engine, fuel cell, fusion heat source, generator, or another converter.
- **Traction motor:** converts electricity to mechanical shaft work.
- **Transmission/drive:** gearset, reduction drive, chain/belt, hub motor, or another mechanism.
- **Vehicle/service system:** the combined machine and its human, thermal, repair, charging/refueling, manufacturing, and lifecycle constraints.

No design may claim “engine efficiency” while omitting the energy source, generator, charger, power electronics, cooling, pumps, controls, storage, containment, or power supply needed to deliver its advertised output.

## 2. Candidate architecture register

| Candidate | Role in first study | Main promise | Principal blockers / falsifiers |
|---|---|---|---|
| Conventional gasoline ICE | Frozen reference baseline | Fast refueling, established supply/service ecosystem, high stored-fuel energy per mass | Fuel conversion losses, emissions, heat/noise, many wear and service items |
| Battery-electric direct drive | **Primary design target** | Short conversion chain, controllable torque, regenerative braking, fewer moving parts | Pack-level energy density, battery cost and ageing, charging access/time, thermal derating, repair and recycling |
| Battery-electric with swappable pack | System/service variant | Potentially less downtime for high-utilization fleets | Interface safety, pack inventory, capital tied up in spares, standardization, charging logistics, pack custody |
| Series hybrid | Secondary comparator | Engine can operate near selected efficient regions while motor supplies wheel torque | Two conversion stages, extra mass/cooling/service needs; test whether reduced fuel use compensates |
| Parallel hybrid | Secondary comparator | Direct engine path plus electric assist/regeneration | More mechanical/control integration and service complexity |
| Hydrogen fuel-cell electric | Conditional comparator | Electric traction with rapid replenishment where hydrogen supply exists | Hydrogen production/storage/transport, tank mass/volume, cost, service network, full system efficiency |
| Hydrogen ICE | Research comparator only if use case warrants | Retains some engine-production/service knowledge | Storage, combustion abnormality/materials, NOx and conversion efficiency; hydrogen is an energy carrier, not a free source |
| Biofuel/e-fuel ICE | Fuel-pathway sensitivity | Compatibility with certain existing assets where validated | Actual fuel pathway, land/feedstock/renewable-electricity needs, engine compatibility and cost; green label alone proves nothing |
| Plasma-assisted ICE | **Small experimental side branch** | Potentially improve ignition stability/lean-burn limits/emissions | Electrical parasitics, electrode/insulator life, high-pressure discharge behavior, operating envelope, emissions certification and net cycle efficiency |
| Ion/Hall plasma thruster | Physics screen only; not a road candidate without a new credible thrust concept | High specific impulse and efficient propellant use in space | Low thrust relative to required road-wheel power, electrical-power and radiating-area burden, ambient atmosphere, electrode/channel lifetime |
| MHD/atmospheric plasma propulsion | Speculative screening only | Electromagnetic force on ionized gas in principle | Ionization energy, achievable force density, power electronics, hot/reactive gas materials, mass, exhaust/ambient coupling; no road-vehicle advantage assumed |
| Lattice-confinement fusion (Spark Engine context) | **Excluded from the viable vehicle frontier**; hypothesis tracking only | If a controllable net-energy process were independently established, an entirely different energy pathway could follow | Current Spark Engine baseline does not establish net energy. Require independent replication, complete energy accounting, nuclear-product accounting and useful conversion before reconsideration |

The [machine-readable register](data/two_wheeler_powertrain_candidates_v0_1.json) is a hypothesis screen, not a ranking based on measured candidates. Lack of evidence means “not established,” not “zero performance” and not proof that something is impossible.

## 3. Physics gates for exotic concepts

Each high-novelty proposal must clear these gates before costly vehicle integration:

### Gate P0 — A complete energy path exists

Draw the energy flow from primary source to stored energy to converter to transmission to wheel. If the proposed thruster requires an electrical source, that source belongs in the system mass, loss and cost. If an LCF experiment requires an X-ray/beam source, target heating and auxiliaries, those are experiment inputs—not optional details.

### Gate P1 — System-level power and energy density close

Check continuous and peak wheel power, all-up propulsion system mass/volume, cooling, containment, control electronics and reserve energy. Compare complete systems, not a bare motor, cell, electrode, plasma chamber or fuel's theoretical specific energy.

### Gate P2 — Correct propulsion physics

Use a validated road-vehicle dynamics model for tire/wheel force and required traction power. For a plasma thruster, independently estimate thrust as a function of input power, propellant and ambient conditions. A high exhaust speed/specific impulse does not by itself yield enough force for a motorcycle.

### Gate P3 — Net gain has a declared boundary

Different claims must remain different:

- energy conversion efficiency (useful output / declared source energy);
- target or reaction gain, where relevant, with an explicitly defined coupled-input boundary;
- system electrical/thermal net output after recirculating loads;
- useful shaft/wheel energy under an equivalent vehicle mission.

A net-energy claim requires calibrated input/output measurements, uncertainty, full auxiliary consumption and an agreed boundary. For the LCF question, neutron count or nuclear-product identification is not a calorimetric net-energy result.

### Gate P4 — Durability and safe service are plausible

Record heat flux, cycling, erosion, vibration, electromagnetic compatibility, shock, water/dust, thermal transients, fault containment, maintenance access, and qualified failure response. Do not place ionizing-radiation, high-voltage plasma or fusion experiments on an ordinary vehicle prototype path.

### Gate P5 — An experiment can falsify the hypothesis

Specify predicted effect size, detector/instrument capability, controls, calibration, blind analysis where applicable, stopping/exclusion rules, uncertainty and competing explanations. Use Spark Engine's experiment-design work only within its actual model coverage; existing issues #857 and #868 document important outcome-ordering and missing-prediction/likelihood gaps that must be qualified before treating EIG rankings as full-scope calibrated probabilities.

## 4. Common mission and fair baselines

Freeze a use-case protocol before ranking candidates. Include at minimum:

1. urban commuter duty: stops, starts, realistic speed trace, modest payload and parking/charging profile;
2. utility/cargo duty: declared payload and CG distribution, repeatable starts/grades and service schedule;
3. mixed/rough-road duty: a stated surface/grade/profile with load, dust, heat and wet-condition variants;
4. long-range/fleet duty: higher daily distance, rapid replenishment, downtime penalty and parts logistics.

Use the applicable World Motorcycle Test Cycle and legal requirements for the specific vehicle class when relevant; custom cycles must be labelled engineering stress tests, not regulatory certification cycles.

For every mission, freeze payload, vehicle utility, route/speed trace, environment, temperature, reserve, tire generation, rider/machine loading and accessory state. Calculate or measure the required wheel-work trace first; feed that exact demand to each candidate. Candidate-specific gearing and thermal management may vary but must be recorded.

### Common wheel-demand calculation — first executable slice

The pointwise Rust screening kernel lives in [`two_wheeler_wheel_demand.rs`](../../xtask/src/two_wheeler_wheel_demand.rs), with [13 synthetic oracle cases](data/two_wheeler_wheel_demand_v0_1.json). A separate [trace-energy integrator](../../xtask/src/two_wheeler_trace_energy.rs) now integrates signed wheel work and distance with [six synthetic trace cases](data/two_wheeler_trace_energy_v0_1.json). The full contract and limitations are in [TWO-WHEELER-WHEEL-DEMAND-001](TWO_WHEELER_WHEEL_DEMAND_BENCHMARK_2026-10-18.md). The pointwise kernel computes signed wheel-force demand from:

```text
F_inertial = (mass + rotating_equivalent_mass) * acceleration
F_rolling  = Crr * mass * g * cos(grade) * rolling_direction
F_grade    = mass * g * sin(grade)
u_air      = ground_speed - along_track_wind
F_aero     = 0.5 * air_density * CdA * u_air * abs(u_air)
F_wheel    = F_inertial + F_rolling + F_grade + F_aero
P_wheel    = F_wheel * ground_speed
```

Positive wheel power is traction demand. Negative wheel power is overrun/braking opportunity only; it is not credited as battery-regenerated energy. The pointwise v0.1 kernel is instantaneous; the trace utility integrates its signed mechanical wheel power using a piecewise-linear interpolation and explicit zero crossing. Neither utility predicts battery power, fuel energy or conversion efficiency. They provide a shared road-demand input, not a competing full-powertrain digital twin. The source-reported validation summaries are frozen at [`published_sa_electric_motorcycle_validation_summary_v0_1.json`](data/published_sa_electric_motorcycle_validation_summary_v0_1.json); these summaries are not raw telemetry or reproduction evidence.

### Recent validation evidence that should set the bar

The research register [freezes source-reported summaries](data/published_sa_electric_motorcycle_validation_summary_v0_1.json), not raw traces, from two relevant studies:

- Adams, Botha & Booysen (10 February 2026) used dynamometer-derived motor/inverter maps and six Nairobi trips; their published energy-per-distance errors range from 0.4% to 11.3%. Their study discusses gradient estimation, surface, rider posture and payload effects: https://doi.org/10.3390/wevj1702090.
- The same group’s full-powertrain digital twin (20 August 2026) was validated on six Stellenbosch trips. It reports battery-power correlations of 0.905–0.981, motor-speed correlations of 0.912–0.996, and energy predictions within 2.71% for five trips; a short downhill/high-throttle-transient trip had 10.65% error: https://doi.org/10.3390/wevj1708432. The reported limitations include omitted mechanical braking/auxiliary load and unmodelled battery temperature/ageing, with estimated rather than directly measured wind.

These results mean the next contribution should **not** be a redundant full twin. We should develop the portable Rust force-demand contract, then seek permitted raw telemetry and component-map data to independently cross-check it and extend validation across braking, wind, thermal state, degradation, surface and payload. The papers’ summary tables are not enough to claim exact reproduction.

For an electric candidate, meter wall-plug energy and battery-terminal energy separately; for ICE, meter fuel and declare its heating-value source; for hybrids/fuel cells, include every generator, storage and converter stage. Measure required auxiliaries such as cooling, pumps, controls and plasma ignition supply at the same boundary. Use measured operating maps rather than multiplying peak component-efficiency numbers.

For Spark Engine/LCF, distinguish energy used by the radiation/beam source, total facility electricity, measured nuclear/thermal output, useful converted output, and net exported electricity. Neutron counts do not establish useful energy gain.

## 5. Required metrics—never collapse them prematurely

| Layer | Metrics |
|---|---|
| Wheel/task | Wheel work (MJ/mission), peak/continuous wheel power, useful payload, hill/grade capability, acceleration and controllability within the same declared envelope |
| Storage/source | Battery wall-charge energy, fuel lower heating value and amount, hydrogen pathway, generator/fuel-cell source, reserve, usable vs nominal capacity |
| Conversion | Motor/inverter efficiency map, engine efficiency map by operating point, generator losses, charger losses, auxiliaries, regeneration recovered and actually reused |
| Complete system | Wh/km or MJ/km at a declared boundary, total propulsion-system mass/volume, power density, efficiency vs speed/load/temperature, thermal derating |
| Service | Range with reserve, wall-charge or refueling time, uptime, diagnostic time, repair task times, downtime, warranty and parts lead time |
| Durability/safety | Degradation with cycles, thermal limits, brake integration, failure-mode behavior, maintenance interval, component lifetime and safe fallback evidence |
| Economics | Cost per completed mission and 1,000 km; acquisition, energy, storage replacement, maintenance, downtime, equipment amortization, scrap/rework, warranty and end-of-life |
| Lifecycle | Upstream primary energy and emissions; battery/fuel production; electricity mix; fuel pathway; materials/recycling; water and local pollutants where relevant |
| Uncertainty | Measurement uncertainty, repeatability, model residual by operating envelope, source quality, missingness and sensitivity ranges |

For electric candidates, report wall-plug-to-wheel and battery-terminal-to-wheel separately. For ICE, report fuel-chemical-to-wheel and well-to-wheel separately. For a hybrid, report the complete combined source chain and how energy is apportioned. Never compare an ICE's brake thermal efficiency to an electric motor's efficiency as if both cover the same energy boundary.

## 6. Proposed benchmark targets

All targets are research hypotheses—not standards, achieved results or promises.

| Benchmark | Initial target / rule |
|---|---|
| Baseline quality | Same useful task, payload, duty cycle, reserve and ambient conditions; freeze source and configuration identity |
| Energy improvement | Stretch goal: at least 10% lower well-to-wheel energy per completed mission versus the matched baseline, only if the baseline measurements and uncertainty support a meaningful comparison |
| Whole-life cost | Stretch goal: at least 15% lower lifecycle cost per completed 1,000 km under documented regional prices and maintenance assumptions |
| Serviceability | After timed physical trials, target ≥25% less median and P90 routine repair downtime against a matched baseline |
| Manufacturing | Report yield-adjusted cost at 1/10/25/100/1,000 units; candidate goals such as ≥30% lower assembly touch labor at 25+ units require observed production runs with unchanged acceptance quality |
| Efficiency map | Provide maps across the actual duty-cycle operating points, not one peak laboratory number |
| Robustness | Repeat with battery ageing, temperature extremes, variable payload, road grade, wet/low-grip surface, supplier outage and degraded subsystem controls |
| Evidence | Every result binds exact configuration, data, equipment/calibration, model revision, boundary definition and uncertainty |
| Adversarial verifier | Zero false accepts in the frozen suite for missing energy inputs, omitted parasitic loads, incompatible units, stale revisions, incorrect duty cycles, unsupported LCF gain claims and mislabelled plasma- thruster metrics |
| Exotic engine claims | No universal numeric target until a reproducible model and measurement protocol exist. A proposal fails P0–P5 if its source, power density, net gain, measurement boundary or falsifier is absent |

The numerical targets should be adjusted after the first frozen baseline; they are neither universal nor meant to conceal trade-offs. Publish the Pareto frontier and worst-case results, not merely the winner.

## 7. What to build in Symthaea/Symtropy now

### Tranche A — Evidence contract and static energy ledger

Represent each energy stream with quantity type, unit, value/domain, uncertainty, boundary, evidence class, source identity and timestamp where applicable. Missing energy remains unknown. The validator catches inconsistent boundaries and missing auxiliary loads, but does not infer physical truth from a well-formed record.

### Tranche B — Source-verified reference dataset

Collect official/manual/source data for comparable real vehicles and components, recording exact model years/variants, load and test-cycle conditions, date and data rights. Manufacturer peak data, catalog listings, theoretical cell figures and measured duty-cycle results are separate evidence classes.

### Tranche C — Transparent reference models

Build an energy ledger and use-case-level simulation around existing Symthaea engineering/physical-type/solver infrastructure. Compare candidates under the same wheel-load trace. Validate motor/inverter and engine models against independent measured data before using them to make quantitative predictions.

### Tranche D — Robust optimization

Optimize the product, battery/tank sizing, motor operating region, cooling, drive ratio, regen, charging/replenishment network, production route and repair process. Preserve Pareto alternatives and hard constraints. Use model uncertainty to select the highest-value next test, not to inflate confidence.

### Tranche E — Controlled physical evidence

Begin with bench instruments and a serviceable low-energy demonstrator after engineering review. Use an appropriately equipped lab for any plasma-ignition investigation; any ionizing-radiation/LCF experiment belongs in a suitably authorized, independently instrumented facility. Separate simulation, bench, track, durability and legal approval evidence.

## 8. Research findings and source links

- **Two-wheeler electrification:** Nandola et al., *Energy* 324 (2025), 135911, compare gasoline and battery-electric two-wheelers with well-to-wheel accounting under an Indian electricity/fuel model. The paper reports the battery-electric vehicle's tank-to-wheel advantage exceeds upstream electricity disadvantages in its chosen case, while showing the result remains energy-mix dependent. It is a regional scenario, not a universal value: https://doi.org/10.1016/j.energy.2025.135911
- **Plasma-assisted combustion:** the review “Review on Plasma-Assisted Ignition Systems for Internal Combustion Engine Application” describes plasma-enhanced radical production/ignition as an approach to improve combustion/lean ignition, and details the implementation challenges; it is an enhancement to combustion rather than an independent energy source: https://www.mdpi.com/1996-1073/16/4/1604
- **Plasma propulsion:** NASA explains solar-electric propulsion uses electricity to ionize xenon/krypton and accelerate the ions using electric/magnetic fields; this is spacecraft propulsion, with its own thrust/power trade-offs: https://www.nasa.gov/space-technology-mission-directorate/tdm/solar-electric-propulsion/
- **Ion-thruster trade-offs:** NASA's in-space propulsion state-of-the-art describes Hall-effect thrusters as a widely deployed technology, while ion thrusters provide higher specific impulse but typically less thrust-to-power: https://www.nasa.gov/smallsat-institute/sst-soa/in-space_propulsion/
- **LCF observation:** NASA's summary and the peer-reviewed 2020 papers describe neutron/nuclear signatures from irradiated deuterated metals, but they do not establish a motorcycle power source or net-electric energy: https://www.nasa.gov/glenn/glenn-expertise-space-exploration/lattice-confinement-fusion/ ; https://doi.org/10.1103/PhysRevC.101.044609 ; https://doi.org/10.1103/PhysRevC.101.044610
- **Spark Engine's own claim boundary:** https://github.com/Luminous-Dynamics/spark-engine/blob/main/README.md and https://github.com/Luminous-Dynamics/spark-engine/blob/main/docs/TECHNICAL_REPORT.md
- **Lifecycle accounting:** ISO 14040 defines the principles/framework for lifecycle assessment: https://www.iso.org/standard/37456.html

## 9. Claim ceiling

This is a comparative protocol, not a design for a functioning engine. It provides no net-energy, propulsion, durability, safety, emissions, cost, roadworthiness or certification claim. NASA's reported LCF observations warrant scientific investigation but must not be converted into claims of a proven fusion power source. A software PASS can establish only the narrow scenario/ledger rules actually exercised, never a physical energy result.

## First implementation task

The first static ledger verifier and synthetic fixture have now been added to the two-wheeler research branch:

- Fixture: [`two_wheeler_powertrain_energy_ledger_v0_1.json`](data/two_wheeler_powertrain_energy_ledger_v0_1.json)
- Rust source: [`xtask/src/powertrain_energy_ledger.rs`](../../xtask/src/powertrain_energy_ledger.rs)
- CLI entry point: [`xtask/src/main.rs`](../../xtask/src/main.rs)

On a Rust-capable checkout, the intended commands are:

```sh
cargo test -p xtask powertrain_energy_ledger
cargo run -p xtask -- powertrain-energy-ledger \
  --input docs/research/data/two_wheeler_powertrain_energy_ledger_v0_1.json \
  --out reports/two_wheeler_powertrain_ledger_report.json
```

The fixture has one positive control and eight negative controls; ten Rust unit tests are authored for the verifier, including frozen-case replay and boundary/uncertainty controls. Its numeric streams are deliberately synthetic and exist only to exercise accounting rules. The Rust source/command are present but have **not** been compiled or executed in this environment; no verifier pass is claimed. This module is not a vehicle model or net-energy oracle.

Do not start with exotic plasma hardware, create a competing physics engine, or conflate Spark Engine's LCF hypothesis toolkit with a vehicle prime mover.
