# InstantClean hybrid system architecture (research draft)

Status: design hypothesis; not a released product design or validated performance claim.

## Design thesis

Build a resource-proportional cleaning appliance, not merely a dishwasher with a fast programme.

The machine should choose between a dedicated single-item/micro-batch cell and a conventional efficient full-load chamber based on urgency, validated cleaning requirements, current load, and measured marginal resource use.

The unit of optimization is:

`resource per verified successful clean`

where a successful clean must satisfy independently defined soil-removal, rinse-residue, material-compatibility and (where requested) sanitization endpoints.

## Physical architecture

### 1. QuickCell

A compact enclosed cell with interchangeable item fixtures:
- Plate cassette with two-sided access.
- Cup/mug dock with a retractable internal lance and external jets.
- Bottle / tall-vessel lance where geometry permits.
- Individually separated cutlery cartridge.

Start with one shared cell and exchangeable fixtures. Add parallel cells only if measured demand and lifecycle economics justify the additional valves, seals, dead volume and service burden.

Each fixture must define:
- Allowed materials and geometry envelope.
- Nozzle arrangement and validated flow/pressure envelope.
- Detergent compatibility.
- Maximum load and soil classes.
- Drainage and retained-water behaviour.
- Cleaning and final-rinse acceptance criteria.

### 2. FullLoad chamber

Retain a high-capacity chamber for normal household batches. It must compete independently against current full-size machines under the relevant IEC 60436 method. Do not weaken full-load objectives in order to make QuickCell look attractive.

### 3. Hydraulic service spine

Candidate shared components:
- Pump with measured pump curve.
- Instrumented supply manifold.
- Independently controlled branches and shutoff valves.
- Pressure and flow measurement at representative branches.
- Coarse debris separation and serviceable filtration.
- Drain and backflow protection.

Do not assume that one pump with many nozzles is optimal. Compare shared pumps, per-cell pumps and hybrid arrangements using measured pump curves, branch-loss models, acoustic performance, serviceability and real electrical power.

### 4. Separate fluid-quality states

Represent water as explicit process streams:
- Incoming fresh water.
- Recirculated wash water.
- Filtered/conditioned wash water, if validated.
- Fresh final-rinse water.
- Waste / purge water.

Do not silently mix final-rinse and wash-water circuits. Reuse is allowed only while a specified, experimentally validated water-quality envelope is satisfied. Turbidity alone is not a complete contamination or detergent-residue measurement.

The controller should fail closed to fresh rinse or drain when quality sensors are invalid, disagree or exceed calibrated ranges.

### 5. Thermal architecture

Use measured effective heated mass and thermal losses rather than assuming the whole cabinet/load reaches target temperature simultaneously.

Investigate and compare before choosing hardware:
- Low-volume local heating for QuickCell.
- Resistance heating, heat-pump heating/drying, zeolite/desiccant drying, AutoOpen/passive airflow and variable hot-air drying.
- Thermoelectric heating/cooling integrated with sensible or phase-change thermal storage, using the June 2026 TEHC-DW paper as a replication lead rather than assumed SOTA.
- Pre-wetting / pre-soak during delayed start (e.g. V-ZUG OptiStart) to soften soil and lower subsequent wash temperature, only within validated soil-specific limits.
- Shared thermal storage only when measured duty-cycle and standby losses justify it.
- Heat recovery from hot discharge streams, condensation and selective heating/cooling of fluid paths.
- Insulation and thermal isolation between concurrent QuickCell and FullLoad operations.

Compare total wall energy, internal water-heating energy, peak power, storage standby loss, drying energy/time at equal residual moisture, material dryness (especially plastics/recessed items), mass/volume, service burden and reliability. See [thermal-storage-benchmark.md](thermal-storage-benchmark.md) for the physical comparison protocol.

Report cold-start and warm-start results separately. Include heater, pump, controls, standby, drying and water-preheat energy in the system boundary.

## Scheduling paradigm: clean now or batch intelligently

Treat dirty items as jobs with:
- Arrival time.
- Deadline / urgency.
- Soil family and estimated severity.
- Geometry/material.
- Validated process eligibility.
- Cleaning and rinse endpoint.
- Resource and capacity demand.

At each decision point compare:
1. Run eligible items in QuickCell now.
2. Wait for a justified batch threshold and use FullLoad.
3. Split urgent and non-urgent items across the two systems.
4. Run parallel cells only if pressure/temperature/contamination isolation remains within its validated envelope.

The scheduler must not reduce a hard cleaning or rinse requirement to save time or water. It optimizes only among programmes that have already passed the relevant endpoint for that load class.

Use an explicit cost model:
- Electricity (kWh).
- Fresh water (L).
- Time / lateness.
- Detergent dose.
- Noise.
- Wear / expected service cost.
- Wastewater / filter loading.

Publish Pareto alternatives rather than hiding trade-offs in a single opaque score.

## Control and safety partition

### Deterministic safety supervisor

Owns:
- Door/lid interlocks.
- Leak and overfill shutdown.
- Overtemperature protection.
- Pump dry-run / blocked-flow response.
- Pressure limits.
- Valve sequencing and isolation.
- Electrical protection.
- Final-rinse quality rules.
- Sanitization programme constraints.

The safety supervisor is deterministic, independently testable and able to stop actuators without the AI.

### Adaptive optimizer (Symthaea)

May:
- Propose a validated recipe for a known soil/material class.
- Estimate resource use and uncertainty.
- Select among programmes inside their validated envelopes.
- Use camera-based load/item recognition as a *prior* for geometry and item type, fused with process sensors and user constraints.
- Request extra sensing or a conservative fallback.
- Learn from physical test data.

Must not:
- Invent a sanitization guarantee.
- Extrapolate a kinetic model outside its calibration envelope.
- Override interlocks or hard temperature/pressure/time constraints.
- Treat camera recognition, a visual classifier or turbidity/conductivity sensor as proof of cleanliness.
- Assume the new system's camera-based load recognition is novel: Miele announced the G8 Diamond AutoVision camera-AI system on 2026-09-02 for load/type recognition, wash adaptation, bottle interior cleaning and plastic drying. G8 DACH launch is announced for 2027-04; matched test data are not yet published.

The observer benchmark must compare manual selection, existing turbidity/auto sensing, camera-only load recognition (including the announced Miele incumbent when testable), and camera + turbidity + conductivity + hydraulic/thermal fusion. Camera use must have explicit privacy settings, a no-camera operating path and a retention policy; these are InstantClean design requirements, not statements about another manufacturer's implementation.

Every control action should be logged with model version, camera/model version where enabled, recipe version, sensor quality, uncertainty, scope validity and fallback reason.

## Simulation division

### Symtropy geometry screen

Use geometric visibility and incidence-angle metrics to rank nozzle locations, dish arrangement and likely line-of-sight shadows. These metrics are not CFD, wetting or soil removal.

### Symthaea analytical model

Use lumped pump/system curves, measured branch resistance, thermal accounting and calibrated soil-specific response models. Preserve model uncertainty and validity ranges.

### Higher-fidelity physics

Use validated CFD / multiphase solvers only where the question needs them: rotating spray arms, jet breakup, transient pressure waves, splash, wall films or droplet deposition. Calibrate against open physical rig data before using results for design claims.

## Prototype sequence and gates

### Gate 0 — data and safety design
- Define measurement units, data contract, instrument calibration and interlocks.
- Freeze soil recipes, surfaces, detergent batches, water hardness and endpoint definitions.
- No cleaning claims.

### Gate 1 — hydraulic rig
- Characterize pump pressure-flow curve.
- Measure branch pressure/flow and leakage.
- Compare analytical network predictions to at least three independent operating points per branch configuration.
- Target <=10% mean absolute relative error on held-out flow predictions, <=10% pressure prediction error, and disclose every outlier. These are proposed research gates, not an industry standard.

### Gate 2 — geometric coverage and deposition
- Compare Symtropy visibility/incidence estimates with witness-film or calibrated deposited-mass measurements.
- Test sample-count and mesh convergence.
- Report geometry-to-deposition correlation, not only visual overlays.

### Gate 3 — soil-specific cleaning kinetics
- Separate protein, carbohydrate/starch, oil/fat, tea/coffee and mixed soils.
- Use independent day/batch holdouts.
- Compare first-order, two-stage and nonlinear models.
- Reject extrapolative predictions.

### Gate 4 — QuickCell
- Single plate and cup protocols.
- Measure complete fresh-water, energy and elapsed-time boundaries.
- Compare with the smallest fair conventional reference cycle.
- Include rinse residue and material compatibility.

### Gate 5 — integrated full load
- Test the exact IEC 60436 protocol using a licensed copy of the standard.
- Compare with at least three market reference machines.
- Report cleaning/drying, energy, water, duration, capacity and noise separately.

### Gate 6 — durability and service
- Repeated-cycle wear.
- Filter blockage and cleanability.
- Seal/valve leakage.
- Material/chemical compatibility.
- Repair time and replacement-module cost.
- Cross-contamination and microbiological studies where applicable.

## First decision rules

1. Never compare quick-cycle data to eco-label data as if they were the same test.
2. Never call a visible surface clean, a wetted surface clean, or a clean surface sanitized.
3. Resource targets are valid only when the cleaning and rinse endpoints pass.
4. Compare both total system resources and per-item resources.
5. Retain raw runs; report central tendency, spread and confidence intervals.
6. Freeze protocol, source, firmware, geometry and analysis before confirmatory testing.
7. Negative controls must fail the verifier; invalid sensors must trigger conservative fallback.
8. A failed simulation/model validation should narrow the model's scope, not weaken the gate.
