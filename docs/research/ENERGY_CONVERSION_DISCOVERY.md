# Symthaea Energy-Conversion Discovery Program

**Status:** research proposal plus initial analytical baselines  
**Owner:** Luminous Dynamics / Symthaea  
**Last reviewed:** 2026-10-15

## Mission

Use Symthaea to discover and validate energy-conversion systems that outperform established options on a specified mission. The search must be architecture-neutral: combustion is one candidate family, not the default; electric motors, fuel cells, heat engines, turbines, waste-heat recovery, storage-assisted hybrids, and combinations can all be considered when the energy source and mission make them relevant.

The claim being tested is not that Symthaea can ignore physical limits. The claim is that a cognitive search-and-design system may explore a wider, better organized design space, identify high-value experiments, and reduce engineering iteration while every performance claim remains grounded in independently checkable physics and measurement.

## Current repository substrate

The work should compose existing public Symthaea capabilities rather than start a parallel monolith:

- `symthaea-thermofluids`: applied heat-transfer and fluid-flow relations. The `cycles` module provides ideal Carnot, Brayton, Otto, and Diesel efficiency baselines with input validation and analytical reference tests. The `energy_balance` module checks steady-state first-law closure and gross/net efficiency accounting; `energy_interval` checks transient interval closure and validates each interval before duty-cycle aggregation.
- `symthaea-materials`, `symthaea-structural`, and `symthaea-fabrication-kernel`: material/structural constraints and eventual geometry/fabrication hand-off.
- `symthaea-circuits` and `symthaea-control-theory`: electrical conversion, instrumentation, and control models.
- `symthaea-operations-research`: optimization techniques and constrained trade-offs.
- `symthaea-sim-bridge`, `symthaea-digital-twin`, and `symthaea-core` physics/dynamical-system modules: simulation integration and trajectory reasoning.
- `symthaea-formal-safety` and `symthaea-lean-bridge`: safety cases and formal obligations where applicable.

These components do not by themselves establish a high-fidelity engine-design capability. Each solver and coupling still needs to be qualified for the question it is used to answer. In particular, the generic dynamical-system integrators are not a substitute for validated combustion CFD, turbomachinery, electrochemistry, structural fatigue, or material-property solvers.

### System-boundary energy accounting

The new `energy_balance::EnergyBalance` API accepts four mutually exclusive steady-state power flows in watts: input, net useful output, auxiliary consumption, and rejected/other loss. It computes gross output as net useful output plus auxiliary draw, checks that gross output does not exceed input, and rejects a first-law residual that exceeds a caller-supplied relative tolerance. The tolerance must be chosen and justified before examining a candidate's score; do not relax it to make a design pass.

This is intentionally a narrow accounting primitive, not an engine simulation. It assumes steady-state operation and cannot account for changing stored energy, start-up, or transient thermal inventories. For a transient interval, use `energy_interval::EnergyIntervalBalance` with input, net useful output, auxiliaries, rejected/lost energy, and signed change in stored energy. A positive stored-energy change means energy accumulated; a negative value means a store discharged. Its reported throughput includes external input plus released stored energy, and it intentionally does not report a naive output/input efficiency for a storage-discharge interval. The `evaluate_duty_cycle` aggregator validates every interval before summing, then checks aggregate closure as a second gate so opposing residual errors cannot hide each other. It reports a net cycle efficiency only when net stored-energy change is within the declared tolerance of total throughput and is non-negative, so a small net discharge cannot inflate useful-output/input. All interval values must use the same registered boundary and measurement interval. A balance that fails closure is a failed data-quality gate, not a reason to tune the efficiency calculation.

## Define “better” before searching

There is no universal score for an engine. A road vehicle, an aircraft, a ship, and a stationary generator have different system boundaries and constraints. Every benchmark must register its mission and comparison basis before candidates are generated.

Required common fields:

- **Mission:** duty cycle, ambient conditions, load profile, start/stop behavior, altitude or pressure regime, required life, maintenance capability, and safety constraints.
- **Energy boundary:** name the energy entering the system and the useful output. For vehicles, distinguish tank-to-wheel from well-to-wheel; include upstream energy only in the latter. For generators, report net electricity after auxiliaries, not gross turbine output.
- **Hard constraints:** output power/torque or thrust, response time, operating envelope, mass and volume limits, temperature/pressure/material limits, emissions, safety, and service life.
- **Economic/ecological outcomes:** manufacturing and operating cost, maintainability, supply-chain concentration, critical-material demand, embodied emissions, operational emissions, and end-of-life handling when data exists.
- **Evidence quality:** record source, units, uncertainty, solver and version, inputs, output artifacts, and whether results are analytical, simulated, calibrated, or measured.

Use a vector of metrics and a Pareto frontier, not a single opaque scalar that can hide severe weaknesses. If a scalar is needed to rank a particular procurement decision, publish the weights and sensitivity analysis. A Pareto frontier is not, by itself, a decision rule: preregister how an implementable candidate will be selected (hard constraints, explicit preferences, and tie-breaks) and report whether the recommendation changes under plausible preference and uncertainty variations. Recent reviews identify this last step as commonly under-specified in multi-objective energy-system studies.

## Candidate architecture families

Search within families and across them. Do not assume the best system is a single prime mover.

1. **Electric drive:** synchronous and induction motor families, rare-earth-reduced or rare-earth-free designs, motor/inverter/gear integration, cooling, regenerative recovery, and control.
2. **Thermal power conversion:** air-standard Otto/Diesel/Brayton baselines, recuperated Brayton, Rankine/organic Rankine, supercritical-CO2 cycles, Stirling, and waste-heat recovery where a real heat source exists.
3. **Chemical-to-electric:** fuel cells and battery/fuel-cell hybrids. Include reactant storage, balance of plant, power electronics, lifetime, and upstream energy—not merely cell-stack efficiency.
4. **Combustion and propulsion:** conventional reciprocating/turbine designs and researched alternatives such as rotating detonation propulsion only when the mission fits. Rocket engines must be compared using appropriate propulsion metrics such as specific impulse, thrust-to-mass, and propellant-system mass, not road-vehicle efficiency.
5. **System-level hybrids:** multiple converters, thermal cascades, storage, waste-heat use, and mission-aware control. A hybrid is worthwhile only if gains survive extra mass, conversion losses, cost, and reliability penalties.

These are search categories, not recommendations that all technologies are mature, commercially available, or superior.

## Physics and evidence ladder

Use the cheapest model that can answer the current question, then raise fidelity as candidates survive.

| Tier | Model/evidence | Permitted claim |
|---|---|---|
| T0 | Closed-form equations and dimensional checks | Analytically consistent under stated assumptions |
| T1 | Reduced-order numerical model with conservation/property tests | Behaves as expected within the validated model envelope |
| T2 | Qualified external solver or calibrated digital twin, reproducible inputs | Predicted performance within its validated domain and uncertainty |
| T3 | Component or subsystem bench data | Measured component result under stated conditions |
| T4 | Integrated prototype tested on the registered mission cycle | Measured system-level result for that prototype and operating envelope |
| T5 | Independent replication or independent test facility | Reproduced claim, limited to the replicated configuration and protocol |

Map these states to the repository's evidence model without upgrading claims automatically. In the existing LEM empirical axis, E1 is local simulation output, E2 means replay evidence is available, E3 means integrity-protected evidence, and E4 means public reproduction. A higher self-reported confidence or HDC similarity is not a substitute for any evidence transition.

Every candidate result should produce a durable run record containing: immutable candidate ID, source revision, configuration hash, solver/version, model fidelity, all inputs and units, boundary conditions, random seeds where relevant, outputs, uncertainty estimates, constraint violations, and artifact hashes. The independent checker should recompute cheap invariants instead of accepting the search system's own declaration that a design is valid.

## What Symthaea should do—and not do

Symthaea can contribute to candidate generation, cross-domain analogy search, parameter-space prioritization, experiment selection, anomaly explanation, and multi-objective trade-off analysis. HDC representations may help index and compare design structures; liquid dynamics and active inference may help choose the next informative evaluation. Those are hypotheses to evaluate through ablation, not presumed advantages.

The numerical solver remains responsible for physical predictions. Learned surrogates may accelerate candidate screening only after calibration and out-of-sample error bounds are established; finalists must be re-evaluated by the authoritative higher-fidelity solver. A candidate that violates energy or mass conservation, dimensional consistency, material limits, or registered constraints is rejected regardless of its score.

## Initial experimental sequence

### E0 — analytical baseline (started in this change)

The `symthaea-thermofluids::cycles` module adds input-validated, idealized efficiency calculations for Carnot, Brayton, air-standard Otto, and air-standard Diesel cycles. Unit tests cover reference values, parameter trends, and invalid inputs.

This baseline is deliberately modest. It is a small trustworthy floor, not a design optimizer and not proof that a real engine achieves these efficiencies. Carnot is a reversible upper bound; the other functions use ideal-gas/constant-specific-heat assumptions and isentropic idealizations.

### E1 — benchmark contract and known-system reproduction

Before proposing a novel system, freeze at least one target mission and reproduce a known design within stated tolerances. Require the steady-state energy-balance checker to close for steady-state records and the interval-integrated checker to close for transient tests, including energy gained or released from storage. Populate measured or authoritative baseline data with sources and compatible system boundaries. Do not mix ideal-cycle values with measured engine results in the same leaderboard.

Use negative controls: deliberately invalid ratios, unit/boundary mismatches, impossible material properties, and candidate outputs that violate constraints must be rejected. Preserve them as regression tests.

### E2 — calibrated multi-fidelity search

Expose a typed design representation with explicit ranges and dependencies. Enumerate established candidate families first, then use an optimizer (grid/random or Latin-hypercube baseline, followed by Bayesian or evolutionary multi-objective search if justified). Compare search methods under equal solver budgets and the same constraints.

Optimize across a registered operating envelope and mission duty cycle, not just the most favorable design point. Report both the best-point result and duty-cycle-weighted net output/efficiency, start/stop costs, constraint violations, and robustness to uncertain inputs. The 2025 supercritical-CO2 cycle design/control review specifically notes that design-point-only optimization underutilizes system potential and argues for off-design points and coupled control design. This is a general lesson for thermal systems, not evidence that one particular cycle is universally superior.

The Symthaea-generated search should beat simple baselines on at least one preregistered measure—e.g., time to find a feasible candidate, hypervolume of the Pareto set, or validated improvement over the incumbent—without increasing false-feasible rates. Report failed candidates and the full experiment budget, not only winners.

### E3 — candidate down-selection and independent engineering review

Select designs for high-fidelity simulation only when the expected information gain or mission value justifies the cost. Check sensitivity to uncertain inputs, off-design operation, degradation, and manufacturing tolerances. Run structural, thermal, controls, and fault analyses appropriate to the candidate. Complete an independent model review before hardware is built.

### E4 — instrumented prototype

Define the physical test plan before fabrication: calibration records, sensor uncertainty, safety interlocks, environment, duty cycle, acceptance thresholds, sample size, abort criteria, and raw-data retention. Compare against an incumbent on the same rig or use a documented normalization method. Distinguish peak bench readings from continuous duty-cycle performance.

No operational autonomy or real-world actuation should be enabled merely because a simulation passes. Hardware testing requires ordinary engineering controls, qualified operators, and independent safety review.

## Initial acceptance gates

A candidate cannot advance unless all relevant gates pass:

- **Physics:** steady-state and interval-integrated conservation balances close within declared tolerances; transient studies account for stored-energy change; assumptions and domains are explicit; no over-unity claims or impossible property combinations.
- **Software:** deterministic test vectors, invalid-input tests, reproducible runs, immutable artifact hashes, and pinned solver/tool versions.
- **Model validity:** validation data is separate from optimization data; uncertainty and extrapolation are surfaced; a low-cost surrogate cannot certify itself.
- **Comparison fairness:** identical mission and boundary, compatible units, equal accounting for auxiliaries and storage, and transparent baseline sources.
- **Robustness:** acceptable performance across an agreed operating envelope, not just at one tuned point.
- **Evidence:** all changes in evidence tier are backed by artifacts and review; queued CI is not a pass, and simulation is not hardware validation.

## Immediate follow-on engineering work

1. Run `cargo test -p symthaea-energy-bench`, `cargo test -p symthaea-thermofluids`, and `cargo fmt --all -- --check` on the exact PR head. The authoring environment lacks Rust, so authored tests have not been executed locally.
2. Run a JSON Schema Draft 2020-12 validator against the schema and synthetic fixture, then add a verifier integration test that applies structural and semantic validation together. Keep the independent Rust semantic verifier separate from the search logic.
3. Strengthen manifest provenance by binding the preregistered policy/manifest digest to an immutable commit or signed append-only registry before search starts. Timestamp ordering in a manifest is not cryptographic evidence of preregistration.
4. Add uncertainty propagation and comparison logic that preserves relevant correlations; unknown material uncertainty must make a superiority decision indeterminate rather than favorable.
5. Reproduce at least one authoritative or measured system with a common boundary before novelty search. EPA/NREL drive schedules define demand, not by themselves a measured energy-conversion baseline; verify that the selected reference dataset contains required input/output energy channels.
6. Only after the benchmark verifier and reference case are reproducible should Symthaea introduce architecture generation and multi-objective optimization. Compare search strategies under equal model-evaluation budgets and preserve failed candidates as evidence.

## Research references

- U.S. Department of Energy, [Typical EV efficiency compared with conventional gasoline vehicles](https://www.energy.gov/cmei/vehicles/articles/fotw-1360-sept-16-2024-typical-ev-87-91-efficient-compared-30-conventional) (2024). Useful vehicle-level motivation; its accounting assumptions must be preserved when making comparisons.
- NASA, [Integrated Rotating Detonation Engine System (InRoDES)](https://www.nasa.gov/integrated-rotating-detonation-engine-system-inrodes/). An example of a specialized propulsion architecture progressing through hardware tests; it does not establish superiority for road vehicles or other missions.
- [A review on supercritical CO2 and CO2-based mixtures in power cycles](https://doi.org/10.1016/j.enconman.2024.119295), *Energy Conversion and Management* (2025). Reviews cycle configurations, components, heat-source integration, and limitations.
- [Review of system design and operation control technology of supercritical CO2 power cycle](https://doi.org/10.1016/j.enconman.2024.119462), *Energy Conversion and Management* (2025). Emphasizes joint system-design and off-design control issues; its review notes the shortcomings of design-point-only optimization.
- [Machine learning techniques in thermodynamic cycle system design and control for energy harvesting](https://doi.org/10.1016/j.rser.2025.115802), *Renewable and Sustainable Energy Reviews* (2025). Survey of machine-learning work on fluids, configurations, components, dynamics, and control.
- [Multi-objective stochastic optimization problem: a systematic literature review](https://doi.org/10.1016/j.apenergy.2025.127237), *Applied Energy* (2026). Synthesizes 572 studies published from 2018–2025, emphasizing uncertainty and conflicting objectives in energy and sustainability applications.
- [From Pareto Front to Preferred Design: Human-in-the-Loop Preference-Guided Decision Making in Multi-Objective Energy Systems Optimization](https://www.mdpi.com/2076-3417/16/10/4966), *Applied Sciences* (2026). Reviews explicit and reproducible ways to select preferred solutions from Pareto sets, including sensitivity to changing preferences.

## Claim discipline

The success criterion for this program is not “Symthaea invented an engine” in the abstract. It is: **for a registered mission, the workflow produces a reproducible candidate that meets all hard constraints and demonstrates a measured, independently reviewed advantage over a fair incumbent baseline.** Until that final evidence exists, results remain hypotheses or model-level findings.
