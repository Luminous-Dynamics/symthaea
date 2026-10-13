# Subterranean Rock Mechanics and Excavation Physics Roadmap V1

**Status:** Research/design proposal; not an implemented or qualified physics model  
**Date:** 2026-10-13  
**Scope:** Symthaea subterranean domain, physics interfaces, and robot-control validation  
**Next step:** Freeze typed material/evidence contracts and a benchmark corpus before adding runtime physics.

## 1. Executive position

Yes: Symthaea can become a much stronger platform for characterizing, planning around, excavating, sampling, and stabilizing hard rock such as granite and basalt. The route is not to treat an HDC representation as a substitute for mechanics. It is to compose:

1. measured geology and rock-mass evidence;
2. a bounded, calibrated rock/tool model;
3. numerical mechanics where stress redistribution and fracture matter;
4. uncertainty-aware perception and mapping;
5. a robot planner plus an independently enforced safety envelope; and
6. falsifiable tests against analytical, numerical, laboratory, and eventually field evidence.

This is a proposal for engineering capability, not a claim that Symthaea can currently predict real granite or basalt excavation accurately. Software cannot manipulate rock without suitable machinery, energy, tooling, access, and operator authorization.

“Hard rock” is not one scalar property. Intact-rock strength, tensile strength, fracture toughness, elasticity, abrasivity, porosity, permeability, thermal behavior, saturation, confining stress, weathering, and the orientation and condition of joints can all change the outcome. A model that labels a stratum “granite” and assigns a hardness score is useful for a lightweight scenario, but is not yet a calibrated granite constitutive model.

## 2. Current baseline found in the public repository

This review inspected the default-branch source for:

- crates/domains/symthaea-subterranean/src/geology.rs
- crates/domains/symthaea-subterranean/src/simulator.rs
- crates/domains/symthaea-subterranean/src/causal_attribution.rs
- crates/domains/symthaea-subterranean/src/embodiment.rs
- crates/domains/symthaea-subterranean/SUBTERRANEAN_HARDENING_PLAN_2026-07-20.md
- crates/domains/symthaea-subterranean/SUBTERRANEAN_CAPABILITY_CAMPAIGN_V_2026-07-20.md

Source inspection is not a build or runtime qualification.

### Existing strengths to preserve

- A stratified geotechnical profile, sampling, and geology lookahead already exist.
- Rock characteristics influence penetration, cutter heat, spoil, and hazard-related behavior in the deterministic reference simulator.
- The simulator's principal dynamics use rates/continuous-time relaxation, with an existing timestep-invariance goal.
- Command/response attribution can retain ambiguous and contradicted outcomes instead of automatically promoting correlation to proof.
- The embodiment has a substantial, separate physical-hazard arbitration and authority/safety boundary.
- Existing design documents already identify scenario curricula, external sensor/plant boundaries, independent observation, and the gap between reference simulation and hardware qualification.

### Current fidelity boundary

The current geology model stores normalized properties in [0, 1], including hardness and abrasiveness, and its material enum includes granite but not basalt. The simulator uses reduced expressions such as a hardness-based penetration factor and simplified thermal/load dynamics. This is a legitimate deterministic control testbed; it is not yet a stress/strain, joint-network, fracture-propagation, or calibrated rock–cutter model.

Do not replace this useful fast simulator wholesale. Keep it as a cheap reference plant and add a separate, explicit model layer whose fidelity and evidence level are stated.

## 3. Architecture: separate five different questions

The system should not confuse these claims:

| Question | Required evidence/model | Does not establish |
|---|---|---|
| What rock is likely present? | Sensor observations, survey context, material hypotheses and calibrated uncertainty | Exact strength or safe excavation |
| How will the rock mass respond? | Stress, joints, groundwater, boundary conditions, constitutive model and uncertainty | Tool performance without tool/rock coupling |
| How will this cutter interact with it? | Tool geometry, rotation, thrust, feed, wear and calibrated cutting model | Stability of the whole excavation |
| What should the robot do next? | Planning over mission, energy, map, uncertainty, tool limits and hazards | Permission to bypass safety/authority checks |
| Did the real action behave as predicted? | Independent, calibrated post-action measurements and provenance | Proof merely because the simulator agrees with itself |

Recommended composition:

~~~text
raw observations and test records
    -> provenance, unit and calibration checks
    -> geological/material belief with uncertainty
    -> rock-mass model + tool/rock interaction model
    -> predicted response distribution and alternatives
    -> mission/exploration/excavation planner
    -> independent hard safety/authority arbitration
    -> actuator interface
    -> independent measured response
    -> discrepancy, causal attribution and model requalification
~~~

Symthaea's appropriate role is to bind evidence, manage hypotheses and uncertainty, perform fast prediction/control, select experiments and plans, and monitor discrepancies. High-fidelity FEM/DEM or a validated specialist solver should remain a replaceable physics component—not be silently approximated by a cognition layer.

## 4. Material contract: replace “hardness = physics” with conditional measurements

Create an additive, versioned rock-material contract before altering runtime behavior. Candidate intact-rock properties include:

- lithology/mineralogical description and specimen/source identity;
- bulk density (kg/m³);
- unconfined compressive strength (Pa), with test method and specimen condition;
- tensile strength (Pa), with explicit test method;
- Young's modulus (Pa) and Poisson's ratio;
- cohesion (Pa) and friction angle with explicit angle units, when the constitutive model needs them;
- fracture toughness (Pa·√m), with mode and test method;
- porosity (fraction) and intrinsic permeability (m²);
- P-wave/S-wave velocity (m/s);
- thermal conductivity (W/(m·K)), heat capacity (J/(kg·K)), and thermal expansion (1/K) where relevant;
- abrasivity test result and test identity, including CERCHAR where used;
- water saturation, temperature, confining stress, loading rate, orientation, weathering and alteration state.

Intact specimens do not describe the whole rock mass. A separate rock-mass structure should represent joint/fracture orientation and coordinate frame, spacing, persistence, aperture and roughness, joint friction/cohesion and infill, groundwater/pore pressure, in-situ stress estimate, faults/crushed zones/contacts/voids, spatial extent and uncertainty.

### Property provenance and uncertainty

Each measurement or estimate must identify units, method, specimen/site, conditioning state, uncertainty representation, source digest/reference, calibration state, and whether it is observed, inferred, synthetic, or externally supplied. Correlated measurements must not count as independent support merely because separate software paths delivered them.

Do not invent universal “granite values” or “basalt values.” Both lithologies vary with mineralogy, texture, cracks, weathering, saturation, confining pressure, specimen scale, and test method. Initial defaults must be marked synthetic-reference-only; they are not site characterization.

## 5. Physics ladder: add fidelity only when a benchmark needs it

### Tier 0 — Existing fast reference simulator

Retain the current deterministic reduced-order plant for controller tests and large scenario sweeps. Document every proxy, units convention, omitted mechanism, and admissible claim. Do not label it “high-fidelity rock physics.”

### Tier 1 — Calibrated excavation-response model

Add a small, deterministic, unit-aware reduced-order model for drilling/cutting. Candidate inputs include material belief and confidence, tool/cutter geometry and wear, torque, rotational speed, thrust, axial feed/penetration rate, removed volume, spoil state, cutter temperature and cooling, plus groundwater/abrasion/load conditions where supported by data.

One useful observable is mechanical specific energy (MSE):

~~~text
MSE = measured mechanical input power / measured rock-volume removal rate
~~~

Rotary and axial power must be defined consistently from torque × angular velocity and axial force × feed velocity. The measurement contract must state which powers, losses, and removed-volume estimate are included. MSE has energy-per-volume units (Pa); it is not a universal rock-strength constant. Use measured removed volume and operating data, and model cutter wear rather than assuming all deterioration is geology.

Start with a transparent equation and calibrated parameter ranges. Compare against independent single-cutter/drilling experiments; do not fit and evaluate on the same specimens or sites.

### Tier 2 — Continuum stress and deformation

When the question involves excavation-induced stress redistribution, deformation, yielding or support loads, connect a continuum solver through a narrow, versioned interface. Finite-element or finite-difference approaches can be appropriate depending on the problem and solver.

Required inputs include geometry/mesh, boundary and initial stress conditions, constitutive law and parameters, excavation sequence, pore pressure/water boundaries where applicable, and declared numerical tolerances. Outputs need units, model version, convergence status and digests of the inputs.

### Tier 3 — Discontinuities and fracture

When explicit joints, block motion, crack initiation/growth, fragmentation or cutter-induced fracture determine the answer, evaluate discrete-element, bonded-particle, phase-field, or coupled finite/discrete approaches. Choose a solver from the validation case and failure mechanism, not because the method is fashionable.

A smooth continuum approximation must not erase a known joint set. Conversely, a particle model should not be used when its parameters cannot be identified from available observations. Coupling and GPU acceleration are later options after physical correctness and benchmark coverage—not evidence of correctness by themselves.

### Model-fidelity policy

The runtime must answer: what mechanism was represented, which were omitted, which parameters were fitted, what evidence validates this regime, and whether this input is outside the validated envelope. Out-of-envelope inputs should yield Unknown/InsufficientEvidence or a restricted disposition, not inherit a nominal prediction as a safety fact.

## 6. Calibration and validation campaign

Use the International Society for Rock Mechanics and Rock Engineering (ISRM) suggested methods as a testing-method reference. The exact procedure must match the property and model being qualified; do not replace standards with ad hoc labels.

### R0 — Freeze the benchmark manifest

Each case includes unique case and specimen/site IDs, material description and sample condition, procedure and instrument/calibration metadata, units and measurement uncertainty, raw-data digest and access/licence provenance, model/parameter version, calibration versus held-out partition, expected observables, a tolerance justified by its source, and known limitations/confounders. Keep synthetic, literature-derived, laboratory and field cases distinct.

### R1 — Material characterization

For selected granite and basalt samples, prioritize UCS/deformability, tensile strength, fracture toughness, abrasivity, density/porosity and wave velocities. Add triaxial behavior, permeability, thermal response and joint behavior only for target use cases. The objective is to measure what affects a declared prediction, not maximize the number of fields.

### R2 — Reduced-order drilling/cutting tests

Measure torque, thrust, rotational speed, penetration/feed, removed volume, wear and temperature. Test controlled comparisons with appropriate repeats. Report prediction error and interval coverage on held-out samples; set acceptance targets before examining held-out results.

### R3 — Analytical and numerical verification

Before matching laboratory outcomes, verify implementation against known limiting cases:

- dimensional/unit checks and finite/range validation;
- elastic response against a chosen analytical case;
- fracture or joint cases with a documented reference result;
- energy accounting and mesh/time-step convergence where relevant;
- stable results under unit-preserving rescaling;
- explicit handling of non-convergence and missing boundary conditions.

A plausible visualization is not enough if conservation, convergence or limiting-case tests fail.

### R4 — Adversarial and negative controls

At minimum, test that:

- a missing property is not silently replaced by a confident “typical granite” constant;
- greater intact strength does not *by itself*, with all other conditions fixed, increase predicted penetration in the reduced-order model;
- greater cutter wear cannot be reported as an unexplained improvement in cutting efficiency;
- stale or conflicting geological observations cannot raise confidence;
- duplicate/correlated sensor channels cannot satisfy an independent-evidence requirement;
- non-convergence or out-of-envelope input cannot produce an ordinary qualified result;
- simulation output is never mislabeled as physical measurement;
- changing units without changing the physical quantity preserves predictions within the declared numerical tolerance.

These are scoped model properties, not universal laws for every rock/tool configuration.

### R5 — Independent predictive evaluation

Report more than mean error: include interval coverage, worst-case error, residual trends by lithology and test regime, out-of-distribution behavior, and known failure cases. Group train/test splits by independent specimen/site or test campaign to prevent near-duplicate leakage.

### R6 — Hardware-in-the-loop and field evidence

Only after earlier stages close, use instrumented, constrained test equipment, controlled specimens and approved procedures. Compare predictions with independent sensors and measured removed volume. Set a discrepancy stop rule before testing. Field trials require qualified engineering oversight, site-specific risk assessment and independent emergency stop. Simulated success is not field qualification.

## 7. Subterranean robotics: upgrade world modelling and task planning together

### A. Shared spatial/geological map

Move beyond a depth-only profile for branching environments. Build a frame-aware 3-D/3-D-plus-time map retaining free space and tunnel topology, rock/material hypotheses and discontinuities, pose/map covariance, sensor coverage and unobserved volumes, water/gas/temperature observations with freshness, traversability/clearance/slope uncertainty, drilling/cutting history, tool condition and provenance for each hypothesis. Do not conflate geometry, semantics, geological inference and safety authority. Two maps derived from one physical sensor are not two independent witnesses.

### B. Active geological probing

Choose a probe or measurement when expected information can change the decision. Candidates include a short low-risk drill, acoustic/ultrasonic measurement, a new scan viewpoint, or a water-pressure observation. Value-of-information calculations must include energy, tool wear, exposure, return reserve and the cost of being wrong. If critical uncertainty remains unresolved, pause or retreat rather than reasoning through missing evidence.

### C. Task- and robot-specific planning

Plan over energy and return reserve, tool wear, excavation rate, spoil capacity, thermal margin, water/gas hazards, roof/support requirements, communication windows, uncertainty and repair resources—not distance alone. A tunnel-boring scout, tracked inspection vehicle, quadruped and aerial mapper should share evidence contracts and map semantics, not unexamined dynamics or safety envelopes.

### D. Closed-loop prediction discrepancy

For every significant action, record predicted response distribution, command identity, timing window, model version and independently observed outcome. Update model parameters only through a declared calibration procedure; online learning must not silently redefine safety margins or erase contradictory evidence.

### E. Multi-agent mapping and work allocation

Retain existing conservative team/tunnel-graph work, then add geometric uncertainty and model/version provenance. A teammate's optimistic map cannot clear a local hazard. Partitioned communications must not silently authorize stale work plans or recovery.

### F. Independent safety and operator authority

Rock-model confidence may inform planning only within the reviewed envelope. It must not loosen physical interlocks, bypass operator restrictions, convert rescue consent into implied consent, or mint actuator authority. Safety decisions remain independently testable and fail closed when critical evidence is insufficient.

## 8. Symthaea integration boundary

Use Symthaea for multimodal evidence fusion, uncertainty-aware hypotheses, active geological observation selection, fast reduced-order prediction within its validated envelope, planning and counterfactual comparison, deterministic trace recording, model-discrepancy detection, and connecting sensor/test/model/action provenance.

Use specialist physics components for constitutive stress/strain, fracture initiation/propagation/fragmentation, validated contact and tool–rock mechanics, meshing and numerical error estimation.

The interface should be language/runtime neutral at the evidence boundary, typed in the Rust host, versioned, memory-bounded, and explicit about time, units, coordinate frames, solver exit status and input/output digests. A specialist solver must not run synchronously on a safety-critical loop unless its worst-case runtime and failure behavior have been independently demonstrated. The controller should use bounded estimates, monitor validity/freshness and retain an independent fallback.

## 9. Incremental implementation plan

### Phase A — Contract and corpus

Define versioned material, rock-mass, tool state, model request/result and benchmark schemas. Mark every property with units, condition, uncertainty, method and provenance. Add Granite and Basalt to the lithology vocabulary without assigning unqualified canonical constants. Define a model-claim ledger: validated regime, evidence, limitations and promotion requirements.

**Exit:** schemas reject missing units, non-finite values, contradictory ranges, unsupported provenance claims and synthetic-as-observed substitutions.

### Phase B — Benchmarked reduced-order model (Rust)

Add a separate excavation-response module rather than overloading present simulator hazard proxies. Implement a dimensionally checked MSE/penetration/wear baseline. Add analytic tests and frozen, provenance-bound benchmark fixtures. Keep its output advisory to planning; it does not grant actuator authority.

**Exit:** deterministic replay, analytic/unit/monotonicity gates, held-out benchmark report and explicit unknown/out-of-domain cases.

### Phase C — Rock-mass geometry and sensing

Represent discontinuities, groundwater and stress assumptions separately from lithology. Add sensor timestamps, calibration, source independence and spatial-frame validity. Fuse drilling telemetry with non-invasive measurements as evidence, not ground truth.

**Exit:** fault-injection tests show stale, correlated, missing and contradictory data reduce or preserve confidence, never increase it.

### Phase D — Solver adapters and fracture benchmarks

Select continuum and fracture solvers only against frozen cases requiring their mechanisms. Maintain adapter contracts and reproducible solver manifests. Compare against public or permissioned reference cases after licensing review.

**Exit:** independent reproducibility and a declared validity domain; a library integration compiling alone is not the gate.

### Phase E — Robot-in-the-loop workcell

Run repeatable, controlled cut/drill trials with independent metrology. Test degraded sensors, wear, jams, thermal/flood/roof interventions, communications loss and return reserve. Capture exact hardware/software, specimen, tool and calibration identities.

**Exit:** predefined prediction-error and safety-intervention criteria met; external review approves any broader trial.

No phase automatically authorizes operation in an occupied underground space or proves suitability for mining, tunnelling, construction or rescue.

## 10. Promotion and reporting rules

Report evidence by tier, not with one blended “intelligence” or “fidelity” score:

- **D0 — Specified:** contracts and limitations written.
- **D1 — Source checked:** implementation exists and is reviewed.
- **D2 — Deterministic/analytical:** local executable tests or analytical comparisons actually ran.
- **D3 — Laboratory held-out:** performance evaluated on independent samples.
- **D4 — Solver/robot composition:** model and downstream robot behavior checked together.
- **D5 — Exact artifact/reproducible evaluation:** source, parameters, data, solver and environment frozen and reproduced.
- **D6 — Controlled field evidence:** bounded physical trial and independent outcome measurements.

A test definition is not an executed test; a CI queue is not a pass; a validated component does not automatically qualify a composed robot. Every result must name its exact commit, data/model digests and accepted operating envelope.

## 11. Research references

1. **International Society for Rock Mechanics and Rock Engineering (ISRM), Testing Methods Commission.** Suggested rock characterization, testing and monitoring methods, including UCS/deformability, triaxial strength, Brazilian tension, fracture toughness, abrasivity, porosity, water saturation and acoustic emission.  
   https://isrm.net/isrm/page/show/305  
   https://w.isrm.net/page/show/1305

2. **Mechanical specific energy versus depth of cut in rock cutting and drilling.** *International Journal of Rock Mechanics and Mining Sciences* (2017). Connects MSE, cutter wear, depth of cut and rate of penetration; emphasizes accurate removed-volume measurement.  
   https://doi.org/10.1016/j.ijrmms.2017.11.004

3. **A review of computational approaches for simulating fracturing mechanisms in layered rock formations.** *International Journal of Coal Science & Technology* (2025). Reviews continuum, discrete-element and combined finite-discrete methods for fracture and discontinuities.  
   https://doi.org/10.1007/s40789-025-00836-8

4. **Advances in discrete element modeling of rock fracture for next-generation comminution models.** *Computational Particle Mechanics* (2025). Reviews bonded DEM, lattice-element, particle-replacement and level-set approaches.  
   https://doi.org/10.1007/s40571-025-01092-y

5. **A Systematic Review of Data Fusion Techniques for Digital Twin Applications in the AEC Sector: Perspectives for Geotechnical Engineering.** *Applied Sciences* (2026). Discusses sparse/indirect subsurface evidence, heterogeneous data fusion, uncertainty and interoperability limitations.  
   https://doi.org/10.3390/app16105170

6. **Team CERBERUS Wins the DARPA Subterranean Challenge: Technical Overview and Lessons Learned.** (2022). Reports multi-robot mapping/localization, terrain-aware mobility and resilient autonomy in GPS-denied underground environments.  
   https://arxiv.org/abs/2207.04914

7. **NASA/JPL NeBula robotics.** Describes multi-modal sensing, degeneracy-aware fusion, large-scale GPS-denied mapping and traversability for caves, mines and extreme terrain. This is a robotics reference, not a rock constitutive model.  
   https://www.jpl.nasa.gov/capabilities/robotics/nebula/

These sources establish candidate methods and research directions, not the performance of Symthaea's current implementation. Reference-data access, licensing, specimen/site representativeness and fitness for a chosen use case must be reviewed before admitting a benchmark.

## 12. Non-claims

This roadmap does not claim that Symthaea currently predicts real granite/basalt failure with laboratory-validated accuracy, performs FEM/DEM/phase-field rock-fracture analysis, estimates site-specific properties without measurements, proves safety from material labels or model confidence, physically excavates/changes bedrock without suitable equipment, or is qualified for underground human occupancy, autonomous mining, rescue or construction.

The intended deliverable is a path from a useful deterministic subterranean control simulator to a scientifically testable, provenance-aware engineering platform, while keeping cognition, numerical physics and physical authority distinct.
