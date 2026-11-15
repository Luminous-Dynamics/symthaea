# Marine Generative Engineering Benchmark Protocol (M-GEB-001)

Status: research protocol / targets, not a report of achieved performance.

This protocol links MARINE-GEOMETRY-MANIFOLD-001 (#12302), validated multifidelity evaluation (#11513), propeller-plane wake fidelity (#11729), and solver verification and validation (#11442). It defines the comparison needed to evaluate a generative hull system against current research and engineering practice.

## 1. Research hypothesis

The hypothesis is not that a learned model can generate visually plausible hulls. The hypothesis is that a geometry-manifold workflow can reach a better **independently validated, manufacturable, mission-useful design frontier per unit total engineering cost**, while reducing false-feasible decisions and preserving traceable evidence.

The system boundary is:

1. Candidate generation and geometry identity (Symthaea).
2. Geometry and hydrostatic checks, with explicit scope (Symthaea / independently qualified backends).
3. Wake, propulsion, seakeeping and mission evaluation using qualified solvers (Symthaea orchestration; solver-owned numerical physics).
4. Process-specific DFM, block/assembly, handling, access and repair constraints (Symtropy #2048).
5. Metocean/site/test context (Sol-Atlas #809).
6. Provenance, authority and immutable cross-stack references (Mycelix #5968).
7. Independent experimental and operational validation (#11442 and subsequent test campaigns).

A generator score, surrogate prediction, mesh-admission result, or CFD result alone is not a validated engineering improvement.

## 2. Current external comparators

The frozen literature baseline includes:

- Anagnostopoulos et al. (2026), *Ship Hull Generation Using GANs & VAE-PCTs: A Comparison Study*, JMSE 14(18), 1707, DOI: https://doi.org/10.3390/jmse14181707. This work compares a convolutional GAN with a VAE plus Point Cloud Transformer, performs a discrete-to-continuous CAD pull-back to B-spline lofted surfaces, and evaluates generated hull quality/diversity with Noise Score (NS), Diversity Score (DS), Jensen–Shannon Divergence (JSD), Minimum Matching Distance (MMD), and Coverage (COV). It reports distinct quality/coverage trade-offs between the architectures and notes limitations in geometric coefficients. The paper's data and task-specific metrics are a comparator, not a universal SOTA leaderboard.
- Zheng, Wang and Chen (2026), *The generative marine hull form optimization based on offline data-driven evolutionary algorithm*, Ocean Engineering 358, 125935, DOI: https://doi.org/10.1016/j.oceaneng.2026.125935. This combines a learned latent hull manifold with an expensive data-driven evolutionary optimizer under a limited CFD budget, explicitly addressing parent-hull dependence and surrogate bias.
- Romero-Tello et al. (2026), *AI-assisted seakeeping operability screening for early-stage hull-form optimisation*, Ocean Engineering 366, 127663, DOI: https://doi.org/10.1016/j.oceaneng.2026.127663. The paper reports an independent BEM-verified improvement of approximately 11 percentage points in its study-specific annual seakeeping operability metric. Treat this as a methodological comparator; do not transfer its numeric gain to unrelated vessels, routes or operating envelopes.
- Gerhardt et al. (2026), *ITTC benchmarking campaign on the added resistance of KVLCC2 in regular and irregular waves*, Ocean Engineering, DOI: https://doi.org/10.1016/j.oceaneng.2026.127723. Ten facilities participated. Use this as an inter-facility reproducibility and uncertainty benchmark for added resistance and motion in waves, not as a propeller-plane wake dataset.
- ITTC 7.5-03-01-01 Rev.05 (2024), *Uncertainty Analysis in CFD Verification and Validation, Methodology and Procedures*: https://www.ittc.info/media/11950/75-03-01-01.pdf.

Commercial engineering systems are capability baselines, not claims that Symthaea must reproduce every proprietary feature. Simcenter STAR-CCM+ documents transient VOF free-surface simulation and up-to-six-degree-of-freedom dynamic fluid-body interaction in its hull-performance workflow. SHIPFLOW documents potential-flow, RANS, hull/propeller interaction and seakeeping/manoeuvring capabilities. OpenFOAM supplies an independently configured open solver path. We should initially interoperate with qualified solvers rather than claim an unvalidated replacement.


## 2A. Additional subsystem and fleet-reconstruction comparators (15 Nov 2026)

- Liu et al. (2026), *An integrated data-driven and experimental methodology for toroidal propeller design improvement*, Engineering Applications of Artificial Intelligence 184, 116126, DOI: https://doi.org/10.1016/j.engappai.2026.116126. Published 15 Nov 2026. It describes an experiment-simulation-optimization loop, with high-fidelity CFD checked against open-water experiments, PCHIP parametric geometry and adaptive radial-basis/evolutionary sampling-assisted optimization across 14 variables. It reports a study-specific 7.26% power reduction at matched thrust and 1.74 dB noise reduction against its best sampled design. Add this as a propulsor subsystem benchmark, not a hull generator arm. It is particularly relevant to the wake/propulsor co-design path.
- *Raster2Hull: Learning 3D Ship Hull Geometry From Raster 2D General Arrangement and Lines Drawings*, IEEE Access 14 (31 Jul 2026), DOI: https://doi.org/10.1109/ACCESS.2026.3717961. Add this as a separate legacy-vessel geometry-reconstruction comparator. It addresses 2D drawing-to-3D hull reconstruction, not forward generative hull design or hydrodynamic optimization. Evaluate reconstruction error, lines-plan consistency, section recovery, topology, hydrostatic properties and uncertainty on held-out drawing sources.

These extend the benchmark matrix without conflating generation, reconstruction and propulsor optimization into one score.

## 3. Matched comparison arms

Every arm must use the same rights-cleared input hull families, displacement/payload and loading constraints, speed and mission envelope, environmental distribution, solver release, mesh policy, high-fidelity CFD budget, and independent holdout.

| Arm | Method | Why it is included |
|---|---|---|
| A0 | Existing reference hull / naval architect baseline | Practical engineering anchor |
| A1 | Conventional section/parameter-based hull optimization | Strong conventional design-space baseline |
| A2 | GAN CAD-to-CAD generation + B-spline recovery | Published 2026 generative comparator |
| A3 | VAE-PCT CAD-to-CAD generation + B-spline recovery | Published 2026 generative comparator |
| A4 | Latent-space generator + expensive evolutionary optimization | Published 2026 performance-driven comparator |
| A5 | AI-assisted seakeeping / operability screening | Published 2026 mission-condition comparator |
| A6 | Symthaea geometry manifold + wake/mission-aware multi-fidelity selection | Proposed system under evaluation |

If published implementations or weights/data are unavailable or not licensed for reuse, reproduce the method from the paper only where feasible and label the arm as a faithful reproduction, partial reproduction or paper-reported reference. Never imply direct implementation equivalence without evidence.

### Controls and ablations

- Same hull, different propulsor; same propulsor, different hull/wake.
- Uniform-wake diagnostic control versus propulsor-conditioned target wake; do not assume uniformity is the objective.
- Parent-hull deformation versus multi-family latent generation.
- Surrogate-only ranking versus uncertainty-aware multi-fidelity selection.
- Calm-water-only ranking versus route/sea-state/operability ranking.
- Geometry-only feasibility versus geometry plus process-specific DFM.
- Blind candidate scoring before revealing high-fidelity and experimental reference results.

## 4. Data and leakage controls

1. Freeze each input dataset by source URL, edition, rights statement, raw digest, conversion pipeline and immutable artifact ID.
2. Split by hull lineage/family and vessel type, not random points from the same hull. Reserve at least one complete geometry family for blind testing.
3. Keep environmental/wave-regime holdouts separate from training and hyperparameter selection.
4. Freeze solver executable/container digest, numerical settings, mesh-generation settings, turbulence/free-surface models, timestep and convergence criteria.
5. Record all failed runs, retries, mesh-generation failures, failed candidates, rejected geometries, manual intervention and reviewer time in total cost.
6. Keep a frozen benchmark manifest and evaluation script. Any change creates a new benchmark version; do not silently change the test set after seeing outcomes.

## 5. Metric groups

### G: geometry fidelity and useful design-space coverage

Report NS, DS, JSD, MMD and COV using the published study's definitions and normalization where its raw-data format permits a faithful implementation. Add 1-Nearest-Neighbor Accuracy (1-NNA), geometric descriptor coverage and distribution calibration as supplementary metrics. Do not compare raw MMD magnitudes across differently normalized datasets.

For recovered continuous CAD, measure:
- B-spline/NURBS recovery error against the generator's discrete representation;
- surface continuity and fairness (including declared C0/C1/C2 requirements);
- principal curvature and sectional-area/volume distributions;
- displacement, beam, draft, LCB, LCF, KB/VCB, waterplane and other declared coefficients;
- valid-solid, valid-surface and repair rates;
- mode collapse, duplicate rate, novelty and out-of-distribution distance.

All metric implementations must be versioned and validated on known-answer synthetic fixtures.

### P: physics and propulsion

Report resistance and wave-elevation comparisons, sinkage/trim, motions/RAOs, added resistance, propeller open-water KT/KQ/efficiency, self-propulsion, propeller-plane axial/radial/tangential velocity fields, wake fraction/thrust deduction (with explicit definitions), thrust/torque, rudder interaction and cavitation-relevant observables.

Wake targets are **propulsor- and mission-conditioned flow fields**, not a blanket preference for uniform wake. Include full-vector coverage, coordinate frame, phase origin, radial/hub treatment, grid identity and transformation receipts. Axial-only data must be labelled axial-only.

### R: reconstruction of legacy geometry

Keep this track separate from forward generation. For Raster2Hull-like methods, report drawing-to-geometry alignment, section/offset error, deck/keel/centreline consistency, missing-view uncertainty, CAD reconstruction validity, hydrostatic coefficient error, manual repair time and provenance of the source drawing. Split by original vessel/drawing lineage, not patches or derived views. An uncertain or incomplete drawing must produce an uncertainty envelope or abstention rather than an invented surface.

### P2: propulsion architecture subsystem

In addition to hull/propulsor interaction, compare conventional open propellers, ducted/toroidal propellers and waterjets only within matched thrust, speed, diameter/installation and mission constraints. Report KT/KQ, delivered and electrical power, efficiency, pressure distribution, vortex metrics, cavitation indicators, underwater radiated noise and maintenance/ingestion risks. For the toroidal-propeller comparator, reproduce the experiment-to-CFD validation loop before comparing optimizer performance. Study-reported 7.26% power and 1.74 dB noise changes are contextual reference points, not target pass thresholds.

### O: operational usefulness

Primary system outcome: energy per equivalent useful service, for example delivered tonne-nautical-mile or mission energy per completed harbour-assist task. Freeze the propulsion/electrical boundary, payload, service level, route, speeds, weather and reserve policy. Include annual operability and service availability across specified seasonal/sea-state distributions.

### P3: manoeuvring and low-GM vessel dynamics

Use the ITTC/SIMMAN benchmark cases for KVLCC1/KVLCC2, KCS and DTMB 5415 where raw data and reuse rights permit. Include free-running turning circles and zig-zag manoeuvres plus captive PMM/CMT test cases in deep and shallow water. Report trajectories, forces/moments, overshoot angles, tactical diameter, advance, transfer and speed loss with experimental and numerical uncertainty.

Do not treat 3-DOF planar manoeuvring as a universal reference. The ITTC benchmark procedure reports that 3-DOF methods were inadequate for KCS manoeuvring predictions and notes that heel/roll inclusion is important for low-GM vessels; select at least a 4-DOF model where those couplings matter. The added DOF must be justified by the vessel/loading regime and independently checked rather than added as decoration.

### P4: seakeeping screening calibration

For AI screening, report both predicted screening operability and independently reevaluated operability on the selected candidate and parent baseline. The 2026 AI-assisted study evaluated multiple onboard locations, operational areas, seasons, speeds and relative wave headings; its screen's best candidate was reported at 99.05%, while independent BEM verification reported 95.55% versus 84.90% for the parent hull evaluated with the same solver. This is a strong example of why the screen-to-validation gap, rank stability, calibration and final decision regret must be measured separately. Treat all percentages as study-specific.

### M: manufacturing, maintenance and environment

Report process-specific DFM dispositions; plate/section developability; forming and welding constraints; block split; assembly sequence; lifting, transport and yard envelope; access to inspection, repair and cleaning; corrosion/coating; fatigue; maintainability and lifecycle materials/energy. Track radiated noise, cavitation noise/erosion, biofouling, emissions, marine pollution and lifecycle greenhouse gas where the data are available. Missing required evidence is UNKNOWN, not PASS.

### D: decision and research economics

Report:
- false-feasible rate against independent hard checks;
- Pareto rank inversion and selection regret on blind holdouts;
- decision stability under uncertainty and repeated random seeds;
- calibration and out-of-distribution abstention;
- number of expensive solver calls per independently validated improvement;
- total cost to validated frontier, including compute, remeshing, failed runs, storage, expert time and physical tests;
- wall-clock time only as a secondary metric.

## 6. Validation ladder

| Level | Qualification | Representative references |
|---|---|---|
| V0 | Analytical and dimensional correctness | Known-answer force, resistance, effective/shaft/electrical power, efficiency, coefficients and unit tests |
| V1 | Geometry and representation | Topology, CAD recovery, sections, fairness, hydrostatic closure and adversarial negative controls |
| V2 | Calm-water hull hydrodynamics | KCS, KVLCC2, Series 60, DTMB 5415 where data and reuse rights permit |
| V3 | Propulsion and wake | Open-water propeller, self-propulsion and 3D propeller-plane measurements |
| V4 | Seakeeping | Regular/irregular-wave RAOs, added resistance, motions and the KVLCC2 ten-facility campaign |
| V5 | Integrated mission, manoeuvring and operability | Hull + appendages + propulsor + motor/drive + energy + route + environment + manoeuvring/heel dynamics |
| V6 | Manufacturing and service | Independent Symtropy DFM, assembly, transport, inspection and repair disposition |
| V7 | Physical experiments | Towing tank, PIV, load cells, cavitation tunnel, strain and instrumented model |
| V8 | Operational validation | Mission-matched full-scale trials, deterioration, maintenance and lifecycle outcomes |

V2–V8 are not implied by passing a geometry-mesh profile. Each observable must report iterative, temporal and grid uncertainty, validation uncertainty, experimental repeatability, inter-facility scatter and model discrepancy separately where applicable. Judge agreement against combined validation uncertainty, not one universal percent tolerance.

## 7. Pre-registered research targets

These are research goals, not achieved Symthaea results:

- 100% pass rate on analytical known-answer tests.
- Zero false accepts on a frozen set of mandatory geometry and safety negative controls.
- Zero silent promotion of unsupported or out-of-domain checks; explicit UNKNOWN / ABSTAIN.
- Observable-level agreement with independent reference measurements within combined validation uncertainty, with all uncertainty terms reported.
- At least 10% mission-energy reduction (15% stretch) against the strongest matched baseline, with equivalent useful service and no degradation of hard safety, DFM, operability, acoustic or environmental constraints.
- At least 10% validated Pareto-hypervolume improvement against the strongest matched baseline, using a frozen objective set and normalization.
- At least 20% reduction in total cost to reach the independently validated frontier.
- Replicated improvement with confidence intervals and blind holdout families; no selection based solely on surrogate scores.

Do not declare a numeric goal met by comparing separate papers with different vessels, loadings, routes, solvers, boundaries or test conditions.

## 8. Decision rule

A candidate may advance only when:
- identity, input rights and provenance are known;
- all required geometry checks are supported and pass;
- hydrostatic, structural, solver-domain and process-specific DFM evidence is adequate for its stage;
- high-fidelity evaluation is within the validated solver domain or explicitly abstains;
- uncertainty is acceptable for the decision;
- the candidate improves the frozen mission objective without violating hard constraints;
- the evidence receipt points to exact artifacts, solver configuration, mesh, environment and report digests.

A geometry pass is not design approval. A CFD pass is not a physical trial. A physical model test is not full-scale operational validation.

## 9. Implementation order

- G0/G1: candidate schema, deterministic geometry identity, topology and scoped admission (implemented in the companion Rust crate).
- G2: reproducible section-based parametric generator with bounded design envelopes and known-answer geometries.
- G3: neutral CAD/B-rep adapter and independently qualified self-intersection, fairing and hydrostatic backends.
- G4: exact candidate-to-wake/propulsor hand-off to #11729, including 3D fields and transformations.
- G5: multi-fidelity value-of-information policy in #11513, measured by decision improvement and total validation cost.
- G6: Symtropy DFM #2048 and mission/operability co-design.
- G7: blinded benchmark replication and independent physical test selection.
- G8: Mycelix/Sol-Atlas evidence digital thread and test context.

The current Rust G0/G1 crate is only a triangle-mesh integrity adapter. It does not yet implement G2 section generation or a CAD kernel. The first naval hull candidate must come after those geometry contracts and appropriate hydrostatic/DFM checks are ready.
