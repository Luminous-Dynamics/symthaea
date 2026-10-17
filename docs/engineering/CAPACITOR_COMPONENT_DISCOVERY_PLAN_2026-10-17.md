# Capacitor and Electronic-Component Discovery Plan

**Status date:** 2026-10-17  
**Status:** Research and implementation plan; equation/provenance, nominal screening and hard-bound robust screening primitives staged in a draft PR  
**Scope:** Public Symthaea repository; no private monorepo dependencies

## Executive direction

Symthaea should become an evidence-carrying component-discovery and co-design system. It should not claim to replace SPICE, density-functional theory (DFT), finite-element solvers, materials laboratories, or component qualification. Its differentiator should be how it orchestrates those tools: search across hierarchies, use uncertainty to choose the next expensive evaluation, jointly optimize material/process/geometry/circuit, and preserve evidence from hypothesis through measured prototype.

The first vertical is **capacitor dielectric + package + circuit co-design**. It has clear physical equations, widely used passive-component families, meaningful multi-objective trade-offs, and a manageable initial simulation scope. The same interfaces can later support inductors, transformers, sensors, protection components, semiconductor devices, and PCB-level power integrity.

## Existing Symthaea work to compose, not duplicate

This capacitor specialization should consume the repository's current materials/engineering contracts rather than establish competing frameworks:

- [MAT-009 #4340](https://github.com/Luminous-Dynamics/symthaea/issues/4340): reuse the shared materials-discovery campaign and Pareto-objective contract.
- [CAP-DISC-001 #7917](https://github.com/Luminous-Dynamics/symthaea/issues/7917): operationalize the capacitor-specific condition-aware dataset, frozen splits, baseline comparisons, and equal-budget discovery trial.
- [MAT-GEN #5258](https://github.com/Luminous-Dynamics/symthaea/issues/5258): align held-out generalization, domain-shift, and split-integrity evaluation.
- [MAT-PSPP #6224](https://github.com/Luminous-Dynamics/symthaea/issues/6224): use the process → structure → property → performance lineage for composition, microstructure, processing, and device context.
- [MAT-ENG #6185](https://github.com/Luminous-Dynamics/symthaea/issues/6185): connect engineering demand, candidate materials discovery, and qualification evidence.
- [ENG-SPICE-001 #5633](https://github.com/Luminous-Dynamics/symthaea/issues/5633): existing task for real ngspice output parsing, convergence semantics, and exact solver/netlist/output provenance.
- [ENG-SEMI-000 #5671](https://github.com/Luminous-Dynamics/symthaea/issues/5671): preserve the distinction between material properties, device physics/TCAD, compact models, circuit simulations, and measured devices.
- [ENG-DEVICE-000 #5670](https://github.com/Luminous-Dynamics/symthaea/issues/5670) and [architecture PR #5672](https://github.com/Luminous-Dynamics/symthaea/pull/5672): reuse common physical-device packaging, power, thermal, measurement, and safety architecture.

The present increment is intentionally limited to capacitor-specific dielectric-property provenance, transparent first-order equations, and an evaluation plan. Future active-learning orchestration, generic Pareto ranking, simulator execution/evidence closure, and physical qualification should land in their existing owner layers.

## Why this is timely

1. **High-κ ceramic chip capacitors.** The October 2026 *Nature Electronics* paper demonstrated hierarchical ML-guided search: first identify promising structural families, then optimize local atomic features in a tungsten-bronze ceramic. The reported Ba–Nd–Bi–Ti–Ge oxide achieved roughly κ=185, loss tangent ≈0.0005 and temperature coefficient below ±30 ppm/K; a 1-pF prototype was reported at about 85% lower volume than the cited commercial comparator. These are that paper's results, not Symthaea results. Crucially, the authors prioritized κ and temperature coefficient in their first ML stage, then assessed loss tangent in experimental validation because it had high extrinsic noise. Symthaea should likewise choose objectives/fidelities based on how measurable and modelable each property is, rather than force every property into one surrogate. The paper's source data and structural datasets are listed at its data-availability section; inspect source terms and provenance before ingestion. For our own evaluation, add family/source-held-out testing as well as repeatable random-split baselines.  
   Sources: https://www.nature.com/articles/s41928-026-01721-1 and https://www.nature.com/articles/s41928-026-01726-w

2. **High-temperature polymer energy storage.** A 2024 *Nature Communications* study combined AI, polymer chemistry, and molecular engineering; it reported 8.3 J/cm³ at 200 °C for a discovered dielectric and compared it with commercially available polymer dielectrics at that temperature. This is a distinct capacitor regime from MLCC miniaturization, so its numbers must not be compared as if the applications and test protocols were interchangeable.  
   Source: https://www.nature.com/articles/s41467-024-50413-x

3. **Calculated dielectric properties need uncertainty and an explicit applicability domain.** Materials Project dielectric tensors use first-principles density-functional perturbation theory. Its methodology reports a 16.2% mean absolute relative deviation for a specified GGA setup with local-field effects, along with a tendency to overestimate dielectric constants. That is a historical methodology-specific benchmark, not a universal error bar for every material. More importantly, the reported response is for a single crystal and a static/long-wavelength field; a polycrystalline processed ceramic, polymer film, or packaged capacitor at AC frequency and DC bias is a different engineering subject. Use these calculated values for candidate triage, not as direct product-performance labels.  
   Source: https://docs.materialsproject.org/methodology/materials-methodology/dielectricity

4. **Research is moving toward generative, multi-objective dielectric design.** A June 2026 review surveys a transition from passive screening toward AI-driven generative design and highlights the coupled permittivity–breakdown-strength trade-off as a central challenge. Treat the review as a research map, not independent proof that one algorithm or model will outperform baselines.  
   Source: https://doi.org/10.1021/acsami.6c05353

5. **Real solver output must be interpreted.** ngspice supports batch analyses and raw-file output. Symthaea's current ngspice adapter can invoke the executable but explicitly does not yet parse real results into validated metrics; the next implementation step must complete that path rather than fill in fictional values.  
   Sources: https://ngspice.sourceforge.io/docs/ngspice-44-manual.pdf and https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/bridges/symthaea-ngspice-bridge/src/lib.rs

## Paradigm shifts

### 1. From maximize-one-property to application-constrained Pareto design

No single metric defines a better capacitor. Search against application-specific constraints and report the Pareto frontier across:

- Capacitance and capacitance per volume/mass.
- Rated and transient voltage, peak electric field, and breakdown margin.
- ESR/ESL and circuit-level ripple/overshoot/efficiency.
- Dielectric loss and self-heating at the actual frequency and waveform.
- Capacitance drift with temperature, frequency, DC bias, and aging.
- Lifetime/reliability evidence, cost, manufacturing yield, supply risk, and environmental impact.

Hard electrical and safety constraints are not soft objectives that can be traded away for a prettier score. Keep a candidate's hard-constraint feasibility separate from its ranking.

For a first-pass *linear, field-independent* dielectric model, a useful candidate-screening figure of merit is:

```
u_ideal,bd = 0.5 * epsilon_0 * epsilon_r * E_bd^2
```

where `E_bd` is the reported breakdown field. This exposes why maximizing `epsilon_r` alone is a bad search objective: breakdown strength enters quadratically. This equation is only an ideal theoretical bound and must not be reported as usable/rated energy density or safety margin. It omits defects, geometry, reliability statistics, leakage, thermal behavior, aging, and process variation. For nonlinear/ferroelectric storage materials, use measured charge/discharge `D(E)` or `P(E)` curves to calculate recoverable energy and efficiency rather than treating constant-permittivity equations as authoritative.

### 2. From material formula to process–structure–property–device lineage

A chemical formula alone does not define a usable dielectric. Preserve phase, crystal structure, dopants, grain/core-shell structure where relevant, defects, porosity, electrode interfaces, processing/sintering conditions, geometry, and test conditions. The lineage should be queryable end-to-end:

```text
application requirements
  -> capacitor family / package
  -> dielectric family
  -> composition + structure
  -> processing recipe
  -> measured material properties
  -> geometry + electrodes
  -> circuit context
  -> simulated candidate
  -> as-built identity + calibrated tests
```

Unknown links remain unknown; an inferred structure or property must not be silently represented as measured data.

### 3. From static prediction to multi-fidelity active learning

Use cheap filters first, and let quantified uncertainty determine which evaluation should run next:

1. Dimension/unit/schema and composition checks.
2. Known-property and literature retrieval.
3. Empirical baselines and uncertainty-aware surrogate ranking.
4. Reduced-order physical calculations.
5. External high-fidelity calculations where justified (for example, DFT/DFPT for appropriate crystalline candidates).
6. Circuit simulation with actual component models.
7. Lab measurements selected for expected information gain.

The acquisition objective should value both candidate quality and uncertainty reduction. A candidate that may change the design decision is more valuable than one with a slightly better mean prediction but no uncertainty estimate.

### 4. From disconnected simulations to closed-loop discrepancy learning

Bind each prediction to an exact solver version, inputs, model assumptions, convergence result, and artifact digest. Bind each observation to the exact sample/device, instrument, calibration, environment, waveform, and protocol. Compare prediction with observation, diagnose model discrepancy, and update the model only within a declared validity envelope.

### 5. From component selection to component–circuit–thermal co-design

A high-capacitance part is not automatically the best part in an inverter or converter. Evaluate capacitor candidates in representative circuits with actual ESR/ESL, ripple current, parasitics, thermal boundary conditions, and control constraints. Co-optimize the complete system when cross-coupling matters.

### 6. From opaque ranking to evidence-carrying decisions

Every property value and candidate recommendation should say whether it is experimental, manufacturer-supplied, first-principles-computed, model-predicted, estimated, or unknown. Attach provenance, conditions, uncertainty, and independent reproduction status. Formal proofs can check arithmetic, constraints, schema invariants, and some algorithmic behavior; they do not prove that a material is physically stable or that a component will survive qualification.

### 7. From benchmark theater to pre-registered discovery trials

Lock test splits, seed sets, evaluation budgets, baseline implementations, and metrics before running the challenger. Split by material family/composition and publication/source where possible; random row splits can leak near-duplicate compositions or repeated measurements. Report failed predictions, out-of-domain candidates, invalid simulations, and negative controls instead of removing them silently.

## Initial software increment

The initial increment adds `dielectric.rs`, `capacitor_screening.rs`, and `robust_capacitor.rs`, exported from `symthaea-materials`. It includes:

- Explicit evidence classes and source references; uncertainty is attached to each property with a reference describing how that uncertainty was obtained, and semantics distinguishing a hard bound, statistical estimate, or confidence interval.
- Permittivity/loss test temperature, frequency and optional characterization field; separate evidence metadata for breakdown strength.
- Multilayer parallel-plate geometry with input validation.
- First-order capacitance, peak stored energy, active volume, ideal linear energy density, sinusoidal dielectric-loss, ideal breakdown-limited energy-density figure of merit, and optional breakdown-field ratio.
- Application-conditioned screening constraints and distinct outcomes for constraint violation versus missing/out-of-domain evidence.
- Fail-closed checks for characterization temperature, frequency and field; evidence-class allow-lists; per-property uncertainty limits; and breakdown evidence when a breakdown ratio is required.
- An additional hard-bound path that calculates lower/upper ideal metric intervals and evaluates constraints against the conservative endpoints. Statistical uncertainty and confidence intervals are explicitly refused as deterministic hard bounds.
- Reproducible reports retain the exact inputs to each nominal and robust decision.
- Deterministic tests for equations, condition mismatch, invalid inputs, missing evidence, uncertainty semantics, nominal-vs-robust disagreement and hard-constraint failures.

It intentionally does **not** include new real-material constants, a trained property predictor, temperature/frequency extrapolation, probability-of-failure calculations, correlated uncertainty propagation, geometry/process variation, a component qualification/safety gate, manufacturing guarantees, or a validated capacitor design. Sample records are synthetic test fixtures explicitly labelled as such. Compilation/test execution remains an outstanding verification gate.

The general physical-device integration should continue to reuse the broader engineering architecture in PR #5672 rather than add a parallel engineering framework.

## Benchmark suite and target gates

All thresholds below are **proposed acceptance targets**, not current results. Publish actual values, errors, dataset manifests, and evaluation commands before making performance claims.

### B0 — Equation and schema correctness

**Purpose:** Make physical arithmetic and evidence boundaries testable.

- Compare capacitor calculations against independently implemented analytic reference equations and dimensional checks.
- Exercise zero, negative, NaN, infinity, missing provenance, missing breakdown protocol/evidence class, missing property-specific uncertainty, out-of-domain temperature/frequency/field, and overflow paths.
- Require invalid inputs to fail closed; no invalid/non-finite derived metric may be emitted.
- Distinguish `PassesNominalIdealizedScreen`, `FailsDeclaredConstraints`, and `NotEvaluable`; a candidate outside the characterization domain must never be ranked as if its properties were known.
- The nominal screen gates on property-specific uncertainty magnitudes but does not propagate them.
- The separate robust screen propagates uncertainty only when it is explicitly declared as a hard relative bound; statistical estimates/confidence intervals are not accepted as deterministic ranges. Its "pass" means only the chosen ideal equations satisfy constraints throughout those declared property bounds. It is not a physical reliability or manufacturing-yield guarantee.
- The robust interval path currently assumes exact geometry and operating inputs, independent property bounds, linear field-independent permittivity, and the stated sinusoidal loss model. Geometry/process uncertainty, model discrepancy and uncertainty correlations are not included.
- Target: relative error no greater than 1e-12 on finite well-conditioned analytic fixtures; all negative controls rejected.
- Test fixtures must be synthetic and must never be counted as empirical material evidence.

### B1 — Circuit-simulation integrity

**Purpose:** Prove that Symthaea's metrics come from a real solver run.

- Complete the ngspice adapter's input staging, isolated working directory, return-code handling, stdout/stderr capture, raw-file or deterministic table parsing, and result provenance.
- Start with analytically solvable RC, RL, and RLC networks; then add capacitor ESR/ESL and frequency-dependent loss fixtures.
- Exercise malformed outputs, missing raw files, nonzero exits, convergence failures, wrong units, missing vectors, and stale output artifacts.
- Proposed target: 100% correct parse/rejection across versioned golden fixtures; key transient/AC metrics within 1% of analytical references for appropriately conditioned circuits.
- A successful process exit is not by itself proof of convergence or model correctness.

### B2 — Materials-property prediction

Evaluate each property separately: relative permittivity, loss tangent, breakdown strength, temperature coefficient, and (only where sufficiently supported) aging/lifetime.

- Baselines: constant/mean predictor, a transparent descriptor-based model, a strong conventional tabular model, and published/computational property references where the same task and units are comparable.
- Use group-held-out splits by composition/material family and source/publication, with a completely untouched final test partition.
- Report MAE, RMSE, median absolute relative error, calibration/coverage of prediction intervals, error by material family and condition, and out-of-domain detection.
- Proposed initial stretch target for in-domain permittivity screening: median absolute relative error <= 20%, with an approximately 90% prediction interval achieving 85–95% empirical coverage. Treat these as a go/no-go hypothesis to validate against dataset size and noise, not guaranteed achievable metrics.
- Do not combine different temperature/frequency/bias protocols as though their values were directly interchangeable. Do not use high permittivity as a proxy for low loss or high breakdown strength.

### B3 — Discovery efficiency

Compare the proposed Symthaea-guided search against random feasible search, a space-filling design, random-forest and Gaussian-process Bayesian-optimization variants where appropriate, and a conventional multi-objective optimizer under equal candidate-evaluation budgets.

A useful external benchmark across five experimental materials-science domains found that Bayesian optimization with random-forest surrogates was comparable to GP with automatic relevance detection, and both outperformed a GP baseline without ARD on those datasets. That result is not specific to dielectrics, but it demonstrates why Symthaea must compete against tuned conventional baselines rather than a deliberately weak strawman:
https://www.nature.com/articles/s41524-021-00656-9

- Pre-register at least 10 random seeds when evaluation is affordable.
- Report feasible-candidate yield, time/cost per useful candidate, best feasible objective, Pareto hypervolume, uncertainty calibration, and number of expensive calculations avoided.
- Stretch goal: at least 10% improvement in normalized Pareto hypervolume or at least 20% fewer expensive evaluations to reach the same reference frontier, without regression in hard constraints. Require confidence intervals and a held-out search space.
- Run ablations: conventional optimizer alone; HDC retrieval alone; uncertainty acquisition alone; Symthaea-assisted combined workflow. If the combination does not beat simple baselines, keep the simpler method.

### B4 — Robustness and reliability

- First verify interval algebra with synthetic hard-bound fixtures: nominal metrics lie inside each calculated interval; upper-loss and lower-breakdown cases are exercised; statistically described uncertainty is rejected from deterministic interval claims.
- Next perturb geometry, manufacturing dimensions, material parameters, ESR/ESL, temperature, ripple waveform, and aging assumptions using separately supported models/data.
- Report deterministic hard-bound feasibility separately from statistical constraint-violation probability. Only estimate probabilities when distributions/correlations are justified and calibration can be assessed.
- No lifetime claim from a generic CfC state similarity. Lifetime prediction needs appropriate labelled degradation data, model calibration, and independent validation.
- Proposed gate: every required conservative endpoint satisfies the declared constraints, or the candidate is marked as failing/not evaluable. Nominal score must not hide a conservative-bound violation.

### B5 — Physical prototype / external validation

- First reproduce a known reference capacitor under an explicit measurement protocol.
- Compare candidates and reference devices under the same fixture, temperature, frequency/waveform, DC bias, and measurement uncertainty.
- Replicate across multiple samples/batches before asserting a robust improvement; publish raw data or a lawful, reproducible data manifest.
- Initial demonstration target: at least one independently measured candidate achieves a pre-registered improvement (for example, >=10% on one targeted objective) while meeting all hard constraints and showing no material degradation in the other required metrics.
- Simulation-only results never satisfy this gate. Component qualification and standards compliance remain separate engineering processes.

## Work order — do not wait for CI to discover the next tasks

### P0: Complete the trust boundary

1. Run formatter and focused `symthaea-materials` tests for the newly added dielectric module.
2. Make B0 a stable deterministic test target and add dimensional/unit review.
3. Finish the ngspice result parser against the actual chosen raw-file/table format; reject all unparsed or non-converged results.
4. Save run manifests containing source revision, input hashes, solver binary/version, command, environment contract, exit status, output hashes, metric units, and evidence classification.

### P1: Build a small, clean dataset

1. Define a versioned dielectric-property schema with condition-specific values and per-property provenance.
2. Start with a small, legally reusable dataset of known reference materials and explicit citations/licences; avoid unbounded API scraping.
3. Record experimental and computed values as separate evidence classes.
4. Establish train/validation/test groups and simple baselines before training any new model.

### P2: First component-level benchmark

1. Specify two separate tracks: (A) MLCC miniaturization/temperature stability and (B) high-temperature polymer energy storage.
2. Keep objective definitions, material families, constraints, and references separate between these tracks.
3. Optimize existing catalog parts in an RC/RLC or converter test circuit before pursuing novel compositions.
4. Compare the optimizer and candidate selection methods at a fixed evaluation budget.

### P3: Materials discovery

1. Integrate appropriate dielectric tensor/property sources through documented and rate-limited interfaces.
2. Add uncertainty-aware screening and a queue for DFT/DFPT tasks, retaining workflow status and convergence details.
3. Add process/structure metadata and distinguish generated hypotheses from realizable, synthesizable structures.
4. Select experimental measurements by expected information gain and feasibility, not novelty score alone.

### P4: System co-design

Extend to inductors/transformers, switching devices, thermal paths, sensing/protection, PCB parasitics, and manufacturability once the first component family produces reproducible gains.

## Definition of done for any claimed discovery

A candidate can be called **promising** when a reproducible, scoped model or solver result supports it. It can be called **simulation-validated** only when the relevant independent numerical and model checks pass. It can be called **experimentally demonstrated** only after measured data from a documented physical sample support the stated claim. It can be called **qualified** only after the applicable reliability, safety, manufacturing, and standards processes have been completed.

These labels are not interchangeable.
