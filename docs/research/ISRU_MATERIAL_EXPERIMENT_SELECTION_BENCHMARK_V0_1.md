# ISRU Material-Response Experiment Selection Benchmark v0.1

**Status:** synthetic software benchmark protocol and reference implementation  
**Owner:** Symthaea experiment-selection policy; composes Symtropy observation/evidence contracts  
**Issue:** [ISRU-MATERIAL-DOE-001 #12476](https://github.com/Luminous-Dynamics/symthaea/issues/12476)  
**Source data layer:** [Symtropy #5577](https://github.com/Luminous-Dynamics/symtropy/issues/5577), [draft PR #5578](https://github.com/Luminous-Dynamics/symtropy/pull/5578)

## 1. Why this benchmark exists

Optical/thermal ISRU must identify a material's process regime, not merely maximize nominal irradiance. Relevant transitions include brittle spallation, sintering/melting, volatile liberation, optical fouling, particle ejection and capture failure. The best next experiment is the one that most reduces uncertainty relevant to a decision, subject to rig capability, measurement quality, cost and safety.

This v0.1 benchmark isolates the narrow question: can Symthaea's currently exported hypothesis-disagreement selector choose a small set of synthetic experiments that distinguishes a declared material-response family and abstains on an out-of-domain family?

The implementation reuses:

- `epistemic_value`;
- `select_most_informative_experiment`;

from `symthaea_core::hdc::conjecture_engine`.

**Important algorithm boundary:** the current primitive computes population variance of candidate-hypothesis predictions at an experiment. It is a query-by-committee disagreement heuristic. It is **not** Bayesian expected information gain, mutual information, posterior entropy reduction, Bayesian optimization, noise-aware acquisition, or cost-normalized information gain. This benchmark says so explicitly rather than assigning it capabilities it does not have.

## 2. Synthetic test design

The runnable example is `crates/core/symthaea-core/examples/isru_material_experiment_selection.rs`, registered in `crates/core/symthaea-core/Cargo.toml`. Its expected deterministic output is frozen in `crates/core/symthaea-core/examples/fixtures/isru-material-experiment-selection-reference-v0.1.csv` and asserted by an authored Rust test.

### Candidate design space

- 7 nominal irradiance coordinates: 100, 200, 300, 400, 500, 600, 700 W/cm².
- 3 dimensionless morphology indices: 0.1, 0.5, 0.9.
- 21 possible candidates; one predeclared ineligible negative control is excluded before ranking.
- 20 eligible candidates remain.
- Every eligible experiment has equal synthetic cost (1 experiment-hour) and common nominal measurement noise (sigma 0.01).
- The maximum campaign budget is six experiments.
- A decision requires at least two separate observations. In-domain identification additionally requires the best model's mean-squared residual ≤0.0016 and a runner-up error gap ≥0.0016. Otherwise the result remains unresolved or is classified out-of-domain when every declared model exceeds the error threshold.

These are dimensionless synthetic response surfaces expressed against an irradiance-like coordinate. The units in the candidate axis are labels for the coordinate grid only. These surfaces are not calibrated to Nectar, CI/CM material, CARS-1/SARS-LL, lunar regolith or any real asteroid. They are not operating setpoints, predicted excavation rates, safety limits or physical evidence.

### Four hidden evaluation worlds

1. Low-flux peak family.
2. High-flux peak family.
3. Morphology-shifted peak family.
4. Held-out out-of-domain response family that is not one of the three hypotheses.

The hidden-world response is not passed to the selector. A candidate is selected first; only then does the evaluator generate a synthetic observation. The selector gets only the remaining candidate pool and the current top-two hypothesis set.

## 3. Policies compared

All policies share the same 20 eligible candidates, per-experiment cost, measurement-noise descriptor and six-experiment maximum.

| Policy | Design |
|---|---|
| Fixed canonical | Precommitted 3 irradiance × 2 morphology grid; six runs |
| One-factor-at-a-time | Precommitted irradiance sweep at central morphology; six runs |
| Deterministic space spread | Six predeclared points spread over both axes; a simple heuristic, not formal Latin-hypercube or Sobol sampling |
| Seeded random | One deterministic random permutation, fixed seed; first six runs |
| Symthaea disagreement | Select maximum population prediction variance; retain the best two hypotheses after each observation and select a confirmation experiment that separates them; stop only after the two-observation evidence rule is met |

The three fixed schedules and the seeded-random schedule execute their complete precommitted six-run design. They are not allowed to stop early merely because one model happens to fit the first point. The Symthaea policy is sequential and may stop after two qualifying observations. This distinction prevents an unfair “early stop” advantage from being handed to a non-adaptive baseline.

## 4. Deterministic reference results

A separate implementation of the synthetic equations and selection/stopping rules was used to calculate the following expected output rows. These values are a source-level reference for the executable, **not an executed Rust result**. The Rust example and its authored tests still require `cargo check --example isru_material_experiment_selection` and `cargo test -p symthaea-core --example isru_material_experiment_selection` on the exact head.

| Hidden world | Policy | Decision | Correct | Runs | Candidate IDs |
|---|---|---|---:|---:|---|
| Low-flux | Fixed canonical | Low-flux | Yes | 6 | 3;5;9;11;15;17 |
| Low-flux | OFAT | Low-flux | Yes | 6 | 7;4;10;13;16;19 |
| Low-flux | Space spread | Low-flux | Yes | 6 | 0;5;7;9;14;17 |
| Low-flux | Seeded random | Low-flux | Yes | 6 | 3;18;13;10;5;0 |
| Low-flux | Symthaea disagreement | Low-flux | Yes | 2 | 12;15 |
| High-flux | Fixed canonical | High-flux | Yes | 6 | 3;5;9;11;15;17 |
| High-flux | OFAT | High-flux | Yes | 6 | 7;4;10;13;16;19 |
| High-flux | Space spread | High-flux | Yes | 6 | 0;5;7;9;14;17 |
| High-flux | Seeded random | High-flux | Yes | 6 | 3;18;13;10;5;0 |
| High-flux | Symthaea disagreement | High-flux | Yes | 2 | 12;15 |
| Morphology-shifted | Fixed canonical | Morphology-shifted | Yes | 6 | 3;5;9;11;15;17 |
| Morphology-shifted | OFAT | Morphology-shifted | Yes | 6 | 7;4;10;13;16;19 |
| Morphology-shifted | Space spread | Morphology-shifted | Yes | 6 | 0;5;7;9;14;17 |
| Morphology-shifted | Seeded random | Morphology-shifted | Yes | 6 | 3;18;13;10;5;0 |
| Morphology-shifted | Symthaea disagreement | Morphology-shifted | Yes | 2 | 12;9 |
| OOD | Fixed canonical | Out-of-domain | Yes | 6 | 3;5;9;11;15;17 |
| OOD | OFAT | Out-of-domain | Yes | 6 | 7;4;10;13;16;19 |
| OOD | Space spread | Out-of-domain | Yes | 6 | 0;5;7;9;14;17 |
| OOD | Seeded random | Out-of-domain | Yes | 6 | 3;18;13;10;5;0 |
| OOD | Symthaea disagreement | Out-of-domain | Yes | 2 | 12;15 |

In this constructed four-world fixture, the disagreement policy reaches the same categorical decisions in two experiments where the precommitted baselines use six. That is a **3× experiment-count reduction inside this synthetic fixture only**. It is not a measured acceleration factor, not a statistically supported performance estimate, and not evidence that Symthaea beats Bayesian optimization or any industrial SDL. The surfaces, hypotheses, noise and thresholds were authored together, so an independent benchmark curator must create sealed worlds before any external superiority claim.

## 5. What v0.1 does not demonstrate

- No real asteroid/lunar simulant experiment was run.
- No literature points are used to train or calibrate the toy response surfaces.
- No Bayesian posterior, likelihood model, confidence calibration or formal uncertainty interval is computed.
- The common `measurement_sigma` is a matched campaign descriptor; the current variance selector does not use it.
- No costs differ in this fixture; the current selector does not optimize information per cost.
- Only one seeded-random schedule is shown, not a Monte Carlo distribution over seeds.
- No physical safety envelope or laboratory equipment authority is implemented.
- No robot manipulation, chamber operation, optical power delivery, capture or product assay is modeled.

## 6. Industry/SOTA comparison and next benchmark ladder

The relevant industry/research baseline is not random sampling alone. It includes factorial/response-surface designs, Latin-hypercube/Sobol space filling, OFAT, Gaussian-process Bayesian optimization, constrained and multiobjective BO, knowledge-gradient/EI variants, physics-informed BO and cost-/fidelity-aware active learning.

The 2025/2026 review *Benchmarking Self-Driving Labs* reports a literature median acceleration factor of about 6 versus random sampling, with enhancement-factor behavior depending strongly on the property space and often peaking around 10–20 experiments per dimension. Those are literature aggregates across disparate materials campaigns, not a universal target and not a directly comparable score for this toy four-world fixture. The review also warns that benchmark design and search-space properties affect comparisons. See https://doi.org/10.1039/D5DD00337G and https://arxiv.org/abs/2508.06642.

The 2026 review of physics-informed Bayesian optimization for self-driving materials laboratories emphasizes practical effects missing from simple demonstrations: noisy and missing experiments, drift, mixed variables, constraints, multiple objectives, variable cost/fidelity, historical transfer, batching/asynchrony and prior physics. See https://arxiv.org/abs/2608.26016.

For autonomous robotics, the 2026 ADePT perspective separates adaptability/learning, dexterity, perception and task complexity, and recommends per-step error, inter-/intra-run variation, failure/recovery outcomes and full protocol/data provenance. It also calls out interoperability (e.g. SiLA 2 / OPC UA), safety and sustainability. See https://pmc.ncbi.nlm.nih.gov/articles/PMC12923569/.

A July 2026 *Nature Reviews Chemistry* review argues that the next self-driving-lab phase depends on scalability, generalizability and provenance-complete experiments spanning preparation, synthesis, characterization and performance assessment. See https://www.nature.com/articles/s41570-026-00847-2.

### Qualification ladder

| Level | Benchmark | Required comparison |
|---|---|---|
| 0 | Deterministic synthetic controls | Identity, OOD, missingness, unit, eligibility and stop-rule negatives |
| 1 | Algorithmic design benchmark | Equal-budget factorial, OFAT, formal LHS/Sobol, random over many seeds, classical EIG, GP-BO, constrained BO |
| 2 | Source reproduction | Nectar A/B and cryogenic-vacuum laser source points; preserve measurement-vs-derived classes and source uncertainty |
| 3 | Held-out material transfer | Entire feedstock batches and morphology families held out before any tuning |
| 4 | Cross-lab replication | Same preregistered protocol and blinded batches on independent rigs/labs |
| 5 | Physical closed-loop validation | Calibrated instruments, failure/recovery, capture accounting, assay and safe human-authorized experiment execution |
| 6 | Accepted-product performance | Accepted kg per source kWh, deployed kg-year, operating hours, availability, maintenance and life-cycle burden |

### Proposed targets (not achievements)

- On a sealed, independently curated algorithm benchmark, show at least 1.5× lower experiment count than the strongest eligible classical baseline to reach the same preregistered decision-quality threshold; stretch target 2×. This comparison must include a robust GP-BO/constrained BO baseline, not just random or OFAT.
- No degradation in false-discovery, OOD abstention, held-out-batch transfer or uncertainty calibration while reducing experiment count.
- At least nominal 95% predictive interval coverage on truly held-out batches when 95% intervals are declared; report sharpness and group-wise coverage as well.
- 100% refusal on frozen impossible/out-of-envelope/stale-calibration/unsupported-sensor safety controls.
- Report acceleration factor (experiments to threshold), enhancement factor (performance after fixed budget), cost-adjusted acceleration, and campaign-level energy/material consumption. Publish per-seed distributions and confidence intervals rather than only the best run.
- At physical qualification, add per-step robotic success/error, recovery time, inter-/intra-run repeatability, provenance completeness and safety-stop effectiveness.

The 1.5×/2× figures are programme targets chosen for a stringent prospective comparison, not published universal standards. Do not claim victory unless the lower confidence bound of the paired, predeclared comparison clears parity and the product/safety constraints remain satisfied.

## 7. The next paradigm shifts

1. **From optimizing a scalar response to identifying a regime map.** Learn spallation/melting/volatile/fouling boundaries and where the model ceases to apply.
2. **From entropy reduction to decision value.** A measurement is valuable when it changes a qualified process decision per total experimental cost, not merely because model predictions disagree.
3. **From a single global model to an evidence-stratified ensemble.** Keep published measurements, author-derived estimates, synthetic controls, new laboratory measurements and model outputs separately typed.
4. **From passive uncertainty to active falsification.** Every experiment names a competing hypothesis, predicted discriminating observation and a negative-control outcome.
5. **From fixed thresholds to calibrated, group-aware uncertainty.** Calibrate by batch/morphology family; abstain on OOD and sensor drift.
6. **From laboratory automation to interoperable industrial service.** Bind instrument commands, calibration, run identity, material custody, telemetry, capture and assay to a common campaign record.
7. **From process yield to accepted product.** Reward only captured, purified, independently assayed product satisfying a frozen contract. Uncaptured ejecta, leaked gas and purification reject remain losses.

## 8. Next executable increment

- Add true cost/noise/fidelity-aware Bayesian EIG and GP-BO baselines, using held-out worlds inaccessible to the selectors.
- Add many seeded randomized trials, paired uncertainty analysis and bootstrap confidence intervals.
- Replace the heuristic space-spread sequence with actual Latin-hypercube and Sobol designs.
- Add missing/censored observations, failed runs, sensor drift, batch shift, calibration revocation and variable experiment duration.
- Add a source-transcription adapter that reads Symtropy's typed observation register without ever treating author-derived or not-reported values as direct measurements.
- Move from toy surfaces to source-grounded response envelopes only when exact data and uncertainty are available; keep figure-digitization uncertainty explicit.
- Do not send autonomous laboratory commands. Experiment proposals remain recommendations for human/rig-authority approval.

## 9. Validation record

At the time this document is authored:
- the observation register PR #5578 has 5 source records and 78 observation records with a separate evidence-class field;
- the policy reference fixture contains 20 expected world/policy output rows, derived by an independent deterministic reimplementation of the toy equations;
- this example contains 13 authored Rust tests for the candidate eligibility filter, matched cost/noise, reuse of the current selector score, two-observation stopping, in-domain identification, OOD abstention, fixed-design budget adherence, deterministic random ordering and duplicate-candidate prevention;
- the repository has no `cargo`, `rustc` or `rustfmt` in the current editing environment;
- no Rust test, cargo check, physical experiment or hosted status-check pass is asserted.

The intended local commands are:

```bash
cargo check -p symthaea-core --example isru_material_experiment_selection
cargo test -p symthaea-core --example isru_material_experiment_selection
cargo run -p symthaea-core --example isru_material_experiment_selection
```

Passing those commands qualifies compilation and synthetic behavior only. Physical truth, calibration, assay authenticity and flight/mission viability remain external qualification responsibilities.
