# ISRU Material-Response Tournament v0.2
## Preregistration and sealed-evaluation contract

**Status:** research protocol; implementation and execution are not yet qualified  
**Parent:** [ISRU-MATERIAL-DOE-002 #12483](https://github.com/Luminous-Dynamics/symthaea/issues/12483)  
**Predecessor:** [v0.1 synthetic experiment-selection benchmark](https://github.com/Luminous-Dynamics/symthaea/blob/research/isru-material-experiment-selection-review-v0.1/docs/research/ISRU_MATERIAL_EXPERIMENT_SELECTION_BENCHMARK_V0_1.md)  
**Composition:** Symthaea selects; Symtropy accounts for source observations and physical stages; independent laboratory authority approves and executes; Mycelix supplies custody, calibration and assay attestations.

This document freezes the comparison rules for v0.2. It is not a claim that any v0.2 algorithm exists, compiles, passes tests, or outperforms current practice.

## 1. The research question

Can an experiment-selection policy identify a material's process-regime map and improve the probability of producing *accepted product* under a fixed resource budget, while preserving calibration, uncertainty, safety and out-of-domain (OOD) refusal?

The programme must not collapse three distinct problems into one optimizer score:

1. **Regime identification:** infer the boundaries between spallation, sintering/melting, volatile release, fouling and failure.
2. **Constrained process optimization:** choose conditions that improve a declared objective without violating a frozen operating envelope.
3. **Stage-chain qualification:** determine whether released material can be transported, captured, purified, independently assayed and accepted against a product contract.

A policy can be good at one task and poor at another. Report each task family separately; a weighted aggregate may be shown only as a secondary view with weights frozen before unsealing.

## 2. Paradigm shift: select the next *decision*, not the next most uncertain point

The v0.1 selector maximizes population variance among candidate predictions. This is a useful query-by-committee primitive, but it is not posterior entropy reduction, Bayesian expected information gain (EIG), or expected reduction in decision loss.

V0.2 separates three acquisition objectives:

- **Information value:** expected reduction in uncertainty over declared hypotheses or model parameters.
- **Decision value:** expected reduction in downstream decision loss, including the possibility that more data will not change the selected process or product disposition.
- **Evidence value:** expected reduction in uncertainty caused by metrology, calibration, batch identity or capture/assay limitations.

The experiment planner should sometimes select a calibration check, blank, reference material, replicate or assay before another high-energy material test. This is not wasted budget: an instrument-fault diagnosis can prevent a false physical conclusion. Include a fault-aware policy that may choose between a material experiment and a metrology experiment.

For a candidate action a, a decision-value acquisition can be defined as:

    V(a) = current Bayes risk - E[posterior Bayes risk | result(a)] - cost(a)

The risk function, admissible actions, loss matrix and resource prices must be preregistered per task. Do not substitute raw entropy for decision value when the hypotheses imply materially different engineering consequences.

## 3. The physical evidence graph

Represent the process as an explicit directed stage graph, not a single black-box yield:

    feedstock → preparation → heating → spallation/melting → release → transport → capture → purification → assay → accepted product

Each stage has:
- an input and output material identity;
- a state or condition record;
- an energy and time boundary;
- sensors and calibration identities;
- uncertainty and failure states;
- prerequisites for any downstream stage.

Unknown or failed upstream stages block downstream inference. Ejected material is not captured material; captured raw material is not purified product; a purity estimate is not an independent assay; an assay is not acceptance unless the product contract is satisfied.

Keep separate metrology planes for source input, optical output, receiver-incident energy, receiver-absorbed energy and useful process energy. Do not combine solar incident energy with electrical energy at a different boundary without an explicit, auditable conversion chain.

### 3.1 Multi-stage and multi-fidelity experiments

The evaluator must include partial evaluations and resumable campaigns. For example, a low-cost surface-temperature or optical diagnostic may be available before a high-cost assay; a process may be paused after release and resumed for capture or purification. A monolithic optimizer that treats every experiment as a single scalar observation is not an adequate only-comparator.

Include:
- a stage-aware policy with conditional downstream acquisition;
- a monolithic full-evaluation BO baseline;
- a multi-fidelity policy where low- and high-fidelity measurements have declared costs and a measured relationship;
- a failure-aware policy for missing, censored and below-detection-limit outputs.

The recent multi-stage BO literature demonstrates why stage order, intermediate results, pause/resume and nested acquisition functions matter in process cascades. Use it as a baseline class, not as proof that it transfers unchanged to optical ISRU.

## 4. Baseline tournament

Baselines are grouped by task. A method is eligible only where its assumptions match the target.

### 4.1 Design-of-experiments and reference policies

- Predeclared full-factorial or response-surface design.
- One-factor-at-a-time (OFAT).
- Genuine Latin-hypercube sampling (LHS), generated by a documented stratification/permutation algorithm.
- Genuine Sobol low-discrepancy sequence, using a documented direction-number implementation and scramble policy.
- Uniform random feasible sampling, at least 100 preregistered seeds.
- Current Symthaea population-prediction-disagreement selector, retained as a distinct baseline.

Do not label a hand-written point list as LHS or Sobol. Publish the generator, implementation version, scramble/seed convention and discrepancy diagnostics. Baselines receive identical candidate eligibility, observation availability and resource accounting.

### 4.2 Hypothesis discrimination and decision support

- Discrete noise-aware EIG with explicit prior probabilities and likelihoods.
- Cost-normalized EIG, reporting the unnormalized EIG and cost separately.
- Expected decision-loss reduction / value of information.
- Robust or risk-sensitive acquisition when likelihoods or priors are misspecified.
- Fault-aware acquisition that can choose a calibration or reference check.

EIG must integrate the declared observation-noise model. The evaluator must contain heteroscedastic, heavy-tailed, censored and missing observations; a noiseless variance score is not a substitute.

### 4.3 Continuous and mixed-variable optimization

At minimum, compare the strongest eligible implementations of:
- Gaussian-process BO with expected improvement (EI), probability of improvement (PI) and an exploration-aware acquisition such as UCB or knowledge gradient;
- a tree/forest surrogate for categorical and mixed spaces;
- a TPE or equivalent density-estimation baseline where mixed/conditional variables make it appropriate;
- constrained BO with explicit feasibility probabilities plus independent hard refusal checks;
- multi-objective BO where product quantity, quality, energy, throughput, uptime and contamination conflict.

A GP-EI baseline should not be omitted simply because the current implementation is Rust-first. A Rust implementation is welcome, but a pinned, reproducible external reference implementation is preferable to an invalid in-house approximation. Do not invent a Rust GP or acquisition function merely to avoid a cross-language benchmark.

### 4.4 Physics-informed methods

Include a physics-informed prior, kernel or constraint only when its source, derivation, validity range and calibration are declared before the sealed run. Report the non-physics-informed counterpart. If the prior is wrong or outside its validity domain, the system must detect that condition or show the degradation; a physics prior must not become a hidden source of evaluation leakage.

An oracle policy may be computed for diagnostic upper bounds only. It is never a peer baseline.

## 5. Sealing, leakage control and reproducibility

A public benchmark cannot credibly claim to be sealed if the optimizer authors can inspect the hidden response functions, labels or evaluation seeds.

Use three evidence tiers:

1. **Public development tier:** v0.1 and openly specified synthetic worlds, used for implementation and debugging only.
2. **Sealed synthetic evaluation tier:** independently authored hidden response families and seeds, held outside the optimizer branch until the evaluation is frozen.
3. **Source/physical tier:** published observations and prospective laboratory observations with provenance, calibration and measurement boundaries. Keep this separate from synthetic performance.

Before evaluation:
- publish the benchmark protocol, policy versions, candidate-space schema, resource prices, loss functions and statistical analysis plan;
- record a cryptographic commitment to the sealed world bundle and seed manifest with an independent custodian;
- freeze each policy's code commit and configuration;
- derive paired trial seeds from the committed manifest;
- log each proposed candidate and its hash *before* the evaluator reveals any observation;
- make evaluator calls one-way: policies cannot query hidden labels or inspect future samples;
- unseal only after all frozen runs and analysis outputs have been archived.

The public repository may contain the commitment, schema, evaluator interface and later the released worlds. Do not put hidden ground truth into a public PR before the blind phase is completed. A hash proves commitment to bytes, not authorship independence or physical truth; record the custodian and review procedure separately.

## 6. Workload matrix

The frozen tournament should cover at least these scenario families:

| Family | Failure mode being tested | Required comparison |
|---|---|---|
| Non-monotone response | Spallation peak followed by melting/sintering | LHS/Sobol, GP-EI, EIG |
| Material and morphology shift | Different response boundaries by family/batch | Hierarchical or transfer model vs no-transfer |
| Delayed volatile release | Release optimum differs from excavation optimum | Multi-output / stage-aware policy |
| Capture limitation | Ejecta or gas release does not imply capture | Stage-chain decision value |
| Heteroscedastic metrology | Noise changes with signal/condition | Noise-aware EIG and calibrated surrogates |
| Drift and common-cause failure | Shared calibration or sensor drift corrupts several outputs | Fault-aware acquisition and fail-closed controls |
| Missing/censored outcomes | Failures and below-detection-limit values are informative | Censored/count models vs naive zero fill |
| Carry-over and apparatus degradation | Fouling changes later runs | Nonstationary model and reset/check policy |
| Variable cost and fidelity | Assay, run time and energy differ | Cost-aware and multi-fidelity BO |
| Batch/asynchronous runs | Results arrive out of order or campaigns are interrupted | Batch and resumable acquisition |
| OOD response | True family is outside model support | OOD detection, abstention and false-confidence |
| Safety envelope | Impossible, stale-calibration or out-of-envelope action | Hard refusal, independent of acquisition score |
| Multiobjective product | Purity, accepted mass, energy and uptime conflict | Pareto quality and decision regret |
| Budget-limited stopping | Further data cannot justify cost | Stop / continue / abstain comparison |

Whole batches, morphology classes and at least one entire material family must be held out. Never random-split observations from the same specimen or batch across training and test.

## 7. Resource matching

Do not equate one experiment with one unit of cost. Each campaign reports a resource vector:

    C = (rig_hours, elapsed_hours, source_energy, sample_mass, consumables, assay_cost, compute_time)

Report outcomes against each resource dimension and the Pareto frontier. A scalarized cost may be used only when the prices and weights are fixed before evaluation. Separate algorithm compute cost from physical experiment cost, but report both.

When comparing policies:
- use the same admissible candidate pool and hard constraints;
- pair policies on the same hidden world, batch and seed;
- charge calibration checks, failed runs, retries, sample preparation and analysis;
- charge partial-stage work even if the campaign never reaches final assay;
- do not give an adaptive policy early stopping while forcing a baseline to consume a full schedule unless the comparison explicitly reports both equal-budget and time-to-decision variants.

## 8. Metrics and statistical plan

Do not compress these into one score.

### 8.1 Decision and efficiency

- **Acceleration factor (AF):** baseline resource cost divided by policy resource cost to reach the same predeclared decision-quality threshold.
- **Enhancement factor (EF):** policy decision quality minus baseline decision quality at a fixed resource budget, with the exact metric and direction stated.
- Experiments, rig-hours, energy, sample mass and analysis cost to decision.
- Paired decision regret against the evaluator's true best admissible decision.
- Accepted-product disposition accuracy and product-contract violation rate.
- Regime-boundary error and held-out material-family transfer.

### 8.2 Uncertainty and trustworthiness

- 95% predictive interval coverage, sharpness and group-wise coverage by held-out batch/family.
- Proper scoring rules: log predictive density for suitable distributions; Brier score for binary/categorical dispositions.
- False-confidence rate: confident assertion when the declared model family or measurement chain is invalid.
- Premature-stop and false-OOD rates, plus OOD detection recall and precision.
- Calibration-revocation detection, missing-data handling and provenance completeness.

Coverage alone is insufficient: arbitrarily wide intervals can achieve coverage. Report sharpness and a proper score with coverage.

### 8.3 Safety and evidence

- 100% refusal of the finite frozen set of impossible, unsupported-sensor, stale-calibration and out-of-envelope controls.
- Zero accepted-product promotion without independently identified assay, issuer, batch and contract evidence.
- Zero synthetic model output relabeled as a measurement.
- Every proposed and executed experiment carries an immutable campaign ID, policy version, candidate identity, material batch, instrument/calibration identities, measurement boundaries and source/evidence tier.

The 100% refusal criterion applies to the enumerated controls; it is not a claim of universal safety.

### 8.4 Paired inference

Use at least 100 paired trial seeds per sealed scenario family as an initial floor, then confirm adequacy with a prospective power analysis. For each seed, all policies receive the same hidden world, candidate constraints and initial observations. Resample at the independent unit (world/batch/lab), not at individual rows from one specimen. Report paired effect estimates and 95% intervals, all failures, all unresolved campaigns and the full distribution; never publish only the best run.

A performance advantage is not qualified unless the paired lower 95% confidence bound for cost-adjusted acceleration is above 1.0 against the strongest eligible baseline, while non-inferiority limits for decision regret, calibration, OOD refusal and product gates are met. Predeclare the limits and adjust for multiple primary comparisons.

## 9. Targets (aspirational, not achieved)

| Measure | Initial programme target | Stretch |
|---|---:|---:|
| Cost-adjusted acceleration vs strongest eligible baseline at equal decision quality | ≥1.5× | ≥2× |
| 95% predictive interval coverage on held-out batches | Near 95%, with sharpness reported | Near 95% across every declared family |
| Frozen hard-control refusal | 100% | 100% |
| Accepted-product promotion without independent assay and contract | 0 | 0 |
| Held-out family transfer | No material degradation vs in-family non-inferiority bounds | Improved transfer with calibrated uncertainty |
| Decision regret | Non-inferior to strongest baseline | Lower with the same or lower resource cost |

The self-driving-lab literature review reports a median acceleration factor around six relative to random sampling across heterogeneous published campaigns. That figure is not a target against strong BO, not a universal baseline and not directly comparable to our multi-stage, constrained task. Our threshold is deliberately against the strongest eligible comparator.

## 10. Industry and SOTA comparison to reproduce

The following sources are useful comparators, with their scope kept explicit:

- **NASA/TM-20260001102 (March 2026), “Benchmarking Bayesian Optimization Frameworks and Acquisition Strategies for Materials Discovery and Autonomous Laboratories.”** Compares Ax/Bayes GP, Monte-Carlo neural networks, random forests and TPE across continuous, categorical and hybrid benchmarks, with posterior mean, PI and EI; reports 100 trials from randomized initial samples and identifies GP-EI as a strong low-evaluation baseline, while forests are competitive on categorical/mixed spaces. V0.2 should reproduce the relevant baseline class or document why it is ineligible. [NASA NTRS PDF](https://ntrs.nasa.gov/api/citations/20260001102/downloads/TM-20260001102_Rework.pdf)
- **Multi-stage Bayesian optimization for dynamic decision-making in self-driving labs (Digital Discovery, 2026).** Adds intermediate-stage evaluation, saved partial workflows, nested acquisition and heterogeneous costs. This is the closest algorithmic challenge to our process cascade. [Article](https://doi.org/10.1039/D5DD00572H)
- **Cost-aware Bayesian optimization of real-world nanoindentation workflows (Digital Discovery, 2026).** Reports nearly 30-fold mapping-efficiency improvement over grid-based indentation in its specific thin-film workflow, with motion, drift stabilization and reconfiguration included in a detailed cost model. This is a workflow-specific result, not a transferable ISRU performance target. [Article](https://doi.org/10.1039/D5DD00518C)
- **Benchmarking self-driving labs (Digital Discovery, 2025/2026).** Reviews acceleration and enhancement metrics and reports a literature median AF of roughly six versus random; highlights dependence on property-space complexity. [Article](https://doi.org/10.1039/D5DD00337G)
- **The ADePT framework for assessing autonomous laboratory robotics (Communications Chemistry, 2026).** Evaluates adaptability/learning, dexterity, perception and task complexity; interoperability is an orthogonal practical requirement. Use ADePT for a future physical-lab execution tier, not as an optimizer-only score. [Article](https://www.nature.com/articles/s42004-026-01932-9)

Physical competitors remain ahead of this software programme in hardware integration and physical demonstration. NASA's MRE work processed 25 kg of lunar simulant with a reported average oxygen production rate of 0.07 kg/h over a nine-hour electrolysis phase; its public performance report lists 0.3 mol O2/kWh against a 0.6 threshold and a 1.0 goal, with distinct melt-phase energy figures. NASA CaRD separately demonstrated an integrated solar-carbothermal prototype and confirmed CO production. Neither is a direct asteroid-optical-mining comparator, and these reported signals must not be promoted to an accepted purified product without the corresponding evidence. [MRE ICES 2026 record](https://ntrs.nasa.gov/citations/20260006167) · [MRE public performance report](https://ntrs.nasa.gov/api/citations/20250003208/downloads/MRE_PublicRelease4072025.pdf) · [CaRD 2026 record](https://ntrs.nasa.gov/citations/20260003603)

The 2025 Nectar A/B work demonstrates why the response should be a regime map: morphology affected spallation/melting and peak excavation occurred near different irradiances for the two simulant cakes; excessively high irradiance can suppress useful spallation. Those values are campaign-specific observations, not universal operating setpoints. [Optical mining of carbonaceous chondrite simulants](https://doi.org/10.1016/j.actaastro.2025.04.033)

## 11. Execution sequence and evidence gates

1. Freeze public schemas, scenario interfaces, costs, task loss and statistical plan.
2. Implement genuine LHS/Sobol/random generators and test their invariants independently.
3. Add a noise-aware discrete EIG baseline with explicit priors/likelihoods and numerical sanity checks.
4. Add decision-value and fault-aware acquisition; test cases where an instrument check is more valuable than a material run.
5. Integrate a pinned GP-EI baseline and an eligible categorical/mixed-space forest or TPE baseline.
6. Add stage-aware / multi-fidelity policies and compare them with monolithic BO on the same campaign graph.
7. Have an independent custodian create and seal held-out response families and seeds.
8. Run paired campaigns, publish raw trial tables, confidence intervals, calibration plots, failures and reproducibility receipts.
9. Only after synthetic qualification, connect published observations and independently instrumented laboratory campaigns through the Symtropy evidence register and Mycelix authority records.

**Compilation gate:** source authorship, fixture audits, successful compilation, unit tests, integration tests, hosted CI and physical execution are separate states. No skipped or queued check is a pass. No benchmark claim is made from authored tests or an unexecuted design.

## 12. Claim ceiling

Passing v0.2 demonstrates only performance on its frozen synthetic evaluation worlds. It does not establish real material response, safe laser operation, industrial throughput, captured resource, purified product, asteroid reserves, mission economics or flight readiness. Those require separate, independently instrumented physical campaigns and product acceptance evidence.
