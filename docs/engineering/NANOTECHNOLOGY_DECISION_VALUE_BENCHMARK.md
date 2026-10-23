# Nanotechnology Decision-Value Benchmark (NANO-009)

Status: **protocol proposal; no benchmark result claimed**

Issue: [#7679](https://github.com/Luminous-Dynamics/symthaea/issues/7679)  
Related protocols: [NANO-008 trajectory reliability (#7654)](https://github.com/Luminous-Dynamics/symthaea/issues/7654), [NANO-005 selector kernel](https://github.com/Luminous-Dynamics/symthaea/issues/7570)

## 1. Research question

Under a fixed, auditable budget, does a candidate-selection system find more *useful, feasible, independently verifiable* materials than strong simple baselines?

The benchmark intentionally changes the optimization target. Low energy/force error, a stable molecular-dynamics run, or a high acquisition score is not itself a successful discovery. A successful computational recommendation must satisfy a property target and every preregistered hard constraint against an eligible reference. Experimental success is a separate, higher evidence tier.

The unit of evaluation is a **decision episode**: a frozen candidate pool, target utility, feasibility rules, information available at the start, cost schedule, selection policy, and budget. An episode ends at the fixed budget or the predeclared stop rule, not when a method appears to win.

## 2. Why this benchmark exists

Recent research motivates several distinct validation gates:

- A September 2026 CdSe nanocrystal preprint reports that, for its tested 149-atom chloride-passivated cluster and 1 ns trajectories at 300 K, the lowest held-out force error did not imply the best dynamic robustness. It is a case study/preprint, not a universal ranking: https://arxiv.org/abs/2609.15299
- A 2026 supported-nanoparticle study explicitly evaluates finite-temperature dynamics and structural exploration, emphasizing that energy/force accuracy and successful structural search can diverge: https://www.nature.com/articles/s41524-026-02043-8
- A July 2026 elemental-system benchmark combines equation-of-state tasks with minima-hopping searches rather than relying on one static metric: https://www.nature.com/articles/s41524-026-02251-2
- UniFFBench compares universal-potential predictions with experimental mineral data and reports that computational performance alone can fail to capture experimental-property accuracy. Its mineral results are motivation for an independent experimental gate, not transferable accuracy thresholds for nanocrystals: https://www.nature.com/articles/s43588-026-01019-4

These publications do not establish that Symthaea currently predicts nanomaterial properties or can run qualified nanocrystal dynamics.

## 3. Freeze the decision before evaluating

Every benchmark task must preregister a machine-readable task card containing:

- Material family and candidate-generation domain, including allowed elements, composition, structure/provenance, surface/passivation, defect range, and boundary conditions.
- Target observable, direction of improvement, unit, operating conditions, reference method, property threshold, and any measurement tolerance.
- Hard feasibility constraints: chemistry, charge/state, stability, synthesis/manufacturability, availability, safety, and maximum estimated cost, as applicable.
- The candidate pool or a precise generation protocol; duplicates and equivalent structures must be canonicalized before splitting.
- Initial information available to every policy, budget limits, cost schedule, stop rule, and maximum candidate count.
- Baselines, aggregation, uncertainty method, statistical test, non-inferiority margins, exclusions, and success criteria.
- Dataset and checkpoint licenses, data provenance, version IDs, file hashes, and permitted use.
- For experimental work: specimen preparation, measurement protocol, environment, number of independent replicates, blinding, failure accounting, and preregistered comparison.

Do not modify criteria after inspecting held-out results. Any amendment creates a new benchmark version.

## 4. Decision outcomes

### 4.1 Feasibility and hit rate

For a candidate to count as a **qualified computational hit**, the trusted reference must show that it meets the frozen property threshold and every hard constraint. Missing, failed, unsupported, or out-of-domain candidates are not successes. Report separate counts for unknowns, abstentions, invalid candidates, and reference-evaluation failures.

For each episode, report:

- **Qualified-hit rate @ K** = qualified candidates among the policy's first K recommendations divided by K.
- **First-hit cost** = cumulative accounted cost through the first qualified hit; report no-hit episodes as right-censored or as budget exhaustion, never as zero cost.
- **Hits per budget** = qualified hits divided by total budget consumed, alongside absolute counts.
- **Constraint-violation rate** = evaluated recommendations violating one or more hard constraints divided by evaluated recommendations; also report each constraint separately.
- **Abstention rate** = cases where the policy refuses to make a supported recommendation divided by opportunities to recommend.

State whether the top-K list is unique by chemistry/structure family. Multiple near-identical candidates must not inflate the hit count.

### 4.2 Regret and ranking quality

For an enumerable frozen candidate pool and frozen utility function U, define:

- U* as the best utility among all candidates that are feasible according to the trusted reference.
- U_B as the best feasible utility found by the policy by budget B.
- **Simple regret at B** = U* - U_B for a maximization task.

If no feasible candidate was found, report that outcome separately rather than silently assigning an arbitrary utility. If the pool is sampled or incomplete, call U* the best in the *frozen pool*, not the global materials optimum. For non-enumerable generation, predeclare a valid best-known reference set and use a separate, clearly labeled metric.

Also report top-K ranking metrics where meaningful, but do not use ranking correlation as a substitute for qualified-hit rate, regret, or feasibility.

### 4.3 Cost accounting

Compare policies using total cost, not only the number of selected configurations. Record actual values where available and otherwise retain separately labeled estimates for:

- Reference electronic-structure/force calls, retries, failures, and wall-clock or accelerator time.
- Training, fine-tuning, uncertainty calibration, model selection, and acquisition overhead.
- Candidate generation, geometry relaxation, trajectory generation, failed simulation time, and data conversion.
- For experimental qualification: synthesis attempts, materials, labor or facility usage, measurement attempts, failed specimens, and repeat measurements.

Publish both raw resource counts and any monetary/carbon conversion with the conversion source and date. Do not collapse unlike units into one scalar unless the weighting is fixed before the study.

## 5. Mandatory baselines

At identical information and budget, compare only applicable and eligibility-qualified baselines:

1. Random sampling.
2. Diversity-only or farthest-first selection.
3. Uncertainty-only selection with a declared uncertainty estimator.
4. Existing Symthaea cost-aware selector.
5. Target-specific expected improvement or constrained Bayesian optimization when the task supports it.
6. Passive/no-active-learning sampling as a control.

Use common starting pools and paired episodes where possible. Tune baselines only on training/validation material groups. Freeze acquisition hyperparameters before opening held-out results. Missing or ineligible baselines must be disclosed, not treated as defeated.

For selector evaluation, the same reference labels and constraints must be available to each method only after it selects a candidate, as if in a real closed loop. An oracle ranking can be used to compute a frozen-pool regret bound, but must never be provided to a policy.

## 6. Grouping and leakage controls

Split by the entity most likely to make data correlated: parent material/composition, crystallographic prototype, surface/passivation family, trajectory, source dataset, synthesis batch, or specimen. Use multiple simultaneous grouping rules when needed. Adjacent trajectory frames, randomized replicas from one parent, duplicate CIFs, and transformed copies must remain in one group.

Record a similarity/duplicate audit across splits. Reserve a genuinely held-out material family when claiming transfer beyond interpolation. Report in-family and out-of-family results separately. If no defensible group split or rights-clear evaluation set exists, status remains **protocol-only / blocked**, not a failed scientific result or a pass.

## 7. Decision calibration and safe abstention

When a system exposes the probability that a candidate will satisfy all the constraints, evaluate that decision probability directly against the binary qualified-hit outcome. Report Brier score or log loss, reliability diagrams, grouped confidence intervals, and the calibration procedure. Scalar property-interval calibration is useful but not a proxy for calibration of the joint feasibility decision.

For claimed nominal 90% property intervals, report empirical coverage and sharpness separately, with grouped calibration/evaluation splits. A provisional goal is coverage in the 85–95% band across sufficiently large, relevant held-out groups; this is a target to test, not a guarantee. Small-sample, shifted, or chemically unsupported cases should be allowed to abstain. Include abstention utility/cost so an always-abstaining model cannot appear excellent.

Test negative controls: shuffled targets, random acquisition, duplicated candidates, deliberately overconfident uncertainty, and at least one out-of-domain task. The system must not earn credit for confidence on unsupported chemistry.

## 8. Separate experimental qualification gate

A computational hit is never an experimentally confirmed discovery.

An experimental claim requires:

- An independent measurement or specimen set not used for model tuning or candidate selection calibration.
- Blinded measurement/evaluation where feasible, explicit experimental conditions and units, independent replication, and predeclared tolerance to measurement/reference uncertainty.
- All attempted syntheses and measurements accounted for, including failed synthesis, no result, decomposition, unusable specimens, and repeat attempts.
- Comparison with a declared incumbent/material baseline; measure both target-property difference and practical feasibility.
- Exact provenance for material identity, sample preparation, instruments/method, analysis pipeline, and raw-data rights.
- Independent reconstruction of the reported metric from the frozen manifest and raw/derived observations.

Experimental outcomes must be reported separately from DFT agreement and from simulation stability. A candidate should not be described as synthesized, validated, or useful in practice merely because an atomistic model predicted a favorable property.

## 9. Statistical protocol

The benchmark owner must choose a sample-size/power rationale before unblinding. Prefer paired comparisons on the same episodes; resample or estimate uncertainty at the material-group level rather than treating correlated candidate frames as independent observations. Report per-task outcomes and heterogeneity, not only one pooled average. Correct for multiple comparisons if many methods/tasks are used for claims.

For initial selector studies, a practical design target is at least 20 independent, preregistered policy seeds per task where stochasticity exists, subject to a documented power and cost analysis. Seeds are not substitutes for independent material families. Publish the denominator and confidence interval for every success-rate claim.

## 10. Provisional go/no-go criteria

These are starting targets for task-specific preregistration, **not measured Symthaea results**:

- At a fixed total reference budget, improve qualified-hit rate @ K by at least 10 percentage points against the strongest eligible baseline, **or** reduce total reference cost by at least 20% at matched hit rate and regret.
- Do not worsen the hard-constraint violation rate. Start with a maximum +2 percentage-point non-inferiority margin, but finalize the margin from task risk and sample-size/power analysis before unblinding.
- The primary benefit must survive grouped held-out evaluation, with uncertainty excluding no improvement under the preregistered method.
- Include all failed reference calls and invalid proposals in the budget accounting.
- No experimental superiority claim without the independent experimental gate.

If the candidate pool is too small, the comparator is ineligible, data rights are unclear, or statistical power is inadequate, report the blocker and do not lower the bar after seeing results.

## 11. Evidence tiers and claim ceilings

| Tier | Evidence | Allowed claim |
|---|---|---|
| 0 | Protocol, source audit, or synthetic fixtures only | Design is specified; no performance claim |
| 1 | Rights-cleared labels, frozen manifest, reproducible baseline runs | Baseline behavior on the named computational task |
| 2 | Grouped held-out comparison with independent metric reconstruction | Comparative decision value on the named benchmark |
| 3 | Prospective predictions plus blinded independent experiments | Experimentally supported result under the named conditions |
| 4 | Independent replication across groups/labs/conditions | Replicated result within the tested scope |

Never generalize a result beyond the supported element, chemistry, size, surface, temperature, environment, and property domain. A successful Tier 2 benchmark is not Tier 3 experimental qualification.

## 12. Implementation sequence

1. **Adopt before rebuilding:** audit the public MADE (MAterials Discovery Environments) release and its official benchmark tasks/metrics; use a compatible, rights-cleared task as the first closed-loop baseline rather than building a second discovery simulator from scratch.
2. **Reproduce an upstream baseline:** run a small, official MADE baseline with the original harness, freeze the exact commit/config/data identifiers, and preserve the upstream result artifacts. This is a research comparator only, not a Symthaea runtime dependency.
3. **Independent Rust verification:** write a Rust-first, read-only parser/metric checker for exported episode records. Verify candidate counts, oracle-call budgets, hit/regret summaries, cost accounting, and provenance independently. Do not port an unverified upstream metric by name alone; freeze formulas and test fixtures from the paper/release.
4. **Adapt Symthaea policies through a boundary:** add the current cost-aware selector and later Symthaea planning as policy implementations behind the environment/oracle interface, without making the benchmark depend on a Python production runtime.
5. **Extend beyond convex-hull tasks:** add trajectory reliability, structural exploration and dynamics checks from NANO-008, and only then add experimental qualification for tasks with rights-cleared independent measurements.
6. **Paired held-out evaluation:** compare against upstream random and generative baselines plus appropriate diversity, uncertainty and expected-improvement policies, with fixed oracle budgets and grouped test systems.

Current Symthaea molecular AIMD uses molecular, non-periodic RHF/STO-3G with finite-difference forces and is not an eligible periodic nanocrystal reference. A new benchmark must not use it as experimental truth or imply it is a qualified nanocrystal simulator. NANO-008 separately covers trajectory reliability.

## 13. Keep three scientific benchmark lanes separate

A paradigm shift is to share a *decision/receipt/audit architecture* without collapsing distinct physical questions into one composite "nanotechnology score". A discovery policy can succeed on one lane and fail on another; pooling incomparable targets, units or failure modes hides this.

| Lane | Scientific question | Core outcomes | Admission gate |
|---|---|---|---|
| A. Convex-hull discovery | Can a policy find new reference-stable compounds under a fixed oracle budget? | Official discovery curves, qualified stable-hit rate, paired regret, setup-inclusive cost | Reproduce a pinned MADE task; identify each phase-diagram/source revision, the exact oracle and effective-ground-truth filters |
| B. Historical measured-property optimization | Can a policy choose candidates with better *measured* performance? | Hit rate at K, simple regret, target-coverage calibration, cost to first hit, failure/repeat accounting | Local snapshot/hash, units, measured-feature availability/cost, sample/replicate grouping, source rights and attribution |
| C. Atomistic trajectory reliability | Does a potential preserve credible nanostructure dynamics and explore plausible structures? | Trajectory survival, structural fidelity, severe-error recall, warning lead time, calls to reliability threshold | Exact starting geometries, reference trajectory/config, model/data rights, group splits, material-specific failure rules |

Use one policy/evidence interface across lanes where possible, but keep distinct task contracts, units, thresholds, denominators, and reports. Report lane-specific results side-by-side; do not average them into a single score. A cross-lane transfer claim requires a fourth, explicitly preregistered experiment in which the transfer mechanism and held-out target domain are defined in advance.

### Distribution shift is not just another random test split

The atomistic lane must include a first-class shift-manifest and worst-group protocol, tracked in [NANO-012](https://github.com/Luminous-Dynamics/symthaea/issues/7762) and the [trajectory reliability benchmark](https://github.com/Luminous-Dynamics/symthaea/blob/research/nano-008-trajectory-reliability/docs/engineering/NANOTECHNOLOGY_TRAJECTORY_RELIABILITY_BENCHMARK.md). Kreiman & Krishnapriyan analyze distribution shifts for universal interatomic potentials (2026, https://doi.org/10.1039/D5DD00260E); a later 2026 elemental-system benchmark compares potential generalization across structural search settings (https://www.nature.com/articles/s41524-026-02251-2); and a multicomponent-alloy study reports separate transferability, finite-temperature robustness and uncertainty behavior (https://doi.org/10.1103/1qc9-ypyb).

For every admitted material task, freeze the physical shift family, severity definition, independence keys, sample-to-split mapping, and source/reference hashes before fitting or evaluating. Report performance versus shift severity and all-in budget; show worst-group error, trajectory survival, calibration, failure-detection recall, false alarms and adaptation cost independently. Never summarize those outcomes into a composite score that conceals a failed chemistry/surface/temperature regime.

NANO-010's optional shift-manifest checker validates the metadata/hash chain and rejects shared sample IDs or independence keys across data splits. That is a split-integrity guard only; domain review must still justify that the shift categories and severities represent physically relevant deployment conditions.

### Current candidate tasks and rights status

- **MADE convex-hull campaign:** the official repository says its run harness can use ORB/MACE/analytic oracles, and some data/agents rely on Materials Project, checkpoints, and API services. The README's MIT code license does not automatically settle upstream dataset, model-weight, or API terms. Freeze one task, oracle, cached/source artifacts, data rights, start state, setup invocations, and actual external-call ledger before using it for a fair cost comparison.
- **NIST/REMI Fe-Co-Ni historical measurements:** issue [NANO-007 (#7644)](https://github.com/Luminous-Dynamics/symthaea/issues/7644) records that the pinned source tree lacks an explicit dataset license. Do not bundle the source data or derived redistribution artifacts until the rights question is resolved. This is measured-property benchmarking if admitted, not a nanocrystal simulation benchmark.
- **MOCLIP measured nanophotonics:** the public [release repository](https://github.com/RodionovSA/MOCLIP_release) provides a 10k subset of measured geometry–spectrum pairs and has an MIT code license. The repository README/license alone does not establish separate rights for the dataset or pretrained weights; issue [NANO-004 (#7531)](https://github.com/Luminous-Dynamics/symthaea/issues/7531) remains blocked pending exact dataset/model terms. Its measured spectra could provide a useful historical inverse-design lane if the rights gate is cleared; they do not validate atomic-scale nanocrystal dynamics.
- **CdSe nanocrystal reliability:** the 2026 preprint is a useful candidate, but its exact input/reference artifacts and terms must be independently admitted before reproduction. The current Symthaea molecular RHF/STO-3G implementation is not an eligible periodic/nanocrystal reference solver.

At the time this version was authored, **no one task above is declared rights-cleared and experimentally qualified by this protocol**. Treat that as a concrete admission blocker, not permission to silently choose whichever public files happen to be downloadable. The immediate deliverable is a source/rights decision record plus a reproducible baseline package; no real scientific improvement can be claimed until one lane clears admission.

## 14. Candidate benchmark: experiment-grounded nanophotonic inverse design

This is a **separate measured-photonics lane**, not a proxy for atomistic nanocrystal chemistry or material stability. It offers a useful paradigm shift because the outcome can be tied to fabricated geometries and measured spectra rather than treating simulator scores as reality.

The [MOCLIP Nature Communications paper](https://www.nature.com/articles/s41467-026-76714-x), published 19 August 2026, reports a silicon-on-glass metasurface dataset with 466,537 unique measured geometry–spectrum pairs, 42 design degrees of freedom, inverse-design throughput around 200,000 target spectra/s on GPU, and source-reported latent optimization accuracy above 95%. Its public repository offers a 10,000-sample subset. These are **source-reported reference numbers**, not independently reproduced Symthaea results. The paper also reports fabrication and hyperspectral measurement; a benchmark should exploit that empirical link instead of scoring only generated spectra.

### Task contract and controls

**Input:** a target spectral amplitude profile and allowed geometry/fabrication constraints.

**Decision:** return a ranked, deduplicated list of geometries plus uncertainty, expected constraint/yield risk, and the reason each candidate is selected.

**Outcome:** the candidate's measured spectrum, deviations from target by wavelength/polarization, hard fabrication constraint results, and actual per-candidate fabrication/metrology cost.

Mandatory comparators are: direct nearest-neighbor/library search, the published MOCLIP method where rights permit, a simple diversity-aware selector, an uncertainty-aware selector, a constrained Bayesian/latent-space optimizer if the exact task supports it, and random valid geometries. Every method must use the same candidate pool, initial information, feature availability, constraints, and defined evaluation budget. A training-free retrieval baseline is important: a larger model is not a discovery advance if it only matches a library lookup at more total cost.

### Evaluation protocol

1. **Historical reproduction:** pin exact code/checkpoint/data artifacts, split, preprocessing, wavelength/polarization channels, normalization, and the publication's exact Top-k and MSE formulas. Reconstruct reported metrics before adding new ones.
2. **Leakage-resistant generalization:** retain a random held-out split only for comparability, then add group-held-out tests by wafer/substrate, geometry family, fabrication batch, or target-spectrum family. Keep replicas of the same geometry and near-duplicates in the same partition. Freeze exact grouping before training or candidate selection.
3. **Fabrication-robust inverse design:** perturb design variables within measured process tolerances and report feasible-yield curves, lower-tail spectral quality, hard-constraint violation rates, and risk-sensitive quality (e.g. worst-decile quality). The tolerance model must come from source/facility process evidence, not arbitrary noise injected only to improve a robustness score.
4. **Calibrated abstention:** on independent group holdouts, evaluate whether predicted spectral-error intervals cover measurements at their declared levels, and let the policy abstain when uncertainty or fabrication yield is insufficient. Measure coverage, interval width, selective-risk curves and failure-detection recall.
5. **Prospective measurement:** select candidates before measurements exist; preserve candidate IDs, model/policy revision, expected outcome, all rejected candidates, and call receipts. Fabricate/measure under randomized order and include controls and replicates. Report measured success per *all-in* resource budget, including failed fabrication, unsuccessful candidates, repeats, retries, and metrology.

### Proposed benchmark targets (design goals, not achieved results)

- **Baseline gate:** reproduce the published metrics from pinned eligible artifacts with the published formulas and disclose the full random/group-held-out gap. Do not require an arbitrary tolerance until the upstream evaluation and measurement noise are exactly specified.
- **Generalization gate:** report separate performance for each held-out group and each major geometry/fabrication regime; no score averaged across groups may hide a catastrophic subgroup failure.
- **Calibration gate:** as a first target, nominal 90% spectral-error intervals should achieve 85–95% empirical coverage on genuinely held-out groups, with interval width and selective-risk curves reported. Coverage alone is insufficient if intervals are uninformatively wide.
- **Budget-efficiency gate:** against the strongest eligible baseline, seek either at least **+10 percentage points** prospective measured-hit rate at equal fabrication/metrology budget, or at least **20% lower all-in cost** at matched measured hit-rate/regret, without increased hard-constraint violations. Preregister the spectral objective and hit threshold.
- **Robustness gate:** report nominal and tolerance-perturbed feasible yield separately; a proposed initial target is at least 90% feasibility over the empirically justified tolerance distribution. Treat this as a target to review with process engineers, not a universal physical constant.

### Current admission status

This is a candidate task card only. The MOCLIP code repository's MIT license does **not** by itself establish the terms for the measured data, checkpoint weights, or derived artifacts. Do not download, redistribute, or treat the model/data as rights-cleared until those exact terms and attribution requirements are resolved. See [NANO-004](https://github.com/Luminous-Dynamics/symthaea/issues/7531). The 10k subset can support a retrospective harness only after its archive/data terms are verified; prospective measurement requires an authorized facility and independent measured outcomes. Until then, implement the schemas and synthetic negative controls, but do not make a scientific-performance claim.

## 15. Interoperability with MADE (ICML 2026)

Before inventing a new closed-loop benchmark stack, evaluate [MADE: Benchmark Environments for Closed-Loop Materials Discovery](https://proceedings.mlr.press/v306/malik26a.html) and its [official implementation](https://github.com/diffractivelabs/MADE). MADE models discovery as an iterative campaign under a constrained oracle budget, using convex-hull stability tasks and interchangeable generators, filters, scorers, planners and oracles. Its release lists AF, EF, AUDC and mSUN as discovery metrics and provides per-episode trajectories, summaries, configs and logs. The ICML 2026 paper reports that adaptive planning becomes more important as the search space grows.

This changes our plan: **reuse and reproduce a credible upstream benchmark first; add Symthaea-specific trustworthiness and physical-evidence gates around it**. Do not claim a novel benchmark is needed until this compatibility audit shows a real gap.

- Stage A: verify upstream license, data-source terms, supported oracle access, reproducibility, metric formulas, and whether a small task can run without paid/secret services. MIT licensing of the code does not automatically settle licenses for downloaded datasets, model weights, APIs, or derived structures.
- Stage B: run the smallest official baseline task and archive its source commit, resolved config, dataset/system hashes, oracle version, per-episode trajectories, stdout/logs and resource accounting.
- Stage C: independently reconstruct official discovery metrics from raw episode records. Add a Rust implementation for integrity checking and metric reconstruction; keep the upstream benchmark harness as an optional external research tool rather than introducing Python into Symthaea's production runtime.
- Stage D: adapt a Symthaea policy only after the baseline reproduces; maintain exact same oracle budgets, allowed information, task candidates, and seed/episode definitions.
- Stage E: report separate results for convex-hull discovery, atomistic trajectory reliability, and experimental confirmation. MADE's convex-hull task does not certify nanocrystal dynamics or laboratory outcomes.

If upstream reproduction is blocked by API keys, data licensing, or irreproducible dependencies, document the blocker and select a rights-cleared alternative. Do not silently substitute a different oracle or reuse the official method name for a changed metric.

### Initial source-audit risks to resolve before trusting headline metrics

A preliminary read of the upstream release identifies specific items that the interoperability pilot must settle rather than assume away:

- `ConvexHullEnvironment.step()` increments the environment's `query_count` for each proposed structure, while setup paths compute or retrieve elemental reference energies and construct the initial observed phase diagram. The budget denominator must establish whether those setup operations actually trigger paid/expensive oracle work, whether results are cached, and how that cost is reported. Do not equate `query_count` with total compute cost without reconciling the call graph and logs.
- `start_with_all_stable=True` initializes the observed diagram by re-evaluating stable ground-truth structures. That is a different and potentially oracle-expensive task setting than discovery from elemental references only. Freeze the setting and report initial oracle calls separately; do not compare policies under unlike initial information.
- The environment can filter its ground truth by SMACT validity, maximum stoichiometry and near-stability tolerance before computing the target set. Those filters define an **effective ground truth**. Record raw-source counts and post-filter counts, exact filter parameters, excluded formulas/structures, and the effect on recall/precision/novelty denominators.
- The official analysis utilities use several similarly named metrics and fallbacks (e.g. structure discoveries vs stable structure discoveries; episode histories may be reconstructed when fields are missing). Freeze exact metric keys/formulas and fail closed on incomplete trajectories instead of accepting fallback-derived values without a provenance flag.
- A read-only Rust verifier must reconcile per-episode JSON with summary JSON/CSV, reported query counts, and logged oracle calls; intentionally malformed or truncated outputs must fail verification.

These points are an audit checklist derived from source inspection, not a claim that MADE is invalid. Preserve the official results and names when reproducing the paper, and publish any corrected or extended metric as a separately named metric.

## 16. Research references

- Usman et al. (2026), *Benchmarking Machine-Learning Interatomic Potentials for Dynamical Stability in Inorganic Semiconductor Nanocrystals: A CdSe Case Study*, preprint: https://arxiv.org/abs/2609.15299
- Xu et al. (2026), *Benchmarking universal machine learning interatomic potentials for supported nanoparticles: decoupling energy accuracy from structural exploration*, npj Computational Materials 12, 189: https://www.nature.com/articles/s41524-026-02043-8
- Tahmasbi et al. (2026), *Benchmarking universal machine learning interatomic potentials on elemental systems*, npj Computational Materials 12, 251: https://www.nature.com/articles/s41524-026-02251-2
- *UniFFBench: evaluating universal machine learning force fields against experimental measurements* (2026), Nature Computational Science: https://www.nature.com/articles/s43588-026-01019-4
- Malik et al. (2026), *MADE: Benchmark Environments for Closed-Loop Materials Discovery*, ICML / PMLR 306: https://proceedings.mlr.press/v306/malik26a.html; official implementation: https://github.com/diffractivelabs/MADE
- Ho, Ortner & Wang (2026), *Flexible uncertainty calibration for machine-learned interatomic potentials*, npj Computational Materials: https://www.nature.com/articles/s41524-026-02080-3
- Wang et al. (2026), *Training-free active learning framework in materials science with large language models*, npj Computational Materials: https://www.nature.com/articles/s41524-026-02136-4
