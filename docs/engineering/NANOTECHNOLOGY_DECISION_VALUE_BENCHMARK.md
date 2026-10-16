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

1. **Protocol-only:** finish metric definitions, baseline eligibility, data rights, task card, and leakage checks.
2. **Baseline-only pilot:** freeze one manageable task with ground truth; produce manifest, per-episode costs, metric reconstruction, and negative-control results before asserting any model advantage.
3. **Selector comparison:** evaluate current and strong baselines with paired budgets and held-out material groups.
4. **Reference-in-the-loop:** add active learning only after its label-eligibility, stopping, and budget accounting are independently verified.
5. **Prospective experiment:** only where real experimental capacity and data rights exist, run blinded confirmation under a separate protocol.

Current Symthaea molecular AIMD uses molecular, non-periodic RHF/STO-3G with finite-difference forces and is not an eligible periodic nanocrystal reference. A new benchmark must not use it as experimental truth or imply it is a qualified nanocrystal simulator. NANO-008 separately covers trajectory reliability.

## 13. Research references

- Usman et al. (2026), *Benchmarking Machine-Learning Interatomic Potentials for Dynamical Stability in Inorganic Semiconductor Nanocrystals: A CdSe Case Study*, preprint: https://arxiv.org/abs/2609.15299
- Xu et al. (2026), *Benchmarking universal machine learning interatomic potentials for supported nanoparticles: decoupling energy accuracy from structural exploration*, npj Computational Materials 12, 189: https://www.nature.com/articles/s41524-026-02043-8
- Tahmasbi et al. (2026), *Benchmarking universal machine learning interatomic potentials on elemental systems*, npj Computational Materials 12, 251: https://www.nature.com/articles/s41524-026-02251-2
- *UniFFBench: evaluating universal machine learning force fields against experimental measurements* (2026), Nature Computational Science: https://www.nature.com/articles/s43588-026-01019-4
