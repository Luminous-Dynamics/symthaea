# Nanotechnology Benchmark Charter: Metrics, Thresholds and Evidence Gates

**Status:** Proposed acceptance policy for future nanotechnology benchmark work; no benchmark is claimed to have passed.  
**Date:** 2026-10-16  
**Domain atlas:** [NANOTECHNOLOGY_DOMAIN_ATLAS.md](./NANOTECHNOLOGY_DOMAIN_ATLAS.md)  
**Nanophotonics benchmark:** [NANOPHOTONICS_MOCLIP_BENCHMARK_SCORECARD.md](./NANOPHOTONICS_MOCLIP_BENCHMARK_SCORECARD.md)  
**MADE benchmark:** [NANOTECHNOLOGY_CLOSED_LOOP_BENCHMARK_PLAN.md](./NANOTECHNOLOGY_CLOSED_LOOP_BENCHMARK_PLAN.md)  
**OC22 program:** [NANOMATERIALS_RESEARCH_PROGRAM.md](./NANOMATERIALS_RESEARCH_PROGRAM.md)
**Measured thin-film active-learning pilot:** [NANOTECHNOLOGY_NIST_FECO_NI_BENCHMARK_PLAN.md](./NANOTECHNOLOGY_NIST_FECO_NI_BENCHMARK_PLAN.md), tracked by [NANO-007](https://github.com/Luminous-Dynamics/symthaea/issues/7644). Its 921-row historic pool is blocked on explicit data-use terms and local schema/integrity validation.
**Rust policy reference:** [tools/nano-policy](../../tree/research/nano-001-atomistic-discovery/tools/nano-policy) is a small isolated prototype, not the canonical selector. The main kernel/evaluator is [PR #7568](https://github.com/Luminous-Dynamics/symthaea/pull/7568), tracked by [NANO-005 issue #7570](https://github.com/Luminous-Dynamics/symthaea/issues/7570). Neither tool is a physical predictor or a benchmark result.
**NIST Fe–Co–Ni measured thin-film benchmark:** [NANOTECHNOLOGY_NIST_FECO_NI_BENCHMARK_PLAN.md](./NANOTECHNOLOGY_NIST_FECO_NI_BENCHMARK_PLAN.md), tracked by [NANO-007](https://github.com/Luminous-Dynamics/symthaea/issues/7644). It is an offline replay of 921 historic measured samples and is blocked on explicit data-use terms, local integrity/schema validation, and baseline reproduction.
**Rust selection kernel:** [PR #7568](https://github.com/Luminous-Dynamics/symthaea/pull/7568) adds four target-blind batch baselines, SHA-bound selector receipts, and a separate [post-outcome metrics evaluator](https://github.com/Luminous-Dynamics/symthaea/blob/research/nano-policy-kernel/crates/domains/symthaea-nano-policy/src/metrics.rs) with its [ledger schema](https://github.com/Luminous-Dynamics/symthaea/blob/research/nano-policy-kernel/crates/domains/symthaea-nano-policy/schema/outcome-ledger.schema.json) and [metrics report schema](https://github.com/Luminous-Dynamics/symthaea/blob/research/nano-policy-kernel/crates/domains/symthaea-nano-policy/schema/benchmark-metrics-report.schema.json). The evaluator recomputes success-at-budget, normalized success AUC, failures/repeats/cost and paired group-bootstrap comparisons from frozen post-run outcomes; it never exposes those outcomes to the selector. Predictive uncertainty, expected improvement, and expected-value-of-information remain future arms tracked in [NANO-005](https://github.com/Luminous-Dynamics/symthaea/issues/7570); the cost-normalized geometry heuristic is not Bayesian information gain. Tests are authored but not yet verified as passing.

## 1. Core principle: no universal nanotechnology accuracy score

An optical spectrum, energy above hull, permeability/selectivity, sensor detection limit, nanoparticle-size measurement, biological endpoint and synthesis yield are different observables. They require different units, reference methods, error costs and baseline models.

Use two layers:
1. **Universal evidence gates** that every scientific claim must pass.
2. **Domain-specific outcome metrics** with the appropriate physics, reference data, operating conditions and acceptance thresholds.

A failed hard gate cannot be offset by a high mean score or a weighted overall index. Do not create a single "nano intelligence score" that hides scientific differences or risks.

## 2. Universal gates (apply to all projects)

### G0 — Source, rights and provenance

Before scoring any scientific outcome:
- freeze exact repository commit, source data release, individual artifact digests, transformation code/config, checkpoint/model digest and license/attribution/use decision;
- specify target definition, units, conditions, sample identity, measurement method and allowed input fields;
- distinguish measured, simulated, predicted, derived and synthetic data;
- account for every record as valid, duplicate/replicate, ambiguous, excluded (with reason), or failed;
- preserve raw receipts and preprocessing lineage; never report a provider-reported digest as a locally computed digest.

**Acceptance:** zero unaccounted records; zero unresolved rights blockers for the planned use; all source/model identities known. Unknown remains "unknown", not pass.

### G1 — Leakage-resistant evaluation

- Freeze splits before model/selector tuning; hold out groups at the scientific unit that could leak (e.g. structure family, slab/bulk parent, trajectory, wafer/substrate, batch, device, patient/organism cohort or instrument).
- Keep all duplicate/replicate records and correlated samples in the same group unless the intended claim explicitly concerns replicate-level generalization.
- Prevent target labels, test membership, post-outcome fields and future observations from reaching the candidate-selection policy.
- Include at least one negative control appropriate to the task (label shuffle, hidden-label canary, family-leak canary, random-selection control, or corrupted feature control).

**Acceptance:** no known leakage; all required controls pass. A metric produced with failed leakage controls is invalid regardless of its value.

### G2 — Executability and independent reproducibility

- Pin an immutable source revision and dependency lock; record runtime/compiler, OS, hardware, deterministic flags, batch size, precision, resource caps and fallbacks.
- Every claimed scored run has a run receipt with artifact identities, config, inputs, raw predictions/trajectories, metric implementation and outputs.
- A checker that does not trust the selector's metric values independently regenerates reported metrics from saved outputs.

**Target:** 100% of reported scored runs must have reconstructible environment/provenance and independently regenerated metrics. Failed reproductions remain visible in the denominator. Queued, skipped, or timed-out jobs never count as a scientific pass.

Why this gate matters: a 2026 MatBench reproducibility study reported only 2 of 28 submitted environments installed and passed import checks from original dependency metadata, while 13 of 28 completed a selected task after remediation. See https://cname.oaepublish.com/articles/aiagent.2026.31

### G3 — Fair comparison and uncertainty

A recent critical review of materials-science active learning emphasizes that acquisition results depend on candidate-pool construction, surrogate/preprocessing choices, implicit sampling bias and the chosen metric; random is a necessary baseline but does not establish a fair comparison by itself. See Nair & Foppa (2026), *A critical examination of active learning workflows in materials science*, https://doi.org/10.1039/D6DD00081A. Accordingly, compare against the strongest eligible task baseline (e.g. Expected Improvement and, where applicable, Add-GP-UCB/CAMEO), preserve the starting design and all attempts, and report full regret-versus-budget curves rather than only acceleration/enhancement factors.

- Include a simple baseline (constant/mean where applicable, nearest neighbour/descriptors, random search or conventional method) plus the strongest eligible published baseline that can actually be reproduced.
- Give all arms the same task definition, data access, compute/oracle budget, allowed information and stop criteria.
- Predeclare one primary metric, practical margin, sample/episode count or power rationale, aggregation, confidence interval procedure, exclusion rule and seed policy.
- Report effect sizes and uncertainty, not only the winning point estimate. Report all failed/invalid runs and all preregistered metrics.

**Pilot policy-effect target:** where the task supports a budgeted selection policy, aim for either **at least +5 percentage points in a prespecified success-at-budget metric** or **at least 20% lower median best-achieved task error/cost** than the strongest eligible baseline, with a 95% paired/bootstrap interval excluding zero/no improvement on held-out groups. These are proposed go/no-go thresholds, not literature-established universal standards. Preflight/power analysis can justify a more stringent or feasible domain-specific threshold before the holdout is exposed; never weaken it after seeing outcomes.

### G4 — Applicability, uncertainty and abstention

- Use group-aware out-of-domain tests and meaningful distribution shifts.
- If probabilistic intervals are claimed, evaluate empirical coverage on held-out groups and calibration/interval width jointly.
- Report performance/coverage curves when the system can abstain; don't hide difficult cases by silently excluding them.

**Initial target:** for a claimed nominal 90% prediction interval, observed grouped holdout coverage should be 85–95% with useful interval width and no critical subgroup collapse. This is a provisional screening criterion; sample size and confidence intervals still matter. If the interval is heuristic rather than calibrated, label it as such.

### G5 — Physical and safety claim boundary

- Simulation accuracy is evaluated against a numerical/reference calculation; physical performance requires calibrated measurements and specimen/process lineage.
- Safety, toxicity, exposure, environmental fate, manufacturability and economic viability are distinct claims requiring their own evidence.
- Absence of hazard, process, replication or exposure data is not evidence of safety or feasibility.
- No autonomous synthesis, biological application, environmental release, medical use or laboratory actuation follows from a computational benchmark pass.

OECD nanomaterial measurement guidance emphasizes sample preparation, physical-chemical characterization and protocol details for data comparability; its 2025 Safe-and-Sustainable-by-Design report recommends integrating risk and sustainability questions at early design stages:
- https://doi.org/10.1787/70c84148-en
- https://doi.org/10.1787/e411a4b7-en

## 3. Domain-specific benchmark map

The thresholds below indicate **what to measure and how to set acceptance**, not a claim that one number is universally appropriate for every material.

| Domain | Primary outcome metrics | Required baseline / split | Initial acceptance logic |
|---|---|---|---|
| Atomistic energies / interatomic potentials | Energy MAE/RMSE (eV or eV/atom), force MAE (eV/Å), stress error where appropriate; stability/relaxation success; cost per valid query | Task-matched reference; hold out structure/composition/surface/trajectory families; exact units and DFT settings | Reproduce task-matched published baseline within preregistered tolerance, then demonstrate held-out improvement with uncertainty. Never substitute energy error for catalytic activity. |
| OC22 IS2RE / catalyst surfaces | Relaxed total-energy error on official task; label-efficiency curve; split/pair reconciliation; cost and failure rate | Official source reader/split; SID/trajectory/parent-surface grouping; leakage-free reference | Admission and pair-canonicalization pass first; independent baseline reproduction before selector comparison. No claim until exact pairs/splits are verified. |
| Nanophotonic metasurfaces | Top-k pair retrieval; spectral MSE across wavelength/polarization; Success@K where MSE is below a frozen target; robust success under geometry perturbations; cost per successful design | MOCLIP or conventional library/adjoint/forward-solver baseline; held-out geometry families and properly mapped official split | For the MOCLIP-compatible benchmark, paper-reported context is ~75/95/98% Top-1/5/10 and 95% of inverse-design predictions with MSE <1e-2 at probe sets >10k. Independently reproduce only under compatible split/protocol. Test Symthaea using +5 pp Success@K or >=20% lower median best MSE at equal budget. |
| 2D materials / heterostructures | Task-specific property MAE/RMSE and uncertainty; rank correlation/top-k discovery; coverage across compositions/stacking/strain; compute cost | MatBench or exact database task; group holdout by parent/material family; separate computed labels from experiments | Reproduce the chosen task baseline and demonstrate a preregistered improvement on family-held-out data; no transfer from one property to another. MatBench offers 13 standardized property tasks and fixed evaluation protocols: https://docs.materialsproject.org/services/ml-and-ai-applications/matbench |
| Optical cavities / photonic crystals / plasmonics | Spectral peak/linewidth/phase/efficiency error as appropriate; forward-solver convergence; robust success under fabrication tolerances | Full-wave electromagnetic solver and standard inverse optimizer; geometry/fabrication-aware holdout | Optimize and report the actual target observable; independently re-simulate finalists with a grid/mesh convergence check. Do not apply MOCLIP numbers outside its metasurface task. |
| Nanopores / membranes / separation | Permeability/flux, selectivity, energy per separated quantity, fouling and regeneration performance | Validated transport simulation plus relevant-condition experiments; hold out material families and solution conditions | Predeclare minimum viable selectivity/flux plus durability constraints; compare Pareto coverage and cost at equal budget. No single generic accuracy target. |
| Nanosensors / biosensors | Sensitivity, specificity/precision-recall, detection limit, false-positive rate, response time, drift, calibration and instrument robustness | Blank, positive, interferent and matrix controls; instrument/batch/site holdouts | Meet predeclared application-specific false-positive and detection limits; calibrate on an external or held-out source; report drift and calibration failure. |
| Nano-bio interfaces / nanomedicine | Endpoint-specific error and calibration; formulation stability; release kinetics; biological assay variability; adverse/negative outcomes | Expert-reviewed, assay/route/dose-aware data; hold out source studies, batches and relevant biological groups | Retrospective research or triage only until qualified validation; no clinical, toxicity or safety claims from descriptor similarity. |
| DNA/colloidal self-assembly and molecular machines | Assembly yield, defect fraction, state-transition rates, cycle completion, directionality and energy/input coupling | Time-resolved trajectories and independent microscopy/biophysical measurements; hold out sequences/protocols | Predict on held-out sequences/process conditions; report failures and kinetics, not only end-state structure. |
| Nanocomposites/coatings/construction additives | Strength/modulus, barrier/corrosion measures, aging/fatigue retention, dispersion and process variability | Standardized tests and process-aware splits; hold out batches/facilities/protocols | Meet a declared practical engineering improvement with uncertainty; measure stability/aging and transfer across batches/facilities. |
| **Fe–Co–Ni measured thin-film active-learning replay** | Simple regret and queries-to-target for Kerr rotation (mrad) and coercivity (mT), reported separately; best-so-far curve, diversity and cost | NIST REMI 921-sample historical pool; fixed candidate IDs/features; composition-only as primary arm; repeated seeds; compare random, Expected Improvement, Add-GP-UCB and CAMEO if eligible | Source-reported anchor: for Kerr, the paper reaches within 0.1% of maximum within 5% of the pool and Add-GP-UCB in ~11 samples; coercivity is harder. Proposed pilot target: ≥10% lower median low-budget simple regret than random with seed-bootstrap uncertainty, and no material degradation on the other property. Rights and local MAT validation are hard gates; this is offline replay, not a new experiment. See [NANO-007](https://github.com/Luminous-Dynamics/symthaea/issues/7644). |
| Nanomanufacturing / scale-up | Yield, coefficient of variation, throughput, defect rate, energy/material use, reproducibility across batches | Process/protocol versioning, control charts and batch/facility holdouts | Improve yield or reduce variation at fixed quality/cost; confirm with repeat batches before scaling claims. |
| Nano-characterization / metrology | Bias, MAE/median absolute error versus traceable reference, precision, repeatability/reproducibility, limits of agreement and calibration drift | Calibrated standards/annotations, cross-instrument/operator holdouts, frozen preprocessing | Meet instrument/task uncertainty requirements; quantify uncertainty and failures. Image-segmentation IoU alone does not establish measurement accuracy. |
| Environmental remediation / lifecycle | Removal/selectivity, regeneration cycles, energy, secondary waste, leaching/by-products, fate/exposure and life-cycle indicators | Relevant matrices and operating conditions, reference analytical methods, material-specific risk evidence | Satisfy predeclared treatment performance and lifecycle/risk constraints together; do not count removal while ignoring transformation products or spent media. |

## 4. MEEP-gated power-splitter benchmark: budget audit before reuse

A particularly actionable public research benchmark is [nanophotonics-inverse-design](https://github.com/pberlizov/nanophotonics-inverse-design), reviewed at immutable commit `f2475999c0ad24f8b5ae2f26982c0ad19b257236`. Its frozen `phase0_v1` target is split ratio 0.5 at 1550 nm, with in-spec defined as `abs(split_ratio_upper - 0.5) <= 0.05`. It retains documented negative results and compares search policies under configured budgets B=30, 50 and 100.

The detailed results document reports a five-replicate study and, at configured B=100, these **source-reported** summaries:

| Policy | In-spec count, mean ± SD | Best absolute split error, mean ± SD |
|---|---:|---:|
| `surrogate_rank` | 15.0 ± 3.2 | 0.0042 ± 0.0031 |
| `hierarchical_35` | 14.8 ± 3.5 | 0.0052 ± 0.0029 |
| `random_perturb` | 12.8 ± 1.8 | 0.0058 ± 0.0061 |
| `sigma_meep` | 10.6 ± 4.8 | 0.0074 ± 0.0070 |

The study's interpretation explicitly reports an overlap/tie between the surrogate and `hierarchical_35` for in-spec count; it does **not** show that the surrogate dominates every baseline at every budget. A robust new selector should compare against the strongest eligible baseline, not only a weak uncertainty-only arm.

**Two implementation discrepancies block using these figures as a verified equal-MEEP-call reference without further audit:**

1. The current README describes a **six-seed** B=100 pilot, while `docs/SIM_BUDGET_REPLICATION_RESULTS.md` says five replicates. The published summary numbers are similar, but per-run receipts/raw artifacts must resolve the provenance. See [README](https://github.com/pberlizov/nanophotonics-inverse-design/blob/f2475999c0ad24f8b5ae2f26982c0ad19b257236/README.md#L199) and [detailed results](https://github.com/pberlizov/nanophotonics-inverse-design/blob/f2475999c0ad24f8b5ae2f26982c0ad19b257236/docs/SIM_BUDGET_REPLICATION_RESULTS.md).
2. The pinned `scripts/run_sim_budget_study.py` skips DRC-rejected proposals before calling `meep_one` in the random and sigma paths (lines 82 and 125), while `src/nano_inv/sim_budget.py` counts only rows with finite split values in `n_meep` (lines 56 and 83). Therefore proposal-loop iterations, actual MEEP invocations, failed/non-finite simulations, and reported finite rows are not demonstrably the same denominator from these summary artifacts alone. This does not prove the research result is invalid; it means an equal-actual-call comparison needs call-level instrumentation or raw ledgers. Audit tracked by [NANO-006](https://github.com/Luminous-Dynamics/symthaea/issues/7586).

The repository's [recipe-sensitivity audit](https://github.com/pberlizov/nanophotonics-inverse-design/blob/f2475999c0ad24f8b5ae2f26982c0ad19b257236/docs/RECIPE_SENSITIVITY.md) also reports substantial disagreement between the legacy `phase0_v1` split at resolutions 25 and 50 for the same mask, and keeps the historical labels at r25 separate from a mesh-stability recipe. Do not mix results across these recipe identities.

### Proposed confirmatory protocol for this task

- Freeze exact source, configuration, candidate-generator/surrogate digests and `phase0_v1` resolution 25 before any scored run.
- Define the primary budget as **100 actual MEEP invocations per policy/seed**, with a separately reported and capped proposal-attempt count. Failed solver invocations consume a MEEP-call budget; heuristic rejection is a proposal attempt but not a solver call.
- Preserve a ledger of every proposal: generated, DRC-rejected, duplicate, accepted, MEEP-started, MEEP-failed/non-finite, finite-output, and in-spec. Count exact unique geometries separately.
- Reproduce all applicable baselines with equal actual MEEP-call counts, and resolve the five/six-seed discrepancy before treating prior means as a baseline. If raw receipts are unavailable, describe numbers only as source-reported pilot results.
- Run at least 20 paired seeds for a confirmatory comparison, or document a prospective power analysis before revealing held-out outcomes. Keep candidate source, baseline checkpoints and recipes frozen across arms.
- Report in-spec count / 100 calls, best absolute split error, median/best-so-far error curve, insertion loss separately, failed/non-finite solver calls, DRC proposal rejection rate, unique-design rate, wall time and cost.
- A candidate acceptance target for further work is **at least 18 in-spec designs per 100 actual calls** or **at least 20% lower best-error metric than the reproduced strongest baseline**, with paired/seed-aware 95% uncertainty intervals excluding no improvement. These are proposed go/no-go targets, not published standards. Finalize the primary endpoint/effect-size and sample size before holdout evaluation and do not switch endpoints afterwards.
- Run mesh-stability and fabrication-robustness as separate gates. A split-ratio success under one frozen 2D TE simulation is not foundry DRC, calibrated insertion loss, fabricated yield, or a general photonic-device qualification.

The target-blind Rust selection kernel in [PR #7568](https://github.com/Luminous-Dynamics/symthaea/pull/7568) only selects candidate IDs; a downstream runner records actual oracle outcomes in a separate ledger, and [the metrics evaluator](https://github.com/Luminous-Dynamics/symthaea/blob/research/nano-policy-kernel/crates/domains/symthaea-nano-policy/src/metrics.rs) computes outcome metrics from that completed ledger. Never conflate selected IDs with completed simulations or pass the outcome ledger back to the selector. Comparison intervals require at least five declared independent groups and remain conditional on the validity of those group definitions and the preregistered sampling plan. Its cost-aware geometric coverage arm is not expected information gain, and predictive/uncertainty-guided arms require separate validation.

## 5. MOCLIP metasurface inverse design: a separate worked example

The MOCLIP paper provides a concrete numerical starting point for the measured metasurface task:
- Dataset: 466,537 unique geometry–spectrum pairs after quality filtering; official split of 373,230 train, 46,654 validation and 46,653 test.
- Retrieval: approximately 75% Top-1, 95% Top-5, 98% Top-10.
- For probe libraries greater than 10,000 candidates, 95% of inverse-design predictions had spectral MSE below 1e-2 under the paper's definition.
- The paper reports full-wave validation where, in their particular solver and geometry, a 5 nm grid had spectral MSE 1e-4 against a 2.5 nm reference, around one order of magnitude lower than their measurement error.

Primary source: https://www.nature.com/articles/s41467-026-76714-x

Do not treat those as universal optical constants or guaranteed future results. The released model checkpoint was trained only on its specified training split. The public 10k random subset does not, from the examined release metadata alone, certify the official split membership of each item. Match exact identities to the split or obtain proper test artifacts before reproduction claims.

### Proposed Symthaea policy challenge

Choose a single primary measure before the held-out test:
- **Success@K:** best candidate after K probes has spectral MSE under 1e-2 with constraints satisfied; or
- **best-so-far error curve:** median best spectral MSE by candidate/solver budget.

Compare against random valid candidate, spectral nearest-neighbour/library search, and the applicable MOCLIP/adjoint baseline, plus a Symthaea ablation and negative control. Target +5 percentage points Success@K or >=20% lower median best-so-far MSE, with a paired/cluster-aware 95% interval excluding no improvement, at equal budgets. Include fabrication-robustness perturbations as a separate metric, not folded silently into the same score.

## 6. Separate immediate discovery from learning the response space

A useful additional paradigm shift is to optimize not only which candidate looks most promising *now*, but also which measurement will most improve the model used for later decisions. A 2026 materials-discovery study reports that deliberately sampling both high and low extrema of a response distribution improved generalization versus uniform or one-sided sampling in its tested benchmarks and materials datasets: [Wu et al., *Data Distribution Matters*](https://doi.org/10.1016/j.actphy.2026.100291). This is a task-specific result to reproduce, not a guarantee that extreme sampling always wins.

Keep at least four policies distinct:
- **Immediate exploitation:** choose candidates predicted closest to the application target.
- **Uncertainty acquisition:** query where a suitable, validated estimator predicts high uncertainty.
- **Diversity/coverage:** sample descriptor or family regions that are underrepresented.
- **Two-sided response coverage:** query candidates from both high and low pre-query predicted-response tails to improve future model coverage/generalization.

This fourth objective is not identical to maximizing immediate Success@K. Report full held-out prediction error/uncertainty by response quantile and candidate family *separately* from discovery-success curves. The current Rust kernel [PR #7568](https://github.com/Luminous-Dynamics/symthaea/pull/7568) deliberately excludes predictions and labels from its v1 request contract; do not smuggle them into descriptor fields. A two-sided policy belongs in a separately versioned sequential loop after the task-specific predictor is frozen, with each label revealed only after the candidate is selected.

## 7. How to set numerical targets responsibly

1. **Match the scientific task.** Use reported results only when target, input, split, units, oracle, and metric implementation match.
2. **Establish the baseline first.** If the paper's run is unreproducible, report that; do not make an arbitrary target stand in for the missing baseline.
3. **Define practical significance.** A 0.5% improvement could be meaningful in one use case and irrelevant in another; define the decision impact and costs.
4. **Power and uncertainty.** Preflight query counts, variance and sample size; bootstrap by independent system/family, not correlated rows alone.
5. **Keep hard constraints hard.** No accuracy gain compensates for leakage, invalid candidates, failed rights review, unsafe data use, or missing critical controls.
6. **Use staged claims.** Source verified → execution verified → baseline reproduced → policy improvement demonstrated → out-of-domain robustness examined → independently measured physical performance. Each is a distinct evidence state.

## 8. Immediate implementation order

1. Finish NANO-001 data admission and exact-head Rust verifier validation; do not wait idle for unrelated CI, but never relabel skipped/queued as passed.
2. For NANO-003, pin and inspect MADE's exact source/configs, then run plumbing-only smoke and reproduce baselines before a Symthaea arm.
3. For NANO-004, resolve MOCLIP subset/split membership, terms and local bytes; run isolated inference smoke; measure VRAM/runtime before designing a larger experiment.
4. Compare nanophotonics with 2D materials and metrology on NANO-002's readiness criteria, then select one follow-on domain. Avoid concurrent uncontrolled implementations.
5. Use a Rust-first Symthaea selector adapter only after the baseline tasks and interface semantics are frozen. Scientific tools may retain isolated research runtimes; never add an unqualified Python model stack to production Symthaea by accident.

## 9. Active Rust selector baseline

A candidate-policy prototype is under active development in [PR #7628](https://github.com/Luminous-Dynamics/symthaea/pull/7628), tracked by [NANO-005](https://github.com/Luminous-Dynamics/symthaea/issues/7627). It provides three policies: seeded random control; Euclidean descriptor maximin with incremental nearest-selected-distance updates; and `hdc_maximin`, which uses Symthaea core's actual 16,384-bit `BinaryHV`, a deterministic 256-level scalar codebook, dimension-role binding and bundling. The contract includes strict input/output schemas, exact-request BLAKE3 receipts, smoke fixtures, 21 active unit tests plus one ignored manual scaling-profile harness, and a targeted Rust workflow.

This is still a research-only selector, not an atomistic predictor, optical solver or scientific result. HDC distance is a descriptor representation score, not physical similarity or calibrated uncertainty. For nanophotonic work, only permitted geometry-derived features may reach the selector before acquiring labels; spectra, test membership and unacquired outcomes remain forbidden. The benchmark should compare random, Euclidean maximin, HDC maximin and task-native baselines under equal label/oracle budgets, using the same downstream model trained from scratch on acquired labels and a sealed, group-aware test set. Rust compilation/tests remain pending the targeted workflow and must not be reported as passing before exact-head evidence exists.

## 10. Success definition for the program

The program succeeds when it can **reproducibly show where Symthaea improves evidence-gathering decisions and where it does not**, across carefully chosen scientific tasks, while respecting distinct physical models, data rights, risk constraints, and uncertainty. Discovering a physically useful nanotechnology is a later, more demanding claim requiring independent experimental evidence.
