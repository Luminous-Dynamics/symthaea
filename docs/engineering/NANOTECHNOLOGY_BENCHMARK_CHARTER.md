# Nanotechnology Benchmark Charter: Metrics, Thresholds and Evidence Gates

**Status:** Proposed acceptance policy for future nanotechnology benchmark work; no benchmark is claimed to have passed.  
**Date:** 2026-10-16  
**Domain atlas:** [NANOTECHNOLOGY_DOMAIN_ATLAS.md](./NANOTECHNOLOGY_DOMAIN_ATLAS.md)  
**Nanophotonics benchmark:** [NANOPHOTONICS_MOCLIP_BENCHMARK_SCORECARD.md](./NANOPHOTONICS_MOCLIP_BENCHMARK_SCORECARD.md)  
**MADE benchmark:** [NANOTECHNOLOGY_CLOSED_LOOP_BENCHMARK_PLAN.md](./NANOTECHNOLOGY_CLOSED_LOOP_BENCHMARK_PLAN.md)  
**OC22 program:** [NANOMATERIALS_RESEARCH_PROGRAM.md](./NANOMATERIALS_RESEARCH_PROGRAM.md)

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
| Nanomanufacturing / scale-up | Yield, coefficient of variation, throughput, defect rate, energy/material use, reproducibility across batches | Process/protocol versioning, control charts and batch/facility holdouts | Improve yield or reduce variation at fixed quality/cost; confirm with repeat batches before scaling claims. |
| Nano-characterization / metrology | Bias, MAE/median absolute error versus traceable reference, precision, repeatability/reproducibility, limits of agreement and calibration drift | Calibrated standards/annotations, cross-instrument/operator holdouts, frozen preprocessing | Meet instrument/task uncertainty requirements; quantify uncertainty and failures. Image-segmentation IoU alone does not establish measurement accuracy. |
| Environmental remediation / lifecycle | Removal/selectivity, regeneration cycles, energy, secondary waste, leaching/by-products, fate/exposure and life-cycle indicators | Relevant matrices and operating conditions, reference analytical methods, material-specific risk evidence | Satisfy predeclared treatment performance and lifecycle/risk constraints together; do not count removal while ignoring transformation products or spent media. |

## 4. Nanophotonics targets: a worked example

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

## 5. How to set numerical targets responsibly

1. **Match the scientific task.** Use reported results only when target, input, split, units, oracle, and metric implementation match.
2. **Establish the baseline first.** If the paper's run is unreproducible, report that; do not make an arbitrary target stand in for the missing baseline.
3. **Define practical significance.** A 0.5% improvement could be meaningful in one use case and irrelevant in another; define the decision impact and costs.
4. **Power and uncertainty.** Preflight query counts, variance and sample size; bootstrap by independent system/family, not correlated rows alone.
5. **Keep hard constraints hard.** No accuracy gain compensates for leakage, invalid candidates, failed rights review, unsafe data use, or missing critical controls.
6. **Use staged claims.** Source verified → execution verified → baseline reproduced → policy improvement demonstrated → out-of-domain robustness examined → independently measured physical performance. Each is a distinct evidence state.

## 6. Immediate implementation order

1. Finish NANO-001 data admission and exact-head Rust verifier validation; do not wait idle for unrelated CI, but never relabel skipped/queued as passed.
2. For NANO-003, pin and inspect MADE's exact source/configs, then run plumbing-only smoke and reproduce baselines before a Symthaea arm.
3. For NANO-004, resolve MOCLIP subset/split membership, terms and local bytes; run isolated inference smoke; measure VRAM/runtime before designing a larger experiment.
4. Compare nanophotonics with 2D materials and metrology on NANO-002's readiness criteria, then select one follow-on domain. Avoid concurrent uncontrolled implementations.
5. Use a Rust-first Symthaea selector adapter only after the baseline tasks and interface semantics are frozen. Scientific tools may retain isolated research runtimes; never add an unqualified Python model stack to production Symthaea by accident.

## 7. Success definition for the program

The program succeeds when it can **reproducibly show where Symthaea improves evidence-gathering decisions and where it does not**, across carefully chosen scientific tasks, while respecting distinct physical models, data rights, risk constraints, and uncertainty. Discovering a physically useful nanotechnology is a later, more demanding claim requiring independent experimental evidence.
