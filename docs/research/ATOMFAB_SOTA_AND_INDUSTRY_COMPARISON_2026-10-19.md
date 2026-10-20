# AtomFab SOTA and Industry Comparison

**Reviewed:** 2026-10-20  
**Status:** Competitive research plan; no AtomFab or Symtropy result is claimed to beat a public leaderboard.  
**Purpose:** Define what a fair state-of-the-art comparison means for the Symthaea + Symtropy atomic-scale manufacturing effort.

## Executive view

AtomFab should not try to beat every pretrained interatomic potential at its own task. Current specialist models and large datasets are extremely capable. The better opportunity is to improve the **end-to-end decision loop**: find candidates that survive stronger tests, make fewer high-cost evaluations, avoid false-feasible process decisions, and carry evidence from atomic prediction to a manufacturable process and (later) physical measurement.

That is a different, testable claim from “our model is more accurate.” It must be evaluated against both specialist scientific baselines and simpler orchestration baselines.

## Competitive landscape as of 2026-10-19

| Reference / system | What it establishes | What AtomFab should do with it | Important caveat |
|---|---|---|---|
| **OMat24 / current pretrained MLIPs** | OMat24 comprises more than 110 million DFT calculations. Its paper reports leading Matbench Discovery performance, F1 above 0.9 for stability prediction, and around 20 meV/atom formation-energy accuracy for models trained on it. | Treat strong pretrained potentials as the starting baseline; reuse them rather than retraining a universal potential from scratch. | Dataset/model scale is a major industry/research advantage. AtomFab must not imply its orchestration layer replaces this capability. |
| **Matbench Discovery** | Public evaluation framework for high-throughput crystal-stability prediction and related atomic-property tasks, including standardized F1 and task-specific metrics. | Run the exact published protocol, retain the official split and reference values, and report a compatible score before claiming leaderboard competitiveness. | A good crystal-stability score is not an overall materials-manufacturing score or evidence of experimental validity. |
| **UniFFBench / MinX** | A 2026 benchmark uses 1,500+ mineral systems across 85 elements and experimental references. It reports a reality gap between computational benchmarks and experimental property predictions. | Prioritize a held-out experimental-validation lane; test whether model confidence predicts actual error and whether the workflow abstains when it should. | Experimental reference values can have their own uncertainty, conditions, and coverage gaps; comparison must account for them. |
| **MS25** | Diverse materials-relevant systems including MgO surfaces, liquid water, zeolites, Pt surface reactions, high-entropy alloys, and disordered Zr oxides; evaluates energy/force/stress and derived observables. | Use a pinned subset as the first reproducible task-level scientific benchmark. Preserve grouped material-family splits. | Its reference method and exact split must be respected; do not compare unlike DFT settings as equal ground truth. |
| **2026 EOS + minima-hopping benchmark** | Evaluates near-equilibrium equation-of-state properties and far-from-equilibrium structural searches, showing search efficiency can diverge from structural fidelity. | Benchmark structure recovery, energetic ordering, EOS curves, and off-equilibrium robustness separately from MAE/RMSE. | The published study focuses on elemental systems; it is not proof of broad multicomponent generalization. |
| **2026 supported-nanoparticle search benchmark / UniAlCu** | Explicitly examines how energy accuracy can decouple from structural exploration for supported nanoparticles. | Add a separate, preregistered lane that measures unique low-energy structures discovered, diversity, duplicate minima, matched-structure energy error, and cost per validated structure. | The chosen particle/support family and its DFT protocol define the scope; this is not universal catalytic generalization. |
| **Open Catalyst Project OC20/OC22** | Open catalyst datasets, baseline models, and relaxation benchmarks at large scale; OC20/OC22 together include about 1.3 million relaxations and more than 260 million DFT calculations. | Add a catalyst route as a later domain-specific lane if AtomFab selects catalytic materials. Use task-specific relaxation/energy metrics and matched reference configurations. | Do not compare the OC22 energy-within-threshold metrics directly with Matbench F1 or MS25 errors; tasks differ. Public leaderboard submissions may be dated. |
| **Matbench property-prediction suite** | Standardized materials-property tasks covering different dataset sizes and target properties. | Use only the tasks that match a declared AtomFab objective, as sanity checks for data splits and downstream property predictors. | It is not an atom-manipulation or process-planning benchmark. |
| **Self-driving laboratory benchmark literature** | Uses acceleration factor (AF) and enhancement factor (EF) to compare experiments required and objective improvement. Published results vary greatly by search-space properties and protocol. | Compare campaign curves against random, diversity-only, uncertainty-only, and cost-aware baselines with the same objective and query cost. | An “AF of 6” from another domain is not a comparable guarantee for AtomFab; campaign difficulty and target definitions matter. |
| **Multi-fidelity / active-learning strategy benchmarks** | A 2025 benchmark compared 17 acquisition strategies across nine materials datasets and found early-stage performance advantages for uncertainty and hybrid diversity strategies, with strategy gaps narrowing as data grew. A 2026 supported-nanoparticle study separately showed model accuracy and structural exploration can diverge. | Compare fixed cheapest-first, random-eligible, uncertainty-only, calibrated value-of-information and Symthaea routing under identical domains, budgets, targets and seeds; report the realized value-estimate calibration and full cost. | The 2025 work is on small-sample property regression, not atomistic global optimization. Acquisition performance must be revalidated in the actual AtomFab task; don't transport a ranking across domains. |
| **NIST manufacturing digital-twin guidance** | Identifies interoperability, verification/validation/uncertainty quantification (VVUQ), cybersecurity, and trustworthy traceable data as persistent challenges. | Treat Symtropy's first scenario as a surrogate workcell, then calibrate and validate individual model channels before any physical claim. | A visual simulation or deterministic replay alone does not establish twin credibility. |

### Primary references

1. Barros-Luque et al. (2026), “The Open Materials 2024 (OMat24) inorganic materials dataset and models,” *Nature Computational Science*. [https://doi.org/10.1038/s43588-026-00996-w](https://doi.org/10.1038/s43588-026-00996-w)
2. Riebesell et al. (2025), “A framework to evaluate machine learning crystal stability predictions,” *Nature Machine Intelligence*. [https://doi.org/10.1038/s42256-025-01055-1](https://doi.org/10.1038/s42256-025-01055-1). Official guide: [Materials Project Matbench Discovery](https://docs.materialsproject.org/services/ml-and-ai-applications/matbench-discovery).
3. Mannan et al. (2026), “UniFFBench: evaluating universal machine learning force fields against experimental measurements,” *Nature Computational Science*. [https://doi.org/10.1038/s43588-026-01019-4](https://doi.org/10.1038/s43588-026-01019-4).
4. Maxson et al. (2025), “MS25: Materials Science-Focused Benchmark Data Set for Machine Learning Interatomic Potentials,” *Journal of Chemical Information and Modeling*. [https://doi.org/10.1021/acs.jcim.5c01262](https://doi.org/10.1021/acs.jcim.5c01262).
5. Tahmasbi et al. (2026), “Benchmarking universal machine learning interatomic potentials on elemental systems,” *npj Computational Materials*. [https://doi.org/10.1038/s41524-026-02251-2](https://doi.org/10.1038/s41524-026-02251-2).
6. Xu et al. (2026), “Benchmarking universal machine learning interatomic potentials for supported nanoparticles: decoupling energy accuracy from structural exploration,” *npj Computational Materials*. [https://doi.org/10.1038/s41524-026-02043-8](https://doi.org/10.1038/s41524-026-02043-8).
7. Open Catalyst Project, [OC20/OC22 datasets and baselines](https://opencatalystproject.org/).
8. Adesiji et al. (2026), “Benchmarking self-driving labs,” *Digital Discovery*. [https://doi.org/10.1039/D5DD00337G](https://doi.org/10.1039/D5DD00337G).
9. Frechette, Shao, and Pease (2026), *Digital Twins Workshops Summary Report*, NISTIR 8620. [https://doi.org/10.6028/NIST.IR.8620](https://doi.org/10.6028/NIST.IR.8620).
10. Bi et al. (2025), “A comprehensive benchmark of active learning strategies with AutoML for small-sample regression in materials science,” *Scientific Reports* 15, 37167. [https://doi.org/10.1038/s41598-025-24613-4](https://doi.org/10.1038/s41598-025-24613-4).
11. “Probing out-of-distribution generalization in machine learning for materials” (2025), *Communications Materials* 6, 9. [https://doi.org/10.1038/s43246-024-00731-w](https://doi.org/10.1038/s43246-024-00731-w).

## Concrete pretrained-model baseline pool

The current public Matbench Discovery model catalog includes model families/variants such as **MatRIS-10M-OAM, eSEN-30M-OAM, Orb v3, MatterSim v1.5M, MACE-MP-0, and OMat-trained equivariant models**. These are candidates for a frozen baseline pool, not an asserted ranking across all materials tasks. The catalog is maintained by the [Matbench Discovery project](https://github.com/janosh/matbench-discovery/blob/main/matbench_discovery/enums.py).

For each selected MS25 subset, first inspect supported elements, compatible structures and reference methods, checkpoint/model revision, licensing and use conditions, numerical precision, hardware requirements, and known validity domain. Then choose the strongest compatible model that can be evaluated reproducibly. Don't select the baseline after seeing which model AtomFab happens to beat. If the best public model cannot be run under the available constraints, report the capability gap and use the strongest reproducible compatible baseline—do not describe the result as a global SOTA win.

## Fair comparison protocol

Every competitive result needs a frozen experiment manifest recording:

- Dataset name, version, license, source URLs, exact file list, hashes, and reference-method details.
- Train/validation/test identity; structure or material-family groups; data-overlap screening; and any temporal or out-of-domain split.
- Solver and model names/versions, checkpoint hashes, model license, precision mode, cutoff/radius, relaxation settings, convergence criteria, and all effective defaults.
- Hardware, wall-clock time, peak memory, energy or compute budget where measurable, reference-calculation count, and preprocessing cost.
- Random seed, acquisition rule, number and cost of evaluations, stopping condition, failure count, and all runs—not only the best run.
- A provenance class for every value: reference, prediction, simulation, measurement, or independent reproduction.
- A typed campaign record containing dataset/model/evaluator/shared-environment/policy-configuration digests, target and budget definition digests, policy ID, seed, evaluation count, elapsed time, total cost and cost unit, terminal state, objective, and evidence references. Completed claims without evidence and failed runs without a reason are invalid records.

**No leakage by convenience:** keep related geometries and configurations from the same parent material in the same split; audit training-data overlap where feasible; report in-distribution and family-held-out results separately; and reserve a prospective/experimental holdout if later lab access becomes available.

## Campaign artifact contract

The campaign schema is [campaign-record-v1.schema.json](../../crates/domains/symthaea-atomfab/schemas/campaign-record-v1.schema.json); the Rust implementation is [campaign.rs](../../crates/domains/symthaea-atomfab/src/campaign.rs). Paired-policy helpers reject results with different model, dataset, evaluator, shared execution environment, target, budget contract, objective, seed, or incompatible cost units. Separate gates exist for target-reaching acceleration, equal-evaluation comparisons, and equal-cost comparisons.

This is important because an apparent improvement can be manufactured unintentionally by changing the solver, model, data split, cost assumptions, or target between baselines. The campaign record is intended to make those mismatches explicit before aggregation. JSON Schema validates shape; the Rust semantic validator enforces cross-field rules such as evaluations used not exceeding the budget. These tests are authored but have not been executed in this environment.

## How to beat SOTA honestly

### Separate energy accuracy from structural exploration

The supported-nanoparticle benchmark published in April 2026 makes a second distinction alongside the EOS/minima-hopping results: an interatomic potential may score well on energy prediction for structures it sees, yet the search driven by that potential may fail to discover the right structures. Therefore, every structure-search campaign should publish **two scoreboards**:

1. **Conditional scientific accuracy:** for matched candidate/reference structures, report energy error, energetic-ranking agreement and relevant property error.
2. **Search discovery quality:** at equal reference-calculation budget, report the number of unique reference-validated low-energy structures found, structural diversity, duplicate-minimum rate, and cost per discovered structure. Also report pairwise energetic-ordering agreement and the new **normalized finite-prefix top-weighted ranking overlap** implemented in the AtomFab Rust kernel. The latter uses preregistered persistence (p=0.94), requires unique IDs and equal-length rankings, and assumes stable canonical identity for structures across policies. Its finite-prefix score is explicitly RBO-style—not a claim to implement every extrapolated RBO variant. The implementation uses rank-position thresholds and a cumulative prefix pass with expected linear work in ranking length, rather than repeatedly intersecting growing sets. Its optimized score is also compared in unit tests against a brute-force prefix-intersection reference over multiple ranking pairs.

For actual search policies, the paired metric now reports candidate-set Jaccard separately and calculates RBO-style ranking overlap only on the intersection of shared canonical structure IDs. This prevents a policy's unique discoveries from being silently counted as ranking mistakes; discovery quality remains measured by recovered reference structures, unique validated minima, diversity, and cost per discovery. If the intersection is empty, the helper returns candidate-set Jaccard = 0 with ranking overlap represented as `None`; it does not invent a numeric ranking score. Deduplication tolerances and structure-identity matching must be frozen before evaluation.

Never infer one scoreboard from the other. A model that predicts energies accurately on a selected set can still guide exploration poorly; a broad search that finds many minima can still rank them incorrectly. The new `supported-nanoparticle-search` registry lane is separate from the elemental EOS lane so material family and search protocol remain explicit. Its ranking metrics include the finite-prefix overlap score, with duplicate-ID, length, and persistence validation.

### Track A — Scientific baseline parity

1. Select one tractable MS25 subset and reproduce a published reference baseline.
2. Add an equilibrium/off-equilibrium split and derived observables.
3. Compare at least one strong, available pretrained potential and one simpler baseline under matched settings.
4. Do not claim “SOTA” unless the exact task, split, reference method, metric, and protocol match a current public benchmark/leaderboard.

### Track B — Decision-layer advantage

Hold the atomistic model constant and compare these policies:

- random selection;
- diversity-only selection;
- uncertainty-only selection;
- cost-aware active learning;
- Symthaea planning/evidence policy;
- Symthaea with the relevant memory/history mechanism ablated.

Measure **best validated objective at equal budget**, area under the learning curve, queries to a predeclared target, AF and EF, decision regret, calibration, and OOD abstention. This isolates what Symthaea contributes over the solver and conventional optimizer.

### Track C — Manufacturing feasibility

Use the contract in [Symtropy PR #2140](https://github.com/Luminous-Dynamics/symtropy/pull/2140) to test whether each process proposal satisfies workcell constraints under nominal and adversarial scenarios. Compare against a simple conventional planner. Report false-feasible accepts, missed-feasible plans, fault-detection performance, stop latency, recovery, and replay fidelity separately.

The gate to a physically calibrated twin is not a prettier simulation; it is prospective residuals, measured uncertainty coverage, and a defined applicability domain for each channel.

### Track D — End-to-end experimental value

Only after the previous gates pass, conduct a supervised experimental campaign in a suitably equipped laboratory. Compare to a conventional scientist/planner or a declared automation baseline on the same task, using the same success definition and recording failed syntheses, rework, characterization, and total cost. A genuine new-material claim requires independent characterization and a reproducible procedure.

## Proposed numerical gates

These targets must be preregistered, and thresholds should be chosen per task before test results are seen.

| Gate | Initial target |
|---|---|
| Baseline reproduction | Reproduce the published reference result within its declared tolerance before modifying the method. |
| Core scientific metrics | Report energy/force errors and task-level observables; no aggregate-only claim. |
| Off-equilibrium correctness | Correctly report structure recovery and energetic ordering; measure EOS deviations separately. |
| Experimental prediction | Report normalized/absolute errors and interval coverage against experimental references, including reference uncertainty and out-of-domain cases. |
| Selective prediction | Report the full risk-coverage curve and stepwise AURC; compare uncertainty-ranked abstention against random-score and shuffled-score controls at identical retained coverage. |
| Abstention accounting | Report accepted coverage, error among accepted cases, false-feasible count/rate, and missed-feasible cases together; abstaining on everything cannot count as success. |
| Active learning | Stretch target: median AF ≥ 2 across at least 20 fixed-seed campaigns, with 95% bootstrap lower bound > 1, while objective quality at equal budget is not worse than the strongest matched baseline. |
| Decision quality | Lower decision regret and/or higher validated objective at equal budget versus the best conventional acquisition baseline; publish non-wins as well. |
| False-feasible safety | Zero accepted hard-constraint violations on a frozen adversarial suite; if no candidate is accepted, false-feasible rate is undefined, not zero. |
| Reliability | Report failure/recovery distributions and per-fault critical misses; no average score may hide a critical miss. |
| Twin credibility | Calibrate each measured channel against a reference; publish residual, uncertainty coverage, drift sensitivity, and validity envelope. |
| Independent reproducibility | A clean checkout with pinned artifacts reconstructs every accepted score; missing artifacts or mismatched digests fail closed. |

A 2× AF is a deliberately achievable research stretch target, not a claim to beat the best literature results. The self-driving-lab literature reports a median AF around 6 across surveyed studies, but protocols and search spaces vary substantially; direct comparison requires reproducing a matched task. Where justified, a stronger later goal is to outperform the best matched baseline's AF/EF while preserving physical validity and reliability.

## Selective-risk benchmark protocol

AtomFab now includes a pure Rust `summarize_risk_coverage` kernel. For each candidate prediction, record the scalar prediction, reference value, and a non-negative uncertainty score. Samples are sorted by uncertainty; ties enter as a group so input ordering cannot change the curve. Report a stepwise AURC in the supplied error units, the full set of curve points, and coverage at each preregistered operational threshold. Freeze uncertainty construction and thresholds on validation data only. In addition to the complete AURC, report risk at the nearest attainable 50%, 75%, 90%, and 100% coverage operating points. Never split samples tied at the same uncertainty merely to hit nominal coverage; report the realized coverage.

The negative controls are deliberately simple: shuffle uncertainty scores across the same held-out samples and compare with a random score under the same retained coverage. A selective model only demonstrates useful uncertainty if low-uncertainty subsets reduce held-out risk relative to these controls while retaining meaningful coverage. Report false-feasible and missed-feasible counts separately; the scalar-error curve does not replace physical feasibility validation.

## What we must not claim

- Not “SOTA” because Symthaea's architecture is novel, has more code, or uses a higher-dimensional representation.
- Not a scientific improvement because an internal benchmark score rises on a random split.
- Not experimental validity because a pretrained potential is stable or a simulator replays deterministically.
- Not a manufacturing win because fewer evaluations were used if the resulting material is worse, less manufacturable, or less reproducible.
- Not an AtomFab win unless the same fixed solver/baseline ablations show the Symthaea decision layer adds value.
- Not atom-by-atom printing until a physical process has independently demonstrated specified placement fidelity, yield, throughput, and defect rates.

## Recommended near-term objective

The first defensible achievement is **a reproducible benchmark report proving whether AtomFab's decision layer improves candidate-selection quality per unit cost compared with strong conventional policies**, alongside a separate Symtropy workcell benchmark for false-feasible planning decisions.

This creates two potentially valuable research contributions even before laboratory hardware access: scientific decision efficiency and verifiable process feasibility. If either adds no measurable value, narrow or redesign that part rather than preserve a claim by changing the metric.
