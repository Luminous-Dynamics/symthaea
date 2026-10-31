# AR-5B4 external benchmark map — 2026-10-31

**Purpose:** establish a defensible comparison programme for direction allocation, scientific search, autonomous experimentation and repository-level optimization. This is a research map, not a claim that Symthaea currently beats any external system.

## Executive conclusion

Do not aim at one aggregate leaderboard. Current external evaluations measure different capabilities and expose different failure modes. AR-5B4 should compare along four separate axes:

1. **Discovery:** can the system select candidates that are actually good enough to test?
2. **Learning:** after receiving measurements, can it rank and predict unseen candidates?
3. **Evidence integrity:** can it demonstrate that the experiments ran, numbers are traceable, and claims are supported?
4. **Transfer and utility:** does the mechanism generalize to held-out landscapes, workflows, and real repository/scientific problems?

Autonomous Science Bench's explicit discovery and learning gates are particularly important: finding a high-quality candidate and predicting the quality of unmeasured candidates are distinct capabilities. SCI-AGENT-EXAM similarly combines executable artifacts with written outputs and exposes a gap between top average score and highly reliable runs.

## External benchmark comparison

| Benchmark / system | Published scope and useful signal | What AR-5B should adopt | Important limitation |
|---|---|---|---|
| [Autonomous Science Bench 0.1](https://www.autonomous-science-bench.ai/) | Six pilot campaigns across biology, chemistry and materials. Its tasks budget experiment selection and measurement, then require both a discovery gate and a learning/ranking gate. The current leaderboard reports GPT-6 Astra/Codex 72.2% and Opus 5.5/Claude Code 66.7%; the site explicitly cautions that three trials per task do not make small configuration gaps meaningful. | Two independently scored endpoints: measured-design quality and ranking quality on unmeasured designs. Report cost and elapsed time alongside resolution rate. Preserve all trials and per-task results. | Version 0.1 is a six-task pilot, not a statistically stable general-purpose leaderboard. |
| [SCI-AGENT-EXAM](https://arxiv.org/abs/2610.32441) | 120 expert-curated tasks across 11 scientific fields and seven research-workflow task types. Requires executable and textual artifacts. Uses deterministic programmatic checks plus calibrated expert-validated model judging. Reported top overall score is 80.9, yet only 26% of top-system runs exceed the 0.95 success threshold. | Add trajectory stability, artifact verification, report/evidence consistency and a high-confidence success-rate endpoint. A strong average with a low pass-at-0.95 rate is not a reliable scientific system. | Composite scores and LLM-judged text are not directly interchangeable with deterministic search metrics; preserve raw task-level evidence. |
| [SciAgentArena](https://sciagentarena.github.io/) | Approximately 200 interactive, stepwise-verified tasks across five scientific domains; agent-agnostic execution and evaluation. Reports uneven domain generalization and difficulty with genuinely novel insights and sustained self-directed exploration. | Use as a cross-domain workflow transfer target; measure stepwise correctness, stability, cost and novelty separately. | Broad task mix differs from AR-5B allocator microbenchmarks. |
| [FIRE-Bench](https://proceedings.mlr.press/v306/wang26d.html) | Scientific finding rediscovery through agent-designed and executed experiments; claim-level evidence matters, not merely a plausible report. | Require claim-to-artifact traceability, reproducible experiment receipts, negative controls and independently checked evidence. | Rediscovery is not identical to open-ended discovery or optimizer efficiency. |
| [Into the Unknown](https://arxiv.org/abs/2610.19159) | Open research problems with domain-scientist rubrics. The reported evaluation found none of 135 submissions met all conditions and 114 failed integrity checks. | Make unsupported claims, fabricated runs, missing receipts and contradicted metrics first-class negative controls. | A small open-problem benchmark is not a direct policy-allocation leaderboard. |
| [EurekaBench](https://arxiv.org/abs/2610.00492) | Long-horizon, expert-verified scientific tasks that separate scientific constraints, prediction and genuine insight. | Separate “predicts accurately” from “discovers a useful mechanism”; require held-out validation and expert review for mechanism claims. | Expert scoring and scientific interpretation require rubric calibration. |
| [FormulaCode](https://proceedings.mlr.press/v306/sehgal26a.html) | Repository-level code optimization on 957 performance bottlenecks, with many real workloads per task and correctness-constrained, multi-objective performance metrics. | Add an end-to-end code/engineering track: triage, patch, correctness, latency, memory, energy and workload robustness; compare on equal measured resource budgets. | The published dataset is heavily grounded in scientific Python repositories; Rust/Nix systems need separate language/build-system strata. |
| [CodeEvolve](https://aclanthology.org/2026.findings-emnlp.1345/) | Open-source island-based evolutionary search with inspiration crossover, meta-prompting, CVT-MAP-Elites and model ensembles. Reports matching/surpassing reported AlphaEvolve results on 5/9 tasks and outperforming OpenEvolve/ShinkaEvolve on 6/9 under its matched conditions. | Include island/novelty/crossover/archive baselines in the proposal-generation track, with ablations and exact backbone/tool/compute matching. | These are task-suite-specific results; they do not imply superiority on scientific discovery or AR-5B policy allocation. |
| [Coral Discovery](https://arxiv.org/abs/2610.33281) | Asynchronous agentic discovery with shared memory, intervention and direction-aware allocation. Its reported 6/1/2 win/tie/loss summary uses best-of-three runs per condition. | Reproduce exact task configurations and run every trial; measure means, intervals, cost and time, not only best-of-three. Keep its four-seat advisory semantics exact. | Tiny margins and best-of-three selection make its headline unsuitable as a population-level effect estimate. |

## The benchmark architecture we should build

### Track A — mechanism-matched policy evaluation

All policies receive the same pre-decision offered choices, evaluator labels, task recipe, evaluator/verifier build, worker schedule and normalized budget model. This isolates allocation/search policy.

Policy set:
- score-only / winner-takes-all;
- fixed equal-share / round-robin;
- seeded random;
- global UCB;
- discounted UCB;
- Thompson sampling;
- Symthaea DirectionBoard;
- exact Coral scheduler equations and four-seat execution adapter;
- constrained-seat Coral variants only as separately labelled ablations;
- MAP-Elites/CVT-MAP-Elites only when behavior descriptors have a defensible interpretation;
- oracle only as a non-deployable synthetic upper bound.

### Track B — best-practical-system evaluation

Each system may use its native proposer, memory, orchestration, workers and tools, while the campaign holds total measured resources and evaluation tasks constant. This compares whole systems rather than scheduler mechanisms. Do not combine Track A and Track B into one leaderboard.

Include ShinkaEvolve/CodeEvolve-style proposal-generation systems here. Report full model, agent, harness, toolchain, prompt/context, concurrency and environment identities.

### Track C — scientific evidence and reliability

Use FIRE-Bench, Into the Unknown, EurekaBench and SCI-AGENT-EXAM-style checks:
- every claimed result maps to a source artifact and derivation;
- executable artifacts are tested independently;
- negative controls and label-leakage tests are mandatory;
- verifier and solver have separate access boundaries;
- claims contradicted by receipts fail closed;
- expert review is required for claims of scientific mechanism or novelty.

### Track D — real engineering optimization

Use FormulaCode-like tasks plus Symthaea-native Rust/Nix workloads:
- correctness is a hard constraint;
- measure throughput/latency, CPU/GPU time, peak and integrated memory, energy and tool/token cost;
- run multiple workloads and perturbations;
- report Pareto frontiers, not a single weighted score alone;
- include build-system integrity, reproducibility and rollback as explicit dimensions.

## Metrics and acceptance targets

| Dimension | Required reporting |
|---|---|
| Discovery | Target-hit probability at fixed spend; cost to first verified improvement; best verified objective at budget exhaustion; simple regret against a declared feasible optimum/lower bound. |
| Learning | Held-out ranking correlation / NDCG / regret; calibration; prediction error on unmeasured candidates; discovery and learning gates reported separately. |
| Evidence | Receipt completeness; claim-to-artifact coverage; independent replay pass rate; false-NEW, duplicate, rejection and verifier-disagreement rates; negative-control false-positive rate. |
| Robustness | Worst-family regret; performance under ranking reversal, deceptive proxies, correlated directions, heteroscedastic rewards, null opportunities and verifier errors. |
| Diversity | Unique verifier-admitted directions; meaningful descriptor coverage; allocation entropy/HHI; starvation and duplicate rates. |
| Efficiency | Full resource vector; equal realized normalized spend; elapsed latency distinct from worker-time; energy and memory measured from instrumented telemetry. |
| Transfer | Held-out landscape retention; held-out task-family success; cross-domain workflow performance; independent replication. |
| Reliability | Mean and median plus dispersion, 95% intervals, per-task results, failure rate and fraction of runs exceeding a stringent success threshold (e.g. 0.95). |

Confirmatory studies should use at least 100 paired independent landscape instances/templates per family as an initial floor, with sample size justified by pilot variance and a preregistered minimum practical effect. Multiple seeds from one payoff template are repeated measures, not independent landscapes. Use landscape-cluster uncertainty and predeclare the primary endpoint; correct secondary endpoint families for multiplicity.

## What would count as “better than SOTA”

A valid superiority claim requires all of the following:
1. exact external baseline version, code, prompts, tool configuration and task conditions reproduced;
2. matched total measured resources, not merely equal candidate counts;
3. enough independent landscapes/tasks and trials to estimate uncertainty;
4. superior primary endpoint with a preregistered practically meaningful margin;
5. no material regression on worst-case families, reliability, evidence integrity or held-out transfer;
6. independently replayable artifacts and a separate verifier;
7. public negative results and ablations, including the oracle ceiling.

A result on the nine synthetic AR-5B landscapes (eight deterministic surfaces plus one seed-matched stochastic heteroscedastic surface) is a useful mechanism-control result only. The stochastic surface uses common random numbers keyed by campaign seed × arm × within-arm evaluation index, so policies receive identical potential outcomes. Simple regret is computed from expected reward, not a lucky observed draw. This is not evidence of a scientific-agent SOTA claim. The immediate goal is to make AR-5B a better experimental instrument than a self-authored score table: same offers, exact action semantics, executable budgets, sealed labels, full receipts, independent inference and transfer.


## AR-5B synthetic benchmark claim ceiling

The current scalar suite has eight deterministic surfaces plus one seed-matched stochastic heteroscedastic surface. The seeded stochastic surface uses common random numbers keyed by campaign seed, direction ID and within-direction evaluation index, so policies receive identical potential outcomes. Simple regret uses evaluator-side expected reward, not a lucky observation. These are mechanism controls, not a scientific-agent SOTA result.

## Benchmark update: October 31, 2026

Two practical lessons sharpen the competitive plan:

- CodeEvolve's October 2026 EMNLP Findings paper reports matching or exceeding reported AlphaEvolve results on five of nine tasks and outperforming OpenEvolve/ShinkaEvolve on six of nine under its matched conditions. It combines island evolution, inspiration crossover, depth refinement, CVT-MAP-Elites and an LLM ensemble. This is a real end-to-end competitor, not a scheduler-only baseline; replicate its components as ablations and compare the complete system only in the best-practical-system track. The task-specific results do not establish general scientific-discovery superiority.
- FormulaCode (ICML 2026) evaluates 957 repository performance bottlenecks with expert patches and roughly 264.6 workloads per task, emphasizing correctness-constrained, multi-objective optimization. We should borrow its lifecycle framing—triage, diagnosis, patch, correctness, workload performance—but keep Rust/Nix repository tasks in a separate language/build-system stratum rather than pretending the Python-heavy benchmark is directly representative.
- OpenScience's October 1, 2026 benchmark update explicitly says its evaluation setup had comparison problems and that results would be rerun after harness improvements. Treat those public rankings as provisional until the corrected protocol and rerun are published; this is a useful example of why every benchmark release needs versioned conditions and a correction log.

Sources: CodeEvolve, https://aclanthology.org/2026.findings-emnlp.1345/; FormulaCode, https://proceedings.mlr.press/v306/sehgal26a.html; OpenScience benchmark update, https://www.openscience.sh/benchmark.

## Multi-objective optimization standards and new AR-5B5 implementation

The new `pareto_objectives.rs` module is distinct from the existing `resource_pareto.rs`: the latter selects actions under hard multi-resource limits, while the new module evaluates candidate outcome vectors, feasibility, nondominance and dominated volume.

The external method baseline should follow COCO's biobjective assessment design: evaluate the nondominated archive with hypervolume, measure function-evaluation/runtime cost to predeclared target values, and preserve the full archive plus versioned reference values so analyses can be recalculated. COCO explicitly warns, through its workflow, that a single HV number is insufficient without normalized objective space, a defined reference point and target-attainment behavior.

For algorithmic coverage use ZDT (2-objective sanity), DTLZ (scalable objectives and front geometry), WFG (nonseparable/deceptive/mixed fronts), and constrained DTLZ feasibility cases. Compare NSGA-II, NSGA-III, MOEA/D, SMS-EMOA and RVEA alongside the Symthaea direction allocator. The v0.1 exact-hypervolume implementation is intentionally capped at 2–4 objectives; higher-dimensional runs must use an explicitly versioned approximation or separate indicator set.

Do not use HV alone. An indicator-reliability study found that HV, IGD, IGD+ and additive epsilon can rank real-world algorithms differently, and even HV rankings can change with the chosen reference point. AR-5B5 therefore records reference sensitivity and should add IGD+, additive epsilon, attainment curves and uncertainty over independent instances before confirmatory claims.

Sources:
- COCO biobjective performance assessment: https://numbbo.github.io/coco-doc/bbob-biobj/perf-assessment/
- Reliability of indicator-based comparison results: https://scholars.cityu.edu.hk/en/publications/reliability-of-indicator-based-comparison-results-of-evolutionary/
- DTLZ/WFG/ZCAT/LSMOP benchmark families: https://academic.oup.com/jcde/article/13/2/1/8407630
- PlatEMO and standard multi-objective algorithm/test-suite coverage: https://link.springer.com/article/10.1007/s42979-020-00265-1

Current implementation boundary: the Pareto evaluator is a batch analysis primitive and accepts a digest reference for each evidence artifact. It does not yet adapt EventTape receipts into per-candidate objective vectors or retrieve and independently verify the candidate artifacts. Those are explicit follow-on tasks in issue #9469.

The current AR-5B5 Pareto primitive now provides three separate indicators: exact hypervolume (2–4 objectives), IGD+, and additive epsilon. IGD+ and epsilon are evaluated in transformed maximization-utility space against a frozen, unique, mutually nondominated reference front. The combined indicator receipt binds the plan, approximation-front digest and canonical reference-front digest. This is not a composite score and does not replace the required per-objective and worst-family reports.

Still outstanding: EventTape resource-to-objective adaptation, artifact-byte resolution, R2/attainment analysis, uncertainty-aware dominance, and inference across independent multi-objective instances.

The first executable benchmark functions are now present as deterministic generators: ZDT1 for two-objective minimization and DTLZ2 for three objectives, each with an analytic reference-front sampler. A cross-module integration test feeds the ZDT1 front through Pareto analysis and the digest-bound IGD+/epsilon receipt. This is a fixture/integration check only; it does not implement NSGA-II/NSGA-III/MOEA/D/SMS-EMOA/RVEA, WFG, constrained DTLZ or a comparative performance result.

## New AR-5B5 execution and verifier controls — 2026-10-31

The EventTape-to-Pareto adapter now turns strict, fully metered action tapes into candidate-level objective vectors. It preserves failed retries in candidate costs, reports direction-proposal overhead separately, reconciles both to whole-campaign resource totals, and can deterministically reconstruct a report from the source tape and frozen metric mapping. Reported uncertainty can itself be an objective; missing uncertainty on any verified repeat fails closed when that metric is selected.

The semantic-alias benchmark now keeps policy-visible proposal surface forms separate from evaluator-only semantic identity and utility. It measures pairwise duplicate precision/recall, false merges, false splits, false-new admissions, false novelty rejections, and synthetic downstream discovery regret under a fixed canonical-slot budget. Its evaluator-only adjudication digest is included in the final receipt. The initial fixture is deliberately synthetic and small: it validates the control logic, not natural-language semantic understanding or real verifier accuracy.

### PlatEMO industry comparison

The July 2026 PlatEMO 4.16 release notes list 360 algorithms and 630 problems, including constrained, large-scale, expensive and multimodal multi-objective methods. This is a reminder not to make unsupported “state of the art” claims from two analytic functions and a hand-picked baseline. Pin the exact PlatEMO version, algorithm identifiers, problem identifiers, parameter files and termination budgets. For a Rust-first research stack, port or reimplement selected baselines with reference cross-checks; keep a separate pinned MATLAB/PlatEMO reference export only as an external reproducibility oracle rather than a production dependency.

Sources:
- PlatEMO 4.16 release notes (July 2026): https://github.com/BIMK/PlatEMO/blob/master/Doc/releasenote.md
- COCO biobjective target-attainment protocol: https://numbbo.github.io/coco-doc/bbob-biobj/perf-assessment/
- Reliability study showing HV/IGD+/epsilon disagreement and HV reference sensitivity: https://scholars.cityu.edu.hk/en/publications/reliability-of-indicator-based-comparison-results-of-evolutionary/
- 2024 constrained multi-objective optimization under uncertainty, including adaptive Pareto dominance and Pareto-optimality probability: https://onlinelibrary.wiley.com/doi/10.1002/nme.7403
- September 2026 robust multi-objective optimality under data uncertainty: https://link.springer.com/article/10.1007/s10479-026-07410-8

This expands the immediate benchmark plan: go beyond deterministic front convergence to (1) expensive/limited evaluation budgets, (2) noisy or epistemically uncertain objective estimates, (3) constrained feasibility, (4) multimodal decision-space fronts, and (5) semantic alias/verifier failure injection. Report the applicable problem class, not one pooled score.

## Latest AR-5B5 implementation tranche — 2026-10-31

Three practical pieces now make this a runnable research programme rather than only a list of objectives:

1. **Measured-resource adapter:** `event_tape_pareto.rs` consumes a strict EventTape and builds candidate-level quality, verification, uncertainty and resource objectives. Failed retries remain charged; proposal-generation overhead is reported separately and reconciled with whole-campaign totals. A deterministic verifier rebuilds the report from the tape and frozen mapping.
2. **Verifier negative controls:** `semantic_alias_benchmark.rs` separates policy-visible surface forms from an evaluator-only truth key. It measures pairwise false merges/splits and downstream utility regret. Seeded corruption schedules freeze the seed, category rates and proposal-level corruptions into a separate digest, and that digest is carried into the result receipt.
3. **Reference evolutionary baseline:** `pareto_nsga2.rs` implements deterministic NSGA-II for ZDT1 and three-objective DTLZ2, with seeded variation, binary tournament, nondominated sorting, crowding distance, exact evaluation budget, run replay verification and conversion to the same Pareto indicator path as Symthaea.

### Updated competitive baseline scope

PlatEMO 4.16 (released July 19, 2026) lists 360 algorithms and 630 problems, including new large-scale, expensive constrained and multimodal multi-objective coverage. This is much broader than our current ZDT1/DTLZ2 + NSGA-II starting point. Our next priorities should be cross-validation against a pinned PlatEMO 4.16 NSGA-II configuration, then MOEA/D and SMS-EMOA; add NSGA-III and RVEA for higher objective counts. Constrained DTLZ, WFG, expensive/noisy and multimodal tests should be added as separate problem classes, not pooled into one leaderboard.

COCO's performance-assessment protocol supports reporting time/evaluations to reach multiple fixed indicator targets, instead of comparing only final hypervolume. It also preserves nondominated archives and versions reference values for future recalculation. We should copy this discipline: publish runlength-to-target curves and every archive point, then report HV, IGD+ and additive epsilon together.

Sources:
- PlatEMO 4.16 release notes and current algorithm/problem inventory: https://github.com/BIMK/PlatEMO/releases/tag/v4.16
- COCO biobjective target/indicator performance assessment: https://numbbo.github.io/coco-doc/bbob-biobj/perf-assessment/
- NSGA-II algorithm lifecycle and elitist parent/offspring selection: https://link.springer.com/article/10.1007/s42979-020-00265-1

**Claim ceiling:** the new Rust NSGA-II implementation is not yet cross-checked against PlatEMO or original reference code; the synthetic alias fixture is not yet a large semantic benchmark; and all new Rust tests remain unexecuted. No algorithm is reported as outperforming SOTA.

### Second evolutionary reference: MOEA/D-DE

Added deterministic Rust `pareto_moead.rs` for ZDT1 and three-objective DTLZ2. It uses uniform 2D or simplex-lattice 3D weights, nearest-weight neighborhoods, neighborhood/global mating, DE/rand/1/bin, Tchebycheff scalarization, ideal-point updates and bounded neighborhood replacement. The report preserves weight-vector/subproblem order, exact objective-evaluation count and a replayable digest; `analyze_moead_report` converts the verified archive into the shared Pareto indicators.

The active baseline set is therefore now NSGA-II plus a compact MOEA/D-DE variant on common ZDT1/DTLZ2 fixtures. Neither implementation has been compiled, run or cross-checked against PlatEMO 4.16 yet. They are reference implementations in development, not evidence that either method converges well or that Symthaea beats it.

The two baseline implementations guard population size at 2,048 for their current quadratic sorting/neighborhood operations. This is an implementation safety limit, not an algorithmic limitation.

## Third algorithm baseline and common comparison harness — 2026-10-31

Added the compact Rust SMS-EMOA reference variant, using the same ZDT1/DTLZ2 functions and exact hypervolume primitive. It performs one-offspring iterations and removes the lowest exclusive-HV-contribution point from the worst nondominated front against a frozen utility-space reference point. Its particular tournament/operator choices are documented and must be cross-checked against a pinned external implementation before claiming algorithm equivalence.

Added `pareto_baseline_comparison.rs`, which replay-verifies NSGA-II, MOEA/D-DE and SMS-EMOA, then refuses to compare unless problem ID, seed, objective-evaluation count, objective specs/scale anchors, hypervolume reference point and reference Pareto front match exactly. It reports all three indicators (HV, IGD+, additive epsilon), archive/population size and the SHA-256s of the run, analysis and comparison receipts. A deterministic verifier reconstructs the comparison. The integration fixture gives all three methods exactly 60 objective evaluations on ZDT1 and checks the mismatch gates.

### Critical reference-point rule

The baseline fixtures now use a hypervolume reference point below the normalized feasible utility range (for example -0.1 in every objective). A zero coordinate would give no dominated box volume to a Pareto point lying exactly on that normalized boundary, which can remove endpoints from SMS-EMOA's contribution accounting. The reference point and normalization are shared and receipt-bound; it must never be chosen post hoc to favor a method.

### Claim ceiling

Three algorithms can now be compared on identical synthetic budgets by one shared deterministic analysis harness. This is not yet a benchmark leaderboard: runs have not been executed, methods have not been cross-checked against PlatEMO 4.16, and no wall-clock, energy or hardware telemetry comparison is included. We have not established any algorithm wins.

## SMS-EMOA and matched three-algorithm comparison — 2026-10-31

The evolutionary baseline set now includes three distinct methods:
- NSGA-II (rank/crowding and elitist population selection);
- MOEA/D-DE (weight-vector decomposition and Tchebycheff scalarization);
- SMS-EMOA (exclusive hypervolume contribution in the worst Pareto front).

Added `pareto_smsemoa.rs`, `pareto_baseline_comparison.rs`, and the corresponding baseline protocol documents. The shared comparison entry point replay-verifies every optimizer, then requires the same benchmark problem, seed, objective-evaluation count, objective definitions/scale anchors, hypervolume reference point and reference Pareto front. It reports HV, IGD+ and additive epsilon through the same analysis implementation and binds all run/analysis/comparison digests.

The integration fixture compares all three algorithms on ZDT1 with 10 decision dimensions and exactly 60 objective evaluations each. It rejects a valid 61-evaluation run, a different seed, duplicate method IDs and incompatible objective/reference contracts. This validates protocol behavior only; it is not an optimization result.

### Benchmarking methodology: fixed-budget and fixed-target

COCO explicitly distinguishes:
- **fixed-budget:** compare achieved quality after a common number of objective evaluations;
- **fixed-target:** compare objective evaluations/runtime needed to reach one or more predeclared quality-indicator targets.

The programme should report both. Target-attainment curves and runlength distributions are more informative than one final HV value because they show whether one method improves early, steadily, or only near budget exhaustion. COCO recommends retaining every nondominated solution in the run archive so indicator values can be recalculated as the reference front/reference values improve. It also versions its HV reference values in the benchmark data.

Sources:
- PlatEMO 4.16 release notes (July 19, 2026; 360 algorithms, 630 problems): https://github.com/BIMK/PlatEMO/releases/tag/v4.16
- COCO biobjective target-based assessment and retained nondominated archives: https://numbbo.github.io/coco-doc/bbob-biobj/perf-assessment/
- COCO general fixed-budget/fixed-target performance assessment: https://numbbo.github.io/coco-doc/perf-assessment/
- COCO official benchmark data archive, including bbob-biobj and constrained/noisy suites: https://coco-platform.org/data-archive/index.html

### Updated claim ceiling

The three methods are compact in-repository reference variants on ZDT1/DTLZ2. No results have been compiled, run, cross-checked against PlatEMO 4.16, or measured for CPU/GPU/energy. Do not call the harness an industry leaderboard yet. Next work is external implementation parity, larger test families, multiple seeds, target-attainment evaluation and real resource telemetry.
