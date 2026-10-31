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
