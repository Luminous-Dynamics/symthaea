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
