# CEB External Observation Registry V0 — 2026-11-05

**Status:** Design-only external reference snapshot. These are published external results and methodological findings, not CEB measurements.
**Machine-readable data:** [`benchmarks/ceb/external-observations.v0.json`](../../benchmarks/ceb/external-observations.v0.json)
**Schema:** [`benchmarks/ceb/external-observations.schema.json`](../../benchmarks/ceb/external-observations.schema.json)
**Validator:** [`scripts/validate_ceb_observations.py`](../../scripts/validate_ceb_observations.py)

## Why observations are separate from comparator prose

The measurement matrix is useful for selecting external comparators, but its `published_reference` text can bundle multiple systems and multiple metrics. This registry normalizes those values into atomic observations. One observation has exactly one metric and one model/harness/configuration. No score is reconstructed by joining the best Recall from one system to the best Context F1 from another, or the best functional score from one security configuration to the best secure score from another.

The registry currently contains:
- **17 external source records** (the parent measurement matrix has 16 registered URLs because it also registers one internal documentation-audit implementation reference).
- **19 comparator records**, including the 15 comparator records present in the previous matrix plus AgentAbstain.
- One observation for each reported metric/configuration pair; metric values, displayed uncertainty, denominators and source locators remain separately inspectable.

## Data semantics

- `sample_denominator` is attached to a benchmark and split, not to a general benchmark name. SWE-Bench Pro V2 Full (642) and HARD (51) remain separate.
- `system_id` binds a metric to one named model, harness and configuration. A metric row cannot contain a list of systems.
- `uncertainty` states whether the source reports an interval. The validator preserves the raw `±` display and only labels an interval 95% when the source explicitly specifies that interpretation (PDB).
- A `less_than` metric is an upper-bound statement, not an exact point estimate. The SWE-fficiency finding remains “below 0.23×”.
- The HiL-Bench 79.67 ± 4.17 value remains `unlabeled`; it is not inferred to be ASK-F1 or Pass@3.
- Agent Security League functional and secure measurements are kept per configuration. Muse Code + Muse Wonder has 91.6% FuncPass and 25.1% SecPass; Claude Code + Claude Fable 5.1 has 86.6% FuncPass and 36.9% SecPass.
- AgentAbstain's paired accuracy is the primary measure. Act-side, abstain-side and conditioned abstention (CAR) scores are separate observations, not substitutes for paired accuracy.

## Validation and negative controls

The offline validator checks:
1. JSON Schema Draft 2020-12 validity and unique identities.
2. Comparator-to-source, source-to-lane and observation-to-comparator linkage.
3. Every source is used and every comparator has observations.
4. Exact task denominators and separation of SWE-Bench Pro V2 Full/HARD.
5. Metric/uncertainty binding, including exact interval denominator.
6. One system per observation and no unregistered composite derivation.
7. HiL unlabeled-metric protection.
8. Security metric/configuration pairing and AgentAbstain metric separation.
9. Explicit rejection of unsupported direct-superiority comparison labels.

The self-tests mutate records in memory only. They do not query external websites, execute candidate code, run benchmark tasks, or qualify any runner.

## Source refresh policy

A refresh is append-and-review, not silent overwrite. Preserve the old observation with its original snapshot date; add a new observation for a new model, benchmark revision, task split, score or uncertainty. A changed benchmark protocol creates a new comparator version. If the original source disappears, retain the URL, retrieval date, archived evidence digest and a clearly marked unavailable state; do not invent replacement values.

## Sources and current date boundary

The source URLs and retrieval date are encoded in the JSON. The key methodological references are:
- SWE-Bench Pro V2: https://labs.scale.com/leaderboard/swe_bench_pro_public_v2
- SWE Atlas: https://labs.scale.com/papers/sweatlas and https://labs.scale.com/leaderboard
- ContextBench: https://contextbench.github.io/
- PDB results: https://huggingface.co/datasets/Precise-Debugging-Benchmarking/PDB-Results
- CRBugBench: https://www.microsoft.com/en-us/research/publication/when-should-a-software-factory-be-lights-off-benchmarking-code-review-agents-on-bug-detection/
- Agent Security League: https://www.endorlabs.com/research/ai-code-security-benchmark
- Rust-SWE-bench: https://conf.researchr.org/details/icse-2026/icse-2026-research-track/67/Evaluating-and-Improving-Automated-Repository-Level-Rust-Issue-Resolution-with-LLM-ba
- SWE-fficiency: https://proceedings.mlr.press/v306/ma26l.html
- Epoch AI Terminal-Bench review: https://epoch.ai/benchmarks/terminal-bench-4/review
- AgentAbstain: https://agentabstain.github.io/

All source dates and metric definitions should be refreshed before every formal replication. The registry does not establish that the current Symthaea system is competitive.


## HiL-Bench normalized observations (2026-11-05)

The combined HiL comparator retains the 300-task population. Two additional comparator entries retain the 150-task SWE and 150-task SQL domain populations separately. Four new atomic observations are recorded:
- 44% highest combined ASK-F1; winning system identity is not stated in the overview.
- 24% combined Pass@3 for Claude Opus 4.6.
- 36% highest SWE-domain blocker recall; winning identity and exact blocker denominator are not stated.
- 61% highest SQL-domain blocker recall; winning identity and exact blocker denominator are not stated.

ASK-F1 is the harmonic mean of Question Precision and Blocker Recall. Pass@3 is task success in at least one of three independent assisted runs. These are distinct measures. The consolidated Muse Wonder 1.0 score of 79.67 ± 4.17 remains unlabeled and cannot be assigned either metric. Source: https://labs.scale.com/leaderboard/hil



## CEB-CONTINUITY adjacent method comparators (2026-11-08)

Two additional sources are registered as continuity-adjacent context. Neither is an equivalent CEB claim-disposition score.

### Incremental Program Analysis in the Wild — ASE 2025

The ASE study curates **4,084 real-world Java program changes across 20 open-source projects** and evaluates two representative incremental program-analysis tools. It reports efficiency gains relative to exhaustive analysis alongside inconsistencies and markedly higher peak memory in the evaluated incremental tools.

**CEB interpretation:** use this to motivate measuring both the correctness of selective revalidation and resource burden (including peak memory). The study does not measure retain/revalidate/invalidate/supersede/unresolved dispositions, and its Java-only scope does not establish transfer to Rust or multi-language workspaces.

Source: https://doi.org/10.1109/ase63991.2025.00194

### DiffCI Open Evidence Study 2026 — change-aware test selection

DiffCI's self-published evidence study replays **2,000 historical commit deltas across 20 repositories**. It reports 29.8% of deltas as offering a genuine selective-test opportunity and 4.2% weighted whole-corpus test reduction versus path selection. For 280 historical CI failures, reported job-level recall is 96.4% for its graph selector versus 70.7% for a changed-path rule.

The limits must remain attached to these figures: the aggregate test-reduction confidence interval crosses zero (-4.8% to +47.9%); the study says the population effect may be null; its 32/32 measurable mutation detections are explicitly not a reliability estimate; it reports no production CI runs were skipped, cancelled or blocked; and only 3 of 7 attempted repositories passed canonical suite-green qualification.

**CEB interpretation:** job-level recall here is historical CI-failure test-selection recall, not claim invalidation recall. Test-count reduction is not revalidation-cost-per-accepted-sequence. Use the study as a contextual comparison for C0 path heuristic vs selective test selection, not as a direct CEB-CONTINUITY leaderboard score.

Source: https://diffci.com/research/diffci-open-evidence-2026

At the 2026-11-07 snapshot, the structured registry contained **51 sources, 56 comparator records and 162 atomic observations**. The 56-row metric correspondence crosswalk is unchanged; the new observations are not mapped as direct equivalents to CEB-CONTINUITY primary metrics.


## Branch and merge-history comparators — November 8, 2026

Two adjacent research sources have been added to the machine-readable registry:

- **ConflictBench** evaluates five software merge tools on Java merge-conflict cases. It is a methodological reference for merge-resolution controls, not an agent benchmark and not a measure of evidence continuity. The exact case denominator is not reported in the abstract summary.
- **Change propagation channels mined from evolution history** reports a dataset of 5,032 project versions across five maintained projects and 40 relationship-channel types. These channels can seed candidate impact hypotheses, but co-change/history association does not prove semantic causation or branch-scoped claim invalidation.

These comparators motivate BHC-S02/BHC-S04 and the requirement to compare first-parent/path heuristics with full-parent history and an independent clean-recomputation oracle. They do not supply directly comparable CEB scores. The existing 56-row metric-correspondence crosswalk remains unchanged.

Sources:
- ConflictBench: https://doi.org/10.1016/j.jss.2024.112084
- Change propagation history mining: https://www.sciencedirect.com/science/article/pii/S0164121223003072

## SWE-Milestone v1.0.2 normalized observations (2026-11-05)

The CEB-CONTINUITY lane now carries an atomic contextual snapshot from the official live leaderboard for the top listed GLM-5.3 + Claude Code row:

- Resolve: 29.75%.
- Precision: 60.30%.
- Recall: 75.07%.

The visible leaderboard row does not expose its task/sequence denominator, aggregation details, or a reported uncertainty interval. Those fields are explicitly not reported in the machine-readable observations. The benchmark site explains that a milestone is fully resolved when every fail-to-pass test passes and no previously-passing test regresses; it describes precision as retained P2P tests and recall as passed F2P tests. These are benchmark-specific measures, not direct CEB evidence-disposition metrics.

The same leaderboard row also displays a 58.75 aggregate Score, $42.76 cost, 886K output tokens, 10.61 wall-clock hours and 2,193 turns. The aggregate score formula and full configuration metadata are not sufficiently explicit in the visible table snapshot, so those figures are not promoted into the atomic metric registry. Do not synthesize a best-of-metric system or compare its cost directly with CEB cost per accepted sequence.

Sources:
- Official leaderboard: https://swe-milestone.com/
- ICML 2026 paper: https://proceedings.mlr.press/v306/deng26d.html

The 2026-11-07 structured external-observation snapshot contained 51 external sources, 56 comparator records and 162 atomic observations. The figures in earlier dated sections above are preserved as historical captures; the current machine-readable snapshot is authoritative for registry totals. All remain contextual references; no direct CEB superiority claim is enabled.

## Executable-verifier and benchmark-integrity frontiers — November 8, 2026

### VerifierBench: two-sided verifier correctness

VerifierBench evaluates whether an agent can generate an executable verifier that accepts valid outcomes and rejects invalid ones. Its published project page reports 222 retained tasks from six agent benchmarks and 4,993 hidden cases: 1,566 PASS and 3,427 FAIL. The 1,344 `Alt.` cases combine cosmetic variations and valid alternatives (a subset of PASS); the 798 attack cases are a subset of FAIL. The page reports 45 source-task exclusions after V4V identified source-material defects.

Keep its experimental populations separate:

- The paper's common pi-harness experiment covers 19 models.
- The project page lists 36 model–harness configurations across the broader evaluation.
- The paper reports normalized task-averaged balanced accuracy from 10.6 to 62.1 across the 19 pi-harness models. This is a range, not a single model score.
- Across those 19 pi-harness models, reported TPR is 21.1–70.2% and TNR is 79.2–96.3%; TPR is below TNR for every model. The project page reports all 36 configurations below the TPR=TNR line.
- For nine models with archived task-level records, verifiers that reject every case in a task account for 87.9–96.0% of false rejections.
- The source reference-verifier audit contains six V4V checks and 1,410 task–operator mutation-test configurations.

**CEB interpretation:** a verifier that rejects every candidate can appear strong on negative cases while accepting no valid solution. Qualification must therefore be two-sided: prove acceptance of the gold implementation and diverse spec-valid alternatives, *and* rejection of real violations and attacks. Positive controls cannot be inferred from compilation, a reference patch, or the agent's own self-tests. Keep the 19-model experiment, 36 configuration overview, per-class denominators and task-level all-reject pathology distinct.

The project page identified the submission as under review and showed no released code/data links when accessed on November 8, 2026. Treat this as a paper/project snapshot, not an independent reproduction or a CEB score.

### Unearned-pass maintenance: benchmark integrity is continuous

The benchmark-maintenance study audits 3,810 passing trajectories across 29 model–benchmark cohorts. Its abstract reports descriptive SWE-Bench Pro V1 unearned-pass rates of 24% for Opus 4.7 and 73% for Fable 5 on matched tasks, with later cohort observations of 11% for Fable 5.1 and 0% for GPT-6 Astra. Exact per-cohort denominators are not stated in the abstract, and the full model/harness configurations are not normalized; these are not universal model reliability rates or a monotonic generation trend.

The study's core methodological finding is that fixing a known exploit is insufficient when the same protected information remains reachable through another route. Its repair examples pair a minimal patch with replay of the exploit and fresh-agent evaluation. This changes the benchmark-maintenance standard: a repaired evaluator or task is not qualified merely because the original exploit string no longer works.

**CEB interpretation:** retain a versioned exploit corpus and re-run it after every benchmark/evaluator change; test alternate access paths (Git history, refs, artifacts, caches, logs, environment, tool outputs and stale workspaces); then test fresh agents for both legitimate solvability and unearned passes. Keep benchmark-integrity rates separate from review accuracy, causal diagnosis, and claim-validity metrics. The existing #10138/#10681 verifier-integrity and V4V work remains the owner; this registry adds external evidence, not a parallel integrity framework.

The registry now contains **54 sources, 59 comparator records and 181 atomic observations**. The 56-row CEB metric-correspondence crosswalk remains unchanged. These additions are contextual/methodological references; no direct CEB superiority claim or new primary metric is enabled.

Sources:
- VerifierBench paper: https://arxiv.org/abs/2610.28679
- VerifierBench project page: https://verifierbench.github.io/
- Unearned-pass maintenance study: https://arxiv.org/abs/2609.34262

