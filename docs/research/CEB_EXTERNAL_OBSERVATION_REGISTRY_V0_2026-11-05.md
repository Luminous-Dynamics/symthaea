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

The external-observation registry now contains 17 external sources, 19 comparator records and 41 atomic observations. All remain contextual references; no direct CEB superiority claim is enabled.
