# RITE-Bench

**Research Integrity, Trust & Efficiency** is a deterministic evaluator contract for research agents. It complements CEDAR-Bench (evidence dependence), METHODE-Bench (method fidelity), and REPAIR-Bench (correction propagation) by making the overall qualification profile measurable without hiding failures inside one aggregate.

## Why another benchmark layer?

A final-answer score cannot distinguish a correct answer supported by independent evidence from a lucky answer backed by duplicated sources. A report-quality score cannot establish that its code executes correctly. Mean task accuracy does not disclose how often a system is near-perfect, how often it abstains, or whether it changes its answer when decisive evidence is removed while remaining stable under placebo changes.

Recent benchmarks are moving toward richer evaluation:
- SCI-AGENT-EXAM evaluates 120 tasks across 11 fields and seven workflow types with both executable and textual artifacts. Its reported leading average was 80.9, while 26.0% of runs crossed a 0.95 success threshold. This is a useful warning against relying on means alone.
- Autonomous Science Bench 0.1 evaluates six real laboratory workflows; its pilot leaderboard reports a 72.2% resolution rate for the leading configuration, with substantial cost and time. This is promising but still a small pilot set.

RITE-Bench does not claim to replace those task suites. It provides a common qualification layer that can be applied to task-level evaluator outputs.

## Scored dimensions

All metrics are emitted separately, with numerator and denominator preserved. Rates with no eligible observations are **undefined**, not zero.

| Dimension | Metric | Contract |
|---|---|---|
| Outcome | Decision accuracy | Exact agreement with the evaluator's decision oracle |
| Abstention | Answer coverage | Fraction not returning InsufficientEvidence |
| Selective reliability | Selective risk | Error rate among non-abstaining decisions |
| Calibration | Brier score | Mean squared error of P(submitted decision is correct) |
| Calibration | 10-bin ECE | Weighted absolute calibration gap, fixed equal-width bins |
| Citations | Citation precision | Citations eligible for and supporting the evaluator's expected decision / all cited sources |
| Independence | Decision-support origin adequacy | Required tasks meeting the oracle's distinct eligible-origin minimum for the expected decision |
| Causal robustness | Counterfactual oracle accuracy | Post-intervention decision matches the oracle after the decisive fact changes |
| Causal robustness | Source-removal oracle accuracy | Post-removal decision matches the oracle, including abstention when support becomes insufficient |
| Causal robustness | Placebo invariance | Post-placebo decision matches the unchanged oracle decision |
| Reproducibility | Replay pass rate | Independent replay, exact contract digest, and exact output-manifest match |
| Resource discipline | Budget compliance | All declared resource caps respected |
| Economics | Cost per correct decision | Total measured spend / correct decisions, reported in integer micro-USD |

The evaluator does **not** emit an overall weighted score. High accuracy cannot compensate for failed replay, poor calibration, weak evidence independence, or budget violations. Consumers should compare systems on a Pareto profile and publish each denominator.

## How RITE composes with the existing benchmark contracts

RITE-Bench is a cross-suite decision/evidence qualification layer, not a replacement for domain-specific benchmark oracles.

| Existing contract | What it qualifies | How to use it beside RITE |
|---|---|---|
| CEDAR-Bench | Whether an answer depends on admissible, independent, claim-specific sources and reacts to evidence counterfactuals/placebos | Supplies claim/source eligibility, provenance-origin groups, and expected intervention outcomes |
| METHODE-Bench | Whether the declared scientific method, dataset, split, preprocessing, components, baselines, metrics, budgets and outputs were actually reproduced | Keep method-fidelity disposition as a separate gate; a correct final answer cannot compensate for an unresolved method deviation |
| REPAIR-Bench | Whether corrections and invalidations propagate through dependent claims, artifacts and evidence graphs | Keep correction-propagation and stale-support rates separately visible; an initially correct result can become invalid after a source correction |
| Artifact manifest and bounded delimited parser | Whether evidence artifacts have stable hashes, bounded parsing, typed locators, and reproducible acquisition provenance | Supplies artifact identity/locator receipts; parsing success is not a truth or citation-eligibility judgment |
| WebResearch acquisition/security contracts | Whether acquisition is bounded, source/claim provenance is recorded, and outbound requests meet URL/DNS/redirect policy | Treat acquisition-policy violations as hard security failures, not a small penalty in the scorecard |

The future top-level report should be a **qualification vector** with one result per contract and one result per RITE axis. It should not average these into an "overall research intelligence" number. An unresolved method, failed replay, forbidden network destination, or unpropagated correction remains visible even when final-answer accuracy is high.

## Record semantics

A TaskRecord is constructed by the benchmark harness and independent evaluator, not by the agent being scored.

- expected_decision is the task oracle's label.
- submitted_decision is the agent's final label. InsufficientEvidence is the abstention action.
- confidence_correct means the agent's stated probability that its submitted decision is correct.
- Citation eligibility and support of the expected decision are evaluator-adjudicated against the exact claim. A source may be relevant but contradict the expected conclusion; it does not count as a correct supporting citation.
- origin_id groups mirrored URLs or re-hosted copies under one underlying origin. Distinct URLs do not imply independent evidence.
- Every intervention carries an explicit post-intervention oracle. Counterfactual and source-removal outputs are scored against that oracle, not merely rewarded for changing. Placebo oracles must preserve the pre-intervention decision.
- A required replay passes only when an independent replay succeeds and both the contract digest and output manifest match exactly.
- Resource limits are explicit. Missing limits are not silently interpreted as unlimited budget observations.

The evaluator rejects empty/duplicate task IDs, invalid confidence, empty citation identities, empty budget declarations, and replay evidence on tasks where replay was not required. A missing required replay is scored as a replay failure. Resource aggregation uses checked integer addition and returns an error on overflow rather than saturating.

## Execution

This crate is a standalone Rust 2024 package with no external dependencies:

~~~sh
cargo fmt --manifest-path tools/rite-bench/Cargo.toml -- --check
cargo test --manifest-path tools/rite-bench/Cargo.toml --locked
cargo clippy --manifest-path tools/rite-bench/Cargo.toml --locked --all-targets -- -D warnings
~~~

## Proposed research targets (aspirational, not achieved results)

These are proposed stretch goals for Symthaea and the Luminous evaluation stack. They are not current measurements, leaderboard entries, or guarantees. A target is only comparable when the benchmark version, model, agent scaffold, tool access, prompts, judge, run budget, and scoring implementation are held constant.

| Evaluation | Current reference point | Proposed target |
|---|---|---|
| SCI-AGENT-EXAM | The October 2026 paper reports a leading mean score of 80.9, while 26.0% of that system's runs exceed the 0.95 success threshold. | Mean >=85 and >=70% of runs above 0.95, on the same task release and evaluation protocol. Publish both, plus each task-type/domain breakdown. |
| DeepResearch Bench II | 132 tasks and 9,430 expert-derived binary rubrics; its paper reports the strongest evaluated systems below 50% rubric satisfaction. The public leaderboard has since migrated and reports a GPT-5.5 judge change. | >=60% rubric pass under a pinned judge version; re-run all baselines after any judge migration. |
| Autonomous Science Bench 0.1 | Six pilot tasks; ten configurations ran three trials per task. The top reported configuration resolved 72.2% of trials at $190.23 total. The benchmark authors explicitly caution that small differences with three trials are not meaningful. | >=80% resolution with lower cost per resolution than the matched baseline. Treat >=20 repetitions per task as exploratory only; require more campaigns and a pre-registered paired power analysis before claiming superiority. |
| BrowseComp | 1,266 hard-to-find, short-answer browsing tasks; its authors note that this format is not equivalent to open-ended research reports. | Establish a same-version, same-tool baseline first; target >=5 percentage-point improvement with paired confidence intervals and no hidden increase in cost. |

Internal qualification gates to trial on expert-authored, sealed tasks:

| RITE axis | Proposed threshold |
|---|---:|
| Selective risk at >=80% answer coverage | <=5% |
| Brier score for confidence in decision correctness | <=0.10 |
| Fixed 10-bin expected calibration error | <=0.05 |
| Claim citation precision | >=95% |
| Required independent decision-support origins | 100% of tasks meeting the oracle's minimum |
| Counterfactual oracle accuracy | >=95% |
| Source-removal oracle accuracy | >=95% |
| Placebo invariance | >=99% |
| Required independent replay | 100% |
| Declared resource-budget compliance | 100% |

These internal thresholds are qualification gates, not averages to optimize at the expense of another axis. For example, an agent cannot compensate for an unsafe or unsupported claim by being cheaper or faster. Publish coverage and every denominator next to each rate.

## Statistical reporting protocol

- Match the official task split, evaluator, judge version, tools, prompts, scaffold, and resource caps before comparing scores.
- Use all available tasks; report task-level scores and domain/task-type slices, not only pooled averages.
- Repeat stochastic configurations across independent seeds. Three trials per task are useful as a pilot but do not support small-difference claims. Twenty repetitions per task are still exploratory unless the task set is broad enough and a power calculation supports the claim.
- For a superiority claim, pre-register the minimum effect of interest, use paired runs on the same tasks and seeds, and report task-clustered bootstrap confidence intervals or an appropriate hierarchical model. Account for multiple comparisons when testing many configurations.
- Report cost, latency, resource use, failure rate, abstention, and reliability alongside outcome quality. Separate best-of-N results from single-run results.
- Do not infer broad research capability from BrowseComp alone: short-answer fact finding and long-form research artifacts are different workload distributions. Likewise, six pilot lab campaigns do not establish broad autonomous-science generalization.

## External benchmark references

- [SCI-AGENT-EXAM](https://arxiv.org/abs/2610.32441)
- [Autonomous Science Bench 0.1](https://asb.open-ssi.org/)
- [DeepResearch Bench II paper](https://arxiv.org/abs/2601.08536) and [current leaderboard/methodology notice](https://agentresearchlab.com/benchmarks/deepresearch-bench-ii/)
- [BrowseComp](https://openai.com/index/browsecomp/)

## Qualification and claims boundary

The unit tests are synthetic contract tests. They prove that the evaluator computes its defined metrics and rejects selected malformed records; they do **not** establish model performance, benchmark validity on real scientific work, or superiority over any commercial system.

A public comparative result requires:
1. Expert-authored tasks and decision oracles, with private/sealed evaluation tasks.
2. Pre-registered run budgets, tool access, model versions, prompts, and harness commits.
3. At least three independent seeds per task for instability analysis, with task-level confidence intervals.
4. A frozen candidate stream and independently held auditor stream.
5. Human validation of claim-specific citation and support-origin judgments.
6. Exact run manifests, source hashes, code artifacts, environment lockfiles, and independently replayed outputs.
7. Full reporting of task outcomes, abstention, reliability, costs, and all denominators.

No leaderboard claim should be made from the contract fixtures alone.
