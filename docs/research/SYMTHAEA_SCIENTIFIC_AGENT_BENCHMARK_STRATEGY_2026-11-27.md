# Symthaea Scientific-Agent Benchmark Strategy — 2026-11-27

## Executive thesis

The field is moving from chat-based research assistance toward project-level scientific agents. Symthaea should not compete only on a single resolution-rate leaderboard. The defensible opportunity is **evidence-qualified scientific autonomy**: a system that can select experiments, execute them, preserve provenance, detect invalid evidence, calibrate its uncertainty, abstain when warranted, and transfer improvements to new tasks without silently regressing.

The benchmark programme should therefore optimize a vector:

1. scientific task success and quality;
2. independent evidence integrity;
3. calibration and selective risk;
4. reproducibility and artifact completeness;
5. transfer/generalization;
6. cost, wall-clock time, and tool efficiency;
7. robustness to missing, delayed, contradictory, and adversarial observations.

No single scalar should conceal a severe failure in another dimension. Publish the Pareto frontier and per-domain/per-task results.

## Current external comparison (snapshot 2026-11-28)

All results below are snapshots from public benchmark pages, not Symthaea measurements. Harness, model, effort, trial count, task version, and scoring rules differ. They are not interchangeable.

| Benchmark | Public reference | What it measures | Interpretation for Symthaea |
|---|---|---|---|
| Terminal-Bench-Science | The public official-board snapshot lists 70 tasks and GPT-6 Astra/Codex at 68.1% ±3.2, with Opus 5.5/Claude Code at 63.3% ±3.3. The Nov 28 OpenResearch mirror separately displays model-reported GPT-6.2 Astra 77.1% and Mythos 5.2 75.3%, explicitly not yet on the official board. A third-party Terminal-Bench-Science-AA capture reports GPT-6.2 Astra at 78.6% on 70 tasks × three runs, averaged, with a refreshed public task set; this is not the Snorkel official-board score. Vals AI's Nov 19 Mini-SWE-agent snapshot reports GPT-6.2 Astra at 71.43% and $11.08/test. Keep all four records separate by publisher, harness, effort and scoring method. [Official board](https://snorkel.ai/leaderboard/terminal-bench-science/) · [OpenResearch reported results](https://openresearch.sh/evals) · [Priors/AA capture, 28 Nov](https://www.mypriors.com/benchmarks/aa-terminalbench-science) · [Vals snapshot](https://vals-ai.com/benchmarks/terminal-bench-science) [Official board](https://snorkel.ai/leaderboard/terminal-bench-science/) · [Nov 28 mirror and reported results](https://openresearch.sh/evals) · [Vals snapshot](https://vals-ai.com/benchmarks/terminal-bench-science) | Long-horizon executable research workflows with task verifiers | Primary external task-completion target; match exact task/harness/trial rules; never substitute vendor-reported results for official runs |
| Terminal-Bench-Science cost frontier | Vals AI's Nov 19 snapshot reports GPT-6.2 Astra 71.43%, Gemini 4 Neon 62.86%, GPT-6 Astra 62.86%, with per-test cost/tokens/duration. These are Vals harness results, not the official-board denominator above. [Results](https://vals-ai.com/benchmarks/terminal-bench-science) | Completion plus resource consumption | Optimize cost-quality-time Pareto frontier, not raw success alone |
| ASI-Bench | 60 tasks, 11 domains, progressive methodological-guidance removal (B1–B4); leaderboard lists GPT-6.1 Sol/Codex at 63.90 and GPT-6 Astra/Codex at 62.23. [Leaderboard](https://asibench.apexin.ai/) | Project-level research, method reconstruction, robustness to missing guidance | Essential test of whether the system can reconstruct executable methods rather than follow a supplied recipe |
| Autonomous Science Bench 0.1 | Six pilot tasks; ten configurations × three trials per task. GPT-6 Astra/Codex 72.2%, Opus 5.5/Claude Code 66.7%. Authors explicitly caution that small differences are not meaningful at three trials. [Leaderboard](https://asb.open-ssi.org/leaderboard) · [Methodology](https://asb.open-ssi.org/about) | Real laboratory workflow planning, observation interpretation, iterative improvement | Use as a laboratory campaign target, but require a larger confirmatory evaluation before superiority claims |
| SCI-AGENT-EXAM | 120 tasks, 11 fields, seven workflow task types; paper reports GPT-6 Astra 80.9 overall, with only 26% of runs above 0.95 success threshold. [Paper](https://arxiv.org/abs/2610.32441) | Executable and textual scientific outputs; progress and stability | Track run-level reliability and threshold success, not only average score |
| SciAgentArena | Approximately 200 tasks with stepwise verification across scientific contexts. [Project and benchmark](https://sciagentarena.github.io/) | Agent-agnostic real-world scientific analysis and open-ended work | Use to measure domain unevenness and novel-insight limitations; do not substitute narrow coding success |
| AstaBench | Stanford AI Index summarizes over 2,400 problems and a best agent score around 0.53 at roughly $3.40/problem in its cited 2025 evaluation. [AI Index science chapter](https://hai.stanford.edu/assets/files/ai_index_report_2026_chapter_5_science.pdf) | Broad end-to-end scientific workflow capability | Broad coverage and cost-per-problem anchor; align to the exact benchmark release before comparing |
| OpenScience claims | OpenScience reports 75.7% Terminal-Bench-Science and other leading results, but its page also notes evaluation issues and planned reruns after comparison/scoring problems. [Benchmark page](https://www.openscience.sh/benchmark) | Vendor-reported system results | Treat as a hypothesis to reproduce, not a settled leaderboard fact |

**FrontierPhysics snapshot conflict (28 November 2026):** public leaderboard mirrors do not currently agree on the reported pass-rate aggregation. Benchmark Almanac's capture of the benchmark's own leaderboard reports Claude Opus 5.5 at 27.4% and GPT-6 Astra at 23.2%; Priors currently reports Opus 5.5 at 45.8% and GPT-6 Astra at 39.3%, describing its score as the verifier pass rate averaged across three trials per task. The task set is 56 tasks across nine domains in both descriptions, but the denominators/aggregation snapshots are not reconciled. Preserve both captures and do not choose a winner or set a superiority target from this discrepancy until the exact run IDs, trial denominator, verifier version and scoring aggregation are matched. [Benchmark Almanac, 28 Nov capture](https://www.benchmarkalmanac.com/benchmarks/frontierphysics/) · [Priors, 28 Nov capture](https://www.mypriors.com/benchmarks/bg-frontierphysics).

### Comparison discipline

- Never compare scores unless task set, task version, harness, model access, tool access, time budget, attempt count, grader, and aggregation method match.
- Keep official benchmark submissions separate from local reproductions and vendor-reported numbers.
- Report confidence intervals and per-task outcomes. ASB 0.1 is a useful warning: three trials per task are not enough to interpret small leaderboard gaps.
- For cost claims, include all inference, tool, orchestration, verification, and human-review costs. Report both cost per attempt and cost per successful task.
- Preserve failed runs and incomplete artifacts. Do not silently remove timeouts, invalid outputs, or verifier failures.

## Paradigm shifts to pursue

### 1. From answer generation to evidence-carrying decisions

Every consequential research action should bind to:
- a versioned question and objective;
- the current evidence-prefix digest;
- a complete permitted-action offer and payload digest;
- a prospective prediction;
- explicit resource bounds and stopping conditions;
- an execution receipt;
- observations with event/availability time and provenance;
- an independently adjudicated outcome.

This turns a research trace into a replayable decision object. The scientific-decision replay work in PR #13933 is a prototype contract, not yet a production-isolated execution system.

### 2. From confidence to calibrated, abstention-aware judgment

Accuracy alone does not tell us whether a system knows when it is wrong. Track Brier score, calibration error, valid adjudication sample size, and risk-versus-coverage. Missing labels must remain missing; model confidence must never be promoted to correctness. PR #13969 implements the first bounded step: adjudication-only Brier and ten-bin ECE.

### 3. From one-shot experiments to causal, budget-aware active learning

Select experiments by expected information gain or expected decision value under explicit resource constraints. Compare against random, greedy uncertainty sampling, fixed design, and domain-expert baselines. Use matched initial states and equal realized budgets. Record failed experiments and negative results. Do not optimize a proxy such as novelty or confidence in isolation.

### 4. From static benchmark wins to transfer under distribution shift

Use held-out task families, domains, instruments, and data-generating processes. Measure both average transfer and worst-group regressions. RecurSciBench-style repeated starts and ASI-Bench's guidance gradient are complementary: one tests transfer, the other tests method reconstruction.

### 5. From a monolithic agent to independently checkable roles

Separate proposer, executor, evidence custodian, evaluator, and policy updater. The proposer must not see sealed outcomes; the evaluator must not inherit the proposer’s confidence as truth; the policy updater consumes only adjudicated evidence. Process/OS isolation and sealed evaluation remain future work, not properties established by typed Rust structures alone.

## Proposed Symthaea target ladder

These are **research targets**, not achieved results. Freeze benchmark versions and baselines before claiming progress.

| Workstream | Near-term target | Stretch target | Required controls |
|---|---:|---:|---|
| Terminal-Bench-Science | ≥84% on a frozen, matched 70-task/three-run protocol | ≥89% | Baseline must be re-run under the same tasks, harness, effort and trial rules; require ≥5 percentage-point paired improvement over the strongest matched baseline; report confidence intervals, cost/time and failure taxonomy |
| ASI-Bench | ≥65 overall, no B-level collapse >10 points | ≥70 overall; B4 within 8 points of B1 | Report B1–B4 separately; distractor and method-reconstruction controls |
| SCI-AGENT-EXAM | ≥83 overall and ≥35% runs >0.95 | ≥86 and ≥50% runs >0.95 | Preserve task-type/domain slices and trajectory stability |
| SciAgentArena | +5 absolute points over matched baseline on ≥3/5 selected task groups | +10 points on ≥3/5 with no >5-point regression elsewhere | Frozen task subsets, independent grading, domain-level regressions |
| ASB-style laboratory campaigns | ≥15/18 resolved on a defined 6-task × 3-trial pilot | ≥90% on a larger confirmatory suite | The 18-run pilot is plumbing only; expand to ≥10 trials/task before superiority claims |
| Calibration | Brier and ECE reported with n; improve over uncalibrated baseline | Brier skill >0 on held-out domains; ECE ≤0.05 at adequate n | Reliability diagram, 5/10/15/20-bin sensitivity, confidence intervals, source/time stratification |
| Selective prediction | At 80% coverage, reduce error relative to always-answer baseline | 50% lower risk at 80% coverage without pathological abstention | Risk-coverage curve, AURC, coverage, abstention reasons |
| Reproducibility | 100% successful claims linked to immutable artifacts and replay inputs | Independent clean-room replay of ≥99% of eligible runs | Negative controls, artifact hashes, environment lock, no hidden mutable state |
| Transfer | Non-negative mean transfer; ≤20% held-out tasks regress | ≥5% mean gain; ≤10% regressions | Matched starts, equal budgets, held-out task families, report worst group |
| Claim/evidence attribution | 100% of accepted claims carry a claim-specific relation and a byte-resolved quote span verified against the pinned artifact digest | Zero false-support events on sealed negative controls; zero accepted claims with missing/stale artifact digests or unresolved quote spans | High query similarity alone must never create support; verify byte offsets and quote hashes against exact artifact bytes; deduplicate source families; count independent support only after lineage checks |
| Cost efficiency | Report cost per verified success | ≥25% lower cost at matched quality | Include model, tools, compute, retries, evaluator, and human review |

Targets should be revised only after a baseline run and preregistration. A high target is not evidence that the architecture can achieve it.

## Production researcher boundary audit — 2026-11-28


### Artifact-bound quote verification added to the typed-state work

The typed ResearchState schema already requires a claim/source relation to bind a classifier record to the exact claim and evidence-prefix digests, and a quote span to carry byte offsets plus a quote digest. Draft PR #13665 now also adds `verify_claim_source_quote`: given the retrieved artifact bytes, expected artifact SHA-256, span offsets and quote digest, it checks the full artifact hash, range bounds, UTF-8 validity and exact slice hash before returning the quote. This closes a byte-identity gap; it does **not** establish semantic entailment, authenticate the source host, or prove the classifier's judgment. The resolver and four negative/positive tests are in the standalone typed-state contract crate.


A default-branch source audit identified a concrete production surface and a blocking evidence-attribution defect. The relevant entry point is `WebResearcher::research_and_verify` in [`src/web_research/researcher.rs`](https://github.com/Luminous-Dynamics/symthaea/blob/05ab60f36a43fa0da011dec9870d8a21f3e7167e/src/web_research/researcher.rs#L61-L184). It is a useful retrieval/extraction executor to adapt, but it is **not yet a qualified state-contingent research policy**:

1. It builds a fixed keyword-triggered query list (`plan_queries`), fetches sources, and processes them in a fixed loop. It has no public prospective action/forecast contract and does not expose an offer-scoped next-action decision.
2. Every extracted `SourceEvidence` is constructed with `supports: Some(true)`; its similarity is computed against the original query, not against the individual extracted claim. The same `all_sources.clone()` is then supplied to every claim's `VerificationContext`. See the source construction and claim loop in `research_and_verify` above.
3. `EpistemicVerifier::verify_claim` consumes `SourceEvidence.supports` as the support/refutation label and multiplies it by credibility and similarity; it does not independently derive claim-specific entailment from `relevant_text`. `verify_from_results` has a second circular path that maps `result.confidence > 0.5` to support. [`verifier.rs`](https://github.com/Luminous-Dynamics/symthaea/blob/05ab60f36a43fa0da011dec9870d8a21f3e7167e/src/web_research/verifier.rs#L140-L234).
4. The overall research confidence is the arithmetic mean of claim confidences; the result's summary is a short prefix of the first selected page, not a synthesis of individually attributed evidence. The pipeline caps verification at ten extracted claims per call.
5. `KnowledgeIntegrator::integrate` integrates a claim when its confidence crosses a threshold; it does not require independent correctness adjudication. [`integrator.rs`](https://github.com/Luminous-Dynamics/symthaea/blob/05ab60f36a43fa0da011dec9870d8a21f3e7167e/src/web_research/integrator.rs#L89-L127).
6. In the default branch, `EpistemicConsciousness::research` still maps `claim.confidence > 0.7` into `was_correct`. PR #13969 changes this to an unadjudicated `None` record and prevents pending claims from training calibration; its focused harness is valuable, but it does not modify `researcher.rs`, `verifier.rs` or `integrator.rs`. Until that PR is merged and its exact head requalified, the default-branch behavior remains the old implementation. [`mod.rs`](https://github.com/Luminous-Dynamics/symthaea/blob/05ab60f36a43fa0da011dec9870d8a21f3e7167e/src/web_research/mod.rs#L175-L200).

**Integration ruling:** use `WebResearcher` initially as a retrieval/extraction tool behind the existing permitted-offer boundary; do not treat its present confidence, support labels or aggregate epistemic status as independently qualified evidence. Do not wire `ScientificMethodEngine` as the missing research planner: its current cognitive-loop call site updates a standing input-coherence hypothesis from prediction error and consciousness level, not from web-research decisions or adjudicated scientific outcomes ([`cycle.rs`](https://github.com/Luminous-Dynamics/symthaea/blob/05ab60f36a43fa0da011dec9870d8a21f3e7167e/src/cognitive_loop/cycle.rs)).

The smallest safe repair sequence under #11171 is:

- Replace the boolean `supports` placeholder with an explicit claim/source relation whose default is `UNCLASSIFIED`; retain the exact source span, claim ID, artifact digest, and source-family lineage.
- Make support classification claim-specific and quote/span-backed. A retrieval similarity score may rank passages but must never create a support label.
- Keep extraction, source classification, truth adjudication and decision usability as different fields/stages. Correlated/syndicated pages must not count as independent support.
- Require the evidence-prefix digest and correction state before a claim is eligible for any downstream decision or knowledge-graph integration.
- Make integration store provisional/unadjudicated claims as such; only independently adjudicated outcomes may train calibration or source reliability.
- Expose the existing retrieval actions as permitted offers. A policy selects an offer and emits a pre-action forecast; the executor performs it and returns an immutable receipt. Do not let the policy create its own permissions or label its own outcomes.

**Mandatory negative controls:** a source with high query similarity but no claim-relevant span must remain unclassified; one source that supports claim A but is irrelevant to claim B must not support B; a negated/contradictory quote must not be converted to positive support; duplicate or syndicated sources must not inflate independent-source count; changing irrelevant page metadata must leave the decision invariant; corrupting the evidence-prefix digest must fail closed. These tests are more important than improving headline retrieval accuracy until the attribution defect is repaired.

## Immediate execution sequence


### Source/claim attribution gate added — draft PR #14214

The production audit is now being addressed by [draft PR #14214](https://github.com/Luminous-Dynamics/symthaea/pull/14214). It changes retrieval-originated source relations to unclassified, makes `None` neutral in the verifier, removes confidence-derived support in `verify_from_results`, and adds two regression tests plus a standalone harness importing the production `types.rs` and `verifier.rs`. Latest exact head: `4aaa9f307f05dc5ba9a608a06cf59c661edfe137`; focused Rust 1.96 workflow `46150130304` is queued, not passed. The PR intentionally does not claim claim/span attribution, source-family independence, negation handling, independent truth adjudication or production-policy integration. The main branch remains unchanged until merge.


1. **Close the calibration and source-attribution loops:** PR #13969's 20/20 standalone pass applies only to its earlier head `37e67653b22001dd4b6a97302c1527a1337ae3cb`; its latest head `70d8d53d23262b0ff4ff536ef11c80945fd8b675` has a fresh focused workflow queued. Draft PR #14214 adds fail-closed source neutrality and a path-imported verifier harness at `4aaa9f307f05dc5ba9a608a06cf59c661edfe137`; its focused Rust 1.96 workflow is queued. Neither current head has a pass yet, and neither harness compiles the entire web-research runtime or proves external adjudicator quality.
2. **Complete decision custody:** integrate the proposed offer/context digest contracts with the upstream AgentView owner; then add prefix-only candidate replay and process-level separation.
3. **Make time semantics explicit:** asynchronous experiments need event time, observation/availability time, and decision time. Simulation tick alone is not enough for delayed or censored evidence.
4. **Add selective-risk reporting:** calibration is not sufficient if the agent cannot abstain safely. Evaluate risk-coverage and decision utility.
5. **Run a small matched AS-Bench campaign:** pin API/schema, fixtures, exact design IDs, cost/resource ledger and independent outcome adjudication; do not call a simulated replay a laboratory campaign.
6. **Run R4 matched-start evaluation:** same starting states and realized budgets, independent evaluator, held-out transfer tasks, preserve all failures.
7. **Only then make SOTA claims:** public reproducibility package, frozen benchmark commit, exact run manifests, and independent rerun.



### Terminal-Bench-Science 1.0 contribution opportunity — deadline 1 December 2026

The official call is open for 1.0 submissions until **1 December 2026**. It requires a genuine workflow from a practicing domain expert, objectively verifiable with deterministic pytest checks, and materially difficult for current frontier agents; open-ended hypothesis generation and literature review alone are explicitly out of scope. Tasks pass domain and technical review. [Call and requirements](https://www.terminal-bench-science.ai/contribute) · [Contributing guide](https://github.com/harbor-framework/terminal-bench-science/blob/main/CONTRIBUTING.md).

This is a useful near-term route to make Symthaea's work legible to the industry, but we should not submit a task that merely wraps our internal policy tests or claims domain expertise we have not established. A credible candidate would need a real Luminous engineering/scientific workflow, an authoritative dataset or instrument, a reproducible container and deterministic independent verifier, with a qualified domain expert validating the scientific ground truth. Otherwise, target the post-1.0 release rather than shipping a weak benchmark contribution.

## What would constitute a genuine field-level advance?

A credible claim would be: *at equal or lower total cost, Symthaea produces more independently validated scientific progress, with lower unsupported-claim rate, better calibrated uncertainty, more reliable abstention, and stronger out-of-domain transfer than matched frontier-agent baselines.*

A leaderboard score alone cannot establish that. The evidence package must include the complete action-to-observation chain, negative controls, independent labels, and clean-room replay.

## Expanded external benchmark portfolio — 2026-11-28

The external suite should be a **portfolio of complementary instruments**, not a single leaderboard. These benchmarks measure different outputs, with different denominators and grading semantics. Their numbers must not be averaged together.

| Benchmark | Scope and current reference | What Symthaea should use it to test | Initial target / stretch |
|---|---|---|---|
| Terminal-Bench-Science v0.1 / AA capture | 70 tasks across five scientific domains. Snorkel's official page lists GPT-6 Astra/Codex at 68.1% and Opus 5.5/Claude Code at 63.3%; OpenResearch records 77.1% for GPT-6.2 Astra as a model-reported result not yet on the official board; a third-party AA capture lists GPT-6.2 Astra at 78.6% on 70 tasks × three runs averaged. These are separate score records, not one ranking. [Official leaderboard](https://snorkel.ai/leaderboard/terminal-bench-science/) · [OpenResearch snapshot](https://openresearch.sh/evals) · [Priors/AA capture](https://www.mypriors.com/benchmarks/aa-terminalbench-science) | End-to-end computational science workflows with executable artifacts and deterministic verifiers | ≥84% on a frozen matched 70-task/three-run protocol, then ≥89%; require a paired +5 percentage-point lead over the strongest re-run comparator before claiming SOTA |
| DeepResearch Bench II | 132 report tasks, 22 domains and 9,430 expert-derived binary rubrics. The unified portal currently distinguishes GPT-5.5 scoring from a historical Gemini-2.5 evaluation; tasks 112 and 114 are temporarily excluded from current GPT-5.5 scoring, leaving 130. Pin the judge and eligible task IDs. [Paper](https://arxiv.org/abs/2601.08536) · [Unified leaderboard](https://huggingface.co/spaces/muset-ai/DeepResearch-Bench-Leaderboard) | Evidence recall, analysis, and synthesis quality in long-form research reports | ≥45% rubric pass under the pinned current judge, then ≥50%; superiority means +5 points against a contemporaneous matched submission, not against a different judge |
| PaperBench | 20 ICML 2024 research replications decomposed into 8,316 gradable subtasks. The original paper's best tested agent scored 21%; that is a historical release result, not a current frontier ceiling. [OpenAI benchmark](https://openai.com/index/paperbench/) | Reconstructing methods, writing code, executing experiments, and reproducing published findings | ≥25% mean score, then ≥30%; report per-paper completion and artifact/replay integrity, not only the mean |
| RE-Bench | Seven open-ended ML research-engineering environments; 71 eight-hour attempts from 61 human experts. The paper reports that the best agents beat the human comparison at two hours, while humans gained more with longer budgets and were stronger at 32 hours. [PMLR paper](https://proceedings.mlr.press/v267/wijk25a.html) | Research-engineering productivity as a function of time and compute; compare human and agent scaling curves | At two hours, match the strongest agent on ≥4/7 environments; at eight hours, reach ≥90% of expert-median score on ≥4/7; publish the full 2/8/32-hour frontier |
| FrontierPhysics | 56 expert-authored tasks from 42 physicists across nine theoretical, computational and experimental domains. The paper reports no tested configuration passed more than 46 of 168 rollouts. Nov 28 public mirrors disagree: Benchmark Almanac's capture of the benchmark's own board lists Opus 5.5 at 27.4% and GPT-6 Astra at 23.2%; Priors reports 45.8% and 39.3%, respectively, and describes a three-trial verifier pass-rate average. These may be distinct settings/aggregations. Do not merge the scores. [Paper](https://arxiv.org/abs/2611.06910) · [Almanac snapshot](https://www.benchmarkalmanac.com/benchmarks/frontierphysics/) · [Priors snapshot](https://mypriors.com/benchmarks/bg-frontierphysics) | Deep scientific reasoning, expert intuition, model criticism, experimental design, and publishable outputs | No absolute target until run IDs, verifier versions, trial denominator and aggregation are reconciled; then target ≥5 percentage points above the strongest matched official baseline |
| Autonomous Science Bench 0.1 | Six real laboratory pilot tasks; its published leaderboard reports resolution rates and total agent cost/time separately. It is a pilot, not a large statistical sample. [Leaderboard](https://asb.open-ssi.org/leaderboard) · [methodology](https://asb.open-ssi.org/about) | Laboratory planning, observation interpretation, and iterative experimental optimization | First reproduce the published protocol; then exceed the matched resolution/cost frontier on a larger confirmatory set with ≥10 trials per task |
| SCI-AGENT-EXAM | 120 expert-curated tasks across 11 fields and seven workflow types. The paper reports GPT-6 Astra at 80.9 overall but only 26% of runs above its 0.95 success threshold. [Paper](https://arxiv.org/abs/2610.32441) | Broad workflow capability and trajectory stability | ≥83 overall and ≥35% of runs above 0.95 initially; stretch ≥86 and ≥50%, with full run-level distributions |

### What each benchmark cannot establish

- **DeepResearch Bench II** does not establish that a reported source is causally decisive, that a tool action was authorized, or that a research trajectory is replayable. It is primarily a report-quality benchmark.
- **PaperBench** tests reproductions of a selected ML-paper set; it does not establish general scientific novelty or correct experimental policy across disciplines.
- **RE-Bench** is a narrow seven-environment research-engineering sample. Its time-budget scaling result is valuable, but it should not be generalized to all research work.
- **FrontierPhysics** is an expert physics portfolio; its pass rate combines hard verifiers and rubric criteria. A rollout pass rate is not the same as mean rubric score or pass@3.
- **Terminal-Bench-Science** and **SCI-AGENT-EXAM** exercise computational workflows but do not, by themselves, establish safe external actuation or correction propagation.
- **ASB 0.1** has only six pilot tasks. A 72.2% leaderboard resolution figure is not directly comparable to 68.1% on Terminal-Bench-Science: the task sets, sample sizes, environments and success criteria differ.
- **BrowseComp** remains a retrieval stress test, not evidence of research method fidelity, experimental quality or scientific discovery.

### The missing benchmark layer: a trajectory qualification track

Do not create another disconnected evaluator. Add a trajectory-level qualification profile under Research Policy Adapter #11171 that composes the existing RITE, CEDAR, METHODE, REPAIR, WebResearch, parser, ResearchState, StateGuard, and decision/replay receipts.

A sealed episode should contain:

1. A versioned research objective, initial public state, evidence-prefix digest and resource vector.
2. A candidate-side prospective decision record created before the action: legal `offer_id`, predicted outcome distribution or bounded outcome set, expected decision impact, rejected alternatives, stopping condition and forecast version.
3. An independent authority check that the offer exists, is current and is permitted. The candidate cannot mint permissions.
4. An executor-produced action receipt and observation record with source provenance, event time, availability time and decision time.
5. A new ResearchState version whose claims and dependencies are recalculated from the verified receipt.
6. Sealed evaluator adjudication, counterfactual mutation tests, and independent clean-room replay.

The resulting qualification is a vector, not one confidence number:

| Dimension | Required measurement | Hard gate / research target |
|---|---|---|
| Authorization and custody | Unauthorized actions, invalid receipts, stale offer use, digest mismatches | Zero unauthorized executions; zero accepted invalid receipts |
| Replay and provenance | Exact independent replay rate, artifact completeness, chain-of-custody validity | 100% replay for eligible deterministic episodes; 100% valid provenance for accepted claims |
| Forecast quality | Pre-action Brier/log loss, interval or set coverage, sharpness, calibration by domain and horizon | Improve over an uncalibrated baseline; report confidence intervals and sample size; ECE ≤0.05 only when adequately powered |
| Decision quality | Realized terminal decision loss, normalized regret against an evaluator-only oracle, action value per resource unit | ≤10% normalized regret on tractable oracle fixtures; ≥10% lower regret than matched policy baselines |
| Evidence responsiveness | Sensitivity to decisive changes; invariance to placebo metadata; reaction to contradiction and source removal | ≥95% correct intervention response and ≥99% placebo invariance on frozen fixtures |
| Model adequacy | False acceptance of a misspecified candidate grammar; detection of structural and parameter support shift | ≤5% false-accept rate on sealed misspecification fixtures, with power and abstention reported separately |
| Correction recovery | Downstream claim invalidation/recomputation precision and recall; correction propagation latency | 100% dependent-claim recall on deterministic fixtures; no stale dependent claim may remain marked verified |
| Transfer and robustness | Held-out domains, instruments, methods, source types, missing/delayed/adversarial observations | Non-negative worst-group transfer before claiming broad capability |
| Cost and throughput | Money, wall time, model tokens, CPU/GPU, experiments, retrieval and human review | ≥25% lower total cost at non-inferior verified outcome, or a demonstrably better cost-quality-time Pareto frontier |

Hard gates are conjunctive: a task cannot pass if authorization, provenance, required receipts or replay fail, even when its final answer is correct. Forecasting scores are calculated only against outcomes adjudicated independently of the candidate. Missing labels remain missing; they are not converted to negative or positive truth.

### Paradigm shift: choose actions for decision improvement, not information volume

Keep two quantities separate:

- **Epistemic value:** how much an observation is expected to reduce uncertainty about model structure, parameters or state.
- **Decision value:** how much an observation is expected to improve the final decision under the actual loss function and resource constraints.

An observation can be highly informative and still have no decision value if it cannot change the optimal action. Conversely, a low-entropy measurement may be decisive near a safety or resource boundary.

The policy should therefore evaluate actions with a joint model over model structure (M), parameters (\Theta), and state (S), but score downstream decision impact separately. Use Bayesian EVSI only where priors are justified and calibrated; otherwise report set-valued or prior-free robust regret. Keep the exact sequential-game oracle entirely evaluator-side.

The candidate's prospective forecast must be frozen before execution. Evaluate it prequentially against the actual observation. Do not allow a retrospective rationale to count as a prospective prediction. The same applies to costs: include model inference, tools, orchestration, retries, verification, experiment/material costs and human review.

### Matched-comparison rules before any SOTA claim

1. Pin task IDs, task release, rubrics, verifier versions, judges, model snapshots, prompts, scaffolds, tool access, context limits, time limits and budget vectors.
2. Run the same candidate model with and without the Symthaea orchestration layer. This separates model strength from architecture value.
3. Compare at least random, fixed-design, greedy partition/EIG, calibrated EVSI, prior-free robust/minimax, strongest public agent, and human/expert baselines where available.
4. Use paired task seeds and matched realized budgets. Publish single-run/pass@1, mean over trials, best-of-N and pass@N as separate columns.
5. Preserve every timeout, failed run, abstention, invalid artifact and replay failure. Report per-task and worst-domain outcomes, uncertainty intervals and full cost.
6. Claim a frontier advantage only after an independent rerun shows either a statistically reliable ≥5-point improvement on the same scoring metric, or non-inferior verified quality (predeclared margin, e.g. −2 points) at ≥25% lower total cost.

The initial ambition is not to beat every frontier model on every benchmark. It is to show that the same base model and tools, when governed by Symthaea's evidence/action/replay architecture, produce more verified scientific progress per unit of total cost while making fewer unsupported claims and preserving safety under evidence change.

## Decision, experimental-taste and world-model tracks — 2026-11-28

The portfolio above should be augmented with **diagnostic tracks that test the internal policy and world-model assumptions** which the broad research benchmarks do not isolate.

| Track | External reference and result | How to use it | Symthaea target |
|---|---|---|---|
| RD-Bench / Decision-Centric Research | 2,000 rule-generated cases; 10 decision types instantiated from a 24-type taxonomy. On a held-out research-domain split, a state-aware 0.6B model reached 89.4% Top-1 versus 27.8% for the same recipe using only the question, and 36.4% for DeepSeek-V4.1-Flash. Calibrated prediction sets achieved 97.1% empirical coverage at mean size 1.515. [Paper](https://arxiv.org/abs/2611.00774) | Test typed decisions over a public ResearchState: evidence sufficiency, next action, stopping, conflict, routing and escalation. Treat as closed-option decision quality, not research completion. | ≥90% pooled Top-1 initially, ≥85% next-action Top-1; prediction-set coverage ≥95% with mean set size ≤1.6 and selective-risk reporting on held-out domains |
| TasteVal / experimental taste | Eight private AI R&D tasks, 24 recruited experts and a fixed Coder to isolate experiment selection. Opus 5.5 achieved a reported 2.30× experimental-compute multiplier (95% CI 1.15–4.37) on seven scored tasks, but normalized final-performance multiplier was 1.14× (95% CI 1.01–1.31). Compute multiplier excludes Researcher inference tokens; task coverage differs by model and one refusal was excluded as missing data. [Paper](https://arxiv.org/abs/2610.06824) | Separate experiment-compute efficiency from final held-out outcome. Use one fixed executor/Coder across policy arms and fully account for inference, tools, retries and human review. | First reproduce a matched fixed-executor protocol; stretch to ≥2× experimental-compute efficiency at non-inferior final performance (predeclared −2% margin) and ≥1.1× normalized final performance. No claim without inclusive lifecycle cost |
| Scientist World / Improve and Innovate | Paper reports 1,000 paper-derived executable environments. Its model evaluation used a fixed cohort of ten lower-compute environments: positive held-out improvement over the **reproduced source method** in 84% of Improve and 74% of Innovate attempts. These figures do not mean that the agents beat the original human authors, and they have not yet been independently replicated. [Paper/project](https://arxiv.org/abs/2611.08478) | Measure whether a method improves under protected metrics, fixed data/compute and frozen baselines. Distinguish improvement of a reproduced method from a new scientific discovery. | ≥85% positive held-out Improve and ≥80% Innovate attempts on a larger sealed cohort, with raw metrics, failed attempts, regressions, rollbacks and all compute costs published |
| WorldTest / AutumnBench | A behavior-based world-model protocol instantiated in 43 interactive environments and 129 tasks across masked-frame prediction, change detection and planning. 517 human participants substantially outperformed five frontier models. [ICML/PMLR paper](https://proceedings.mlr.press/v306/warrier26a.html) | Test whether a model supports counterfactual and environment-level queries beyond replaying familiar trajectories. This is a world-model diagnostic, not a scientific-publication benchmark. | ≥90% of human-median performance in each query family; zero false-correct answers on deliberately underdetermined intervention controls |

### Interpretation and caveats

**RD-Bench is a particularly relevant design precedent.** Its key finding is not just a high Top-1 result: exposing a structured research state changed decision performance dramatically compared with asking the same model a question alone. Symthaea should therefore make the state projection and decision trace first-class, then compare state-aware against question-only ablations using identical models and task IDs. The candidate must emit probabilities or set-valued predictions and permit abstention; a hard class label alone is insufficient.

**TasteVal highlights a metric trap.** Experimental-compute multiplier and final normalized performance are different axes. A system may use experiments efficiently without producing proportionally better final outcomes. Moreover, the reported compute multiplier excludes Researcher inference tokens and its seven-task Opus 5.5 cohort differs from the eight-task main cohort. Symthaea's own denominator must include inference, orchestration, tool calls, retries, executor time, experiments, validation, verification and human review. Always publish both compute-efficiency and final-quality results.

**Scientist World is promising but must be interpreted narrowly.** Its reported 84%/74% positive FNI rates apply to ten evaluation environments and use the reproduced paper method as baseline. The 1,000 environments are the corpus/construction claim, not the number of evaluated tasks underlying those percentages. Until an independent reproduction and access/licensing review is complete, treat it as author-reported supporting evidence, not a settled SOTA table.

**WorldTest tests whether the agent learned a useful model of the world.** Its reward-free exploration followed by environment-level queries is a good paradigm for testing causal and dynamics understanding. The large human–model gap makes it a useful stretch target, but a grid-world result cannot be generalized to general scientific research capability.

### Cross-track design

Create no additional evaluator. Under #11171, keep the evaluation layers separate but linked by a common episode ID and immutable receipt chain:

1. **Decision layer:** RD-Bench-style state-conditioned choice and calibrated option set.
2. **Policy layer:** #13571 sequential regret and matched fixed/random/EIG/EVSI/minimax/production-policy comparisons.
3. **World-model layer:** open-world microcosm and WorldTest-style intervention/holdout tests.
4. **Method layer:** METHODE method fidelity, PaperBench replication and Scientist World-style improvement.
5. **Research-outcome layer:** Terminal-Bench-Science, DeepResearch Bench II, FrontierPhysics and ASB.
6. **Integrity layer:** RITE/CEDAR/REPAIR, authorization, provenance, correction propagation and independent replay.

Do not collapse these into a weighted total. The publishable result should be a set of Pareto frontiers and hard-gate pass vectors: verified outcome, unsupported-claim rate, false-unique/model-misspecification rate, calibration, regret, correction recovery, transfer, complete replay and inclusive cost. A candidate that fails authorization or evidence provenance does not receive credit for a correct-looking answer.

## Trajectory-derived research taste and false-unique risk — 2026-11-28

Two additional findings sharpen the programme.

### Taste-Bench: score decisions at the fork, not only the final trajectory

The September 2026 *The Tasteful Agent* paper introduces Taste-Bench, mining decision forks from parallel engineering/research trajectories and within-trajectory detours. The evaluated model sees the prefix and available alternatives but not the downstream outcomes. The paper reports that the best of 14 frontier models answered 59.7% of 502 fork questions correctly; mean accuracy fell from 62.3% to 21.0% as the decisive evidence appeared later. A hindsight teacher distilled into a student improved held-out decision accuracy by 17.9 points and improved a separate held-out SWE-bench Pro executor comparison from 14.6% to 33.7%. [Paper and benchmark description](https://arxiv.org/abs/2609.25804).

This suggests a Symthaea-native **Research Taste track**, but not a copy of the full benchmark:
- Freeze an evidence-prefix and action menu at each fork.
- Record the chosen action, rejected alternatives, prospective outcome distribution, downstream decision relevance and inclusive action costs.
- Hide all post-fork outcomes from the candidate.
- Derive the outcome ranking only from independently qualified downstream trajectories or controlled counterfactual replay.
- Split by source task, project, domain and time so near-duplicate forks cannot cross train/test.
- Include delayed-evidence forks, misleading early proxies, irreversible-action choices, replication-versus-pivot, stop-versus-continue and tool/experiment allocation.
- Score choice quality, proper probability scores, regret against a finite evaluator-side oracle, calibration, resource use and integrity separately.

**Critical safeguard:** a hindsight teacher may label a fork only after its outcome is qualified; the student sees only prefix-time features. Never let post-outcome labels or hidden evaluator state leak into the policy input. Report single-fork accuracy and full-trajectory improvement separately. The evaluator-side oracle is a diagnostic upper bound, not a production policy.

Initial proposed targets: ≥65% held-out fork accuracy on a preregistered balanced binary subset (above the 50% chance baseline), ≥10% lower paired regret than the strongest fixed/random baseline, and a measurable held-out trajectory-success improvement under matched inclusive budgets. Require bootstrap confidence intervals clustered by source task and publish the full fork-generation attrition ledger. Do not claim frontier superiority from a target or from a small mined sample.

### False-unique risk: split structural, composite and finite-budget ambiguity

PR #14163 provides a useful standalone reference for three distinct failure modes: pointwise structural non-identifiability, overlap between composite mechanism families under nuisance uncertainty, and practical ambiguity at a finite sample/design budget. It also demonstrates why mechanism-instance and mechanism-family singleton denominators must remain separate, and why zero singleton denominators mean `NO_SINGLETON_COHORT`, not zero risk. Its six-episode fixture is a reference fixture, not a production SOWRC qualification.

Integrate these estimands with #11060/#11171's open-world episode metrics. In particular:
- A candidate can be a singleton in a declared grammar while the generating mechanism remains outside the evaluator's compatible set.
- Structural identifiability at a nominal parameter point does not imply composite-family identifiability.
- An empty candidate set, an ambiguous set and a false singleton are different terminal outcomes.
- Report joint false-unique risk, conditional-on-singleton risk, commitment rate and coverage separately.
- Keep evaluator-only compatibility sets, hidden mechanisms and sampling probabilities outside the agent process.
- Any sequential confidence bound must reuse the pinned SOWRC theorem implementation and its predictable audit design; the reference script is arithmetic validation, not a replacement theorem.

### Robust experiment design under model misspecification: add R-IDeA

The 2026 AISTATS paper *Representative, Informative, and De-Amplifying* shows why maximizing expected information gain can amplify generalization error when the candidate model is misspecified and the experiment distribution shifts away from the deployment/test distribution. Its R-IDeA acquisition function explicitly combines representativeness, informativeness and error de-amplification; its reported experiments outperform acquisition rules that optimize only informativeness, only representativeness, or their combination. [PMLR 300 paper](https://proceedings.mlr.press/v300/tang26d.html).

Add R-IDeA as a named policy-tournament arm; do not reimplement or claim it as Symthaea's invention. The minimum controlled comparison is:
- matched model, action menu, evidence prefix and vector budget across random, fixed-grid, greedy EIG, calibrated EVSI, prior-free minimax, R-IDeA and the production candidate;
- separate in-support, covariate-shift, structural-misspecification, measurement-process-shift and combined-shift strata;
- evaluate decision regret, held-out predictive error, proper-score calibration, false-unique rate, coverage, representativeness of sampled designs, total inclusive cost and wall-clock latency;
- report paired task-level contrasts and worst-stratum regressions, not a pooled average that lets in-support gains conceal shifted-support failures.

**Proposed qualification target:** on preregistered shift/misspecification strata, the candidate should reduce paired decision regret by at least 10% relative to greedy EIG and be non-inferior to EIG within a predeclared 2% margin on in-support tasks, while meeting the same calibration, integrity and replay gates. This is an aspirational target, not a Symthaea result. If R-IDeA wins, adopt it as a baseline or component only after reproducing the paper's conditions; the differentiator should then be the end-to-end evidence/authorization/decision/replay contract, not an unsupported claim of a novel acquisition formula.

### Add established-discovery and broad science suites to the external validation ladder

Two additional external resources should be used as complementary tracks rather than replacements for the controlled Symthaea-specific tests:

- **AstaBench** is an evolving suite with thousands of scientific research problems across literature understanding and other scientific task categories. It is useful for measuring breadth and retrieval/workflow competence, but its component leaderboards and task-specific metrics must remain separate. [Official AstaBench suite](https://allenai.org/asta/bench).
- **FIRE-Bench** evaluates full-cycle rediscovery of established findings from recent machine-learning literature. The ICML 2026 paper reports that even its strongest tested frontier-backed agents remained below 50 F1, with substantial run-to-run variance and recurring failures in experimental design, execution and evidence-based reasoning. Use it for empirical rediscovery, not report plausibility; pin the task/harness version. [ICML 2026 / PMLR paper](https://proceedings.mlr.press/v306/wang26d.html).

The external evaluation ladder should therefore distinguish:
1. **Breadth:** AstaBench and selected SCI-AGENT-EXAM tasks.
2. **Deep synthesis:** DeepResearch Bench II and live research tasks.
3. **Trajectory choice:** Taste-Bench-inspired held-out forks and RD-Bench.
4. **Rediscovery:** FIRE-Bench and selected known-answer scientific tasks.
5. **Reproduction / engineering:** PaperBench, RE-Bench and MLE-bench.
6. **Experimental improvement:** TasteVal, Scientist World and Autonomous Science Bench.
7. **Scientific integrity:** Symthaea's false-unique, provenance, correction, authorization and replay contracts.

The recent lifecycle survey also cautions that retrieval-grounded research and machine-learning engineering have more repeatable benchmark gains than open-ended ideation and end-to-end discovery; it reports persistent fabrication, mis-citation and overclaiming risks. Use this as a maturity warning, not a leaderboard source. [2026 lifecycle survey](https://www.mdpi.com/3043-0097/1/2/8).

### Correction propagation: measure both damage containment and recovery speed

The MON-045 scoped-invalidation work proposes a typed dependency DAG with required digest-pinned edges, optional context edges, expiry/revocation semantics and three-valued verdicts. This is directly relevant to scientific evidence graphs: retracting one required source should invalidate every dependent claim, but must not damage unrelated claims. It also makes a key missing metric explicit: stale-claim exposure time after a source correction.

For research trajectories, add paired correction episodes:
1. A source is corrected, retracted, expired or digest-shifted.
2. The candidate receives only the public correction receipt.
3. It identifies affected claims and unaffected controls.
4. It recomputes dependent analyses and decisions.
5. An independent evaluator checks dependency completeness, exact invalidation scope, regenerated artifact lineage and final replay.

Report invalidation recall, invalidation precision, unrelated-claim damage, stale-claim exposure time, recovery latency, recomputation cost and replay success. A correct final report with an uncorrected intermediate claim still exposed should not receive a full integrity pass.

## References

- [Terminal-Bench-Science official leaderboard](https://snorkel.ai/leaderboard/terminal-bench-science/)
- [Vals AI Terminal-Bench-Science results](https://vals-ai.com/benchmarks/terminal-bench-science)
- [ASI-Bench](https://asibench.apexin.ai/)
- [Autonomous Science Bench leaderboard](https://asb.open-ssi.org/leaderboard)
- [Autonomous Science Bench methodology](https://asb.open-ssi.org/about)
- [SCI-AGENT-EXAM](https://arxiv.org/abs/2610.32441)
- [SciAgentArena](https://sciagentarena.github.io/)
- [Stanford AI Index 2026, science agents](https://hai.stanford.edu/assets/files/ai_index_report_2026_chapter_5_science.pdf)
- [OpenScience benchmark and its evaluation caveat](https://www.openscience.sh/benchmark)
- [Uncertainty-calibration metrics and risk-coverage discussion](https://link.springer.com/article/10.1007/s44163-026-02240-w)
