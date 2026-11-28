# Evidence-First Research Systems: Cross-Programme Paradigm and Benchmark Scorecard

**Snapshot:** 2026-11-27  
**Scope:** Symthaea / SDCB / MACE Decision-Sufficient Memory / Scientific Decision Replay  
**Status:** Research and execution roadmap. All numeric goals below are proposed acceptance targets, not achieved results, certifications, or external benchmark scores.

## 1. Thesis: optimize qualified decision utility, not isolated prediction or recall

The central shift is to stop treating a good answer, low prediction error, or high retrieval score as sufficient evidence that a system is useful.

A research-capable system should be evaluated as a complete decision process:

`source events → admissible evidence → uncertainty model → candidate action → execution → observation → independent adjudication → replayable receipt`

The unit of evaluation becomes a **qualified decision episode**, not a text answer or model call. An episode records:
- the exact candidate-visible evidence prefix and the time each item became available;
- valid-time interval, source family/root, scope, authority and lifecycle state;
- frozen policy, allowed actions, exact action payload and assignment mechanism;
- predictive distribution and declared decision/loss objective;
- execution state separately from outcome-resolution state;
- measured/estimated/unknown cost by phase;
- independently controlled outcome labels and a replay receipt.

The resulting performance object is a vector, not a weighted sum:

`(decision regret, harm, calibration, evidence sufficiency, lifecycle integrity, latency, memory, compute, cost, replay integrity)`

Hard authority, privacy, custody and mutation violations are **non-compensatory**. They cannot be offset by a higher accuracy score.

## 2. Paradigm shifts

### 2.1 From asserted truth to bitemporal, lineage-bearing evidence

Store what was observed, by whom or what, when it became known, the interval in which it applies, and the chain of corrections, supersession or revocation. Never rewrite old claims in place. Build embeddings, HDC vectors, summaries and graphs as rebuildable projections over canonical source events.

**Research consequence:** every query is a temporal cut. At query time (t), an item is retrievable only if it was available by (t), valid at (t), in the requested scope and not revoked. A correction observed later must not leak backward into what the candidate could have known.

### 2.2 From maximal recall to decision-sufficient evidence

The goal is the smallest admissible evidence packet that preserves or improves the downstream decision. More context is not automatically better: it can add stale, duplicated, correlated, poisoned or contradictory evidence.

A retrieval system must expose two different sets:
- retrieved context;
- evidence actually supplied to the frozen downstream policy.

This enables decision-support precision/recall and intervention tests without pretending all returned text influenced the answer.

### 2.3 From passive retrieval to an active evidence economy

At each decision, route among:
- lexical, dense, entity, date, graph, temporal-state and HDC/HDC-LTC memory;
- raw-source expansion or provenance verification;
- a new observation or experiment;
- abstention, escalation or deferral.

Choose the cheapest admissible operation whose expected reduction in decision loss justifies its cost. The router must itself be evaluated for regret, latency, compute and unsafe routing; it is not allowed to turn hard authority gates into learned preferences.

### 2.4 From score-only learning to causal evidence intervention

Use paired runs with a frozen downstream reader and identical episode, policy digest and seed. Measure the memory/no-memory decision-loss delta. Then remove evidence items, clusters or source roots and rerun the same policy. Leave-one-out is a first diagnostic, not Shapley attribution: interactions require explicit subset interventions.

Historical logs only identify outcomes for actions actually taken. Off-support alternatives remain `unsupported_counterfactual`; never infer their outcomes from a nearby row or a model-generated guess. Use a real experiment when the expected value of information justifies it.

### 2.5 From point calibration to dependence-aware uncertainty

A model should report both its prediction and the limits of its applicability. For sequential and correlated observations, the effective independent unit is a world, campaign or block—not an individual tick.

SDCB should move from a scalar stochastic factor toward coupled vector state-space models, full covariance/model-set uncertainty and finite-block adequacy checks. It must distinguish:
- observation noise;
- latent process variation;
- parameter/model uncertainty;
- dependence/common-cause uncertainty;
- numerical integration error;
- distribution shift and unmodelled residual risk.

No one variance term may silently absorb the others.

### 2.6 From proof-carrying code to proof-carrying observations and decisions

Each action should bind the exact evidence prefix, permitted offer, payload, policy, cost budget and executor receipt. Each result should bind the exact executed payload and independent observation/adjudication. A digest proves byte identity only; it does not prove truth, authority, statistical independence or access control.

A replay service must be physically separated from evaluator secrets. Type separation and separate directories alone are not a security boundary.

### 2.7 From one SOTA number to an assurance/performance frontier

The competitive question is not “which architecture has the highest average score?” It is:

> At the same task mix, evidence budget, latency, compute, hardware and downstream policy, which system achieves the greatest admissible decision value with the lowest harm and strongest independently verified coverage?

Publish the full Pareto frontier and capability profile. Do not collapse incomparable tracks or measurement states.

## 3. External benchmark map and current reference points

The public comparators answer different questions. Their scores must not be combined into one league table.

### 3.1 Agent Memory Leaderboard (AML) Cycle 2

Official results were released on 19 November 2026 and contain separate commercial, open-source and closed-source standings for Textual, Multimodal and Coding Memory. The current open-source leaders include:
- Textual: Resurface, **61.22 overall**.
- Multimodal: Khipu, **43.26 overall**.
- Coding: InsightMem, **63.33% overall resolved**.

The commercial Textual leader MemoraX reports **69.26**; commercial Multimodal MemoraX reports **53.68**; commercial Coding reports **68.00% overall resolved**. These are different tracks and metrics, not directly comparable numbers. AML also reports capability-level scores (for example temporal reasoning, governance, privacy and streaming), which are more informative than the overall rank for system diagnosis. The official result release records 402 full evaluation tasks, 333 completed successfully, 41 corrected after compliance/completeness review and 292 valid leaderboard records; therefore a reproduced comparison must bind to the official task/version and corrected result set, not an informal scrape.

**MACE target:** implement a strict AML-compatible Add/Search adapter only after the live schema and model-use rules are rechecked. Report its official AML track score separately from native MACE causal decision utility. Aim first to reproduce an open-method baseline and its capability profile; then improve the temporal, governance, epistemic-safety and streaming dimensions without regressing the other capabilities.

Source: https://agentmemoryleaderboard.ai/results/cycle-2/

### 3.2 LongMemEval-V2

LongMemEval-V2 evaluates experience acquisition from long web-agent trajectories. Its current specification lists 451 manually curated questions, up to 500 trajectories and up to 115M tokens. It uses an Insert/Query memory contract and a fixed reader, then measures answer accuracy and query latency.

The current online board lists AgentRunbook-C v2 Online at **+2.823 LAFS**, **75.6% accuracy** and **130.5 s** query latency. This is a useful moving frontier for the LongMemEval-V2 task only; it is not an AML score or a MACE decision-value score.

**MACE target:** first match the exact task version, reader, context budget and latency protocol; run the released RAG, file-based and online baselines; compute the official LAFS gain. A credible SOTA claim requires positive gain over the current reference frontier on held-out tasks, with all operating points, accuracy, p50/p95 latency and cost published. Do not optimize accuracy by hiding expensive memory-construction or maintenance work.

Source: https://xiaowu0162.github.io/longmemeval-v2/

### 3.3 Governed memory and memory security

TimlMemBench explicitly evaluates governed use: source-linked memory, lifecycle updates, scope, provenance, activation and restraint. Its current package reports 208 reviewed bilingual samples, 1,566 sessions, 1,715 questions and 1,821 governance/activation cases.

MemSecBench follows poisoning through Write → Execute → Forget and selective repair. Its published benchmark contains 310 cases from 48 contexts across code/science, daily-life and office-work settings; its results show that persistence and downstream execution are distinct failure stages.

**MACE target:** do not claim governed memory or poisoning defence as novel. Reproduce or map the relevant task families, then add decision-loss intervention and full lifecycle cost. For poisoning, report attack persistence, downstream action success, blast radius, selective-repair success and benign-memory retention separately. Any unauthorized action or cross-scope disclosure is a hard failure.

Sources:
- https://aclanthology.org/2026.findings-emnlp.507/
- https://arxiv.org/abs/2607.27080

### 3.4 Numerical multivariate-normal probability

The standard comparator landscape is mature:
- Genz–Bretz randomized quasi-Monte Carlo supports arbitrary covariance structures and dimensions up to 1,000 in mvtnorm.
- Miwa is available for non-singular problems up to dimension 20.
- TVPACK provides specialized two- and three-dimensional semi-infinite regions.
- A 2026 doubly blocked Genz implementation reports speedups close to or above 100× on its tested large-dimensional workloads. Those results are not a fair direct comparison to a two-dimensional conditional-integral reference.

**SDCB target:** benchmark the bivariate oracle first against TVPACK and independent high-precision quadrature; then benchmark trivariate and higher-dimensional solvers against Genz–Bretz, Miwa where applicable, and blocked Genz. Equalize absolute/relative tolerance, hardware, thread count, covariance condition number, integration region and function-evaluation/compute budget. Record numerical error and wall time—not speed alone.

Sources:
- https://cloud.r-project.org/web/packages/mvtnorm/mvtnorm.pdf
- https://ideas.repec.org/a/spr/compst/v41y2026i6d10.1007_s00180-026-01794-8.html

## 4. Programme scorecards and proposed acceptance targets

All targets here are **prospective gates**. They are not achieved results.

### 4.1 SDCB — numerical and statistical reliability

| Dimension | Baseline/comparator | Proposed target |
|---|---|---|
| Analytic identities | Independent/product normal, orthant arcsine, exact marginalization | All analytic fixtures pass within declared absolute/relative tolerance |
| Numerical accuracy | TVPACK for applicable 2D/3D cases; high-precision independent integration; Genz–Bretz/Miwa/blocked Genz at higher dimensions | 100% of accepted fixtures meet declared error tolerance; unresolved cases fail closed |
| Metamorphic invariance | Channel permutation, positive unit scaling, sign transforms, threshold refinement, variable reordering | Invariance residual below the declared numerical tolerance |
| Probability validity | Joint tables and univariate marginals | Probabilities in [0,1]; table mass and marginalization pass declared tolerances |
| Compute transparency | Function evaluations, wall time, peak memory, thread count | Hard global budget across the entire joint table; no per-cell budget multiplication |
| Calibration | Independent-world simulation, then AR(1), latent-factor, common-shock and drifting regimes | Nominal 90%/95% coverage checked with cluster-aware intervals; no claim of iid coverage under dependence |
| Dependence robustness | Static covariance, AR/latent/common cause, block-length sweeps | Publish coverage and decision regret across dependence regimes; no hidden iid assumption |
| Open-world adequacy | Fixed model set vs misspecified/unseen model sets | Unknown model applicability remains visible; no positive score can compensate for unsupported adequacy |

A numerical speedup is only meaningful when the error tolerance and coverage are held constant. Report the Pareto frontier of error, evaluations, time and memory across dimensions 2, 3, 5, 10, 20, 50 and higher as supported. Never compare a 2D semi-infinite specialized routine to a 100D blocked-QMC result.

**Next experiments:** rare-tail probability sweeps; near-singular and ill-conditioned covariance; high/negative correlation; heterogeneous scales; permutation/order invariance; common-factor misspecification; block-length/coverage curves; synthetic distribution shift; and independent high-precision checks.

#### Proposed SDCB qualification gates (not results)

- **In-envelope coverage:** for each declared alpha (at least 0.10 and 0.05), every preregistered held-out in-envelope regime must have a cluster-aware 95% upper confidence bound on empirical miscoverage no greater than `alpha + 0.01`. Publish the lower bound too, so pathological over-conservatism is visible.
- **Out-of-envelope rejection:** the upper 95% confidence bound on false-qualified rate must be below 1% on held-out regimes outside the declared envelope. Target at least 95% detection of material excursions and at most 5% false rejection inside the validated envelope.
- **Model-misspecification controls:** test negative rho, changing lag structure, heavy tails, shared latent scales, multivariate/common-cause factors, abrupt and gradual shifts, missing observations and adversarial diagnostic contamination. Do not treat a scalar rho interval as a complete dependence description.
- **Power and replication:** exploratory grids may use 2,500 trials to find failure modes; confirmatory conditions should use at least 50,000 independent trials per condition, fresh seed domains and a separate held-out regime set. For serial data, the resampling/confidence unit must be a world, path or independently generated campaign—not a time tick.
- **Fail-closed runtime:** any interval partly outside the envelope, insufficient diagnostic sample, unknown model family, failed model adequacy, invalid custody or unsupported dependence state must be `Inconclusive` or `NotQualified`. It must not produce a positive engineering adequacy or downstream authorization.

A fixed sample count is not itself proof of power. The trial design should calculate power and interval width for the smallest materially unsafe excursion before execution, then freeze the test set and analysis code.

### 4.2 MACE DSM — memory that measurably improves decisions

| Dimension | Matched controls | Proposed target |
|---|---|---|
| Retrieval | No-memory, raw/full-history ceiling, temporal BM25, dense, lexical+dense RRF, entity/date, graph, HDC/HDC-LTC | Compare all arms on identical episodes, reader and evidence/context budget |
| Evidence quality | Required/admissible evidence, source-root lineage, contradiction pairs | No unknown source IDs, duplicate evidence or unresolved hard contradiction in an accepted sufficient packet |
| Decision utility | Paired frozen-reader memory/no-memory runs | Positive held-out mean decision gain with cluster-aware 95% interval; publish harm rate and tail harm |
| Decision sufficiency | Oracle evidence set and minimal sufficient subset | Fewer evidence bytes/tokens at no worse decision loss; quantify distance from evaluator oracle |
| Temporal correctness | Transaction/knowledge time and valid time | Zero future-knowledge and expired-validity leakage in all hard negative controls |
| Governance/security | TimlMemBench/MemSecBench families | Zero unauthorized scope disclosure or unauthorized mutation; report poisoning persistence, execution and repair separately |
| Lifecycle cost | Measured/estimated/unknown receipts by phase | Full ingest, index, retrieval, verification, policy, maintenance and storage costs visible; unknown never becomes zero |
| Latency | Fixed hardware and workload; per-query budget | Publish p50/p95/p99 and throughput, both with warm and cold cache, including maintenance contention |
| External retrieval track | AML Cycle 2 and LongMemEval-V2 | Official track-specific score/gain only after reproducing exact submission protocol |

Do not set one arbitrary accuracy target before the memory adapter is integrated with raw trajectory data. First reproduce BM25 and RRF controls, then the best published baseline for each public task. The research contribution is the incremental decision value, reduction in harmful use, less retained context and lifecycle cost—not simply a new retriever name.

#### Proposed competitive gates (not results)

| Track | Current open reference | First surpass target | Commercial/stretch target |
|---|---:|---:|---:|
| AML Textual overall | Resurface 61.22 | ≥63.22 (+2.00 points) | ≥70.26 (above MemoraX 69.26) |
| AML Multimodal overall | Khipu 43.26 | ≥45.26 (+2.00 points) | ≥54.68 (above MemoraX 53.68) |
| AML Coding overall resolved | InsightMem 63.33% | ≥65.33% | ≥69.00% (above MemoraX 68.00%) |
| LongMemEval-V2 LAFS | AgentRunbook-C v2 Online +2.823 | ≥+3.00 on the official held-out evaluation | Preserve ≥75.6% accuracy at ≤100s median query latency, then improve both |

These are separate targets in separate scoring systems. For each AML track, first reproduce the named open-source leader with the exact submission version and metric, then require a paired/bootstrap 95% lower confidence bound above that leader before claiming a win. The commercial/stretch column is a second milestone, not an implied directly comparable target across tracks. For LongMemEval-V2, match the official fixed reader, context budget, task tier and latency measurement; publish all accuracy/latency operating points and full Insert, maintenance and storage cost. Do not treat +3.00 LAFS as an AML score.

For native MACE decision value, use a separate preregistered evaluation: target at least **+5 percentage points in paired task success** over the strongest matched no-memory/BM25/dense baseline, with the paired 95% interval strictly above zero and **≥10% lower decision regret**. Retain all unsupported counterfactuals in the denominator. Report harmful-use and tail-harm rates separately; a mean gain cannot compensate for unauthorized use.

For governed-memory and security tests, require **zero** unauthorized cross-scope retrieval, disclosure or mutation on the complete hard-negative suite; **100%** source-root resolution for accepted evidence; and no positive use after a confirmed revocation or expiry. On held-out poisoning tests, aim for an upper 95% confidence bound below **1%** successful unauthorized action, while reporting benign-memory retention and legitimate activation/abstention. This is intentionally stricter than answer accuracy: MemSecBench reports 84.2% malicious-memory persistence and 50.3% full Write–Execute chain success across its evaluated configurations, so persistence and downstream consequence must be separate metrics.

For every numerical target, report the test set size, independent cluster unit, confidence method, hardware, inference model/version, prompt/reader digest, data version, and missing/unsupported count. Use episode/world/source-root clusters for uncertainty; individual ticks or repeated queries from one episode are not independent samples.

### 4.2A Competitive target ladder (proposed acceptance gates)

These targets translate the current public frontiers into explicit experiments. They are not results already achieved by Symthaea.

| Track | Current public reference | First reproduction gate | Competitive / SOTA-beating gate |
|---|---|---|---|
| AML Textual, open methods | Resurface 61.22 overall | Match official Add/Search contract, task version, capability profile and resource disclosure | Exceed 61.22 with a paired confidence interval; target ≥65 while no capability falls >2 points |
| AML Textual, commercial ceiling | MemoraX 69.26 | Treat as a separate commercial comparison, not a same-class baseline | Stretch target ≥70 on the same official track and division |
| AML Multimodal, open methods | Khipu 43.26 | Reproduce official multimodal request shape, image handling and per-capability metrics | Exceed 43.26; stretch ≥48 without worsening governance |
| AML Multimodal, commercial ceiling | MemoraX 53.68 | Separate hardware, model and cost profile | Stretch target ≥55 on same official track |
| AML Coding, open methods | InsightMem 63.33% overall resolved | Reproduce relevant/noisy splits, exact task set and resolution rubric | Exceed 63.33%; stretch ≥68% with no noisy-memory regression |
| AML Coding, commercial ceiling | 68.00% overall resolved | Separate division and resource disclosure | Stretch target ≥70% |
| LongMemEval-V2 | AgentRunbook-C v2 Online: +2.823 LAFS, 75.6% accuracy, 130.5 s | Reproduce official reader, small/medium tier, web+enterprise pair, and all operating points | First require positive LAFS; SOTA gate >+2.823, with accuracy ≥75.6% at ≤130.5 s or a strictly better full frontier |
| Native MACE decision utility | No directly equivalent public scalar | BM25, BM25+RRF, no-memory and file-based/agentic baselines under one frozen reader | Mean paired memory gain lower 95% cluster CI >0; harm-rate upper CI ≤1%; 0 cross-scope/unauthorized use; report regret, sufficiency, interaction and cost |
| SDCB calibration assurance | Mature mvtnorm/Genz–Bretz/Miwa/TVPACK numerical references; no directly comparable dependence-envelope benchmark | Reproduce oracle error and coverage on held-out Gaussian/t5 regimes | Within declared envelope: absolute coverage error ≤1 percentage point with adequate power; outside envelope: false-qualified upper 95% bound <0.1%; detection power ≥90% for preregistered shifts |

**Important comparison rules**
- AML tracks/divisions are not pooled. Its Cycle 2 release reports 402 full tasks, 333 completed, 41 corrections and 292 valid leaderboard records; bind the result version and corrected task set.
- LongMemEval-V2 LAFS is an accuracy-vs-query-latency frontier score, not AML's score. Report online learning, memory construction and maintenance cost in addition to official query latency.
- MACE native utility is a different construct from retrieval recall or answer accuracy. A higher retrieval score is not a decision-value improvement.
- SDCB numerical integration accuracy, conformal coverage, regime detection power and false qualification are four different metrics; never combine them into one “confidence” score.

## 4.3 Scientific decision replay — research-policy quality

Compare at least:
1. no-memory/static baseline;
2. fixed sequential policy;
3. fixed best-of-N policy;
4. cost-matched random selection;
5. adaptive state-contingent policy;
6. replay-policy learned from historical outcomes;
7. evaluator oracle as a ceiling only.

Use the existing AS-Bench Nuclease and Propylene owners and exact action/result contracts. Do not create a second API client or duplicate result validator.

Report:
- decision regret and selection regret;
- top-k hit/NDCG where appropriate;
- predictive calibration and log score/Brier score;
- support/positivity coverage;
- unsupported-counterfactual rate (retained in denominator);
- execution failure separately from scientific null outcomes;
- latency and complete experiment/evaluation cost;
- rank concordance between replay-selected and independently audited live actions.

**Gate:** no off-support counterfactual is scored as observed. A live audit sample is required before claiming that historical replay predicts the ranking of real experiments.

## 4.4 Scientific-agent capability — research choices, execution and independently verified transfer

This lane is separate from SDCB numerical assurance and MACE memory utility. It evaluates whether an agent can run a scientific workflow, decide what to do next, make testable predictions, revise or abstain under evidence, and produce a result that another party can reproduce.

### External benchmark map (snapshot: 2026-11-28)

| Benchmark | What it actually measures | Current reference / caveat | Symthaea use |
|---|---|---|---|
| Terminal-Bench Science | 70 terminal-based scientific workflows across life, physical, earth, mathematical and engineering sciences; three trials per task | OpenResearch's 2026-11-28 sync lists GPT-6 Astra / Codex at 68.10 ± 3.22, with domain scores and reported cost. The same page separately lists maker-reported maxima of GPT-6.2 Astra 77.10, Mythos 5.2 75.30 and Fable 5.2 74.60 as **not yet on the official leaderboard**. Do not merge those evidence tiers. | End-to-end workflow reference; preserve task-level and domain breakdowns, trial variance and inclusive tool/API cost. |
| FrontierPhysics | 56 end-to-end physics research tasks across nine fields, written by 42 physicists; a pass requires every automated test and every must-have rubric item; three trials averaged | Priors' 2026-11-28 aggregator snapshot lists Fable 5.2 at 33.3%, Opus 5.5 at 27.4%, and GPT-6.2 Astra at 26.8%. This is an aggregator snapshot, not a claim that all 29 models were freshly evaluated or that the source leaderboard has published every result. | High-bar physics workflow lane; strict task-pass definition and rubric transparency are essential. |
| RD-Bench / Decision-Centric Research | 2,000 rule-generated research-decision cases; 24 decision types defined, ten instantiated; held-out research-domain evaluation | The 2026-10-31 paper reports state-aware DCRM Top-1 89.4%, versus 27.8% question-only DCRM and 36.4% DeepSeek-V4.1-Flash; calibrated prediction sets achieve 97.1% empirical coverage with mean set size 1.515. This is a decision-model benchmark with closed candidate options, not a full autonomous-discovery result. | Direct benchmark for the typed ResearchState → decision interface and calibrated option sets. Reproduce exact task splits before comparing. |
| AstaBench | Evolving suite of thousands of scientific problems across literature understanding, coding/execution, data analysis and end-to-end research; includes Core-Bench-Hard, SUPER-Expert and E2E-Bench-Hard | The benchmark is explicitly evolving and exposes multiple sub-benchmarks/leaderboards. Its task families are not one interchangeable scalar. | Reproducibility and end-to-end workflow tracks; pin sub-benchmark version, task set and grader. |
| SciAgentArena | Approximately 200 stepwise-verified tasks, 18 evaluated agents, five scientific domains and four task categories | The public benchmark reports heterogeneous task metrics (for example AUROC, F1, Jaccard and accuracy); no common aggregate should be inferred from those native metrics. The benchmark highlights uneven open-ended exploration, novelty and validity checking. | Domain-specific held-out capability tests, stability/reliability and cost. Compare per task family, not a synthetic average. |
| R-IDeA | Active-learning acquisition under model misspecification and covariate shift | The AISTATS 2026 paper formalizes representativeness, informativeness and de-amplification as distinct acquisition requirements; information gain alone is an incomplete policy baseline under misspecification. | Add as a first-class acquisition comparator beside EIG, EVSI and robust/minimax policies; do not describe R-IDeA as original Symthaea work. |

Sources are listed under References. External results have different harnesses, models, costs, task units and uncertainty treatment. They are calibration points, not a single cross-benchmark ranking.

### Two competition tracks — never confound policy quality with model capability

**Track A: model-locked controller tournament.** Every policy arm receives the same base model/version, context, tools, permissions, prompts, evidence prefix, task seed, wall-clock allowance and inclusive budget. Swap only the research controller / decision policy. This is the clean test of what Symthaea contributes beyond a language model.

**Track B: complete-system tournament.** Compare complete runnable systems under declared hardware, base model, API/model spend, software and tool access. This measures product capability, not the isolated effect of Symthaea's controller. Report model-free/native Symthaea results separately from any model-assisted configuration.

For both tracks, retain an evaluator-only upper reference on tiny worlds. Never give candidate processes hidden truth, future outcomes, score functions, oracle policies or audit-selection randomness. A JSON boundary is not a security sandbox; evaluator isolation is a separate operational requirement.

### Required treatment arms

Use the owners already defined in #11171, #11012, #13691, #13567 and #9230. On identical episodes and inclusive budgets, compare:

1. No-action / fixed workflow and fixed schedule.
2. Matched-cost random action/experiment selection.
3. Entropy-greedy and conventional EIG.
4. One-step EVSI and exact sequential oracle on tiny evaluator-only worlds.
5. Robust minimax / RPSD and R-IDeA acquisition.
6. Spark Engine's LCF specialist EIG baseline only on the LCF domain where its assumptions apply.
7. Direction-board allocation (including a reproducible Coral Discovery-style baseline) versus score-only winner-take-all search.
8. Frozen production candidate selected by #11171, once a real state-contingent callable exists.
9. Hidden-truth or full-information oracle as a diagnostic upper bound only.

Do not relabel a fixed workflow, query planner, literature summarizer or single-step EIG utility as the general Symthaea researcher. The current callable gap in #11171 remains a prerequisite, not an evaluation detail.

### Primary scorecard — report a vector, not a blended score

| Dimension | Required measure |
|---|---|
| Decision quality | Per-decision-type Top-1/Top-k, calibrated option probabilities, selective risk/coverage, safe and unnecessary abstention separately. |
| Research control | Prospective predicted outcome branches, correct stop / continue / pivot / replicate / abstain choice, and regret versus the tiny-world exact oracle. |
| Identifiability | Structural, finite-budget and composite-family false-unique rates with separate commitment denominators; correct set-valued claims on exact-overlap controls. |
| Counterfactual quality | Prospective branch calibration and held-out intervention prediction, including negative controls and prediction log score where adjudication supports it. |
| Discovery validity | Independently reproduced result rate; survival on held-out mechanism, device, site, time and task families; counterexample/negative-result retention. Textual novelty or judge preference alone earns no discovery credit. |
| Robustness | Model misspecification, support omission, covariate shift, measurement shift, common-cause/dependence shift, delayed and missing evidence, and corrupted-source controls. |
| Efficiency | Full cost per independently verified transferable decision: model/API, CPU/GPU, tools, experiments, data acquisition, adjudication, retries, failed runs, latency and maintenance. |
| Integrity | Exact evidence-prefix and task digests, no future-label leakage, no unauthorized action, complete failure accounting and independently replayable receipts. |

The primary economic measure is **cost per independently verified transferable decision**, not cost per token, paper, answer or benchmark attempt. A transfer credit is awarded only when the result reproduces from the recorded artifacts and survives a preregistered held-out family or regime. An unresolved or non-identifiable result can be scientifically correct, but it must be labelled as such and scored on the appropriate decision objective.

### Proposed target ladder (not results)

| Lane | Reproduction gate | First competitive target | Stretch target |
|---|---|---|---|
| RD-Bench decision layer | Reproduce the paper's state-aware and question-only baselines on the exact held-out-domain split | ≥92% pooled Top-1 and ≥85% next-action Top-1; calibrated sets ≥95% held-out coverage at mean set size ≤1.5 | ≥95% pooled Top-1 with no research-domain subgroup below 90%, while retaining calibrated coverage |
| Terminal-Bench Science | Reproduce the official 70-task / three-trial protocol and a listed official reference under the same harness | ≥70% resolution rate, with task-clustered uncertainty and no domain-level integrity regression | ≥80% resolution rate; this would exceed the currently listed official GPT-6 Astra reference and the separately maker-reported 77.1% maximum, if those protocols remain comparable |
| FrontierPhysics | Reproduce the 56-task, three-trial strict all-tests-plus-rubric pass protocol | ≥40% task pass rate | ≥50% task pass rate, with rubric-level and field-level results published |
| AstaBench | Reproduce named Core-Bench-Hard and E2E-Bench-Hard baselines | Beat the strongest version-matched baseline by ≥3 absolute points on at least one named hard track, with no regression on the paired companion track | Beat the best version-matched baseline on both hard tracks with paired task-family intervals above zero |
| SciAgentArena | Reproduce native task/metric baselines and agent-agnostic environment | ≥5 absolute points over the strongest matched baseline on at least four of five domains, without worsening the remaining domain's failure severity | Lead on ≥4/5 domains and demonstrate transfer to a newly held-out domain/task family |
| Native SDCB active-science tournament | Complete all deterministic identifiability, oracle-regret and evidence-integrity controls | Zero false-unique passes on exact structural/family overlap and truth-absent controls; ≥25% lower inclusive cost per independently verified transferable decision than the strongest matched policy without higher false-unique risk | Transfer the advantage to at least three held-out mechanism families and two independently sourced real-data cohorts |
| Economic efficiency | Measure all inclusive costs on matched successful and failed tasks | ≥25% lower cost per independently verified transferable decision at no worse decision regret, false-unique risk, integrity or held-out transfer | ≥40% lower cost with equal or better worst-stratum performance |

The numeric goals are research gates, not predictions. The 80% Terminal-Bench and 50% FrontierPhysics stretch goals are deliberately ambitious; do not present them as achievable until a matched baseline and resource envelope have been reproduced. For closed-set RD-Bench, also report the benchmark's original candidate-set accuracy and calibration separately; a 92% result does not establish open-world scientific discovery.

### Paradigm-changing extensions to test, not merely describe

- **Decision sufficiency before mechanism identity:** preserve a mechanism equivalence class when the mechanism is not identifiable, while still issuing a decision when every compatible model implies the same action. Score decision regret and false-mechanism claims separately.
- **Evidence economy with active acquisition:** choose among retrieval, source verification, simulation, intervention, replication and abstention by expected decision value per inclusive cost. Use R-IDeA and robust EVSI to test the effect of representativeness and misspecification.
- **Direction portfolio rather than winner-take-all search:** preserve multiple causal/architectural directions with delayed payoffs; use verifier-grounded direction identity and budget accounting from #9230.
- **Prospective branch commitments:** before each costly action, freeze predicted outcomes, their probabilities, what decision each outcome changes, and stop/pivot criteria. Score those predictions against later independently adjudicated outcomes.
- **Discovery as a reproducibility event:** separate a plausible idea, a locally reproduced result, a held-out transfer, and an independently replicated result into distinct claim states.
- **Two-channel quality control:** combine adaptive, decision-relevant evidence acquisition with an independent randomized audit channel. The adaptive policy may optimize value; the independent channel estimates residual false-claim risk.

### Immediate implementation sequence

1. #11171 pins the actual state-contingent production policy callable and exposes it through the blind AgentView/AgentAction boundary; do not build another scheduler or posterior owner.
2. Adapt the exact ResearchState and typed decision vocabulary to RD-Bench cases; retain both candidate options and calibrated distributions, then add counterfactual and irrelevant-metadata mutation tests.
3. Freeze one small SDCB active-science campaign with exact equivalence, finite-budget near-equivalence, nuisance-family overlap, truth-absent, and decision-sufficient/non-identifiable episodes. Use #13567 solely as the evaluator oracle.
4. Run model-locked policies first (EIG, EVSI, R-IDeA, robust/minimax, direction-board, #11171 incumbent); only then run complete-system external benchmark submissions.
5. Freeze external leaderboard snapshots, model versions, prompts, tools, trial counts, costs and graders before each run. Publish per-task artifacts, failures and paired cluster intervals.
6. Do not claim “better than SOTA” until the paired held-out lower confidence bound exceeds the version-matched reference and all integrity / false-confidence gates remain satisfied.

## 4.5 Research-method quality, trajectory reliability and independent oversight

The Section 4.4 agent-workflow score should be decomposed into three additional scientific competencies. A system can score well on the final answer while choosing a weak experiment, silently diverging during execution, or failing to expose errors to a scientist.

### Experiment design is a distinct capability, not a by-product of planning or search

SCOPE evaluates experiment planning from 300 papers across 19 domains, separating high-level coverage (main experiments, ablations and analyses) from low-level configuration (datasets, baselines and metrics). Its paper reports a persistent configuration bottleneck: no evaluated model exceeded 3/5 on any low-level subdimension, and search access alone did not improve total design quality for most tested models; in one reported model it reduced the score and nearly doubled the redline rate.

**Symthaea consequence:** split the workflow into (a) scientific objective and falsifiable hypothesis, (b) evidence-backed dataset/baseline/metric selection, (c) executable protocol compilation, (d) preflight verification, and (e) execution. Reuse #12471 for method-to-execution contracts and #13818 for independent review. Search is a source-acquisition mechanism, not a configuration validator.

Before execution, require a machine-checkable manifest for dataset/version, population, split, preprocessing, exclusions, baseline identity, metric direction/definition, uncertainty method, stopping rule, compute/time budget and expected artifacts. The compiler validates structure; the auditor independently checks semantic/method fidelity. Deviations remain explicit and cannot be scored as faithful reproduction.

Proposed internal target (not result): on a reproduced SCOPE-compatible held-out subset, reach at least 4/5 mean on each low-level configuration dimension, zero critical split/dataset/metric substitutions, and no high-level plan-quality regression. Report each subdimension and human-adjudication agreement; do not optimize the average while hiding a fatal configuration error.

### Ideation must separate novelty from grounding and feasibility

AgentIdeaBench evaluates matched Static and Active exploration across 40 densely scored subfields and five disciplines. The paper reports active retrieval improves feasibility, clarity and specificity, while its measured originality change is not statistically significant. The authors also note that originality is their least reliable dimension and that a direct expert novelty study remains an important missing validation.

**Symthaea consequence:** do not let retrieval volume, fluency, feasibility or a critic's novelty score become a discovery claim. Score at least four distinct stages:
1. prior-art coverage as of the proposal timestamp;
2. originality against date-filtered prior art;
3. falsifiability and feasibility under real resource constraints;
4. empirical support and independent reproduction after the proposal.

Use a blind expert panel for close novelty calls, with calibrated adjudicator agreement and explicit known-rediscovery controls. Preserve negative results and previously rejected directions. Cross-domain analogy can be an ideation operator, but it earns novelty credit only when it yields a distinct, testable mechanism and survives prior-art search plus discriminating evidence.

Proposed acceptance gate (not result): active exploration must improve feasibility without reducing originality versus its paired static control; all claimed novel contributions must be judged against timestamped prior art and independently reviewed. If the human panel cannot reliably distinguish the originality cases, publish the metric as unresolved rather than treating the model critic as truth.

### End-state score is insufficient: measure trajectory stability and completion reliability

SCI-AGENT-EXAM contains 120 expert-curated tasks across 11 scientific fields and seven research task types. It requires both executable and textual artifacts and uses deterministic programmatic checks for code/artifacts plus a calibrated, expert-validated LLM judge for text. The current public snapshot lists GPT-6 Astra at 80.9 overall; the paper reports only 26.0% of its runs above the 0.95 success threshold. The leaderboard snapshot has not added more recent frontier model releases, so this is a versioned reference, not the current capability ceiling.

**Symthaea consequence:** record more than final-state correctness. Track artifact correctness, report fidelity, criterion-level regressions/dips, recovery, number of retries, and probability of a fully successful run. The candidate must preserve every consequential intermediate state and never silently overwrite a previously passing artifact with a weaker one.

Proposed target ladder (not results): reproduce the public SCI-AGENT-EXAM protocol; first beat 80.9 overall under the same model/harness/task snapshot, then target ≥84.0 overall and ≥30% Pass@0.95 on three or more independent trials, with no worsening of executable-artifact score or worst-domain results. Report mean, full-success rate, trial variability, cost and trajectory dips separately.

### Domain-specific real-workflow lane: biology

The V1 benchmarks.bio snapshot (updated 25 November 2026) combines 13 capability benchmarks and 23 individual leaderboards. Its published capability aggregate lists Claude Opus 5.5 with LSVP exemption at 58.0% and $4.82/run; the highest listed non-exempt configuration is Claude Sonnet 5.5/Pi at 52.5%, while GPT-6 Astra/Pi is 49.3% at $1.06/run with 13/13 coverage. Missing benchmarks are excluded from the weighted score, and LSVP is a safety-filter-exempt variant, so the scores must retain those provenance labels.

**Symthaea consequence:** benchmark the same V1 task set and report every native capability track, artifact coverage, per-run cost, token/turn usage and safety variant. Do not compare exempt and standard runs as if they had the same operating policy. Proposed initial target (not result): exceed 52.5% on a fully covered, non-exempt, version-matched run; treat 58.0% as a stretch comparison only with the same exemption and coverage regime. Require no hard safety or evidence-integrity failure.

### Scientist–auditor collaboration as a factorial experiment

Reuse #13818, not a new audit agent or checkpoint database. Its six-arm design should be reported as a factorial comparison of:
- self-review vs independent audit;
- scientist steering absent vs present;
- read-only deterministic replay as a verifier baseline.

The primary audit endpoint is **expert-adjudicated valid errors found per inclusive human-review hour**, not number of objections. Also report precision, severity-weighted recall, independently reproduced objection fraction, false objections, missed errors, claim/source traceability, replay cost and downstream correction impact. A valid audit finding must point to a reproducible defect in the immutable checkpoint; the audit agent cannot adjudicate final truth or promote claims.

For the human-steering factor, measure whether a scientist's intervention improves held-out decision quality and hypothesis quality after charging their time. A high intervention count is not inherently good. Record the exact checkpoint, scientist identity/authority, requested change and resulting child checkpoint; missing response is not approval.

DiscoveryAgent's reported 14× increase in valid errors with scientist-plus-audit support is a result from its own task setup, not a transferable benchmark target. Replicate the comparison and confidence interval locally before making any such claim.

### Combined claim-quality ladder

A scientific work product should advance through explicit states:
- **proposed** — hypothesis or method idea, with prior-art search and falsifiable predictions;
- **planned** — executable protocol and dataset/baseline/metric manifest reviewed;
- **executed** — immutable code, data, environment and run receipts;
- **audited** — independent reproducibility and objection resolution recorded;
- **supported** — evidence and independent adjudication support the scoped claim;
- **replicated** — result independently reproduced on the frozen task;
- **transferred** — result survives a preregistered held-out family, site, device, time or regime.

These states are not a confidence ladder and cannot be skipped by a high model score. A claim may remain proposed, unresolved or non-identifiable indefinitely. The system should be rewarded for a correct abstention and penalized for false promotion.

### Updated execution order

1. Keep #11171 as the only production policy/callable owner.
2. Use the typed ResearchState / claim-source relationship contract (#13665 / #13796) and the action-trace/replay contracts (#11015 / #13933) as the public input/output boundary.
3. Use #12471 to compile and validate the full experimental method configuration, not just code shape.
4. Use #13818 for append-only checkpoints, scientist steering and separate-process audit; the auditor only challenges and reproduces.
5. Use #13691/#13567 for active-science episodes and evaluator-only small-world regret.
6. Run matched SCOPE, AgentIdeaBench, SCI-AGENT-EXAM and benchmarks.bio subsets with exact task, harness, model, version, access, cost and coverage parity before external superiority claims.

No new evaluator, source classifier, planner, audit agent, checkpoint service or promotion authority is introduced by this scorecard section.

## 5. Shared architecture that unlocks the combined programme

### 5.1 Canonical source-event projection

Create one reviewed, versioned projection (reuse existing owners; do not create a duplicate ledger) with:
- event/document ID and source ID;
- canonical upstream root and source family;
- observed/available time and valid-from/valid-until;
- tenant/run/purpose scope;
- authority and uncertainty declaration;
- parent/supersession/retraction/revocation relation;
- content digest and transformation lineage;
- schema/codebook/version identity.

BM25, dense vectors, HDC, graphs, summaries and temporal state are derived indexes. Every returned memory ID resolves to this canonical record. A query at time (t) can be replayed exactly from the event log.

### 5.2 Evidence router with a deterministic safety shell

The router may learn which substrate is likely to help, but a separate deterministic shell enforces:
- scope and purpose;
- temporal eligibility;
- revocation and retention;
- trust/authority requirements;
- source-lineage/common-cause rules;
- action and cost permissions;
- explicit abstention on unknown identity or unsupported applicability.

This allows adaptive retrieval without allowing the router to self-authorize evidence.

### 5.3 Reproducible decision receipt

The receipt binds:
- episode and exact context digest;
- decision time and allowed offer list;
- selected action and exact payload digest;
- policy and reader digest;
- selected/retrieved evidence IDs;
- execution/job ID and upstream receipt;
- observation/evaluator receipt;
- resource measurement states;
- replay code/source revision.

Candidate and evaluator domains are separated by actual service/OS permissions; the evidence digest chain alone is insufficient.

### 5.4 Reliability-per-compute allocation

Treat compute as a budget the system allocates to reduce uncertainty:
- use cheap lexical/index lookup when it suffices;
- expand to vector/graph/HDC when additional evidence is likely to change a decision;
- spend quadrature/QMC effort in tails or ill-conditioned regions where it changes decisions;
- run higher-fidelity simulations or real experiments when expected value of information exceeds their cost;
- stop or abstain when further work cannot meet a decision threshold.

Measure decision improvement per unit of measured compute and cost, not merely tokens or nominal FLOPs.

## 6. Execution roadmap

### R0 — reference integrity (partially achieved)
- SDCB chain: exact-source vector, mixture, interval, adequacy and conformal references.
- MACE paired evaluator, cost receipt and LAFS helper.
- Scientific decision/replay and custody contracts.
- Record exact head SHA and workflow per claim; skipped CI is not a pass.

### R1 — matched conventional baselines (in progress)
- Temporal BM25 with scope and bitemporal filters.
- Weighted RRF across independently ranked channels.
- Matched LongMemEval/AML baseline reproduction.
- SDCB TVPACK/Genz/Miwa reference harness and independent high-precision fixtures.

### R2 — common event projection and replay adapter
- Join BM25 retrieval IDs to MACE decision receipts by exact event/root identity.
- Deliver decision-time prefixes only.
- Bind exact payloads to upstream AS-Bench execution receipts.
- Keep all hidden labels, future observations and evaluator artifacts in a separate custody domain.

### R3 — controlled experiments
- SDCB dependence, drift, rare-tail, common-cause and misspecification grid.
- MACE matched reader, BM25/RRF/dense/HDC/graph arms and memory ablations.
- Research-policy replay with unsupported-counterfactual tracking and live audit.

### R4 — independent confirmatory evaluation
- Freeze tasks, seeds, source commit, model/reader and scoring protocol before the run.
- Separate public development tasks from hidden held-out tasks.
- Use independent execution/evaluator custody and pristine replay.
- Publish full distributions, confidence intervals, subgroup failures, raw run manifests and reproducible artifacts.

### R5 — field deployment
Only after R4: shadow operation, human review, canary rollouts, rollback, incident and revocation drills, and continued monitoring for drift/common-cause failures.

## 7. What “better than SOTA” means here

A claim is defensible only when all of the following hold:
1. The exact external benchmark version and current leader are named.
2. The same split, reader, prompts, model restrictions, hardware and task budget are used.
3. The complete resource cost is reported, not just query latency.
4. The improvement is replicated on held-out tasks with uncertainty intervals.
5. Safety, privacy, provenance and revocation gates are reported separately and have no compensating score.
6. The raw per-task output, source revision, configuration, seeds and independent verifier evidence are retained.
7. The result is compared on a Pareto frontier; any new best point states where it wins and where it loses.

A single number cannot establish general SOTA. MACE may lead in decision utility while another system leads in raw recall; SDCB may lead in low-dimensional accuracy but not in high-dimensional throughput; Symthaea's replay system may lead in evidence custody while an external tool still leads in task success. State those outcomes separately.

## 8. Current implementation status and limits

As of this snapshot, the work has advanced the following reference surfaces:
- SDCB five-reference-chain validation previously passed 62/62 synthetic Rust tests on exact head in PR #13946.
- MACE paired evaluator/LAFS helper previously passed 45/45 tests on exact head in PR #13935.
- Scientific decision/replay contract previously passed 29 tests, rustfmt and strict Clippy in PR #13933.
- New child PRs #13952 (bivariate quadrature budget/scaling), #13953 (temporal BM25/RRF plus strict candidate adapter), #13954 (replay-capsule structural checks) and #13959 (AR(1) block-dependence stress) have now produced exact-head reference-test evidence described in Section 9; none is a completed system or scientific qualification.

These are isolated reference tests. They are not real data calibration, external benchmark submissions, production retrieval integration, physical measurements, independent security audits, full workspace CI or demonstrated SOTA results. Keep that distinction in every subsequent report.

## 9. New exact-head evidence since the first scorecard draft

### 9.1 SDCB bivariate numerical reference

The bivariate rectangle increment in #13952 passed its exact-source Rust 1.96 workflow with **23/23 tests** on head 1d00cbe17b99a87b3bf916280a2b05e30451ce37. This includes scale-relative covariance symmetry, channel permutation, positive-unit-rescaling and aggregate table-wide evaluation-budget controls. It remains a 2D conditional-integration reference; it does not validate the general multivariate solvers in #13951.

### 9.2 Block-conformal dependence stress

The standalone Monte Carlo experiment in #13959 passed Rust 1.96 formatting and **4/4 harness tests**, and completed 72 conditions × 5,000 trials on exact source head `419dda0bba8d6f664b78a201172942c078e2c3cd` (360,000 simulations; [workflow run #14 and artifact](https://github.com/Luminous-Dynamics/symthaea/actions/runs/45998581987)). The grid covers stationary Gaussian AR(1) coefficients 0, 0.5, 0.8, 0.9, 0.95, 0.99; block lengths 1, 8, 32; an independent-block control; and serial-path embargo gaps 0, 16, 64.

For block length 1, the serial-path no-gap alarm rate rises from 9.92% at rho=0 to 12.16% at rho=0.90, 13.54% at rho=0.95 and **17.62% at rho=0.99**. At rho=0.99, the 95% Wilson interval for the no-gap rate is 16.589–18.701%, entirely above nominal 10%. A 16-step gap still yields 12.64% (11.748–13.590%); a 64-step gap yields 10.40% (9.584–11.277%, interval includes nominal). Block length does not universally repair the issue: rho=0.99/block length 8/no gap yields 12.12% (11.244–13.054%), while length 32/no gap yields 10.94% (10.105–11.835%).

This is a specific stationary Gaussian AR(1), block-maximum result, not a universally safe embargo, proof of block exchangeability, nonstationary guarantee or physical-data calibration. It motivates negative-rho, heavy-tail, common-cause, drift, independent-seed and held-out-regime controls. Full 72-row CSV and manifest are retained in the artifact linked from #13959.

### 9.3 MACE conventional baseline and source-event adapter

The temporal BM25/RRF reference in #13953 now:
- partitions documents/postings by scope and knowledge-time before scoring;
- computes BM25 corpus statistics over the eligible scope/time cohort;
- preserves event ID, source ID, authority tier, uncertainty, parent relation, source family/root and bitemporal validity;
- resolves retrieval hits to canonical source-event metadata by exact event ID;
- audits archive-wide parent edges as resolved, missing or cross-scope and reports strongly connected cycle components using an iterative SCC pass. This is an integrity diagnostic only: it does not inherit or grant parent authority.

On exact PR head `bc7366eb9820d32fa1d67702bfd96f69de9168d4`, the standalone BM25/RRF reference passed **19/19 tests**, the committed-lock strict adapter passed **26/26 tests**, `cargo metadata --locked` and strict Clippy with `-D warnings` passed, and the parent MACE DSM evaluator passed **45/45 tests** ([adapter workflow](https://github.com/Luminous-Dynamics/symthaea/actions/runs/46000791431), [BM25 workflow](https://github.com/Luminous-Dynamics/symthaea/actions/runs/46000791464), [DSM evaluator](https://github.com/Luminous-Dynamics/symthaea/actions/runs/46000791393)). The adapter includes required-nullable-field and size/top-k negative controls, plus exact-ID lineage closure/cycle audit. BM25 query receipts also expose document eligibility checks, corpus-length scans, query-term lookups, posting rows visited and scored-document counts; these are deterministic operation counts, not measured hardware cost. This is still a fixture-to-index reference, not a live retrieval service.

### 9.4 Scientific replay custody

The structural replay-capsule child #13954 passed rustfmt, 39/39 isolated tests and strict Clippy under its nested lock on exact head 74ae6e3c7958ae7cec5a3a1cd3d22e64622a5577. The tests include deliberately re-hashed malformed artifacts, distinguishing semantic structure checks from raw byte tamper detection. Service/OS evaluator isolation and production package execution remain open.

### 9.5 Workspace metadata recovery

The workspace-glob/lockfile repair #13840 passed the exact-head glob checker, cargo metadata --locked and scoped root-workspace STT formatting on head 08bf1f34b7bef0b131926de0cb5339d445ca989c. This is a build-discovery/metadata gate, not full application/workspace test success. Excluded source-only directories are not thereby repaired.

### 9.6 SDCB identical-marginal dependence-regime shift

The new child research increment #13986 tests calibration-to-test dependence changes while preserving the stationary one-step `N(0,1)` marginal. Its exact-source Rust 1.96 workflow passed formatting and **4/4 unit controls** and completed 36 conditions × 2,500 trials = **90,000 trials** on source head `969a7344bb392ab63b98d40e43b2d673bfe640ec` ([workflow](https://github.com/Luminous-Dynamics/symthaea/actions/runs/46002051862), [CSV + manifest](https://github.com/Luminous-Dynamics/symthaea/actions/runs/46002051862/artifacts/15197659974)).

In the independent-block arm, calibration `rho=0.05` → test `rho=0.95` reduced the nominal 10% alarm rate to **3.88%** for block length 8 (Wilson 95%: 3.191–4.711%) and **3.72%** for length 32 (3.046–4.536%). Reversing the shift (`rho=0.95` → `0.05`) raised the rate to **27.12%** for length 8 (25.413–28.897%) and **30.12%** for length 32 (28.353–31.948%). At block length 1, independent arms remain near nominal in both directions (10.96% and 9.40%), showing that the aggregate score—not the pointwise marginal—has shifted. Serial-path dependence creates an additional violation: the high-to-low, length-1, no-gap condition is 18.68%.

The implication is sharper than ordinary marginal drift: **an unchanged pointwise marginal does not guarantee an unchanged block-score distribution or exchangeable block scores**. Embargo addresses cross-block dependence only; it cannot repair the independent-block score-distribution shift. An evidence-qualified SDCB result therefore needs an explicit, frozen dependence/score envelope and should return `NotQualified` or `Inconclusive` when diagnostics cannot establish that the observed regime remains inside it.

This is specific to two stationary Gaussian AR(1) regimes, block-max scores, 39 calibration blocks, alpha=.10, the tested grid and 2,500 trials per condition. The 36-condition report is an initial diagnostic, not a universal guarantee, production detector or physical calibration. Confirmatory work must add independent seeds, more trials, negative rho, heavy tails, latent common causes, drift and held-out regimes.

### 9.10 SDCB cross-channel common-cause shift (pending exact-head qualification)

PR #14049 adds a two-channel stationary AR(1) stress where both channels retain the same unit Gaussian or standardized Student-t(5) marginal and the same within-channel `rho=0.80`, while contemporaneous cross-channel correlation changes from `gamma=0.05` to `gamma=0.95`. The score is the maximum absolute residual across both channels and all ticks.

This is a genuinely new dimension relative to the univariate rho grid: a channel-wise marginal and autocorrelation check can pass while the joint max-score distribution shifts. The grid covers both change directions, no-change controls, block lengths 1/8/32, two marginal families, 39 calibration blocks and 2,500 trials per condition (24 conditions / 60,000 trials). Seven exact-source controls check marginal variances, realized cross-channel correlation, both direction-specific alarm intervals, replay and complete report rows. Code and report specification are in [PR #14049](https://github.com/Luminous-Dynamics/symthaea/pull/14049) and `docs/research/SDCB_CROSS_CHANNEL_COMMON_CAUSE_STRESS_V1.md`.

The exact-head workflow is queued, so no empirical result is claimed yet. The confirmatory extension should test covariance rotations, 3–8 channels, multiple latent factors, lagged cross-channel structure, time-varying dependence and held-out multivariate telemetry. A validated dependence envelope must cover the joint score—not merely each channel's marginal fit.

### 9.9 SDCB score-aware reduction of redundant dependence experiments

A metamorphic control in #14045 encodes the alternating-sign transform for stationary AR(1): `Y_t=(-1)^t X_t` maps `rho` to `-rho` while preserving `|X_t|` at every tick. For symmetric Gaussian innovations, and for the positive path-level Student-t(5) common scale, the distribution of the max-absolute-residual block score is therefore identical under `rho` and `-rho`.

The exact-source test checks equality of each absolute residual and block maximum for both marginals and `|rho|={0.05,0.50,0.95}`. The workflow is pending. If qualified, the practical result is methodological: do not spend simulation budget on negative-rho duplicates for this stationary AR(1) + absolute-score model. Spend it on genuinely distinguishable regimes—non-AR lag structures, multivariate covariance rotations, asymmetric innovations, signed scores, common-cause shocks and drift. This identity does not transfer to asymmetric innovations or directional score functions.

### 9.8 MACE pairwise evidence interaction (implementation pending exact-head qualification)

The MACE evaluator already supports paired memory/no-memory decision loss, frozen-policy leave-one-out interventions, cost receipts and world-cluster bootstrap. A new research increment adds second-order pairwise interventions to expose non-additive evidence effects that leave-one-out cannot identify.

For full evidence set `E`, define `L(E)` as evaluator decision loss under the exact frozen downstream policy and seed. The pair interaction for items `i,j` is:

`I_ij = L(E\\{i,j}) - L(E\\{i}) - L(E\\{j}) + L(E)`

Positive values mean substitution/redundancy: either item can compensate for the other. Negative values mean complementarity/synergy: the pair is more valuable together than individual removals suggest. The implementation requires a complete single-item ablation set, every unordered pair, and exact policy/system/episode/seed identity. It uses integer micro-loss and does not expose evaluator labels to candidate memory.

Current code head `f096fdc571a6328f42d3611ea35ed1d9755703cf` contains `score_pairwise_interactions` and a known-answer redundancy/synergy control. The spec is in `research/mace-decision-sufficient-memory-bench-v0-1/PAIRWISE_EVIDENCE_INTERACTION_V0_1.md`. Exact-head Rust 1.96 formatting, compilation and test result are pending; the earlier parent evaluator and BM25 adapter qualifications remain the evidence baseline. This extension is not a Shapley value or an estimate of interactions beyond second order.

The next target is to report pair interaction distributions by evidence family, conflict type and cost, with uncertainty resampled at independent world-cluster level. Do not sum pair interactions into a single “memory value” because higher-order effects can remain and pair terms can double-count.

### 9.7 Heavy-tailed common-scale regime shift

The next child increment, #13990, repeats the identical-marginal shift in a standardized Student-t(5) common-scale process. Each independent path draws one `V~chi-square(5)` and shares `sqrt(3/V)` across its Gaussian AR(1) observations. This gives each one-point observation a standardized Student-t(5) marginal with unit variance while introducing a simple shared latent scale factor.

The exact-source Rust 1.96 workflow passed **7/7 controls** and completed 72 conditions × 2,500 trials = **180,000 simulations** on code head `394919be85c65b04510a8ca33b490116a56dcff2` ([workflow](https://github.com/Luminous-Dynamics/symthaea/actions/runs/46003315297), [CSV + manifest](https://github.com/Luminous-Dynamics/symthaea/actions/runs/46003315297/artifacts/15199048151)). Controls included a 50,000-sample unit-variance check, a one-point rank-invariance check, both dependence-shift directions and deterministic replay.

In the t5 independent-block arm at nominal alpha=.10, calibration rho=.05 → test rho=.95 yields 4.56% alarms for block length 8 (Wilson 95%: 3.810–5.450%) and 3.92% for length 32 (3.227–4.754%). Reversing the shift produces 23.20% at length 8 (21.587–24.895%) and 22.60% at length 32 (21.003–24.281%). Independent length-1 controls remain near nominal (10.60% and 9.64%). This reproduces the qualitative over-conservative/undercoverage asymmetry with heavier tails and a shared latent scale.

The serial-path results share the same path-level scale through calibration and test, so that positive scale cancels in within-path rank comparisons; the serial arm is mainly a cross-block-dependence test here. Independent-block rows are the clean evidence for the t5 block-score distribution shift.

This is still a narrow synthetic model with 2,500 trials per condition. It is not a proof, general common-cause model, production detector or physical data result. The confirmatory path is 50,000–100,000+ trials per condition, independent seeds, negative rho, skew, time-varying scale, multi-factor/common-mode shocks and held-out regimes.

### 9.8 Fail-closed dependence-envelope decision contract

The research now has a small typed decision contract in #13991. A positive `QualifiedWithinEnvelope` state requires exact scope/marginal/model/score/block/alpha matching, a passing external model-adequacy diagnostic, sufficient sample count, canonical observation evidence digest, and the entire observed rho interval inside frozen inclusive bounds. Disjoint intervals or metadata mismatches are `NotQualified`; partial boundary overlap or incomplete diagnostics are `Inconclusive`.

The exact-source Rust 1.96 workflow passed Rustfmt, `rustc -D warnings --test`, and **10/10 negative-control tests** on code head `0708fb7a5e20d0c0321aa8f5c3a07e04a15a4031` ([workflow](https://github.com/Luminous-Dynamics/symthaea/actions/runs/46004270255)). The tests cover inclusive limits, fully contained/disjoint/overlapping intervals, model-family pairing, metadata mismatch, rejected/inconclusive diagnostics, insufficient samples, malformed digest and non-finite/reversed intervals.

This contract is only a fail-closed reference. It shape-checks receipt hashes but does not authenticate them, verify reviewer authority, prove model adequacy or detect arbitrary dependence. Production integration must place the envelope behind an independent attestation boundary, bind it to exact calibration/test episode custody, and show that an inconclusive/out-of-envelope state cannot authorize a downstream mutation. It does not change production SDCB assessment.

## References

- Terminal-Bench Science leaderboard snapshot and reported-vs-official distinction (OpenResearch, synced 28 November 2026): https://www.openresearch.sh/evals
- FrontierPhysics leaderboard snapshot (Priors, data as of 28 November 2026): https://www.mypriors.com/benchmarks/frontierphysics
- Decision-Centric Research / RD-Bench preprint (31 October 2026): https://arxiv.org/abs/2611.00774
- AstaBench current benchmark inventory: https://allenai.org/asta/bench
- SciAgentArena benchmark and native task metrics: https://sciagentarena.github.io/
- R-IDeA, AISTATS 2026 / PMLR 300: https://proceedings.mlr.press/v300/tang26d.html
- SCOPE, autonomous experimental design benchmark (4 August 2026): https://arxiv.org/abs/2608.03501
- AgentIdeaBench, active versus static scientific ideation (7 September 2026): https://arxiv.org/abs/2609.07611
- SCI-AGENT-EXAM, scientific workflow benchmark (23 October 2026): https://arxiv.org/abs/2610.32441
- SCI-AGENT-EXAM public score snapshot (retrieved 28 November 2026): https://www.benchmarkalmanac.com/benchmarks/sci-agent-exam/
- DiscoveryAgent, scientist steering and checkpointed audit agents (16 November 2026): https://arxiv.org/abs/2611.21950
- benchmarks.bio V1 leaderboard and methodology (updated 25 November 2026): https://benchmarks.bio/

- AML Cycle 2 official results (19 November 2026): https://agentmemoryleaderboard.ai/results/cycle-2/
- LongMemEval-V2 official benchmark and live LAFS board: https://xiaowu0162.github.io/longmemeval-v2/
- TimlMemBench, Findings of EMNLP 2026: https://aclanthology.org/2026.findings-emnlp.507/
- MemSecBench: https://arxiv.org/abs/2607.27080
- mvtnorm algorithm documentation: https://cloud.r-project.org/web/packages/mvtnorm/mvtnorm.pdf
- Doubly blocked Genz, Computational Statistics 2026: https://doi.org/10.1007/s00180-026-01794-8
