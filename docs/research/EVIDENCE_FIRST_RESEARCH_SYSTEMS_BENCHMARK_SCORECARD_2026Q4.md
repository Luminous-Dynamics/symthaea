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

The commercial Textual leader MemoraX reports **69.26**; commercial Multimodal MemoraX reports **53.68**; commercial Coding reports **68.00% overall resolved**. These are different tracks and metrics, not directly comparable numbers. AML also reports capability-level scores (for example temporal reasoning, governance, privacy and streaming), which are more informative than the overall rank for system diagnosis.

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

### 4.3 Scientific decision replay — research-policy quality

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

The standalone Monte Carlo experiment in #13959 passed 3/3 unit controls and completed 12 conditions × 5,000 trials on exact source head 30a508248f31d595d930224c852925e4762fc3d9. At rho=0.95 with 39 calibration blocks and nominal alpha=0.10, serial-path unit-length blocks without embargo produced a 13.54% alarm rate (Wilson 95%: 12.62–14.52%), above the 10% nominal target. Independent unit blocks produced 10.56% (9.74–11.44%). Embargoes of 16 and 64 steps returned the serial-path unit-block rate to 10.12% and 10.28%, with intervals containing 10%. At block lengths 8 and 32, all tested serial-path intervals contained 10%.

This is a specific stationary Gaussian AR(1), block-maximum result, not a universally safe embargo, proof of block exchangeability, nonstationary guarantee or physical-data calibration. Full rows and manifest are retained in the workflow artifact linked from #13959.

### 9.3 MACE conventional baseline and source-event adapter

The temporal BM25/RRF reference in #13953 now:
- partitions documents/postings by scope and knowledge-time before scoring;
- computes BM25 corpus statistics over the eligible scope/time cohort;
- preserves event ID, source ID, authority tier, uncertainty, parent relation, source family/root and bitemporal validity;
- resolves retrieval hits to canonical source-event metadata by exact event ID.

On exact head 8461d6073b9440789119b96152ba1ba5987ecd47, the dependency-free BM25 reference passed 17/17 tests and the parent MACE DSM evaluator passed 45/45 tests. A separate strict Serde adapter subsequently passed formatting, generated-lock test execution, strict Clippy and fixture/schema checks with **21/21 tests** on its adapter-validation run; the exact committed-lock workflow is being rerun after moving the generated Cargo.lock onto the feature branch. Treat the 21-test run as source/test evidence; do not yet state that the newest committed-lock revision is green until the final locked run completes.

### 9.4 Scientific replay custody

The structural replay-capsule child #13954 passed rustfmt, 39/39 isolated tests and strict Clippy under its nested lock on exact head 74ae6e3c7958ae7cec5a3a1cd3d22e64622a5577. The tests include deliberately re-hashed malformed artifacts, distinguishing semantic structure checks from raw byte tamper detection. Service/OS evaluator isolation and production package execution remain open.

### 9.5 Workspace metadata recovery

The workspace-glob/lockfile repair #13840 passed the exact-head glob checker, cargo metadata --locked and scoped root-workspace STT formatting on head 08bf1f34b7bef0b131926de0cb5339d445ca989c. This is a build-discovery/metadata gate, not full application/workspace test success. Excluded source-only directories are not thereby repaired.

## References

- AML Cycle 2 official results (19 November 2026): https://agentmemoryleaderboard.ai/results/cycle-2/
- LongMemEval-V2 official benchmark and live LAFS board: https://xiaowu0162.github.io/longmemeval-v2/
- TimlMemBench, Findings of EMNLP 2026: https://aclanthology.org/2026.findings-emnlp.507/
- MemSecBench: https://arxiv.org/abs/2607.27080
- mvtnorm algorithm documentation: https://cloud.r-project.org/web/packages/mvtnorm/mvtnorm.pdf
- Doubly blocked Genz, Computational Statistics 2026: https://doi.org/10.1007/s00180-026-01794-8
