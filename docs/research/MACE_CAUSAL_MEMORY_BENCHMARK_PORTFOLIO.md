# Causal Agent Memory: Benchmark Portfolio and SOTA Targets

**Snapshot date:** 2026-11-22  
**Programme:** Broca / Symthaea MACE — Memory-Augmented Causal Experience transfer  
**Status:** Research protocol and targets. Targets below are aspirational; they are not achieved results.

## 1. North-star claim

A capable agent memory system must do more than recover facts. It must improve later decisions by retaining *which action caused which outcome, under which conditions, for which objective, with what evidence and uncertainty*—and then refuse to apply that knowledge when its assumptions no longer hold.

The target architecture is therefore:

`raw events → provenance-bound experience → causal/effect contract → applicability test → decision or safe abstention → observed outcome → correction/revocation`

Memory is treated as an evidence-bearing control surface, not merely a retrieval index. A comparative action choice requires qualified evidence for both alternatives; one-sided evidence must lead to abstention or a bounded information-gathering request.

## 2. Current external benchmark landscape

Scores below are **not interchangeable**. Different benchmarks, tracks, models, judges, and interaction protocols support different claims.

| Benchmark / track | Published result to anchor against | What it measures | What it misses for MACE |
|---|---|---|---|
| [MemoryArena (ICML 2026)](https://proceedings.mlr.press/v306/he26am.html) | No single universal headline score in the paper abstract; the key finding is that strong long-context recall does not ensure success on interdependent multi-session tasks | Memory-Agent-Environment loops across web navigation, preference-constrained planning, progressive information search, and sequential formal reasoning | Explicit source-family attacks, revocation latency, and statistical transportability gates are not its central metric |
| [AMA-Bench (ICML 2026)](https://proceedings.mlr.press/v306/zhao26bs.html) | AMA-Agent: **57.22%** average accuracy, **+11.16 percentage points** over the strongest baseline reported by the authors | Long-horizon real/synthetic agent trajectories, including recall, causal inference, state updating, and state abstraction | Governed revocation and intervention-effect transfer under controlled regime changes are not the sole evaluation focus |
| [Agent Memory Leaderboard — Cycle 1, Textual / Commercial](https://agentmemories.ai/results/cycle-1/) | MemoraX v0.5: **58.02 overall**; **29.99** on Context Learning & Execution | Historical reference for explicit recall, temporal reasoning, governance, execution and privacy | Cycle 1 is not the current leaderboard and does not prove causal transfer |
| [Agent Memory Challenge — Cycle 2, Textual / Open-source Methods](https://agentmemoryleaderboard.ai/results/cycle-2/) | Resurface: **61.22/100**, rank **1/89**; fact recall 62.98, relational/multi-hop 58.94, temporal 57.76, governance 76.55, workflow execution 59.67, epistemic safety/privacy 68.59, streaming 49.65 | Current official open-method textual baseline under the frozen Add/Search → platform Answer/Eval protocol | Does not measure causal intervention transfer or treatment-effect transport; this is an AML score, not MACE |
| [Agent Memory Challenge — Cycle 2, Coding / Open-source Methods](https://agentmemoryleaderboard.ai/results/cycle-2/) | InsightMem: **63.33% overall resolved**, rank **1/20**; new-feature 74.51%, bug-fix 57.58% | Current official open-method coding baseline | Repository task resolution is not the same as evidence-qualified causal memory; report relevant/noisy and new-feature/bug-fix separately |
| [Agent Memory Challenge — Cycle 2, Multimodal / Open-source Methods](https://agentmemoryleaderboard.ai/results/cycle-2/) | Khipu: **43.26/100**, rank **1/18**; retrieval 50.28, relational 45.82, temporal 34.23, continual learning 38.45, personalization 11.04, evidence governance 31.29 | Current official open-method multimodal baseline | Small division and different metric family; no comparison to textual or coding scores |
| [Agent Memory Challenge — Cycle 2, Commercial leaders](https://agentmemoryleaderboard.ai/results/cycle-2/) | MemoraX: Textual **69.26/100** (1/62 commercial entries), Coding **68.00% resolved** (1/11), Multimodal **53.68/100** (1/4) | Commercial context across the three independent tracks | Commercial and open-method divisions are separately ranked; do not treat the commercial score as the open-method target |
| [TimlMemBench (Findings of EMNLP 2026)](https://aclanthology.org/2026.findings-emnlp.507/) | **208** reviewed samples, **1,566** sessions, **1,715** questions, and **1,821** governance/activation cases | Raw-native governed memory: revocation, supersession, scope/boundary control, provenance, activation, restraint | Dialogue-oriented; does not by itself measure action-effect transport in continuous agent-environment loops |
| [WorldMemArena (May 2026)](https://worldmemarena-mem.github.io/) | **400** multi-session multimodal tasks: 38 Lifelong Evolution samples (684 sessions, ~1.9k images) plus 362 Agentic Execution samples (~7.8k sessions, ~13.7k images); stage labels for write/maintain/retrieve/use | Multimodal memory across evolving personal/project state and real GUI/embodied trajectories, including original visual evidence and evidence chains | No single headline score should be inferred from the abstract/site summary; use its official tasks and stage-level evaluator rather than comparing MACE synthetic utilities |
| [ATM-Bench / ATM-Bench-Hard (leaderboard snapshot 2026-10-04)](https://atmbench.github.io/leaderboard.html) | Visible memory entries: ATM-Bench Graphon V1 **81.71% QS / 89.78% Recall@10**; ATM-Bench-Hard Memexa v2 **63.57% QS** | Multimodal memory quality, retrieval under hard distractors, token/cost efficiency and same-model harness comparisons | Distinct boards/configurations/judges; not interchangeable with AML, MemoryArena or MACE transfer scores |
| WorldMemArena | 400-task, four-stage multimodal action-world lifecycle; no single universal headline score used here | Reproduce stage-level write/maintain/retrieve/use results; then target **+5 pp** downstream task success over the strongest reproduced same-backbone baseline with no stage/family regression worse than **2 pp** | Same original images/frames, task split, tool permissions, harness, model and context/call budgets; preserve raw visual/action evidence rather than caption-only surrogates |
| [StateMemBench (Aug 2026)](https://arxiv.org/abs/2608.19652) | 234 multi-session state-tracking scenarios; StateMem reports **0.205 → 0.363** on DeepSeek-V4-Flash (1.8×) and **0.149 → 0.233** on Qwen-3.5-9B against its strongest reported baselines | Current vs superseded facts and relational dependencies over long interactions | Not primarily a causal intervention-transfer or signed-provenance benchmark; backbone, task split and protocol-specific scores must remain attached |
| [TEM (Oct 2026)](https://arxiv.org/abs/2610.21900) | **420 questions / 600 sessions**; historical labels improved reported overall hit rates by **17.14–55.95 percentage points** over otherwise source-ordered facts; 96.67–100% on the historical-residence subset in the reported settings | Versioned fact state, source order, and explicit historical-vs-current answer conditioning | A retrieval/history benchmark, not causal action transfer; results are model/protocol-specific and should be reproduced with a fixed evidence budget |
| [ACL 2026 Memora / FAMA](https://aclanthology.org/2026.findings-acl.1337/) | Introduces **Forgetting-Aware Memory Accuracy (FAMA)**; no single universal score used here | Weeks-to-months user conversations across remembering, reasoning and recommending, with penalties for obsolete-memory use | Personal-memory lifecycle benchmark; not an environment-intervention benchmark. Distinct from Microsoft Research's separate Memora system below |
| [Microsoft Research Memora (June 2026)](https://www.microsoft.com/en-us/research/blog/memora-a-harmonic-memory-representation-balancing-abstraction-and-specificity/) | Reports **86.3% LoCoMo** and **87.4% LongMemEval**; about half as many memory entries as Mem0 and up to 98% token savings versus full-context baselines in its evaluated settings | A harmonic representation balancing abstraction and specificity on long-context memory tasks | Microsoft Research report/blog results; protocol-specific and not directly comparable to AML, MemoryArena or MACE. Reproduce the exact backbone, judge, token accounting and memory interface before claiming a delta |
| [ConsolidationBench (Sep 2026)](https://arxiv.org/abs/2609.33013) | One reported phase: **+22.7% task success at 7× compression**. A later external-transfer study reports pooled Spearman **ρ = −0.24**, n=12, confidence interval spanning zero, between consolidation-quality score and real transfer accuracy | Decisions to retain, compress, promote, abstract or forget; governance and auditability | Its key warning is directly relevant: consolidation quality is not a validated proxy for downstream transfer; score quality, governance and transfer independently |
| Microsoft Research Memora (exact LoCoMo / LongMemEval setup) | 86.3% LoCoMo; 87.4% LongMemEval reported | First reproduce within 1 pp, then target **+2 pp** on a held-out extension with no higher memory-token budget; on mutable-state episodes target **≥90% fewer snapshot tokens** than store-everything without utility regression | Match backbone, task split, judge, memory API and token accounting. These vendor-reported results are not a universal leaderboard score |
| [AgentMemoryBench (Lifelong Agent @ ICLR 2026)](https://openreview.net/attachment?id=MSXbrNExax&name=pdf) | Workshop benchmark; no single cross-track headline score used here | Joint system/experiential memory and personal/factual memory across online learning, transfer and forgetting | Useful for co-evolving procedural and personal memory; its protocol is distinct from MACE causal transfer |
| Anthropic CL-Bench 1.0 database task | Claude Managed Agents: Opus 5.5 **0.838** with memory vs **0.655** baseline; Sonnet 5.5 **0.833** vs **0.644**; 30 questions | **≥0.86** for the declared target model on the same database task, plus a multi-task held-out extension | Small, vendor-authored 30-question task. Reproduce exact model, harness and grader; report paired uncertainty and memory cost; do not compare to AML scores |
| [Mem0 Platform v3 benchmark report](https://github.com/legendPerceptor/mem0-benchmarks) | Repository reports **92.5% LoCoMo** and **94.4% LongMemEval** at Top-200; Top-50 reports **91.8%** and **94.8%**, respectively | A vendor-maintained benchmark harness for memory-augmented QA; includes question-type breakdowns and token accounting | Not a neutral shared leaderboard. Verify backbone, judge, retrieval depth, memory pipeline, benchmark revision and scoring before head-to-head comparison; do not pool its scores with Microsoft Memora, Redis or AML |
| [LongMemEval reproduction (Redis, 2026-09-21; page updated 2026-11-17)](https://redis.io/resources/agent-memory-benchmark-report/) | Redis Remis + Instruct: **86.5%** task-averaged accuracy on LongMemEval; reported at about **$0.0703/session** on the report's common GPT-4o reproduction | Long-term question answering and memory retrieval, including cost and query latency | QA accuracy should not be read as multi-session action competence; this is a vendor-authored reproduction report, not a cross-benchmark SOTA |
| [Survey of World Model Benchmarks (snapshot 2026-11-08)](https://world-model-benchmarks.github.io/) | Catalogues **130** benchmarks across seven target categories, two protocol classes, two metric levels, and four data sources | Clarifies that claims must specify target, protocol, metric family and data source | A taxonomy, not a direct memory-agent score |

### Interpretation

The industry now has distinct work on retrieval, state tracking, obsolete-memory penalties, consolidation decisions, interdependent agent-environment loops, multimodal memory under hard distractors and cost constraints, and representations that balance abstraction against specificity. Microsoft Research Memora is a separate memory-representation system; ACL Memora/FAMA is a separate benchmark/paper despite the naming overlap. ATM-Bench adds same-model/different-harness comparisons that help expose whether apparent gains come from the memory substrate or the agent harness. Its public results are configuration-specific and must not be pooled with other leaderboards.

StateMemBench is especially important because it isolates current-vs-superseded state rather than treating all prior facts as equally useful. FAMA turns obsolete-memory use into an explicit penalty. ConsolidationBench warns that optimizing an internal consolidation score may not improve external transfer at all. StateMemBench and TEM reinforce that a memory item must carry not only what it says, but which revision and time coordinates make it valid. WorldMemArena extends this into the action-world loop: MACE's representation selector should be scored separately at write (persist/locator/cache/discard), maintain (refresh/revoke), retrieve (selective evidence), and use (freshness at decision time). MACE's (valid_at, known_at) replay should be directly evaluated against StateMemBench-style supersession and TEM-style historical questions, in addition to causal-action transfer. MACE should therefore report at least four independent axes: state correctness, governance/lifecycle correctness, causal transfer and downstream task utility.

### Prior interpretation (retained)

The field has moved beyond “can the system remember a conversation?” MemoryArena couples memory with later action; AMA-Bench emphasizes machine-generated trajectories and causal/objective information; TimlMemBench tests governed use from raw sessions; AML separates tracks and divisions. Cycle 2 is now officially published: open-method leaders are Resurface 61.22 (Textual), InsightMem 63.33% resolved (Coding), and Khipu 43.26 (Multimodal). Commercial leaders are MemoraX 69.26, 68.00% resolved, and 53.68 respectively. These are independent boards and cannot be combined or compared with MACE synthetic scores. LongMemEval remains a retrieval/QA control.

The opportunity is not to claim that MACE's synthetic score beats one of these scores. The opportunity is to add a missing, independently testable capability layer: **evidence-qualified causal transfer under distribution shift, source corruption, and lifecycle correction**, then reproduce the official benchmarks under their own protocols.

## 3. Target scorecard

These are programme targets for future matched evaluations, not current measured performance.

### External benchmark targets

| Benchmark | Baseline anchor | MACE target | Qualification rule |
|---|---:|---:|---|
| MemoryArena | Use the official baseline under the selected agent/backbone; do not infer a single SOTA from abstract | **+5 pp** mean task success over no-memory and strongest reproduced memory baseline; no task family worse by more than **2 pp** | Same base agent, task episodes, tool permissions, and token/compute budget; paired 95% interval for improvement excludes zero |
| AMA-Bench | AMA-Agent **57.22%** reported | **≥62%** with the exact published model, judge, data and evaluation protocol | Reproduce the 57.22 result first; publish recall, causal, state-update and state-abstraction sub-scores separately |
| Agent Memory Leaderboard — Textual / Open-source Methods | Cycle 2 Resurface **61.22/100**; governance **76.55**, workflow execution **59.67**, safety/privacy **68.59**, streaming **49.65** | **≥66 overall**, governance **≥80**, workflow execution **≥65**, safety/privacy **≥75**, streaming **≥55** | Same next-cycle track/division and Add/Search contract; match model profile, data, latency and resource budgets; report every capability |
| Agent Memory Leaderboard — Coding / Open-source Methods | Cycle 2 InsightMem **63.33% resolved**; new-feature **74.51%**, bug-fix **57.58%** | **≥68% overall resolved**, new-feature **≥78%**, bug-fix **≥62%** | Same coding task/version, relevant/noisy splits, toolchain and time/token budgets |
| Agent Memory Leaderboard — Multimodal / Open-source Methods | Cycle 2 Khipu **43.26/100**; evidence governance **31.29**, retrieval **50.28** | **≥48 overall**, evidence governance **≥38**, retrieval **≥55** | Same multimodal track/data, image byte limits, Add/Search interface and evaluator |
| TimlMemBench | 1,821 governance/activation cases | **Zero known unauthorized activations** on the supplied revoked/scope-incompatible cases; report provenance recovery and legitimate activation separately | Run the raw-native track; system must ingest raw sessions itself, not receive a prebuilt canonical memory pool |
| LongMemEval | Redis GPT-4o reproduction **86.5%** at about **$0.0703/session**; Microsoft Research Memora reports **87.4%**; Mem0 benchmark repo reports **94.4%** Top-200 and **94.8%** Top-50 in its own setup | Maintain three separate reproduction tracks: Redis matched setup **≥87% at ≤$0.07/session**; Microsoft Memora matched setup **≥89%**; Mem0 Top-200/Top-50 setup: first reproduce the reported score within 1 pp, then target **+1 pp** on a held-out extension at no higher memory-token budget | Do not combine incompatible backbones, judges, retrieval depth or memory pipelines. Report ingest time, query latency, tokens, cost and accuracy together |
| ATM-Bench | Graphon V1 **81.71% QS / 89.78% Recall@10** | **≥84.7% QS** and **≥90% Recall@10** on the same board | Same dataset revision, answer/judge model and retrieval budget; retain frozen evidence and per-question reports |
| ATM-Bench-Hard | Memexa v2 **63.57% QS** | **≥66.6% QS** at no higher measured token/cost budget | Same hard track, judge and memory configuration; report distractor level, cost and per-question evidence |
| ATM-Bench harness-effect control | Same model weights across several agent harnesses | At most **2 pp** performance spread at matched token/tool budgets, or fully report the harness effect | Same questions, memory, judge and model; do not attribute harness gains to memory quality |

### Additional external qualification targets

| Benchmark | Proposed target | Qualification rule |
|---|---:|---|
| StateMemBench | At least **+5 percentage points** absolute current-state accuracy over the strongest reproduced same-backbone baseline | Official split and scoring; matched context/token budget; no future-knowledge leakage; separately report superseded-state errors |
| Memora / FAMA | At least **+5 points** over a matched reproduced memory baseline; **zero** obsolete-memory reuse on deterministic retraction controls | Run the official task protocol and FAMA scoring; report forgetting/false-deletion trade-offs |
| ConsolidationBench | Reproduce the official protocol, then target **+5 pp** downstream task success at a matched memory/storage budget without governance regressions | Do not optimize or claim victory on consolidation-quality score alone; require held-out external transfer, independent labels and separate poison/reversibility/audit metrics |
| AgentMemoryBench | Reproduce its procedural/personal memory protocol, then report per-track gains under matched backbone and budget | Keep personal-memory privacy/governance separate from procedural transfer and causal evidence claims |

### MACE-specific qualification targets

| Dimension | Target |
|---|---:|
| Duplicate event-ID acceptance as independent evidence | **0** |
| Use of retracted, unreconciled, untrusted, confounded, or incomplete-horizon evidence in a qualified decision | **0** in known-control suites |
| Transfer across a different optimization objective | **0** |
| Transport across a different estimand (population, treatment contrast, outcome, or horizon) | **0** on held-out mismatch controls |
| False rejection of the same estimand across different valid randomized designs | **≤20%** on held-out positive controls |
| Canonical estimand descriptor coverage (once semantic descriptor is integrated) | **100%** of transported evidence binds population, treatment versions, outcome/unit, follow-up, censoring and target weighting |
| Known Session-C retraction reused by the next decision | **0** |
| False acceptance of a deliberately changed mechanism | **≤0.1%** across at least **10,000** held-out negative trials; report a binomial confidence bound |
| Acceptance of genuinely invariant, independently witnessed few-shot transfers | **≥80%** on held-out positive cases |
| Mean normalized utility lift vs no-memory | **≥0.05**, with paired cluster-bootstrap 95% lower bound **>0** |
| Worst-family regret | **≤0.05** normalized utility units |
| Confidence calibration (once the policy emits probabilities) | ECE **≤0.03** and Brier score better than frequency-only and retrieval-only baselines |
| Decision-time overhead | Report p50/p95/p99 latency and CPU/energy per decision; target p95 deterministic policy overhead **<100 ms** on the declared reference host |

Do not report an internal metric as a leaderboard win. Synthetic controls qualify implementation properties; external benchmarks qualify external validity.
**Estimand claim boundary:** v0.14 compares receipt-bound estimand digests for identity consistency. It does not yet parse or independently validate the semantic descriptor behind the digest. The production verifier must bind a versioned canonical specification (population, treatment versions/contrast, outcome and unit, follow-up horizon, censoring/intercurrent-event strategy and target weighting) into the receipt and digest. A digest match alone is not proof of a correct estimand.


### Harness and provider portability qualification

ATM-Bench demonstrates why memory evaluations need a model/harness matrix. Freeze the same task episodes, memory, judge and model weights, then vary only the harness (OpenAI Agents SDK, Anthropic Managed Agents, and local Symthaea/Broca). Report task success, evidence fidelity, tool-policy violations, context/token use, cost, p50/p95 latency and recovery from session interruption. A result is not attributable to the memory substrate if the harness changed at the same time.

The architectural goal is stable contracts across replaceable implementations: append-only session log, provider-neutral memory Add/Search, reasoning/harness adapter, isolated execution sandbox, and independently verified authority/provenance. Model/provider upgrades should rerun the same frozen qualification suite; brittle prompts and orchestration workarounds should not become permanent compatibility contracts.

## 4. Paradigm shifts to pursue

### A. From text retrieval to causal contracts

A memory item should compile to a typed contract:

- action and outcome;
- objective and operating regime;
- preconditions and excluded conditions;
- measured effect distribution and uncertainty;
- intervention and confounder-control status;
- source event IDs and source-family lineage;
- observation horizon and delayed consequences;
- revocation/supersession state;
- applicable policy and authority scope.

A vector similarity hit is a candidate pointer, not permission to apply a memory.

### B. From exact-scope rejection to evidence-tested transportability

Exact-scope matching is safe but can under-generalize. Blind similarity transfer is unsafe. The intermediate capability is to test whether the action-effect relationship remains invariant in the target regime, with target-regime witnesses, disjoint source families and upstream roots, and conservative measurement-error intervals over the treatment contrast rather than absolute outcome means.

The current `invariant_transport` policy is only a heuristic prototype. It compares the regime-level treatment contrast `mean(A) - mean(B)`, so a common-mode baseline shift does not invalidate an unchanged effect. The fixed and raw-session integration suites now contain positive controls for this case. It requires the worst-case difference between source and target treatment-effect intervals to fit within a fixed 0.12 normalized-utility margin. This deterministic bound is not a statistical confidence interval; the measurement bounds and source-root IDs require independent attestation. A later version should use preregistered statistical equivalence tests and calibrated uncertainty before any real deployment claim.

### C. From accuracy to regret, calibration, and governed-use

A memory system should be scored on:
- utility and regret of downstream actions;
- wrongful action and wrongful abstention;
- calibration and selective risk;
- unsupported causal claims;
- stale evidence reuse after correction;
- cost and latency across ingestion, consolidation, retrieval, verification and update.

A system that answers more questions but takes worse actions is not a better memory system.

### D. From passive memory to active information acquisition

When no memory is applicable, the agent should identify the cheapest safe experiment that distinguishes plausible mechanisms. Measure value of information, experiment cost, time-to-identification, and whether the experiment changes the next decision. Abstention is a safety property; active, low-risk probing is the capability that makes abstention useful.

### E. From trust labels to independently verifiable provenance

A different source-family string is not proof of independence. The production gate must use source-ledger attestations, event identity, common-cause analysis, and replayable receipts. Evaluator custody must also be independent of the candidate package.

## 5. Protocol needed to make a credible SOTA claim

1. **Reproduce each official benchmark first.** Lock model/backbone, prompts, judges, data versions, token/tool budgets, and stopping rules. No score pooling across benchmarks.
2. **Use matched baselines.** No-memory, full-context, vector retrieval, frequency-only, exact-scope causal memory, graph/tool-augmented memory, invariant transport, and active-probe policies must share the same agent and resource budgets.
3. **Separate development from evaluation.** Split by world seed, source family, mechanism, task template and environment—not random rows. Keep evaluator labels outside candidate dependencies and ultimately in a separately controlled repository/service.
4. **Use paired cluster-level statistics.** Resample environments/tasks, not correlated questions from the same environment. Publish effect sizes, confidence intervals, worst-family results, and failure examples.
5. **Attack the evaluator and evidence pipeline.** Include event-ID aliases, copied/near-duplicate sources, common-cause sources, fabricated success receipts, delayed effects, retractions, supersession, wrong objectives and adversarial regime changes.
6. **Measure the full lifecycle.** Include ingestion/consolidation, index/storage growth, retrieval, verification, tool calls, tokens, CPU/GPU, energy, p50/p95/p99 latency, and cost per successful downstream task.
7. **Publish the negative results.** A transport policy that refuses a true invariant effect or accepts a false one must be visible in the report, not hidden in an aggregate score.

## 6. Current MACE implementation status

- **Receipt-bound consolidation bridge:** exact source receipt bundles, bitemporal coordinates, policy/capability identity and destination artifact digests are bound into a canonical proposal; a separate verifier interface gates Session-B release. The current integration verifier is a fixture, not production Mycelix cryptography. A verified promotion is usable only against the exact knowledge-time snapshot it attests; a later transfer must rebuild the pack and proposal.
- **MACE-lite fixed fixtures:** 22 deterministic scenarios and six candidate policies: no-memory, frequency-only, scope-aware, exact-scope causal-qualified, invariant transport, and active probe.
- **Generated stress suite:** 256 deterministic clusters × 21 scenario families = **5,376** cases. Includes confounding, regime/objective shifts, source retraction, unreconciled tool success, untrusted evidence, incomplete delayed horizons, correlated duplicates, event-ID aliasing, shared-common-root attack, error-interval negative control, common-mode baseline-shift positive control, negative experience, invariant few-shot transport, mechanism break, unseen-target active probing, one-sided evidence requiring a probe, and mismatched-estimand transport rejection.
- **Statistics:** paired cluster-bootstrap utility lift versus no-memory (2,000 resamples); per-family reporting for causal-qualified, invariant transport and active probe; Session-B/Session-C known-retraction control; probe requests/resolutions and net utility after the 0.12 normalized-utility probe cost. The active policy emits a typed, scope-bound request with both actions, a replicate count and maximum normalized-utility budget; the evaluator rejects mismatched-scope, unsupported-action, unsupported-replicate and under-budget requests.
- **Raw-native ingress and consolidation handoff:** broca-mace-ingress plus broca-mace-consolidation; ingress has 36 deterministic mock-verifier tests, the consolidation policy has 7 unit tests and 5 cross-crate raw-session integration tests, including the Session-A→B→C retraction chain and common-mode-shift treatment-effect transfer, tracked by [#12771](https://github.com/Luminous-Dynamics/symthaea/issues/12771). It admits only Mycelix-style `receipt_attested` authority, uses signed server `received_at` for `known_at`, distinguishes Active/Retracted/Superseded lifecycle, validates source/calibration/causal/outcome receipt material, re-verifies duplicates and compares full receipt bundles. Ingress results are non-constructible outside the crate. `build_consolidation_evidence_pack` produces a deterministic receipt-preserving pack with source-family/root counts derived from verified lineage. The MACE-specific proposal consumer now consumes action-scoped summaries and binds all promotion thresholds into each source-linked proposal; invalid policy configuration fails closed. #12731's broader retention/privacy/reversibility policy does **not yet consume the pack**. No live Mycelix cryptographic adapter is claimed.
- **Raw-session replay:** the JSONL evaluator module has 30 authored tests for (valid_at, known_at), server-receipt chronology, A/B/C correction, late arrivals, retroactive/future-effective corrections, invalid targets and correction chronology.
- **Learned predictive-salience baseline:** a deterministic logistic forecaster uses seven observable features, trains on a training partition, Platt-calibrates on a disjoint calibration partition, and evaluates on held-out task families/upstream roots. Its split manifest carries separate dataset, task-family and root digests. Its base-rate Brier comparator uses training prevalence only, not held-out labels. The model/calibrator can emit a raw `FutureUsePredictionV1` bound to the trained model and calibrator digests; it remains ineligible for surfacing until the separate calibration verifier issues a receipt. Four module tests and two evaluator integration tests are authored but have not run. This is a synthetic candidate baseline, not a utility/SOTA result.
- **Predictive-salience forecast admission (MACE v0.10):** raw model forecasts are no longer eligible for candidate surfacing. They must be wrapped in an independently verified calibration receipt binding the prediction/evidence, model/configuration, scope and bitemporal snapshot, training/calibration/held-out datasets, disjoint task-family/source-root splits, minimum cluster counts, Brier-vs-base-rate, ECE/worst-group ECE, verifier and validity window. Surfacing-plan digests bind prediction and calibration-receipt digests. The current verifier is a fixture only, not live Mycelix cryptographic verification.

- **Causal identification and estimand consistency (MACE v0.14):** experiences bind method, design/estimand digests, assignment-overlap bounds, covariate balance and interference status. Exact-scope and transport policies require receipt-verified randomized-design metadata; observational and unsupported quasi-experimental evidence remains retrievable but cannot be promoted. Estimand digests must remain consistent within each action arm, between A/B and across source/target regimes; matching numeric effects for different populations, outcomes, treatments or horizons are not transportable. All seven identification fields are preserved in ingress receipts and included in the canonical consolidation source-binding digest. Propensity/balance thresholds are prototype settings, not universal standards; production randomization/estimand verification remains unimplemented.
- **Target-model-conditioned salience (MACE v0.16):** each predictive-salience example, training/calibration/held-out split, model, calibrator and raw prediction carries one target-agent SHA-256 fingerprint. Mixed-model batches/splits and prediction-to-model mismatches are rejected. Ten unit tests and three evaluator integration tests are authored, not executed.
- **Candidate/evaluator separation:** candidate policy and evaluator are separate Cargo packages; the evaluator source remains public in the same repository, so it is not a sealed independent evaluator.
- **Cross-crate regression:** evaluator integration test `ingress_consolidation_transfer` composes receipt-verified raw ingress, action-specific proposal-only consolidation, an independently verified promotion receipt, Session-B choice, Session-C retraction, stale-receipt rejection, and refusal to re-promote Action A when only one source remains.
- **Verification:** source-level checks have been performed and the integration test is authored. Rust compile/tests/rustfmt have not run in this environment (Cargo/rustc/rustfmt are unavailable); draft PR checks are skipped, not passes. Test counts are source inventory only. No SOTA or physical causal-discovery result is claimed.


- **Provider claim-span grounding:** 9 authored tests in `ingress/src/claims.rs` cover exact Anthropic/OpenAI spans, capture mutation, span digest, non-string pointer, UTF-8 boundaries, assertion digest, provider/thread/native identity and invalid spans. These tests have not run. The validator returns untrusted claims only; it is not a candidate-memory path.

The method-aware extension is specified in the [Causal Memory Identification & Transfer Challenge (CMIT) protocol](./MACE_CAUSAL_MEMORY_IDENTIFICATION_CHALLENGE_V0_1.md).

## 7. Immediate next tranche

1. Implement a production Mycelix verifier for canonical event bytes, signed authority receipt chain, source lineage, calibration, outcome reconciliation and lifecycle receipts (#12771, #9075, #297).
2. Wire ActionConsolidationEvidenceSummaryV1 into the separately governed lifecycle policy (#9139 / #12731), retaining source ancestry and reversible state transitions; explicitly model privacy erasure and supersession events rather than treating active-only ingress as the entire lifecycle. Before calling this raw-native transcript evaluation, add transcript turn/claim extraction with exact source-span receipts. Require per-action root/family support rather than pooled global episode counts.
3. Replace the cross-crate integration test's deterministic fixture verifiers with production Mycelix/Symtropy verifiers and signed receipts; keep the current raw Session-A → consolidation → Session-B → Session-C fixture as the exact interface regression (#9516).
4. Move hidden labels and evaluator ownership to independent custody (#11816), and add source-/seed-/mechanism-held-out worlds.
5. Replace the fixed deterministic interval margin with calibrated uncertainty and preregistered statistical equivalence; retain the deterministic check as an inexpensive negative control.
6. Add official MemoryArena / AMA-Bench / StateMemBench / FAMA adapters only after licensing and evaluator contracts are reviewed; prepare Add/Search for the next eligible Agent Memory Challenge cycle.
7. Measure paired utility, worst-family regret, calibration/coverage, lifecycle cost, supersession/retraction latency and p50/p95/p99 performance under matched budgets (#8427).
8. Obtain an exact-head runner-backed compile, formatting, unit/integration-test and replay bundle for the five-crate workspace; publish immutable run receipts and keep authored test counts separate from executed results.

## References

- He et al., [Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks](https://proceedings.mlr.press/v306/he26am.html), ICML 2026.
- Zhao et al., [AMA-Bench: Evaluating Long-Horizon Memory for Agentic Applications](https://proceedings.mlr.press/v306/zhao26bs.html), ICML 2026.
- [Agent Memory Leaderboard — Cycle 1 results](https://agentmemories.ai/results/cycle-1/), published 2026-08-12.
- Zhou et al., [TimlMemBench](https://aclanthology.org/2026.findings-emnlp.507/), Findings of EMNLP 2026.
- [Redis Agent Memory benchmark report](https://redis.io/resources/agent-memory-benchmark-report/), 2026-09-21; page updated 2026-11-17.
- Zhang et al., [A Survey of World Model Benchmarks](https://world-model-benchmarks.github.io/), final manuscript snapshot 2026-11-08.


Each experience now carries a claimed absolute measurement-error bound. Invariant transport requires:

`abs(mean_source - mean_target) + source_mean_error_bound + target_mean_error_bound <= 0.12`

This is a conservative deterministic interval-containment check, not a statistical confidence interval. Error bounds are candidate-visible claims in this prototype and need sensor/protocol attestation; future work should add calibrated uncertainty models, preregistered equivalence tests and independent target-regime witnesses.

## 8. Provider and benchmark ecosystem snapshot (2026-11-22)

OpenAI's Agents SDK distinguishes durable sandbox memory from conversational session history. Its documented design uses a small summary for initial context, progressive retrieval of detailed rollout summaries, and two-stage extraction/consolidation of memories. The November 17 API update also provides two- and eight-hour background Responses windows for long-running work; these can support offline dreaming, provided every run pins its as-known evidence snapshot and cannot backfill future information. Anthropic's Managed Agents describes independent session-log, harness and sandbox interfaces, and cautions that harness assumptions can become stale as models improve. Anthropic's long-running-agent pattern uses an initializer plus incremental sessions that leave durable handoff artifacts.

For Symthaea, these are useful architectural patterns, not systems to copy wholesale:
Anthropic's November 6 CL-Bench 1.0 database-task report finds that memory can materially improve fresh-session performance: Opus 5.5 scores **0.838 with memory versus 0.655 without**, and Sonnet 5.5 **0.833 versus 0.644**, on a 30-question task. It also reports that a hard-coded “record only failures” rule reduced memory gains for stronger models; allowing the model to judge salience worked better, while dreaming checked notes against original transcripts and repaired stale beliefs. **Design implication:** let the model propose salient successes, strategies, constraints and failures; then independently qualify source grounding, causal status, scope, calibration, reversibility and promotion. Do not replace model salience judgment with a failure-only rule. Treat CL-Bench as a vendor-reported single-task reference, not an independent universal SOTA result.
- retain append-only, bitemporal event history and provenance in Mycelix;
- have Symthaea produce versioned, evidence-bound, progressively disclosed memory views;
- keep OpenAI/Anthropic/local model adapters replaceable;
- keep Symtropy execution isolated and separate from reasoning;
- treat model-generated memory and tool responses as untrusted until verified;
- rerun behavioral qualification on provider/model/SDK upgrades.

TimlMemBench's raw-native track exposes a remaining gap: MACE now tests receipt-gated structured interaction envelopes, receipt-preserving pack projection, action-scoped evidence summaries and a proposal-only consolidation consumer. The fixed/generated MACE evaluators still use structured Experience fixtures, and ingress does not yet parse natural-language transcript turns or bind extracted claims to exact text spans. The broader governed lifecycle policy in #12731 (privacy erasure, reversible storage transitions, retention and forgetting) still does not consume the MACE evidence pack. The next seam is transcript/span provenance plus a single raw transcript → verified evidence → governed lifecycle → Session-B decision → Session-C correction evaluator.

The Agent Memory Challenge Cycle 2 leaderboard was published on **November 19, 2026**. Official open-method results report Resurface leading Textual (**61.22/100**, 1/89), InsightMem leading Coding (**63.33% resolved**, 1/20), and Khipu leading Multimodal (**43.26/100**, 1/18). The commercial leaders are separately ranked: MemoraX Textual **69.26/100**, Coding **68.00% resolved**, and Multimodal **53.68/100**. These are track-specific metrics and must not be compared numerically across modalities or divisions. Cycle 2 submissions and evaluation are closed; prepare the Add/Search adapter for the next eligible cycle. See the [official Cycle 2 results](https://agentmemoryleaderboard.ai/results/cycle-2/) and [competition protocol](https://agentmemoryleaderboard.ai/competition/).


### Mycelix lineage and measurement assurance

The synthetic `source_root_id` and `measurement_error_bound` fields are not trust anchors. Mycelix's FIN-SYS source profile provides a useful existing pattern: multiple publications reusing one upstream BIS/IMF series count as one provenance family. Production MACE should consume equivalent verified lineage receipts, including acquisition mechanism, artifact digest, source family, upstream series/root, transformation and supersession history.

A measurement-error bound must be tied to an identified sensor/method, units, calibration revision, operating interval and evidence record. It must not be an agent-editable confidence claim. This connects MACE to the bitemporal source-event ledger (#9075) and independent evaluator provenance (#11816).


### Raw-native, bitemporal and consolidation-gated coverage

MACE v0.7 adds a provider-neutral JSONL ingestion path that derives experiences from pre-registration, dispatch, tool-result, world-observation and correction events. It evaluates state as of `(valid_at, known_at)`, preserves late-arriving observations, and computes SHA-256 digests over the original session bytes and each event line. Corrections apply only when recorded by `known_at` and effective within their valid-time interval at `valid_at`; explicit `valid_from_ms` permits retroactive correction, otherwise effective time defaults to `occurred_at_ms`. A correction must target an existing `world_observation` and cannot predate the source in sequence, occurrence or recorded time.

Transfer candidates must pass four independent registry boundaries: a Symtropy-style execution receipt; a reconciled world-outcome receipt; a Mycelix-style source-lineage/calibration receipt; and a consolidation receipt that explicitly promotes the exact source event to `PromoteToProcedure`. The raw payload cannot author its own intervention status, pre-registration, observed utility, provenance root, measurement bound or consolidation promotion. An independently verified outcome receipt supplies the authoritative utility and unit; the raw claim must match exactly, and the server-received authority receipt cannot predate the outcome observation or arrive after `known_at`.

This is a provider-neutral ingestion contract, not yet a provider-native parser for OpenAI/Anthropic logs or an end-to-end integration with #12731. The four registries are injected test seams, not cryptographic verification or live connectors. The forgetting/erasure dispositions prevent candidate reuse but do not establish physical deletion. Raw event retention remains separate from active/procedural memory.

### Producer/consumer integration status (2026-11-18)

Draft PR #12731 now emits a candidate-side source-bound consolidation receipt proposal from exact event-level source references and has a structural evaluator for binding and common-root attacks. Draft PR #12743 requires a separately verified PromoteToProcedure receipt for the exact raw observation event before transfer memory admission. They are not yet wired directly together: #12731's proposal is unsigned and uses fixture digests; #12743's receipt registries are injected test seams. Issue #12846 specifies the canonical bitemporal Mycelix/Symtropy verifier adapter, policy/authority digest, signature, supersession and privacy semantics needed for production.

### Canonical 256-bit source-root identity (MACE v0.6)

The numeric `source_root_id: u32` is retained only as a legacy/display identifier. **It is not an independence key.** Candidate qualification, evidence-pack root counts, batch consistency and cross-regime source/target disjointness use the typed `SourceRootDigest([u8; 32])` carried by verifier-returned lineage, receipt bundles, consolidation evidence references and candidate experiences.

The canonical digest contract for a production Mycelix source-ledger verifier is:

```text
SHA-256(
  "mycelix:source-root:v1\0"
  || canonical_upstream_identity
)
```

`canonical_upstream_identity` should identify the upstream dataset/series or primary source lineage, not a particular publisher's copy, acquisition artifact or transformation. Multiple publishers reusing one BIS/IMF series therefore share one root; separately sourced series with the same legacy numeric ID remain distinct when their verified canonical roots differ. Release/revision identity, exact artifact digest, acquisition mode and transformation digest remain separate receipt fields.

The digest is only a stable identity. Trust still requires independent verification of the lineage receipt's issuer, signature, source policy revision, scope, validity, revocation status and canonical payload binding. The all-zero digest is unknown and fails closed. The current MACE `EvidenceVerifier` implementations remain fixtures/mocks; no production Mycelix cryptographic verifier is claimed.


## 9. Provider-native trace capture and evidence boundaries

Provider/session trace means what the agent said, requested, called or received.
It is distinct from execution authority, independently observed world outcomes,
source provenance/calibration, and consolidation promotion.

Anthropic Managed Agents persists event history with native event IDs and
processed_at timestamps. Its event_start and event_delta are stream-only
previews, not persisted history; reconnect should seed seen IDs from event
history and skip overlapping live events. OpenAI Agents SDK session items and
RunItem wrappers distinguish messages, tool calls, tool outputs and handoffs.
The providers expose different identity and timestamp affordances; adapters
preserve what is actually available and never invent action occurrence times.

MACE provider adapters map both into untrusted trace classes. Tool-call outputs
remain claims even when labelled as tool results. They are not world
observations without independently verified execution and outcome receipts.
The normalized provider-trace record carries native payload and capture-line
digests while the original capture artifact remains separately controlled
under the applicable privacy policy.

This adds a provider-ingest path, not an outcome-verification path. It closes
the trace-format interoperability gap without weakening the execution,
outcome, source/calibration and consolidation receipt boundaries. These are
version-specific research adapters and require golden capture fixtures and
requalification when provider event schemas change.


### Multi-agent thread ordering

The provider capture envelope accepts an optional thread_id (or stream_id). Sequence numbers are validated within each thread, not globally across concurrently executing subagents. The normalized event retains provider_sequence and provider_thread_id. MACE's raw JSONL requires one ingest sequence, so the exporter assigns a separate import-order sequence while preserving the provider's per-thread ordering metadata. This records a partial order without claiming a causal order between independent threads.

## Provider trace qualification benchmark

See [MACE Provider-Native Trace Integrity and Outcome Grounding Benchmark v0.1](MACE_PROVIDER_TRACE_INTEGRITY_BENCHMARK_V0_1.md). It defines separate tracks for provider trace fidelity, outcome grounding, source/measurement/causal qualification, consolidation lifecycle, cross-session transfer, and cross-provider harness robustness. It deliberately scores these axes separately from memory recall and official leaderboard accuracy.


## 10. Official Add/Search benchmark adapter

The MACE workspace now includes a separate broca-aml-adapter Rust crate. It encodes the AML Add/Search boundary: ordered Add messages, strict user_id isolation, source-session grouping through session_id, exact response echoes, Add idempotency, Textual chunking through an injected frozen word counter, track-aware Search validation, formal Top K 100, rank-preserving results and multimodal media validation.

The public guide requires Add success only after persistence and immediate Search visibility; retries reuse the same request_id and identical payload; Textual chunks at 20 messages or 2,000 frozen-adapter-counted words; user_id is the sole retrieval-isolation key; and Multimodal uses inline Base64 Data URIs with 10 MiB per image, 30 MiB per Add request and 30 MiB per Search response. Evaluation data is job-only and must be deleted within 30 days after run completion.

**Cycle 2 official results, published November 19, 2026:** the official results page publishes nine distinct track/division leaderboards and 289 entries. Open-method leaders are Resurface Textual 61.22/100 (1/89), InsightMem Coding 63.33% resolved (1/20), and Khipu Multimodal 43.26/100 (1/18); commercial leaders are MemoraX Textual 69.26/100 (1/62), Coding 68.00% resolved (1/11), and Multimodal 53.68/100 (1/4). Textual/multimodal scores and coding resolution rates are different measures. See the [official results](https://agentmemoryleaderboard.ai/results/cycle-2/). The platform owns Answer/Eval; participants must provide hosted Add/Search endpoints. See the [competition contract](https://agentmemoryleaderboard.ai/competition/).

Next-cycle targets: 100% smoke-contract compliance; zero cross-user leakage; zero premature Add acknowledgements; zero idempotency payload rebinding; exact parity with the frozen word counter; and at least +5 percentage points over the strongest same-track/division official baseline, with governance/safety gains measured separately.

The crate is a protocol core only: it does not host endpoints, persist memory, implement embeddings/retrieval ranking, include the verified frozen word counter or perform a Full evaluation. It has 15 authored tests, not runner-verified.

### Raw outcome-receipt mismatch gate (2026-11-19)

The raw-JSONL evaluator's outcome receipt now binds the exact preserved observation-line SHA-256, event/action identity, observation timestamp, normalized utility value and unit. Any mismatch or replay rejects the observation before candidate memory; a receipt no longer silently repairs a conflicting source event. The evaluator has 30 authored raw-ingestion tests for payload mismatch, cross-event receipt replay, timestamp drift and unit substitution. These are authored only, not executed in the current environment. Policy qualification additionally rejects source-family ID zero, aligning direct candidate calls with ingress' unknown-family rejection.

### MACE v0.6 treatment-effect transport

The policy compares `mean(A) - mean(B)` across regimes, rather than separate absolute action means. It requires target witnesses for both actions and independently qualified paired source-regime evidence, uses disjoint family/root lineage, conservatively adds measurement-error bounds, and abstains if the aggregated treatment-effect interval crosses zero. A common-mode baseline-shift fixture verifies that stable treatment effects can transfer even when both raw outcome levels move. The deterministic 0.12 margin remains a prototype constant, not a statistical confidence interval.


### Raw-native treatment-effect transport integration (MACE v0.6)

The consolidation integration suite now has a Session-A → Session-B transport control built from raw `RawInteractionV1` records. Session A is ingested at an explicit (valid_at, known_at) snapshot, grouped into action-/scope-specific evidence packs, promoted through proposal-only consolidation, and released only after an independent receipt verifier accepts the exact source bindings. Session B contributes one receipt-verified witness per action under a shifted regime with a +0.15 common-mode baseline shift. Exact-scope policy abstains; invariant transport uses the stable A-B treatment effect.

The fixture verifier is synthetic. This validates crate composition and receipt/evidence interfaces, not live Mycelix cryptographic verification or physical causal inference.


### Provider/model/harness reproducibility contract

MACE ingress now binds each provider trace capture to a canonical run manifest: provider/session/run, exact model ID/release, API surface/version/endpoint, harness revision, instructions and tool-schema digests, sandbox image, memory policy/state, runtime configuration, original capture digest, run time window and retention mode.

This supports matched experiments for same-model/different-harness and different-model/same-harness. The manifest is self-declared metadata: binding proves consistency with one capture, not that the provider actually executed the declared model or harness. Provider/runtime attestation remains a separate qualification gate. This creates the experiment contract needed to distinguish improvements from the memory substrate from improvements caused by model or harness changes.

### Predictive memory surfacing: bitemporal and scope-safe

The `consolidation::salience` module compares failure-only, recency-first, frequency-first and model-proposed predicted-future-value selection under a bounded token budget. Predictions are timestamped, tied to exact evidence digests and model/configuration identities, and validated against a future horizon. The surfaced plan is deterministic and digest-bound.

Every candidate is now pinned to source scope and the exact `(valid_at, known_at)` snapshot from its receipt-derived evidence reference. All selection modes enforce this, not merely the predicted-value mode. A later retraction or a memory from a different objective/regime cannot be surfaced merely because it is recent, frequently accessed or tagged as a failure. The plan digest includes both temporal coordinates. This remains context selection, not promotion or authority.

Anthropic's November 6, 2026 *The components of agent memory* report motivates the future-use salience hypothesis. Its vendor-reported CL-Bench database task found gains from memory for Opus 5.5 and Sonnet 5.5, while a fixed "record only failures" instruction reduced the gains for stronger models. These are internal-task vendor results, not directly comparable to MACE or AML. The next research step is held-out future-use calibration: compare actual downstream utility/regret under equal token budgets, report ranking quality and probability calibration, and test whether predicted-value surfacing beats failure-only, recency and frequency baselines without increasing stale or cross-scope activation.

### Predictive-salience calibration protocol

A generated development control now specifies 256 task clusters × 16 memory candidates = 4,096 future-use forecasts under a fixed 64-token context budget. It compares failure-only, recency-first, frequency-first and predicted-future-value selectors. It separates:
- forecast calibration (Brier score, base-rate Brier, ten-bin ECE);
- selected expected downstream utility and one sampled realised outcome;
- expected utility per surfaced token;
- regret against a hidden-label oracle knapsack under the same budget.

The selector receives only candidate-visible forecasts, source evidence, access telemetry and the fixed budget. Future reuse labels and true value remain evaluator-side. The development fixture has authored thresholds for forecast improvement, policy lift and oracle regret; no measured result is claimed until the Rust test runs. A credible external claim still requires held-out task-family, source-root, model-version and mechanism splits, with independent labels and paired uncertainty.

Anthropic's *The components of agent memory* (November 6, 2026) motivates this research direction but is not a directly comparable benchmark. Its vendor-reported CL-Bench database task found that persistent memory raised Opus 5.5 from 0.655 to 0.838 and Sonnet 5.5 from 0.644 to 0.833; a fixed "record only failures" instruction reduced gains for stronger models. The lesson is not that predicted-value salience already wins, but that memory selection rules should be tested against downstream reuse instead of assuming failure-only or frequency-only retention is optimal.

## 9. November 20, 2026 update: predictive salience and industry results

### Anthropic: memory is future-oriented, not just failure logging

Anthropic's November 6 engineering report, [The components of agent memory](https://www.anthropic.com/research/components-of-agent-memory), presents memory as a prediction about what future sessions will need. On its CL-Bench database task, the reported stateless → memory mean rewards were Opus 5.5 **0.655 → 0.838**, Sonnet 5.5 **0.644 → 0.833**, Opus 5 **0.564 → 0.828**, Opus 4.7 **0.383 → 0.700**, and Sonnet 5 **0.306 → 0.610**. These are Anthropic's own task results and are not directly comparable to AML, MemoryArena or MACE.

The same report gives two important design lessons:
- On Pokémon Red, Fable 5 wrote 9 memory files and earned four badges, while Sonnet 3.7 wrote 413 files across 225 folders and earned one badge; memory that duplicates currently observable state can become a maintenance burden.
- A fixed instruction to “record only failures” reduced the memory gain for Opus 5 by 0.234 and Opus 4.7 by 0.150 on the evaluated task. Stronger models benefited from judging salience rather than following that hard-coded rule.

For Symthaea, this supports testing predicted future-use value, but not trusting a model's own salience estimate. Keep failure-only, recency and frequency as baselines; require independent calibration evidence and held-out evaluation for predicted-value surfacing. See [Anthropic's report](https://www.anthropic.com/research/components-of-agent-memory).

### OpenAI: memory, session and execution remain distinct

OpenAI's Sandbox Agents documentation explicitly distinguishes durable memory files from conversational Session history. It documents progressive disclosure (summary first, detailed memory only when needed), separate read-only/generate-only modes, and per-agent memory layouts. The Nov 19 API changelog added GPT-Live-1.1 multilingual speech improvements and standby behavior; that is relevant to future voice-session capture, but not evidence of better long-term memory by itself. See [Sandbox Agents memory](https://developers.openai.com/api/docs/guides/agents/sandboxes) and the [API changelog](https://developers.openai.com/api/docs/changelog).

Anthropic's Nov 12 release of self-hosted Managed Agents allows the full agent service, agents, sessions and session history to run on customer infrastructure, with a selectable model provider. Together, these developments reinforce the provider-neutral session/harness/sandbox architecture in this portfolio.

### Agent Memory Challenge Cycle 2: November 19 announcement

As of **November 20, 2026**, Gedo Memory has published a November 20 report stating that it placed third on the Cycle 2 Commercial Products Text Memory leaderboard with an overall score of **62.28**, and fifth on the Closed-Source Methods Text Memory leaderboard. This is a vendor-authored report of the announced results. At the time of this snapshot, the [official competition page](https://agentmemoryleaderboard.ai/competition/) still displays its November 19 expected-release notice rather than complete published result tables; therefore this portfolio does not treat 62.28 as an independently verified official leaderboard extraction or claim a ranking for MACE.

Cycle 2's October 31 submission and November 4 evaluation deadlines have passed. Prepare the provider-neutral Add/Search contract for the next eligible cycle; do not imply that the current MACE harness entered Cycle 2. See [Gedo's report](https://gedo.ai/labs/aml-cycle-2).

### Updated competitive interpretation

The frontier is moving from static retrieval toward memory that helps models learn from feedback, maintain representations, predict future utility, and operate across long sessions. MACE's potential differentiator is to pair that adaptive salience with:
- bitemporal, receipt-bound evidence;
- explicit causal vs associational status;
- independent source-root and task-family splits;
- correction/retraction propagation;
- calibrated forecasts that cannot self-certify;
- matched-context downstream utility and regret.

The goal is not another universal leaderboard score. It is to show, under independent evaluation, that these assurances improve future action quality without increasing stale or unsupported memory activation.

## 9. Agent Memory Challenge Cycle 2: official November 19 results

The official results page reports nine separate leaderboards and 289 entries. Top open-source results:
- Textual: Resurface **61.22/100** (1st of 89).
- Coding: InsightMem **63.33% resolved** (1st of 20).
- Multimodal: Khipu **43.26/100** (1st of 18).

Top commercial results:
- Textual: MemoraX **69.26/100** (1st of 62).
- Coding: MemoraX **68.00% resolved** (1st of 11).
- Multimodal: MemoraX **53.68/100** (1st of 4).

These metrics are track-specific and division-specific; do not pool them or compare MACE synthetic utility to them. For a future Open-Source Methods submission, current programme goals are **66.22 textual**, **68.33% coding resolved**, and **48.26 multimodal**, each 5 points above the relevant Cycle 2 open-source leader. Recalculate when the next cycle freezes its data/API/model rules. The official Add/Search guide also requires `text-embedding-v4` for open-source embedding components and `gpt-4o-mini` for open-source LLM-related components, with unrestricted rerankers; Commercial Products do not have those model restrictions.

### Cycle 2 capability-level comparison and target deltas

The official open-source winners provide diagnostic sub-scores, not only aggregate ranks:

| Track / capability | Cycle 2 leader | Published sub-score | Next-cycle target (+5 pp) |
|---|---|---:|---:|
| Textual — explicit fact recall | Resurface | 62.98 | 67.98 |
| Textual — relational / multi-hop | Resurface | 58.94 | 63.94 |
| Textual — temporal / event sequencing | Resurface | 57.76 | 62.76 |
| Textual — memory governance | Resurface | 76.55 | 81.55 |
| Textual — personalization / care | Resurface | 38.82 | 43.82 |
| Textual — workflow execution | Resurface | 59.67 | 64.67 |
| Textual — epistemic safety / privacy | Resurface | 68.59 | 73.59 |
| Textual — streaming | Resurface | 49.65 | 54.65 |
| Coding — relevant new-feature resolution | InsightMem | 74.51% | 79.51% |
| Coding — relevant bug-fix resolution | InsightMem | 57.58% | 62.58% |
| Multimodal — retrieval | Khipu | 50.28 | 55.28 |
| Multimodal — relational reasoning | Khipu | 45.82 | 50.82 |
| Multimodal — temporal reasoning | Khipu | 34.23 | 39.23 |
| Multimodal — continual learning | Khipu | 38.45 | 43.45 |
| Multimodal — personalization | Khipu | 11.04 | 16.04 |
| Multimodal — evidence governance | Khipu | 31.29 | 36.29 |

The five-point targets apply to each capability in its native track and scale; they are not a claim that the overall score is the arithmetic mean of these rows. Coding uses resolved percentages, while textual/multimodal use the benchmark's score scale. Recompute when AML freezes the next cycle's data, API and model rules. Source: [official Cycle 2 results](https://agentmemoryleaderboard.ai/results/cycle-2/).

The new AML submission preflight checks the declared profile, but it is not a runtime attestation and cannot replace a live, public Smoke test. Cycle 2 is closed; the next eligible cycle is the target.

### AML deployment constraint

The current `aml-adapter` crate validates the Add/Search contract and declaration metadata, but a repository-only adapter is **not** an eligible AML submission. The official workflow requires participant-hosted, publicly reachable Add/Search endpoints, an access/key binding, a passing public Smoke flow, and then a formal Full evaluation. The platform controls Answer, scoring, aggregation and publication; candidate systems supply Add/Search only. Open-source Methods also have the declared model-component restrictions (`text-embedding-v4` for embeddings and `gpt-4o-mini` for LLM-related components, rerankers unrestricted). These requirements should shape an adapter service deployment plan, but no current endpoint, smoke pass or leaderboard entry is claimed.

### Learned forecasting qualification boundary

The separate `salience_learning` module implements deterministic logistic regression, Platt calibration, Brier-vs-base-rate, ECE and worst-family ECE over train/calibration/held-out partitions. Its base-rate comparator is derived from training labels only. The split manifest exposes dataset hashes plus separate task-family and source-root split digests, and requires cluster, task-family, source-root and event disjointness. The evaluator fixture uses 64/32/32 clusters and owns the hidden response function and future-reuse labels; model fitting sees only features and training labels.

This is a candidate-model baseline, not a production forecast, SOTA result or proof of downstream utility. The model now emits raw `FutureUsePredictionV1` candidates whose model-configuration digest binds the trained model and calibrator. It cannot construct the opaque `VerifiedFutureUsePredictionV1` or self-issue a calibration receipt. The independent verifier remains responsible for binding prediction/evidence, model pipeline, training/calibration/held-out dataset digests, task-family/source-root split digests, and held-out calibration metrics before any candidate is surfaced.

### Learned forecasting: next qualification gates

The current forecaster is deliberately small and interpretable. The next meaningful comparison is not a larger neural model; it is to establish whether its features improve realized utility under identical memory-token budgets and under stronger held-out shifts. Required baselines: fixed training prevalence, frequency-only, recency-first, failure-only, retrieval-similarity-only, the existing oracle-informed positive control (as a nondeployable ceiling), and learned logistic prediction. Report Brier skill against training prevalence, ECE and worst-family ECE, realized utility per token, regret, stale-memory activation, subgroup performance, and p50/p95 latency. Use at least five independent seeds and cluster bootstrap at task/environment level. Freeze model/config and splits before held-out scoring; never tune thresholds against held-out outcomes.


### One-sided evidence and value of information

The MACE v0.13 suite includes a challenge where only Action B has qualified historical support but the hidden target environment makes Action A better. Exact-scope causal memory must abstain: absence of qualified evidence for A is not evidence that A is inferior. The active-probe policy must request a bounded experiment, receive observations for both actions, pay the declared normalized utility cost, and only then select the better action. This measures safe acquisition of missing comparative evidence rather than rewarding an always-abstain policy or silently converting missing data into a negative result.


### Estimand identity is part of transportability

MACE v0.14 groups experience only when its receipt-bound estimand digest is consistent within each action arm, between the two candidate actions, and across source/target regimes. A source and target may use different randomized designs, but they must be estimating the same declared causal contrast (population, outcome, treatment definition and horizon). A changed estimand must not be mistaken for a changed regime of the same estimand. The deterministic suite adds a matched-effect / mismatched-estimand negative control. This is a digest-consistency gate, not proof that the verifier correctly specified the estimand; production review must inspect the canonical estimand definition itself.


### Next qualification: estimate the right question, not only the right effect

The next experiment should vary population, outcome definition, treatment version, follow-up horizon, censoring and target weighting independently while holding superficial score patterns fixed. Measure:
- false transport of a numerically similar but semantically different estimand;
- false abstention when independent designs estimate the same canonical estimand;
- sensitivity to omitted or changed covariates and treatment definitions;
- provenance coverage for every estimand field;
- decision regret after transport and after a correctly scoped probe.

The candidate's digest comparison is only as trustworthy as the independent verifier's canonical estimand definition. It must not accept model-authored labels as the estimand oracle.


### E. Model-conditioned memory dose response

The November 19, 2026 IBM Research study *How Much Should an Agent Remember? Deployment Lessons from Self-Distilled Memory for LLM Agents* reports that useful memory quantity depends on the target model: DeepSeek-V3.2 gained +9.5 task-completion points with 191 guidelines, while gpt-oss-120b reached 55.4% with a selected set of about 29 versus 51.2% with all 199. The authors also report failure modes including a high-support harmful guideline and provenance loss during consolidation. These are AppWorld results from the authors' study, not a universal scaling law. ([Paper](https://arxiv.org/abs/2611.26361))

MACE now has a per-target-model dose-response protocol. For each target-model digest and frozen memory-pipeline digest, it evaluates a preregistered set of context budgets (including zero memory) on paired calibration clusters, selects the highest-mean-net-utility budget with a smaller-budget tie break, then requires a positive paired cluster-bootstrap 95% lower bound on disjoint held-out clusters before qualifying a nonzero budget. The qualification binds task-family, source-root, event, model and pipeline digests and can feed the existing evidence-bound knapsack selector.

The next experiment is a two-dimensional surface: **target-model capability × memory token budget**, with separate measurements for task completion, scenario completion, steps, tokens, latency, stale/poisoned activation and governance violations. Require at least five model families/versions, multiple memory-policy families, at least five independent environment seeds per cell, paired cluster bootstrap, and one external held-out environment. Do not infer model capability from parameter count alone; the model digest and measured no-memory baseline are the operative strata.

Target qualification: each model's selected budget must improve held-out paired net utility over zero-memory with a 95% lower bound greater than zero, while no hard governance/safety metric regresses. Publish the full dose-response curve, not only the winning point. The prototype uses deterministic synthetic paired episodes; it does not establish an empirical relationship across production models.


### F. Receipt-qualified model-memory dose response

MACE v0.15.1 requires a `MemoryDoseTrialVerifierV1` before dose-response observations can enter fitting. Each receipt binds the exact trial, episode, environment, evaluation protocol, target model and memory pipeline, with explicit as-known validity. Only opaque `VerifiedMemoryDoseTrialV1` values reach the statistical fitter. Budget arms within a cluster must carry the same episode digest; calibration and held-out partitions are disjoint by cluster, task family, source root, event and episode.

This is a verifier interface, not a live cryptographic implementation. The current fixture verifier creates deterministic test receipts. Production qualification requires an independently authenticated evaluator/run receipt, independent outcome adjudication, and utility accounting for token, latency and execution costs. A self-hashed row is not proof of authenticity.

The runtime selector now rejects a different model fingerprint, binds the held-out dataset digest, and returns a valid empty plan when zero memory is the qualified choice. The dose-response layer is target-model-specific; the predictive-salience model itself remains shared across target models.


### G. Target-model-conditioned predictive salience

The learned salience predictor binds every example, train/calibration/held-out split, trained model, calibrator and emitted raw prediction to one explicit target-agent SHA-256 fingerprint. Mixed target-model rows are rejected within a split and across splits; calibration and evaluation require the same target identity. Raw predictions bind the fingerprint of the agent that will receive the forecast, and mismatches fail closed.

This makes future-use prediction conditional on the target agent instead of pooling different agents into one training population. It is not evidence that all target models require different predictors: that remains an empirical question. The next comparison is pooled vs model-conditioned vs hierarchical prediction under leave-one-model-family-out evaluation, across full memory token-budget dose-response curves. Model capability should be measured with no-memory task baselines rather than inferred from parameter count.


### HDC/HDC-LTC candidate index implementation (MACE retrieval arm v0.1)

A separate `broca-mace-hdc-retrieval` crate reuses the existing `symthaea-hdc-ltc` `ContinuousHV` and `HdcLtcUnifiedNeuron` APIs. It implements static bipolar role/value binding and bundling, plus a zero-input HDC-LTC elapsed-time profile. It ranks event IDs and returns a profile digest, top-k diagnostics, latency and a vector/ID payload-byte lower bound. It does not return `Experience` records or alter trust, source lineage, causal identification, semantic estimand, measurement bounds or authorization. The evaluator resolves IDs against its canonical experience registry and invokes the unchanged MACE policy. The first integration tests cover observational-decoy refusal, retracted-evidence refusal and unknown-ID rejection. This is authored source only; no retrieval or downstream utility gain is claimed. Distractor, irregular-gap and measured cost arms remain future work.

Candidate hits also carry a SHA-256 fingerprint of the candidate-visible `Experience` view at index-build time. The consumer compares it to the current canonical record before resolving the ID. A correction or retraction that changes the view invalidates the old hit and requires re-indexing. This is a consistency check only: it does not bind raw source bytes or authenticate source provenance.


### Re-observation-aware memory representation (2026-11-22)

MACE now has a proposal-only representation selector alongside future-use salience: persist learned rules/procedures/preferences, retain a compact stable fact or verified locator, cache mutable state only through a verified freshness interval, or discard non-positive-value items. Locator value is risk-adjusted by a verifier-calibrated success probability and failure consequence; availability and calibration receipts must remain valid strictly beyond the decision deadline. The selector is bound to the exact bitemporal `(valid_at, known_at)` query and refuses to reuse a profile from an earlier knowledge snapshot.

This adds a new benchmark axis: **memory versus perception**. It is not enough to ask which memories to inject; the agent should decide which information must be learned, which can be looked up at use time, and which snapshots are worth caching. The prototype includes 23 authored module tests and a raw Session-A ingress → future-use prediction → source-profile verification → representation-plan evaluator integration fixture. Fixture verification only; no runner or SOTA claim.


### Authority-scoped memory representation (2026-11-22)

The re-observation representation contract now binds principal, capability, audience, resource/tenant, purpose and policy revision using opaque digests. Both the independent source verifier and the decision policy must match the expected authority scope exactly. This follows Mycelix's principal/capability, destination/audience and purpose binding pattern. The injected verifier is a test seam only; production still requires live Mycelix capability and revocation checks at use.

The module now contains **23 authored unit tests**, including wrong-principal, wrong-audience and wrong-purpose controls; the raw Session-A integration fixture binds an `EnvironmentState` authority scope. These are source counts only, not executed passes.


### Joint representation allocation (MACE v0.17)

MACE now enumerates alternative per-item representations and can allocate them jointly under shared token, observation-call, latency-work, monetary/compute-cost and expected-failure-loss budgets. The exact bounded branch-and-bound solver is compared with a greedy highest-individual-value baseline. The prototype is limited to 24 event groups / 2,000,000 nodes and fails closed on search exhaustion. It does not yet model parallel query scheduling, batching or privacy exposure as a resource.


### Trust-boundary tightening

Portfolio options are now admitted only when their representation-plan digest matches an option independently re-derived by `enumerate_memory_representation_options` from the same verified salience prediction, verified source profile and policy. A caller cannot mutate a plan's cost/disposition, recompute its digest, and present it as an eligible choice. The portfolio constructor still verifies the complete source/prediction/snapshot/authority binding before creating an option.

The exact optimizer assumes the supplied per-item `net_value_micros` are additive across distinct evidence payloads. It now rejects a byte-identical evidence digest being assigned multiple event IDs, but semantically overlapping facts with different payloads can still have interacting or diminishing value. This prototype is not a submodular/interaction-aware optimizer. A credible next version needs independently qualified marginal-value groups or a portfolio-level evaluator that measures the value of combinations against the same no-memory baseline; it must not simply sum correlated single-item predictions and call the result realized utility.
