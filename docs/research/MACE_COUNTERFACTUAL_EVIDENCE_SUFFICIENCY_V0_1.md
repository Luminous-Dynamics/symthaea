# MACE counterfactual evidence sufficiency and dependency-aware knockouts v0.1

**Status:** research protocol and evaluator-hardening increment; diagnostic-only until compiled, executed, independently evaluated and externally reproduced.

**Parent work:** PR #13221 (typed-content HDC and DecisionNeedV1)  
**Implementation issue:** #13241  
**Branch base:** inspected #13221 head f57abba7953790472656a18dc2f39108553d0d54

## 1. Research question

MACE now compares compressed typed-content retrieval with an exact-scope, candidate-visible evidence reference. The next question is not merely whether a memory item can be retrieved:

> Which evidence dependencies are necessary to preserve a decision, which are redundant or complementary, and at what point does compression make the decision no longer sufficiently supported?

This must be answered by intervening on the evidence available to the candidate and measuring the downstream decision, not by inferring importance from vector similarity, attention weights, retrieval rank or the evaluator's hidden labels.

The central unit of intervention must be larger than an event ID. MACE admission requires independent source roots and source families. Removing one event while retaining its correlated descendants, duplicated aliases or same-root siblings can leave the apparent support count unchanged without preserving independent information. The audit therefore operates over dependency closures.

## 2. Paradigm shift: memory as a decision-sufficiency object

Treat a memory representation as a proposal for a compact sufficient decision state, not as a store of facts that happen to be semantically similar to a query.

For a candidate-visible DecisionNeed n, let E(n) be the complete candidate-visible evidence universe admitted to the experiment after exact scope partitioning. Let S be a retained subset or compressed representation, and let pi(n,S) be the unchanged MACE decision policy applied to the records resolved from S. Let y be evaluator-only world truth and U_y(a) be the scenario utility of decision a.

Define:

- Absolute regret: R_y(S) = max_a U_y(a) - U_y(pi(n,S)).
- Regret introduced by compression: G_y(S) = R_y(S) - R_y(E(n)).
- Epsilon decision sufficiency: S is sufficient when all hard evidence/authority gates pass and G_y(S) is no greater than a preregistered epsilon.

The exact-scope reference is an information-complete reference over the generated candidate-visible pool, not a same-cost competitor and not proof that the pool is complete with respect to the real world. Report its resource cost separately.

Decision agreement with the exact-scope decision is useful but insufficient: two policies can choose the same wrong action. Always report absolute regret against evaluator truth, decision flips, abstention, and evidence-gate results alongside agreement.

The benchmark must also test the opposite failure: retaining a misleading item can make a decision worse. Evidence utility is therefore not assumed to be monotonic in subset size.

### 2.1 Contribution and interaction measures

For an evidence dependency group g, define its leave-one-group-out contribution:

    M_g = U_y(pi(n,E)) - U_y(pi(n,E \ closure(g)))

Positive M_g means removing the group reduced utility; negative M_g means the group was harmful under this policy and task.

For groups g and h, a paired second-order interaction can be reported as:

    I_gh = U(E) - U(E \ g) - U(E \ h) + U(E \ {g,h})

Positive interaction is evidence of complementarity under this particular decision policy and scenario; negative interaction is consistent with redundancy or interference. This is a descriptive intervention effect, not a universal causal property of the memory item.

For small evidence sets, enumerate all subsets and compute exact group contributions. For larger sets, sample coalition permutations and report approximate Shapley allocations with seed, sample count and Monte Carlo error. Do not label Shapley values as causal effects without the corresponding randomized intervention design.

## 3. Evidence dependency closures

The audit must create a canonical, candidate-visible dependency graph before running any intervention.

### 3.1 Closure levels

1. **Event closure:** one canonical event and all directly dependent derived views, summaries, cache entries and index rows.
2. **Source-root closure:** every event and derivative whose verified ancestry resolves to the same canonical source root.
3. **Source-family closure:** evidence sharing the same authoritative source-family identity, including mirrors and re-exports where the lineage says they are not independent.
4. **Coalition closure:** the union of two or more root/family closures, used to identify complementary support, common-cause dependence and higher-order interactions.
5. **Lifecycle closure:** the current version plus superseded, retracted or corrected descendants needed to resolve what was knowable at the decision time.

Closure is transitive. If A was derived from B and B from C, removing C must remove or invalidate A and B unless an independently verified reconstruction path remains. A digest link alone does not establish this ancestry; production root identity must be derived from trusted upstream source profiles and verified receipts, not supplied by the candidate.

The current synthetic fixture exposes integer source-family IDs and source-root digests. It deliberately includes a common-cause root attack and correlated echoes. Those fixtures are useful for protocol tests, but do not establish production-grade source identity.

### 3.2 Independence is not cardinality

Count independent roots and families only after canonical identity resolution. Multiple event IDs, distinct display names, different vector embeddings or multiple wrappers around the same source do not make independent evidence.

If two records share a source root, their joint support counts as one root. If the root is unknown, malformed or ambiguous, the evidence is not allowed to increase the independent-root count. The benchmark should report both raw event count and verified independent root/family counts.

## 4. Frozen intervention design

Generate a versioned intervention manifest before evaluation. Its seed and target selection must depend only on public candidate-visible identities, the task contract and a declared sampling rule. It must not depend on evaluator utility, optimal action, hidden causal labels, downstream outcome or observed performance.

The candidate receives the task DecisionNeed and the post-intervention candidate-visible evidence. It does not receive evaluator truth or an intervention label that reveals why records were removed. The candidate-input digest binds the actual post-intervention evidence set. The evaluator separately records the intervention manifest digest.

For each scenario/world pair, retain these contrasts:

| Condition | Purpose |
|---|---|
| No memory | Establish whether memory helps at all |
| Exact-scope full evidence | Candidate-visible information reference |
| Typed-content HDC top-k | Measure compressed representation sufficiency |
| Event knockout | Identify event-level sensitivity |
| Source-root knockout | Test dependence on independent source roots |
| Source-family knockout | Test dependence on independent families |
| Pair/coalition knockout | Identify complementary or redundant evidence groups |
| Matched random dropout | Control for the number, action balance and bytes removed |
| Semantic decoy / wrong-estimand replacement | Test whether plausible but inapplicable evidence misleads |
| Stale/retracted/corrected state | Test current-at-use lifecycle handling |

The first implementation should use the existing seven-arm runner and add a distinct knockout audit over the same frozen generated worlds. Do not inflate the core arm count with every combinatorial subset. Store intervention-level records in a separate, schema-versioned stream and summarize them alongside the paired runner's existing contrasts.

### 4.1 Required intervention families

**Single-group ablations.** Remove each eligible event, root and family closure in turn. Keep the original DecisionNeed unchanged. Record when removal causes a hard qualification failure, an abstention, a decision flip or only a retrieval-rank change.

**Coalition ablations.** Enumerate all root pairs when n is small enough. At larger n, draw deterministic pairs and a preregistered number of larger coalitions. Include combinations across action alternatives because qualification can require matched evidence for both actions.

**Matched random controls.** For each meaningful knockout, create one or more controls that remove the same count and approximate byte budget while preserving action distribution and, where feasible, the number of roots/families removed. Otherwise a large degradation may merely reflect that more evidence was removed.

**Replacement and poisoning controls.** Replace removed support with records that are similar in typed content but have wrong scope, wrong estimand, common-cause ancestry, stale/retracted status, untrusted origin, incomplete delayed horizon or unreconciled outcomes. Similarity may affect rank; it must never turn these records into admissible evidence.

**Correction-at-use controls.** Change a source after retrieval but before decision consumption. Require canonical current-state re-resolution and lifecycle verification at the decision's knowledge-time boundary. A previously retrieved snapshot or matching view digest must not override a later correction or retraction.

### 4.2 Paired design and leakage controls

Freeze the task, DecisionNeed, model/backbone, initial state, permissions, deadline, evaluator, candidate policy and non-memory resources within each pair block. Randomize intervention order deterministically within the pair block. A candidate must not learn from another arm or receive evaluator feedback.

Keep training, calibration and held-out world clusters, source roots and task families disjoint. Where multiple fitted-model seeds are used, resample model-fit seeds first and world clusters within seed second. The existing portfolio-interaction lane's hierarchical bootstrap is the appropriate pattern; do not treat multiple outcomes under one fitted model as independent model replications.

Mandatory metamorphic controls:

- Change hidden evaluator utility/outcome labels without changing DecisionNeed or candidate-visible evidence: candidate input and task digests must remain byte-identical.
- Remove or mutate candidate-visible evidence: the evidence snapshot and candidate input digests must change.
- Permute event IDs while preserving typed content and dependency identities: semantic ranking must remain invariant except a declared deterministic tie-break.
- Duplicate an event under another ID or alias the same source root under multiple families: independent support must not increase.
- Change the task's objective, regime, action alternatives or semantic estimand: DecisionNeed digest must change.
- Mutate a canonical record after indexing: exact record/view revalidation must reject it.
- Insert wrong-scope, wrong-estimand, retracted, untrusted, unreconciled or incomplete-horizon evidence: it must never pass the unchanged MACE admission policy because of a high similarity score.
- Reorder input records: canonical set digests and decisions must remain deterministic.

## 5. Metrics: retrieval is not decision quality

Report metrics at each stage so failure attribution is possible.

### 5.1 Retrieval and evidence closure

- recall@k and precision@k against evaluator-labelled relevant records, reported only in evaluator output;
- canonical candidate-ID resolution rate and view-digest mismatch count;
- decision-support closure completeness;
- raw event count, unique verified root count and unique verified family count by action;
- unresolved ancestry, unknown-root, duplicate-ID and ambiguous-identity counts;
- stale/retracted/wrong-scope/wrong-estimand evidence admitted and actually used.

### 5.2 Downstream decision

- absolute task utility and regret to the evaluator optimum;
- regret delta against exact-scope full evidence;
- decision agreement and action-flip rate;
- correct-abstention and incorrect-abstention rates;
- evidence-gate failure and unsupported-decision rates;
- harmful-memory dependence: utility lower than the declared no-memory baseline;
- per-family, per-regime, per-action and per-evidence-quality subgroup outcomes.

Do not collapse abstention into a real action. The current synthetic benchmark convention is an explicit outside option with utility 0.0; real deployments must supply a scenario-specific abstention cost/value. An abstention must never inherit Action B's payoff by default.

### 5.3 All-in resource accounting

A memory system must not appear efficient by hiding evidence sidecars or background verification. Report a resource vector:

- vector/index/codebook bytes;
- canonical evidence and source-sidecar bytes;
- metadata, journals, tombstones and correction history;
- context tokens and model calls;
- CPU/GPU time, allocations and index build/update/compaction/replay work;
- p50/p95/p99 query and end-to-end decision latency;
- signature, provenance, lifecycle and source re-observation calls;
- monetary cost and privacy/retention exposure;
- deadline misses and failure/recovery cost.

The exact-scope scan is a diagnostic ceiling. Fair practical comparisons require equal all-in storage, token, latency and verification budgets.

## 6. Statistical qualification

Pre-register one primary contrast before examining held-out labels. A reasonable first primary endpoint is the paired difference in absolute regret between typed top-k memory and full exact-scope evidence, with world cluster as the resampling unit.

Report:

- paired point estimate and a 95% interval;
- decision-sufficiency rate at a predeclared regret margin;
- harmful-dependence rate and abstention outcomes;
- all hard-gate failures, invalid blocks, missing blocks and attrition;
- results per family/regime, not just pooled average;
- intervention seeds, coalition sampling count and finite-sample/Monte Carlo uncertainty.

For a multi-fit portfolio model, use hierarchical resampling of fit seeds and then held-out clusters within seed. For the simpler fixed-policy synthetic runner, cluster bootstrap over independent world clusters is appropriate only if clusters are genuinely independent under the generator.

Avoid repeated uncorrected significance tests across many knockout subsets. Treat the intervention family as exploratory unless one primary group-level contrast and multiplicity policy are frozen in advance. Use simultaneous intervals or a preregistered correction for confirmatory families.

## 7. Promotion gates and stretch targets

These are proposed MACE decision rules, not universal causal-inference standards.

| Gate | Proposed criterion |
|---|---|
| Integrity negative controls | Zero known invalid, retracted, wrong-scope, wrong-estimand or unknown-root records admitted as decision support |
| Identity resolution | Zero unresolved or ambiguous support IDs |
| Decision sufficiency | Upper 95% confidence bound on added regret no greater than a preregistered epsilon |
| Memory value | Paired lower 95% confidence bound for memory-vs-no-memory utility above zero on the primary lane |
| Compression efficiency | At least 25% lower all-in memory/context cost than full-history baseline with no more than 1 percentage point task-success loss on a preregistered lane |
| Generalization | Worst-family/regime task-success regression no greater than 2 percentage points |
| HDC-specific claim | Positive matched contrast against strongest reproduced KV/BM25/dense/hybrid baseline at equal all-in budgets |
| Abstention | No post-hoc mapping of abstention to an action; report both false abstention and unsupported action |
| Evaluator integrity | Independently held-out evaluator labels and custody; no candidate-readable truth, labels or hidden-evaluator artifact |

The 25%, 1-point and 2-point thresholds are project stretch targets, not industry consensus. Freeze margins before held-out evaluation. A method that fails the HDC-specific gate should not use HDC simply because it is part of the project. A simpler representation that wins under matched budgets is the correct result.

## 8. Competitive benchmark portfolio

No public benchmark currently substitutes for the causal-intervention and evidence-qualification lane. Keep external scores protocol-specific and use them as independent capability tests.

| Benchmark | What it contributes | How MACE should use it |
|---|---|---|
| Agent Memory Challenge (AML) Cycle 2 | Official textual, coding and multimodal Add/Search tracks | Build a real hosted Add/Search service; repository-only work is not a valid submission |
| Memora / LoCoMo / LongMemEval | Abstraction, cue anchors, long-context retrieval and token efficiency | Reproduce backbone, judge, reader, memory interface and token accounting |
| LongMemEval-V2 | File-based and RAG memory over multimodal web-agent trajectories; accuracy and query latency | Compare file-memory, RAG and fixed-reader lanes separately |
| AMA-Bench / AMA-Agent | Long-horizon agentic trajectories, objective and causal information | Add a causality-graph/tool-retrieval baseline, not just dense retrieval |
| MemoryArena | Interdependent multi-session Memory-Agent-Environment actions | Evaluate whether stored experience improves later environment actions |
| WorldMemArena | Multimodal write, maintain, retrieve and use lifecycle | Test images/action traces, state updates, interference and evidence chains |
| MemoryAgentBench | Accurate retrieval, test-time learning, long-range understanding and conflict resolution | Measure experience gain and selective forgetting, not only retrieval |
| StateMemBench | Current-vs-superseded state tracking | Add bitemporal state and supersession controls |
| FAMA | Penalizes reuse of obsolete/invalidated memories | Report stale-memory harm independently from recall |
| TimlMemBench | Raw-session ingestion, revoke/supersede, scope, provenance and restraint | Add a raw-native governance lane |
| CL-Bench | New context learning with expert rubrics; naive in-context learning can be strong | Retain a strong ICL baseline and measure actual experience gain |
| MCB (Remember, Verify, or Ask?) | Reuse-time persist/use/verify/clarify decision boundary | Use as a small policy diagnostic; do not overstate the statistical strength |
| Agentic Abstention | Timely stopping under interactive budgets | Measure excess tool calls after infeasibility becomes knowable |

### 8.1 Verified public anchors as of 2026-11-22

- **AML Cycle 2 official results**, published November 19, 2026: open-method leaders are Resurface 61.22/100 textual, InsightMem 63.33% coding resolved, and Khipu 43.26/100 multimodal. Commercial leaders are MemoraX 69.26 textual, 68.00% coding resolved and 53.68 multimodal. These are separate tracks/divisions and must not be averaged. Cycle 2 covers 289 track entries across nine leaderboards. See the official results page: https://agentmemoryleaderboard.ai/results/cycle-2/.
- **Textual capability breakdown matters.** Resurface's 61.22 open-method score includes 76.55 Memory Governance, 68.59 Epistemic Safety & Privacy, 57.76 Temporal & Event Sequencing, but 38.82 Personalization & Care. MACE should report capability subgroups and should not optimize only an aggregate memory score.
- **Resurface (official textual open-method leader)** is an Add/Search evidence-retrieval service. Its public repository describes metadata-rich turns, keyword plus dense recall, reciprocal-rank fusion and optional neighbouring-turn evidence. This is the baseline MACE must beat or complement on real textual trajectories—not a deterministic event-ID ordering. Repository: https://github.com/lilysttail/Resurface.
- **Khipu (official multimodal open-method leader)** keeps source messages verbatim and describes BM25, vector, entity, literal-match and date channels, fused with intent-aware reciprocal-rank fusion. Its MM branch retains original image evidence alongside descriptions; its current README describes a cross-encoder reranker over the first 200 fused candidates. Reproduce the exact MM branch/commit and model settings, not only the textual main branch. This suggests an explicit MACE baseline combining exact lexical anchors, temporal/entity channels, reranking and original-evidence pointers. Repository: https://github.com/Lv999APTITUDE/khipu.
- **InsightMem (official coding open-method leader)** combines source-chunk BM25, code-expanded BM25 and dense cosine retrieval with reciprocal-rank fusion, adjacent-slice merging, deduplication and query coverage. This is a stronger coding baseline than generic dense retrieval. Repository: https://github.com/Kline-song/InsightMem.
- **Memora (ICML 2026)** reports 86.3% LLM-judge accuracy on LoCoMo and 87.4% on LongMemEval in its own setup, with up to 98% fewer context tokens than full-context inference. Its abstraction hierarchy and cue anchors are particularly important competitors to a flat typed vector index. These scores are not directly comparable to AML or MACE. See https://www.microsoft.com/en-us/research/blog/memora-a-harmonic-memory-representation-balancing-abstraction-and-specificity/.
- **LongMemEval-V2**'s current site reports a newer AgentRunbook-C V2 online controller at 75.61% on the small split with 130.54-second latency at xhigh, and 65.41% with 26.11 seconds at low reasoning; its online consolidation strategy improves the accuracy/latency frontier. The earlier AgentRunbook-C reports 74.9%/108.3 seconds on small and 70.1%/139.9 seconds on medium, while Codex reports 69.9%/177.2 seconds and 68.7%/185.8 seconds respectively. Keep model/reasoning tier, reader, split and latency settings separate. Reproduce the current V2 online and non-online controllers as distinct baselines. See https://xiaowu0162.github.io/longmemeval-v2/ and https://xiaowu0162.github.io/longmemeval-v2/agentrunbook-c-v2/.
- **AMA-Agent / AMA-Bench (ICML 2026)** reports 57.22% average accuracy and an 11.16 percentage-point gain over the strongest baseline in that paper's protocol. The paper identifies missed causal/objective information and lossy similarity retrieval as weaknesses, then uses causality graphs and tool-augmented retrieval. MACE should reproduce this as a serious causal-graph baseline. See https://proceedings.mlr.press/v306/zhao26bs.html.
- **WorldMemArena** evaluates 400 multi-session multimodal tasks over the write, maintain, retrieve and use lifecycle. It emphasizes action–world interaction, original visual evidence, updating stale state and decision-time use. See https://worldmemarena-mem.github.io/ and https://arxiv.org/abs/2605.29341.
- **MemoryArena (ICML 2026)** couples memory with later environment actions across interdependent multi-session tasks; near-saturated long-context recall does not guarantee good agentic performance. Use its task families to test real experience transfer, not just answer recall. See https://proceedings.mlr.press/v306/he26am.html.
- **StateMemBench** contains 234 multi-session evolving-state scenarios. StateMem reports current-state improvements from 0.205 to 0.363 on DeepSeek-V4-Flash and from 0.149 to 0.233 on Qwen-3.5-9B in its own protocols; it also reports gains from adding explicit state structure to other memory backends. MACE should make supersession and current-state resolution first-class controls. See https://arxiv.org/abs/2608.19652.
- **TimlMemBench (Findings EMNLP 2026)** reports 208 reviewed bilingual samples, 1,566 sessions, 1,715 questions and 1,821 governance/activation cases. Its raw-native evaluation starts from source sessions and tests revocation, supersession, scope, provenance and restraint, exposing weaknesses hidden by a canonical prebuilt memory pool. See https://aclanthology.org/2026.findings-emnlp.507/.
- **CL-Bench** contains 500 contexts, 1,899 tasks and 31,607 verification rubrics. Its central methodological lesson is to compare against strong naive in-context learning: a dedicated memory system has to demonstrate real experience gain. See https://arxiv.org/abs/2602.03587.
- **FAMA** explicitly evaluates reliance on obsolete or invalidated memories. **MCB (Remember, Verify, or Ask?)** supplies a reuse-time decision diagnostic, but its small-sample/significance limitations make it a diagnostic rather than a definitive leaderboard. See https://aclanthology.org/2026.findings-acl.1337/ and https://aclanthology.org/2026.nora-1.2/.
- **Agentic Abstention** evaluates more than 28K instructions across WebShop, Terminal-Bench and QA. One reported held-out WebShop configuration improves timely abstention from 26.7% to 57.4%; this is model/scaffold/task-specific and should not be transplanted as an absolute target. See https://lhannnn.github.io/agentic-abstention/.

### 8.1.1 What the public leaders teach MACE

The current high-performing systems are not merely dense embedding databases:

| Competitor | Publicly described technique | MACE consequence |
|---|---|---|
| Resurface | Keyword + dense recall, RRF and neighboring evidence context | Implement BM25/keyword, dense and RRF controls with matched context expansion |
| Khipu | Verbatim memory, BM25/vector/entity/date/literal channels, intent-aware RRF | Add exact entity/date/numeric anchors and query-intent stratification |
| InsightMem | Source BM25 + code-expanded BM25 + dense cosine + RRF | Add source-preserving and task-specific retrieval lanes for code/trajectory memory |
| Memora | Abstraction hierarchy and cue anchors | Compare hierarchical capsules and cue-to-source traversal against flat typed HDC |
| AMA-Agent | Causal graphs and tool-augmented retrieval | Compare dependency/effect graph retrieval and query-time evidence expansion |
| StateMem | Explicit current state, supersession and relational dependencies | Compare a state-first view and bitemporal revision resolution |
| CL-Bench controls | Naive in-context learning | Demonstrate that persistent memory adds value beyond a strong prompt/context baseline |

**Immutable competitor source snapshots inspected 2026-11-22:**
- Resurface `2e56bdbc9908b3798b08fde87f03a17ef9e4bc7a` (AML textual leader; repository README modified 2026-11-20).
- Khipu deployed MM branch `8cf40aaab0d66442ff25652da4e78a9866c6e514`; deployment commit explicitly declares the application unchanged from `d21d401a00a7afb3410a6a64a7e16e66f9e0b1e0`. Freeze both deployment config and app-code commit; do not assume `main` is the multimodal leader.
- InsightMem `b34b7c00f1994ea042b83bee318d2a1b33f6cdec` (AML coding leader).

**Baseline parity rule:** the current MACE Experience fixture is structured and contains no general natural-language trajectory field. A lexical BM25/dense system over this typed fixture would not reproduce Resurface, Khipu or InsightMem. Build a separate raw-trajectory Add/Search adapter for AML-style comparisons, while preserving the synthetic causal intervention lane as a distinct MACE evaluation. Each external baseline must pin repository commit, container/dependencies, encoder/reranker/tokenizer, prompts, split and checksums, reader/judge, Add/Search contract, token budget, top-k, RRF weights, neighbor expansion, context packing, concurrency, latency and full storage/compute costs. Do not blur the two.

For fair comparisons pin the competitor's public commit, data, encoder/model, tokenizer, reader/judge, prompts, retrieval parameters, token accounting and resource budget. Compare not only output score but index size, source bytes, context size, p50/p95/p99 latency, update/replay cost and verification overhead. Reproduce methods rather than copying their reported leaderboard numbers.

### 8.2 MACE's competitive wedge

Do not attempt to win every benchmark with a single score. Build a capability profile and target the part others do not jointly measure:

1. **Decision-preserving compression:** show that a small capsule preserves decisions and calibrated uncertainty under root/family-level knockouts.
2. **Proof-carrying evidence:** every decision-support item resolves to exact current evidence and a verified dependency closure.
3. **Adaptive representation:** choose persist, locator, snapshot or discard based on expected future decision value, re-observation cost and hard retention/authority constraints.
4. **Use-time governance:** correct for supersession, revocation, permission, time and source changes at the actual decision boundary.
5. **Active sufficiency:** identify the smallest evidence acquisition or probe that can change an unresolved decision; stop retrieving when the expected value of more evidence is below its total cost.
6. **Independent falsification:** publish sealed evaluator outputs and reproducible receipts, not just source counts and self-reported tests.

This creates a stronger claim than “HDC beats RAG”: MACE would demonstrate when a decision can safely rely on a compact evidence state, when it cannot, and exactly which missing dependency explains the failure.

## 9. Immediate implementation sequence

### Authored on the follow-on branch

1. Corrected the synthetic evaluator's abstention mapping to an explicit zero-utility outside option and added a regression test.
2. Added `mace_counterfactual_evidence.rs`, a separate executable audit binary using the existing generated stress worlds and candidate-visible MACE policy.
3. Added an `EvidenceDependencyIndex` that rejects duplicate event IDs, groups events by nonzero source-root digest and source-family ID, and counts unknown root/family identities separately.
4. Added deterministic event, source-root, source-family, explicit unknown-root/unknown-family, bounded root-pair, stratified-random and no-memory interventions.
5. Stratified random controls by regime, objective, action and estimand digest; the manifest reports unmatched target strata.
6. Added five candidate modes for every full-evidence and knockout condition: exact-scope scan; deterministic event-ID/KV top-k; static associative HDC; zero-input HDC-LTC; and typed-content HDC top-2 per declared action.
7. The indexed HDC modes rebuild their respective indexes after each intervention, resolve returned IDs against canonical Experience records and reject any index-time/current-view digest mismatch. Typed HDC additionally converts candidate-visible DecisionNeedV1 to an action-specific query.
8. Bound candidate input digests to DecisionNeedV1 plus the post-intervention canonical evidence-set digest. Candidate observation IDs additionally bind the selected method.
9. Report regret gap versus full exact-scope evidence and incremental regret gap versus each method's own full-evidence baseline, separating information-ceiling, retrieval-substrate and knockout effects.
10. Split candidate decision records from evaluator intervention manifests/scores. Candidate-facing pair identifiers are opaque and derived only from candidate-visible task/evidence digests, not evaluator scenario-family labels. Candidate records expose the retrieval method and returned evidence IDs, but not intervention kinds, removed IDs or evaluator utilities.
11. Bound source-code, fixture and JSON-Schema digests into the run identity/evaluator manifest, and introduced `data/schemas/mace_counterfactual_evidence_v0_1.schema.json`.
12. Added 22 authored intervention-runner tests and 6 authored aggregation tests for duplicate IDs, root aliases, unknown identities and their knockouts, source-family closures, deterministic/truth-blind planning across methods, coalition caps, matched controls, evidence-digest changes, candidate-contract rejection, opaque pair identifiers, KV bounds, canonical HDC resolution/rebuild, cluster-first aggregation, paired task matching and manifest provenance.

### Remaining qualification work

1. Execute rustfmt, compile, unit tests and the runner; validate every JSONL record against the new schema.
2. Add verified transitive source-lineage/derived-view closure. The current Experience fixture has no parent-edge field, so this binary only groups direct root/family identities.
3. Add explicit semantic decoys, correction-between-retrieval-and-use and lifecycle closure.
4. Add exact subset enumeration for small evidence sets, sampled larger coalitions and decision-interaction estimates.
5. The first cluster-first reducer and cluster bootstrap are now authored; preregister primary contrasts/margins, multiplicity policy and hierarchical model-fit-seed resampling.
6. Reproduce BM25, dense, sparse/dense RRF, intent-adaptive channels, context neighbours, causal-graph retrieval and state-first retrieval on the correct raw or structured data lanes.
7. Move evaluator truth to independent custody and run held-out external benchmarks through pinned adapters.

## 10. Current implementation and validation boundary

The follow-on branch is stacked directly on the inspected #13221 child-PR head and contains the following authored artifacts:

- `research/broca-mace-transfer-bench-v0-1/evaluator/src/bin/mace_counterfactual_evidence.rs`
- `research/broca-mace-transfer-bench-v0-1/evaluator/src/bin/mace_counterfactual_aggregate.rs`
- `data/schemas/mace_counterfactual_evidence_v0_1.schema.json`
- `data/schemas/mace_counterfactual_summary_v0_1.schema.json`
- `docs/research/MACE_COUNTERFACTUAL_EVIDENCE_SUFFICIENCY_V0_1.md`

Example invocation after a Rust-enabled checkout:

```bash
MACE_SOURCE_REVISION="$(git rev-parse HEAD)" \
cargo run --manifest-path research/broca-mace-transfer-bench-v0-1/evaluator/Cargo.toml \
  --bin mace_counterfactual_evidence -- \
  64 mace_counterfactual_candidate.jsonl mace_counterfactual_evaluator.jsonl 12345 32
```

Arguments are cluster count, candidate JSONL path, evaluator JSONL path, public intervention seed and maximum source-root pairs. The runner writes separate streams. Evaluator output contains the frozen intervention manifest and hidden-outcome scores. Candidate output contains method, DecisionNeed/evidence digests, retrieved event IDs, chosen action, evidence-support IDs, typed-profile digest when available and resource counts, but not knockout labels, removed event IDs or evaluator utility. The candidate observation identifier allows the streams to be joined without placing intervention type in the candidate record.

The offline cluster reducer consumes the evaluator stream only:

```bash
cargo run --manifest-path research/broca-mace-transfer-bench-v0-1/evaluator/Cargo.toml \
  --bin mace_counterfactual_aggregate -- \
  mace_counterfactual_evaluator.jsonl mace_counterfactual_summary.jsonl 12345
```

The reducer requires exactly one manifest receipt per scored task block, verifies that all manifest rows belong to one run and share source-code, fixture, schema and source-revision provenance, and rejects score rows without a matching manifest. It hashes the sorted manifest set and binds the summary to both input and reducer-code digests. It averages repeated interventions within each opaque task block, averages task blocks within each world cluster, then bootstrap-resamples the resulting world-cluster means (2,000 replicates). The reported field is a world-cluster count; the reducer does not assert that cluster independence has been proven. It emits method summaries and paired method-minus-exact-scope contrasts only when both methods exist for the same task block. The output contract is `data/schemas/mace_counterfactual_summary_v0_1.schema.json`.

This is an initial cluster-aware statistical reducer, not a preregistered confirmatory analysis. It does not yet apply multiplicity correction across intervention families, establish independent-world assumptions empirically, or provide hierarchical model-fit-seed resampling; those remain #13265.

### Important limits

- The root index groups equal nonzero `SourceRootDigest` values. A nonzero digest is an identity key, **not cryptographic authentication of its issuer**. The synthetic `source_family` integer is also a fixture identity, not an independently verified family credential.
- The current `Experience` structure has no explicit parent/derived-view edges. The implementation therefore does not yet compute transitive lineage closure; it groups records sharing the same direct root digest or family label.
- Unknown root/family values are counted separately, do not inflate known-identity counts, and now have explicit knockout groups. The code does not independently verify signatures, source authority, tenant permissions or external receipts.
- Matched-random controls preserve count within regime/objective/action/estimand strata, but do not yet guarantee equal source-root/family multiplicity, exact serialized bytes or equal verification work. They are stratified random controls, not fully dependency- and cost-matched placebo interventions. They are omitted when the candidate-visible pool cannot match a stratum; coverage is reported. The current byte measure is only `size_of::<Experience>()` per record and excludes serialization, allocator, index, sidecar, source and verification costs.
- Root-pair interventions are deterministically enumerated up to a caller-supplied cap. Higher-order coalition sampling, exact subset optimization and Shapley attribution are not yet implemented. An initial offline cluster-first reducer with paired method-minus-exact contrasts and 2,000-replicate cluster bootstrap is authored, but unexecuted; confirmatory preregistration, multiplicity correction and hierarchical resampling remain outstanding.
- The audit reruns five methods on every full-evidence and knockout condition: **exact-scope scan**, deterministic event-ID/KV top-k, **static event-ID HDC**, **zero-input HDC-LTC**, and **typed-content HDC top-2 per action**. All feed the same `Policy::CausalQualified` rule. This separates the exact-evidence reference, simple event-ID selection, associative-HDC substrate, zero-input LTC evolution and typed-content retrieval. The indexes are rebuilt per condition; no incremental update/replay performance is claimed. The KV and event-ID HDC arms are internal controls, not substitutes for strong BM25/dense/RRF competitors.
- Typed-content HDC remains structured feature retrieval over the synthetic Experience type, not natural-language or multimodal memory. The experiment reports candidate IDs and canonical view resolution; it does not yet report labelled retrieval recall/precision because the fixture has no independent per-query relevance annotation.
- DecisionNeedV1 is validated and used by the typed query adapter. The exact-scope method uses its declared scope and action alternatives. A general natural-language/task/environment parser is not implemented.
- The candidate/evaluator are still in one public repository. Separate JSONL files are a data-boundary test, not independent evaluator custody.
- The typed HDC remains structured typed-content retrieval; it is not natural-language understanding, a learned causal model or a production source verifier. The static HDC and HDC-LTC controls encode event IDs rather than typed semantic experience. Zero-input HDC-LTC tests elapsed-time state evolution; it must not be interpreted as event-driven temporal learning.

Abstention is an explicit zero-utility outside option only for the current synthetic benchmark. Every real application must define its own abstention cost/value and deadline policy.

**Validation:** Cargo, rustc and rustfmt are unavailable in this execution environment. The intervention runner, offline aggregate binary, their 28 authored tests and both JSON Schemas are source-authored but unexecuted. No compilation, test pass, formatting pass, schema validation, runner result or hosted-check pass is claimed. The previously inspected PR workflows were skipped; skipped is not passed.

No retrieval gain, decision-utility improvement, official benchmark score, causal sufficiency result, production readiness or SOTA superiority is claimed.

## References

- AML Cycle 2 official results (published 2026-11-19): https://agentmemoryleaderboard.ai/results/cycle-2/
- AML Cycle 2 participation rules: https://agentmemories.ai/rules
- Microsoft Research, Memora (ICML 2026): https://www.microsoft.com/en-us/research/blog/memora-a-harmonic-memory-representation-balancing-abstraction-and-specificity/
- LongMemEval-V2: https://xiaowu0162.github.io/longmemeval-v2/
- AMA-Bench, ICML 2026: https://proceedings.mlr.press/v306/zhao26bs.html
- MemoryArena, ICML 2026: https://proceedings.mlr.press/v306/he26am.html
- WorldMemArena: https://worldmemarena-mem.github.io/
- MemoryAgentBench: https://github.com/HUST-AI-HYZ/MemoryAgentBench
- StateMemBench: https://arxiv.org/abs/2608.19652
- TimlMemBench, Findings EMNLP 2026: https://aclanthology.org/2026.findings-emnlp.507/
- FAMA / From Recall to Forgetting, Findings ACL 2026: https://aclanthology.org/2026.findings-acl.1337/
- CL-Bench: https://arxiv.org/abs/2602.03587
- MCB / Memory–Clarification Boundary: https://aclanthology.org/2026.nora-1.2/
- Agentic Abstention, NeurIPS 2026: https://lhannnn.github.io/agentic-abstention/
- MACE competitive-baseline reproduction issue: https://github.com/Luminous-Dynamics/symthaea/issues/13261


---

## 12. Receipt-backed dependency lineage integration (MACE sidecar v1, 2026-11-22)

### Architecture decision

The canonical mace_counterfactual_evidence evaluator and mace_counterfactual_aggregate reducer remain the only benchmark pipeline. The prior standalone knockout executable is not carried forward: event/root/family interventions are now extended in the canonical evaluator, so every retrieval method is evaluated against the same intervention manifest and cluster-first statistical reducer.

MACE does not create a second persistent evidence graph. This is a bounded, run-local projection over ingress-verified lineage records. The canonical ownership remains:
- #10410 — provider lineage and common-mode source dependencies;
- #4918 — claim-relative independence and shared failure-domain semantics;
- #6283 — measurement validity and benchmark-evidence graph principles;
- #6430 — Mycelix DKG as the authoritative substrate, with deterministic acyclic evidence projections for replay.

The MACE sidecar is a consumer view, not an authority or truth oracle.

### Receipt-backed contract

The ingress VerifiedLineage contract now includes:
- canonical derived_from_event_ids;
- a dependency completeness state (Complete, Partial, Unknown);
- a digest for the exact dependency manifest when the state is Complete or Partial.

Ingress rejects self-references, duplicate or unsorted parent IDs, zero IDs, missing manifest digests for declared dependencies, and contradictory Unknown declarations that carry parents or a digest. The verified fields are copied into EvidenceReceiptBundle, not reconstructed from a caller's Experience booleans.

ConsolidationEvidencePackV1::dependency_sidecar() emits EvidenceDependencySidecarV1 with:
- the exact (valid_at_ms, known_at_ms) query;
- event ID, action and scope;
- verified source family and root;
- upstream source, artifact and transformation identities;
- lineage and raw-event receipt digests;
- the dependency declaration and manifest digest;
- lifecycle status verified at the same as-known snapshot.

The sidecar is deterministically ordered and has a canonical SHA-256 and JSON interchange representation, governed by data/schemas/mace_evidence_dependency_sidecar_v1.schema.json. Its hash detects content/order changes; it is not a signature. Authenticity and authority remain the responsibility of the actual Mycelix/ingress verifier and trust policy.

### Canonical evaluator integration

The main counterfactual evaluator now accepts an optional ingress sidecar when building its evidence index and candidate observation:
- sidecar event/action/scope/root/family identities must match the candidate evidence exactly;
- only receipt-declared, in-pool derived_from edges are followed;
- event, root, family, unknown-root/family and root-pair interventions expand to known transitive derived descendants;
- a candidate whose declared parent is outside the selected evidence pool is marked lineage-incomplete and the unresolved parent reference is retained;
- candidate input/evidence digests bind the sidecar digest; the run identity also binds the sidecar schema digest;
- matched random controls are stratified by scope/action/estimand, exclude target source roots/families, match post-closure root/family/unknown-identity counts, and are rejected if their expanded dependency closure overlaps or descends from the target knockout.

A candidate without a verified sidecar is not treated as having a complete empty graph: every available event is counted as lineage-incomplete. The existing public generated-world runner has no receipt sidecar input yet, so its results remain diagnostic-only and explicitly lineage-incomplete. The receipt-backed path is exercised by authored synthetic ingress→consolidation→evaluator tests, not by the CLI's generated-world benchmark run.

### Lifecycle boundary

The sidecar records lifecycle status as known at one bitemporal snapshot. Ingress currently admits only records that are active at that snapshot; retracted/superseded records are not silently projected into the transferable candidate memory. A later as-known query must re-run lifecycle verification. The current dependency projection models derived_from closures; it does not infer or claim a complete supersession graph. Full historical lifecycle closure requires an explicit verifier-issued supersession edge/history contract, which is a follow-up integration, not something inferred from event order or a status enum.

### Canonical CLI sidecar input (2026-11-22)

The canonical evaluator now accepts an optional sixth positional argument after the existing cluster count, candidate JSONL path, evaluator JSONL path, seed and root-pair cap:

```text
mace_counterfactual_evidence 1 candidate.jsonl evaluator.jsonl 1234 32 dependency-sidecar.json
```

The input must be one canonical `mace_evidence_dependency_sidecar_v1` JSON object for the exact single-world evidence pool (`clusters=1`). The decoder rejects unknown fields, unsupported versions, malformed identities, noncanonical event/parent ordering, digest mismatch and non-active lifecycle records. It verifies the sidecar's canonical SHA-256 commitment; it does not authenticate who issued the receipts. That remains an ingress/Mycelix verifier responsibility.

**Trust downgrade is enforced in code, not just documented:** `from_json_value` always produces an integrity-checked but issuer-unverified sidecar. The private in-process constructor used after ingress receipt admission is the only path that sets the process-local `ingress_receipts_verified` capability. JSON export/import intentionally loses it. An imported sidecar can still drive diagnostic dependency closures, but every event is marked lineage-incomplete, it cannot produce matched-random independence controls, the evidence digest binds the untrusted state, and candidate/evaluator records expose `dependency_sidecar_ingress_verified=false`. Only an ingress-produced object carries `true`; a production-grade verifier implementation is still required before this flag can represent cryptographic Mycelix authority.

The sidecar is now threaded through baseline candidate observations, intervention planning, all five retrieval methods and each knockout. For each intervention, the exact retained event set is paired with `retain_without_event_ids`; a surviving child keeps its parent reference, so the removed parent becomes an unresolved dependency rather than a silent independence claim. The run identity now binds the sidecar digest and hashes the ingress source that defines the decoder. The CLI remains diagnostic-only and the generated-world corpus is not a live external Mycelix evaluation. Sidecar input is a strict integration seam, not proof of live deployment.

The active-only schema, ingress parser, projection and candidate/evaluator output schema now agree on this trust distinction. A self-generated SHA-256 digest is never treated as issuer proof. The JSON parser currently validates shape, integrity and canonical ordering only; it has no signature-verification keyring or Mycelix receipt client.

### Authored regression coverage

The canonical evaluator test suite now includes a two-event synthetic chain:
- ingress verifies event 43 as derived from event 42;
- the consolidation pack exports the exact receipt-backed sidecar;
- the canonical event-42 knockout removes both 42 and 43;
- the sidecar digest changes the candidate input identity;
- removing only event 42 leaves event 43 with an unresolved parent reference and marks it incomplete;
- cyclic ancestry and sidecar/evidence scope/family/root/action mismatches fail closed.

The sidecar adapter also tests incomplete declarations and receipt digest tampering in the ingress suite. At this handoff, the evaluator has 24 authored unit tests and ingress has 47 authored unit tests. No cargo/rustc/rustfmt is installed in the authoring environment, so these tests are authored, not executed; no compile, format, test, or benchmark pass is claimed.

### Industry benchmark frontier and MACE positioning (2026-11-22)

The Agent Memory Challenge Cycle 2 official results were published on November 19, 2026. The open-method leaders were Resurface (textual 61.22), InsightMem (coding 63.33% overall resolved), and Khipu (multimodal 43.26). MemoraX led the separate commercial/closed-source division (textual 69.26, coding 68.00% resolved, multimodal 53.68). These are separate task tracks and divisions, not a single capability score. AML requires participant-hosted public Add/Search APIs; it does not deploy repository-only submissions. MACE has not entered Cycle 2.

Other distinct research anchors:
- TimlMemBench (Findings EMNLP 2026): 208 reviewed bilingual samples, 1,566 sessions, 1,715 questions and 1,821 governance/activation cases; raw-native ingestion and governed activation matter in addition to canonical-pool retrieval.
- FAMA (Findings ACL 2026) explicitly penalizes use of obsolete or invalidated memory. Stale-memory harm should be a separate score, not averaged away by recall.
- StateMemBench has 234 multi-session state-tracking scenarios. Published same-backbone comparisons show current-state accuracy of 0.205→0.363 for one DeepSeek-V4-Flash setup and 0.149→0.233 for one Qwen-3.5-9B setup. These are protocol-specific results, not general model multipliers.
- WorldMemArena evaluates 400 multi-session multimodal tasks across write, maintain, retrieve and use stages; good write/storage scores do not guarantee downstream action competence.
- AMA-Agent reports 57.22% average accuracy on AMA-Bench with an 11.16-point gain over its strongest reported baseline in that paper's protocol.
- MemoryArena shows why long-context recall cannot stand in for interdependent multi-session environment success.

Sources: AML Cycle 2 results (https://agentmemoryleaderboard.ai/results/cycle-2/) and rules (https://agentmemories.ai/rules); TimlMemBench (https://aclanthology.org/2026.findings-emnlp.507/); FAMA (https://aclanthology.org/2026.findings-acl.1337/); StateMemBench (https://arxiv.org/abs/2608.19652); WorldMemArena (https://arxiv.org/abs/2605.29341); AMA-Bench (https://proceedings.mlr.press/v306/zhao26bs.html); MemoryArena (https://proceedings.mlr.press/v306/he26am.html).

### Next qualification gates (targets, not results)

| Evaluation | Target / claim threshold |
|---|---|
| AML textual, coding, multimodal | Enter a future cycle only with a hosted API; aim for ≥3-point improvement over the strongest reproduced open-method baseline on each separately matched track |
| StateMemBench | ≥95% current-state accuracy on a preregistered subset; zero known-superseded answers on critical cases |
| TimlMemBench | Zero unauthorized activation on known revoked, wrong-scope and unknown-trust cases; raw-native results reported separately |
| FAMA | Zero known-invalid reuse on deterministic invalidation controls |
| MemoryArena | ≥5 pp task-success lift over the strongest reproduced baseline with matched scaffold/model/tool budget |
| WorldMemArena | No stage regression across write/maintain/retrieve/use; disclose multimodal and action-grounding costs |
| MACE sufficiency | Upper 95% bound on added regret ≤ preregistered epsilon; paired lower 95% bound for memory-vs-no-memory utility >0 |
| Dependency qualification | Zero false-independence acceptance on seeded shared-root/declared-ancestor attacks; unknown/partial ancestry never counts as complete |
| Compression | ≥50% lower all-in storage/context cost with ≤1 pp task-success loss on a preregistered lane |

The industry scores are not directly comparable with MACE's synthetic utility/regret. Keep separate scorecards. For rare integrity failures, publish one-sided exact upper confidence bounds and the independence assumptions; when samples share scenario/root templates, use cluster-level inference rather than counting every intervention row as an independent trial.

### Remaining work

1. Connect a real Mycelix verifier to the ingress trait; mock verifier receipts are not cryptographic attestations.
2. Make the canonical CLI accept a pinned receipt-backed evidence fixture and sidecar manifest, while keeping the public generated-world lane separately labelled.
3. Add explicit receipt-bound supersession history/edge semantics for lifecycle knockouts; do not infer them from current status.
4. Move evaluator labels/seed custody out of the candidate repository and qualify the manifest under a separately held evaluator.
5. Execute exact-head Rust formatting, compilation and tests on a real toolchain, then publish receipts. Skipped hosted checks remain skipped, not passes.


## 13. New ICML 2026 memory paradigms and MACE comparator design (2026-11-22)

Three recent ICML 2026 methods sharpen the research gap:

1. **MRAgent — active memory reconstruction.** It rejects a rigid retrieve-then-reason pipeline and reconstructs memory paths dynamically as intermediate reasoning discovers new cues. The paper reports gains up to 23% over strong baselines on LoCoMo and LongMemEval with lower retrieval cost. MACE should include an active-reconstruction arm: evidence paths can be expanded, pruned or revised as the task contract is refined, while every resolved node still passes the same scope/provenance/lifecycle gates.
2. **MemoPilot — learned memory update policy.** It treats memory updating as a multi-turn decision problem and uses multi-turn GRPO to optimize updates for downstream performance of a frozen agent. It reports first-place Elo of 1762 on Limit Texas Hold'em and 1590 on Rock-Paper-Scissors. Those game-specific metrics are not comparable to MACE or general memory benchmarks; the transferable research idea is explicit credit assignment for *when and how memory changes*.
3. **Darwinian Memory — utility-driven pruning.** It decomposes long trajectories into reusable units and applies utility-driven selection/pruning, reporting mean gains of 18.0% in success rate and 33.9% in execution stability on its multi-app GUI evaluation, alongside reduced latency. MACE should test governed pruning as a baseline, but add hard anti-poisoning, source-lineage, revocation, and worst-family regret gates. Utility selection alone must not authorize promotion or forgetting.

WorldMemArena independently motivates the same move from snapshot retrieval to a lifecycle: write, maintain, retrieve and use over action–world interaction. Its public benchmark describes 400 multimodal multi-session tasks and notes that recall/storage quality does not guarantee use-time correctness, with costly and brittle performance in harder agentic regimes.

### How MACE should beat the frontier without making a misleading SOTA claim

The target is not to claim one architecture wins all benchmark scores. It is to demonstrate a better *Pareto frontier* on a matched frozen backbone and resource budget:

- **Quality:** downstream task success, decision utility, regret and calibration.
- **Evidence validity:** false independence, stale/revoked use, wrong scope/estimand, unknown ancestry and unsupported causal claims.
- **Adaptation:** useful active reconstruction, selective verification, re-observation and bounded experimentation.
- **Lifecycle:** promote, compress, supersede, quarantine, forget-from-index and privacy erase as separate decisions with independent receipts.
- **Economics:** all-in bytes, receipt/index/tombstone overhead, background consolidation compute, model/tool tokens, p50/p95/p99 latency, and energy/cost where measurable.
- **Distribution shift:** per-family and worst-regime degradation, not only pooled means.

The next baseline matrix under #8427 should include:
- full-history and no-memory controls;
- BM25, dense retrieval, hybrid retrieval and a strong reranker;
- versioned key-value/file memory;
- static HDC, HDC-LTC and typed-content HDC;
- MRAgent-style active associative reconstruction;
- MemoPilot-style learned memory update policy over a frozen backbone;
- Darwinian-style utility pruning;
- receipt-governed MACE consolidation/decision-sufficient memory;
- oracle upper bounds clearly labelled evaluator-only.

These are method comparators, not scores to average together. Reproduce the published systems under their own native evaluation protocols first; then run a separate matched-budget common suite where an exact adapter exists. MemoPilot's board-game Elo, Darwinian Memory's GUI success/stability percentages, MRAgent's LoCoMo/LongMemEval gains, AML's track scores and MACE's synthetic utility/regret must stay in different columns.

### AML Cycle 2 submission implications

Cycle 2 is closed. Its official results were published November 19, 2026. For a future cycle, repository publication is not enough: every entrant must host stable participant-operated Add and Search endpoints; the platform does not deploy a repository-only or Docker-only system. Open-source Methods also have declared embedding/LLM component restrictions, with rerankers treated separately in the current guide. Therefore, preserve an AML-compliant hosted adapter as a distinct integration profile rather than silently substituting the MACE HDC encoder into a component slot constrained by the next official rules. Re-check the next cycle's exact API/model rules when it opens.

Sources: MRAgent (https://proceedings.mlr.press/v306/ji26d.html); MemoPilot (https://proceedings.mlr.press/v306/cai26b.html); Darwinian Memory (https://proceedings.mlr.press/v306/mi26b.html); WorldMemArena (https://worldmemarena-mem.github.io/); AML Cycle 2 results (https://agentmemoryleaderboard.ai/results/cycle-2/); AML participation/API rules (https://agentmemories.ai/rules).


## 14. Agent Memory Challenge Cycle 2: capability-level gap analysis and next frontier (2026-11-22)

### Official current open-method reference

The official Agent Memory Challenge Cycle 2 results were published on 19 November 2026. They report nine separate leaderboards and 289 track entries across textual, multimodal and coding evaluation, with divisions for commercial products, open-source methods and closed-source methods. These divisions and task tracks must remain separate.

| Track / division | Cycle 2 leader | Primary result | Capability-level signal |
|---|---|---:|---|
| Textual / open-source | Resurface | 61.22 overall | Fact 62.98; relational 58.94; temporal 57.76; governance 76.55; personalization/care 38.82; workflow 59.67; epistemic safety/privacy 68.59; streaming 49.65 |
| Multimodal / open-source | Khipu | 43.26 overall | Retrieval 50.28; relational 45.82; temporal 34.23; continual learning 38.45; personalization 11.04; evidence governance 31.29 |
| Coding / open-source | InsightMem | 63.33% resolved | New-feature resolution 74.51%; bug-fix resolution 57.58%; medium input/output tokens; verbose return; fast search; medium total time |
| Textual / commercial products | MemoraX | 69.26 overall | Separate division; not an open-source baseline |
| Multimodal / commercial products | MemoraX | 53.68 overall | Separate division; not an open-source baseline |
| Coding / commercial products | MemoraX | 68.00% resolved | Separate division; not an open-source baseline |

The official result page also lists MemoraX in the closed-source standings. Its commercial, closed-source and open-source values must not be merged into a single SOTA number. Cycle 2 is a retrieval/governance/task-performance reference, not a measure of causal decision sufficiency. MACE has not entered this competition and its synthetic utility/regret results are not comparable to these scores.

Primary reference: https://agentmemoryleaderboard.ai/results/cycle-2/ (published 2026-11-19). The leaderboard's evaluation notes say that ranks are reported separately by track and division. Always cite the system name, track, division, version and metric when using a result.

### What these scores tell us

The open textual leader is strongest in governance (76.55) but has a material personalization/care gap (38.82) and streaming gap (49.65). The multimodal open leader has a still larger personalization (11.04) and evidence-governance (31.29) gap, while temporal reasoning is 34.23. The coding leader resolves 63.33% overall, with a visible new-feature/bug-fix split (74.51% vs 57.58%). Those differences argue against optimizing only a pooled recall score.

MACE's differentiated contribution should be evidence-valid, decision-changing memory:
- a retrieved item is not sufficient unless its scope, provenance, lifecycle and estimand match the current decision;
- unknown or partial ancestry is not independence;
- an obsolete/retracted/superseded item must not be reactivated by a summary or derived skill;
- evidence gaps should cause targeted acquisition, clarification or abstention rather than confident guessing;
- improvements must survive multi-session transfer under held-out source roots, regimes and objectives.

### New paradigm: active evidence reconstruction and value-of-information memory

Move beyond a one-shot \`retrieve → reason → answer\` pipeline. MACE should test a closed-loop process:

\`\`\`text
task contract
  → retrieve a provisional evidence set
  → detect missing / conflicting / stale / weakly identified evidence
  → choose the highest-value evidence action
  → acquire (retrieve another lineage path, inspect source, ask, measure, or intervene)
  → verify and append the receipt
  → revise the evidence set and decision
  → stop, act, or abstain under a declared risk budget
\`\`\`

The evidence action is a first-class policy choice, not an untracked tool call. Each action records its cost, expected information gain, authority/scope, receipt, latency and resulting belief/decision change. A candidate value-of-information score can be operationalized as expected reduction in downstream decision regret minus acquisition cost and risk, subject to hard source, privacy and authorization constraints. It must not be trained or evaluated against oracle labels visible to the candidate.

Required comparator arms:
1. passive top-k retrieval;
2. retrieve-then-rerank;
3. MRAgent-style active memory reconstruction;
4. active evidence acquisition with fixed heuristic value-of-information;
5. learned acquisition policy over a frozen answer agent;
6. full receipt-governed MACE policy;
7. oracle evidence-acquisition policy (evaluator-only upper bound).

The primary question is not “did retrieval find the right passage?” It is “did the agent acquire enough valid evidence to choose a better action, and did it stop when further search was not worth its cost?”

### Proposed qualification targets (future targets, not measured results)

For a future AML open-method entry, use Cycle 2 leaders only as versioned reference points; verify the next official rules and datasets before submission. Stretch targets against the current open-source leaders are:

| Track | Cycle 2 open leader | Initial matched-protocol hurdle (+3) | Stretch target (+5, aligned with #13284) |
|---|---:|---:|---:|
| Textual overall | 61.22 | ≥64.22 | ≥66.22 |
| Multimodal overall | 43.26 | ≥46.26 | ≥48.26 |
| Coding resolved | 63.33% | ≥66.33% | ≥68.33% |

Do not count these as achieved or directly infer them from MACE synthetic results. Per-capability targets should address the observed bottlenecks: textual personalization/care ≥45, streaming ≥55, governance ≥80 and epistemic safety/privacy ≥75; multimodal personalization ≥20, evidence governance ≥40 and temporal reasoning ≥40; coding bug-fix resolution ≥62% without losing new-feature performance. These thresholds are project targets, not official challenge thresholds.

For the MACE-specific benchmark, maintain separate scorecards and preregister:
- paired lower 95% confidence bound of memory-vs-no-memory decision utility greater than zero;
- upper 95% confidence bound on added regret no greater than a preregistered epsilon;
- zero accepted false-independence cases on deterministic shared-root and declared-ancestor attacks;
- zero unauthorized activation of known revoked, wrong-scope, stale or unknown-trust evidence in the deterministic safety panel;
- value-of-information policy improves task utility over passive retrieval at matched model/tool/observation budgets, with net acquisition cost reported;
- under equal all-in memory cost, ≥5 percentage-point final-chain success improvement over the strongest reproduced non-oracle baseline as a stretch target;
- compression target ≥50% lower all-in steady-state and maintenance cost with ≤1 percentage-point task-success loss on a preregistered lane.

For rare integrity failures, report exact one-sided confidence bounds and independent scenario/root counts. Zero observed failures are not proof of zero risk: under independent trials the rough 95% upper bound is about 3/n, and correlated interventions must be clustered by source root/scenario rather than counted as independent samples.

### Implementation sequence

1. Finish the receipt-sidecar reader and exact candidate-pool binding; distinguish integrity hashing from issuer authentication.
2. Add a genuine ingress-produced fixture to the evaluator path; the current CLI sidecar hook still runs the public generated-world corpus and is not a live Mycelix import.
3. Add typed lifecycle history edges (\`supersedes\`, \`retracts\`, privacy erase) with issuer/receipt identity, valid-time, known-time and descendant propagation; do not infer history from status labels or timestamps.
4. Add a value-of-information action schema and sealed evidence-acquisition scenarios, with acquisition receipts and a candidate-visible stop rule.
5. Reproduce the public AML, TimlMemBench, FAMA, StateMemBench, MemoryArena, WorldMemArena and AMA-Bench protocols separately, then run matched-budget MACE extensions.
6. Move hidden labels, seeds and primary evaluator custody outside the candidate implementation before confirmatory claims.



## 15. First executable decision-sufficiency frontier and EVSI planner (2026-11-23)

### Deliverables on the stacked frontier branch

The new Rust binary is \`mace_sufficiency_frontier\` in the existing evaluator crate. It is a separate increment stacked on the receipt-lineage branch and tracked under #13284. It does not modify the receipt-lineage PR's ownership.

Invocation from the repository root:

\`\`\`text
cargo run --manifest-path research/broca-mace-transfer-bench-v0-1/evaluator/Cargo.toml --bin mace_sufficiency_frontier -- 64 candidate.jsonl evaluator.jsonl 557... 0.01 acquisition.jsonl
\`\`\`

Arguments are: world-cluster count (1..=1000), candidate JSONL output, evaluator JSONL output, public random seed, regret-gap epsilon and acquisition-plan JSONL output. Omitted arguments use documented defaults in the binary.

Each generated world has:
- a no-memory reference;
- a full-scope information reference (not an equal-cost competitor);
- event-ID-ordered candidate selection (explicitly not a learned key/value retriever);
- a deterministic seed-clustered random control;
- static event-ID HDC;
- typed-content HDC.

The four budgeted arms are swept at per-action top-k values 1, 2, 4 and 8. The same frozen \`Policy::CausalQualified\` is applied to every selected evidence set. HDC and typed-content indexes are built once per world/arm and their build time and payload lower bound are reported for each corresponding budget point.

### Decision-sufficiency outputs

The candidate JSONL contains task contracts, selected event IDs, candidate decisions, exact candidate-input digests, method/budget identity, profile identity and measured costs. It contains no evaluator optimal action, realized utility, scenario-family label or regret.

The evaluator JSONL separately scores:
- absolute regret against the evaluator-only optimal action;
- regret gap against the full-scope reference;
- epsilon sufficiency by regret alone;
- full-scope decision agreement;
- oracle-optimal agreement;
- per-world Pareto-frontier membership over selected Experience-struct bytes, index-payload bytes, measured index-build plus selection/decision latency, and absolute regret.

The exact-scope/full-scope reference is an information reference, not proof of world completeness. A lower regret gap is not by itself an evidence-qualification certificate.

**Accounting boundary:** current byte values are lower bounds from \`size_of::<Experience>()\` and retrieval index payload. They exclude allocator/container overhead, receipts, sidecars, raw source bytes, journals, lifecycle metadata, serialization, background maintenance and verification work. Token cost is null because the synthetic typed Experience records are not raw conversational text. Do not present these fields as all-in cost.

**Evidence boundary:** the current generated-world lane has no authenticated dependency sidecar. Every available event is counted as lineage-incomplete, imported/verified sidecar fields are null, and \`evidence_qualified_sufficient\` is always false. The runner can explore decision-regret sufficiency only; it cannot claim evidence-qualified or production memory sufficiency.

### Active evidence acquisition: first decision-specific EVSI core

The new \`acquisition.rs\` module implements exact finite-hypothesis expected value of sample information (EVSI). For a candidate query \(q\):

\`\`\`text
prior Bayes risk
  - expected posterior Bayes risk after q
  - acquisition cost
  - delay cost
  - approved-disclosure exposure cost
= net EVSI
\`\`\`

It ranks positive-net-EVSI requests deterministically and otherwise returns \`stop_or_abstain\`. It compares decision-relevant measurement with a high-entropy but decision-irrelevant observation. Source authorization, scope validity, privacy approval and deadline feasibility are hard admissibility gates; no amount of predicted utility can override them.

The planner receives prior probabilities, declared action-loss vectors and candidate observation likelihoods. It does not inspect realized future outcomes or evaluator labels. The current CLI emits a small analytic two-hypothesis fixture to \`acquisition.jsonl\`; it does not contact a source, execute a measurement or obtain a receipt. Cost and loss terms must be expressed in a common, declared decision-loss unit; the code does not invent a conversion from dollars, seconds or privacy exposure. The exposure term applies only after the separate hard privacy-approval gate has passed.

The fixture is a policy unit test, not an empirical agent benchmark. The module has regression tests for decision-relevant versus uncertainty-only evidence, unauthorized high-value queries, malformed observation models and query/outcome order invariance. The frontier also tests retrieval-profile binding in candidate identity and random-control selection invariance under reversed memory ordering.

### Current competitor and baseline boundaries

The latest official AML Cycle 2 open-method references are Resurface 61.22 (textual), Khipu 43.26 (multimodal) and InsightMem 63.33% resolved (coding), published 19 November 2026. Keep the commercial/closed-source divisions separate. These are leaderboard targets, not MACE measurements.

MemoryArena evaluates interdependent multi-session agent/environment tasks and reports a gap between long-context recall and downstream task completion. StateMemBench has 234 evolving-state scenarios and demonstrates the value of explicit current-state tracking. FAMA penalizes obsolete-memory reuse. WorldMemArena covers 400 multimodal multi-session tasks and separates write, maintain, retrieve and use. These external benchmarks provide distinct qualification lanes; their raw percentages must not be pooled with MACE regret.

References:
- AML Cycle 2 official results: https://agentmemoryleaderboard.ai/results/cycle-2/
- MemoryArena, ICML 2026: https://proceedings.mlr.press/v306/he26am.html
- StateMemBench: https://arxiv.org/abs/2608.19652
- FAMA / Memora benchmark, Findings ACL 2026: https://aclanthology.org/2026.findings-acl.1337/
- WorldMemArena: https://worldmemarena-mem.github.io/

### Next qualification gates for this slice

1. Execute exact-head \`cargo fmt --check\`, \`cargo check\`, unit tests and deterministic fixture run on a real Rust toolchain. Authored tests are not passes.
2. Validate both JSON Schemas with an independent Draft 2020-12 validator and replay the JSONL outputs through a separate schema/receipt checker.
3. Add a raw-trajectory adapter; only then add BM25, dense, hybrid retrieval and strong reranker arms. Do not create synthetic language from the Experience struct.
4. Add independent source verification and V2 multi-parent/root-closure/history semantics from #13264 before any evidence-qualified frontier point.
5. Replace the analytic EVSI fixture with a sealed multi-session query/observation dataset; calibrate the likelihood models on train/calibration clusters, evaluate on held-out clusters, and record realized observation receipts.
6. Add cluster-level paired inference and bootstrap over independent world/root clusters. Intervention rows and budget points are not independent samples.
7. Add uncertainty over observation-model likelihoods and robust EVSI bounds. Point-estimate EVSI is only the first planner core.
8. Move evaluator labels, seed custody and primary evaluation outside the candidate implementation before making superiority claims.

The branch is source-authored and statically inspected, not compiled. No frontier metric, EVSI gain, empirical utility lift or SOTA result is claimed yet.



## 16. Frontier runner implementation and active-acquisition fixture (2026-11-23)

### Frontier sweep dimensions

The new \`mace_sufficiency_frontier\` runner emits three separate JSONL streams:
- candidate-only decisions;
- evaluator-only regret/Pareto scores;
- an evidence-acquisition EVSI policy fixture.

The generated stress fixture defines 21 scenarios per independent world cluster. The frontier evaluates 18 points per scenario: no-memory, full-scope reference, and four retrieval/control arms at four per-action budgets. At the default 64 clusters, the expected output is 24,192 candidate rows and 24,192 evaluator rows, plus four EVSI fixture rows. This is a code-derived expected count, not an executed-run result.

The event-ID arm is intentionally named an **ordered control**, not a key/value retrieval system: it ranks by deterministic event ID and does not implement a separately retained KV store. The HDC arms remain typed candidate indexes, not natural-language retrievers. BM25, dense and hybrid retrieval are not fabricated by tokenizing \`Experience\`; they require the raw-session adapter.

The Pareto filter is an exact within-world non-dominance check over four dimensions: selected struct-byte lower bound, index-payload-byte lower bound, measured index-build plus candidate-selection/decision latency, and evaluator absolute regret. It is not an aggregate statistical frontier across independent worlds. The latter requires cluster-level paired aggregation/bootstrapping and is a subsequent increment.

### EVSI core and policy boundary

The finite-state EVSI core validates:
- prior probabilities sum to one;
- query observation likelihoods are finite, normalized per hypothesis and have consistent dimensions;
- action-loss inputs and all cost components are finite and non-negative;
- query IDs and outcome IDs are unique;
- the observation-model digest is canonical lowercase SHA-256.

For every query it computes prior Bayes decision risk, expected posterior Bayes risk, expected risk reduction, acquisition cost, delay cost, approved-disclosure exposure cost and net EVSI. It ranks only admissible positive-net-EVSI queries. When no query qualifies, the output is \`stop_or_abstain\`.

The query carries upstream gate results for source authorization, exact scope validity, privacy approval and deadline feasibility. This module does not authenticate those booleans or issue authority; a real adapter must obtain them from the appropriate verifier/policy service. Rights/privacy prohibition is never converted into an EVSI penalty. The disclosure-exposure cost only describes a burden after the separate privacy-approval gate has passed. All scalar costs must be calibrated into the same explicit decision-loss unit; this module does not invent conversions between dollars, seconds, privacy and utility.

The current fixture has two equally likely hypotheses, one decision-relevant measurement, one high-entropy but decision-irrelevant signal, and one unauthorized perfect measurement. It demonstrates policy mechanics only. No observation is executed, no source is contacted, no receipt is obtained, and no empirical agent improvement is claimed.

### Reproduction and provenance

\`\`\`text
cargo run --manifest-path research/broca-mace-transfer-bench-v0-1/evaluator/Cargo.toml --bin mace_sufficiency_frontier -- 64 candidate.jsonl evaluator.jsonl 557... 0.01 acquisition.jsonl
\`\`\`

The run identity binds the runner source, generated fixture, policy and retrieval implementation sources, both JSON Schemas, arguments and optional \`MACE_SOURCE_REVISION\`. Candidate and evaluator streams remain separate. The candidate stream excludes scenario-family labels and evaluator utilities; the evaluator stream records the world cluster and hidden scores. Evaluator custody is nevertheless co-located in the same binary and repository, so this is still diagnostic rather than independent evaluation.

### Comparator interpretation and claim boundary

Current external paradigm arms to reproduce:
- **MRAgent** dynamically explores and prunes associative memory paths during reasoning, reporting up to 23% gains on its LoCoMo/LongMemEval protocol. It is a direct comparator for active reconstruction, not an EVSI or causal-authority result. https://proceedings.mlr.press/v306/ji26d.html
- **MemoPilot** learns memory updates over a frozen player using multi-turn GRPO, reporting 1762 Elo on Limit Texas Hold'em and 1590 on Rock-Paper-Scissors. Those game-specific Elo values are not comparable to MACE regret or AML track scores. https://proceedings.mlr.press/v306/cai26b.html
- **Darwinian Memory** uses utility-driven pruning in GUI-agent trajectories, reporting average gains of 18.0% in task success and 33.9% in execution stability in its own multi-app protocol. MACE should reproduce it as a governed pruning comparator and add anti-poisoning, lifecycle and worst-family gates. https://proceedings.mlr.press/v306/mi26b.html

The research opportunity is to combine active reconstruction, learned consolidation, utility-aware pruning and receipt-governed evidence eligibility without letting the memory learner certify its own sources or labels. We must reproduce each competitor under its native protocol before building any common matched-budget comparison.

### Current verification

Five frontier unit tests and four EVSI unit tests are authored. They have not been run. No Cargo, rustc, rustfmt or Nix toolchain is available in the authoring environment. JSON Schema files parse as JSON, but independent schema validation, build, test, formatting and deterministic execution remain pending. All results remain unexecuted source work; no SOTA, benchmark, safety or empirical utility claim is made.



## 17. Cluster-paired frontier aggregation (2026-11-23)

### Reducer

A separate Rust binary \`mace_sufficiency_aggregate\` now consumes the evaluator-only JSONL stream from \`mace_sufficiency_frontier\`:

\`\`\`text
cargo run --manifest-path research/broca-mace-transfer-bench-v0-1/evaluator/Cargo.toml --bin mace_sufficiency_aggregate -- evaluator.jsonl summary.jsonl 1234
\`\`\`

It performs the comparison in the correct hierarchy:
1. require exact run/source/schema provenance across every evaluator record;
2. reject duplicate \`(world_cluster, scenario_family, method, per_action_top_k)\` rows;
3. require every method/budget to cover the same independent world clusters and the same scenario-family set within each cluster;
4. average scenario-family outcomes with equal weight inside each world cluster;
5. calculate paired method-minus-no-memory utility and absolute-regret deltas within each same world cluster;
6. bootstrap 2,000 replicates by resampling independent world-cluster deltas, not individual scenario rows or budget points;
7. calculate the aggregate mean-cost/mean-regret Pareto frontier across arm/budget points.

Output includes mean utility/regret, epsilon-sufficiency rate by regret alone, oracle-optimal and full-scope-agreement rates, lower-bound resource metrics, paired mean utility/regret deltas vs no-memory, cluster-bootstrap 95% percentile intervals, resampling unit, provenance hashes and frontier membership.

The reducer labels fewer-than-10-cluster results descriptive-only. It also labels the comparisons exploratory because no multiplicity correction or preregistered primary arm has yet been applied. The confidence intervals are not presented as confirmatory significance claims.

### Inferential boundary

Scenario families within a generated world are not independent samples. The reducer gives each family equal weight within its cluster and treats only distinct world clusters as resampling units. It also requires complete, identical family/cluster coverage across arms before calculating paired contrasts, preventing a method with missing difficult cases from looking artificially better.

This is still a synthetic diagnostic benchmark:
- 21 generated scenario families per cluster; 18 arm/budget points per family;
- default 64 clusters imply 24,192 evaluator rows and 18 aggregate rows (code-derived expected counts, not executed results);
- no authenticated source-lineage sidecars, so evidence-qualified sufficiency remains false;
- resource estimates remain Experience-struct and HDC index-payload lower bounds;
- no multiplicity correction, external evaluator custody or raw-session competitor adapters yet.

The aggregate schema is \`mace_sufficiency_aggregate_v0_1\`. The reducer binds the exact evaluator JSONL bytes, its own source and output schema, input run/source/schema provenance, bootstrap seed and replicate count.

### Tests authored

Five reducer tests cover equal-weight within-cluster aggregation, deterministic cluster bootstrap, paired world-cluster matching/regret-delta direction, multi-dimensional Pareto dominance and canonical source revision validation. They have not been run in this environment.

