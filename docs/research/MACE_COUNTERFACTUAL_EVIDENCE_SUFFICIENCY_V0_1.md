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
- **Khipu (official multimodal open-method leader)** keeps source messages verbatim and describes BM25, vector, entity, literal-match and date channels, fused with intent-aware reciprocal-rank fusion. Its multimodal branch retains original image evidence alongside descriptions. This suggests an explicit MACE retrieval baseline combining exact lexical anchors, temporal/entity channels and original-evidence pointers. Repository: https://github.com/Lv999APTITUDE/khipu.
- **InsightMem (official coding open-method leader)** combines source-chunk BM25, code-expanded BM25 and dense cosine retrieval with reciprocal-rank fusion, adjacent-slice merging, deduplication and query coverage. This is a stronger coding baseline than generic dense retrieval. Repository: https://github.com/Kline-song/InsightMem.
- **Memora (ICML 2026)** reports 86.3% LLM-judge accuracy on LoCoMo and 87.4% on LongMemEval in its own setup, with up to 98% fewer context tokens than full-context inference. Its abstraction hierarchy and cue anchors are particularly important competitors to a flat typed vector index. These scores are not directly comparable to AML or MACE. See https://www.microsoft.com/en-us/research/blog/memora-a-harmonic-memory-representation-balancing-abstraction-and-specificity/.
- **LongMemEval-V2** reports AgentRunbook-C at 74.9% on its small split with 108.3-second query latency and 70.1% on medium with 139.9 seconds; Codex is 69.9%/177.2 seconds and 68.7%/185.8 seconds respectively. It uses a fixed reader and reports accuracy and query latency, so preserve that exact protocol. See https://xiaowu0162.github.io/longmemeval-v2/.
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

**Baseline parity rule:** the current MACE Experience fixture is structured and contains no general natural-language trajectory field. A lexical BM25/dense system over this typed fixture would not reproduce Resurface, Khipu or InsightMem. Build a separate raw-trajectory Add/Search adapter for AML-style comparisons, while preserving the synthetic causal intervention lane as a distinct MACE evaluation. Do not blur the two.

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
4. Added deterministic event, source-root, source-family, bounded root-pair, stratified-random and no-memory interventions.
5. Stratified random controls by regime, objective, action and estimand digest; the manifest reports unmatched target strata.
6. Added two candidate modes for every full-evidence and knockout condition: exact-scope scan and typed-content HDC top-2 per declared action.
7. The typed-HDC mode rebuilds the typed index after each intervention, converts the candidate-visible DecisionNeedV1 to an action-specific query, resolves returned IDs against canonical Experience records and rejects any index-time/current-view digest mismatch.
8. Bound candidate input digests to DecisionNeedV1 plus the post-intervention canonical evidence-set digest. Candidate observation IDs additionally bind the selected method.
9. Report both regret gap versus full exact-scope evidence and incremental regret gap versus the same method's own full-evidence baseline, separating compression/retrieval loss from additional knockout loss.
10. Split candidate decision records from evaluator intervention manifests/scores. Candidate-facing pair identifiers are opaque and derived only from candidate-visible task/evidence digests, not evaluator scenario-family labels. Candidate records expose the retrieval method and returned evidence IDs, but not intervention kinds, removed IDs or evaluator utilities.
11. Added source-code and fixture digests to the evaluator run manifest and introduced `data/schemas/mace_counterfactual_evidence_v0_1.schema.json`.
12. Added 18 authored binary tests for duplicate IDs, root aliases, unknown identities, source-family closures, deterministic/truth-blind planning, coalition caps, matched controls, evidence-digest changes, candidate-contract rejection, opaque pair identifiers, typed-HDC canonical resolution after root knockout and explicit abstention.

### Remaining qualification work

1. Execute rustfmt, compile, unit tests and the runner; validate every JSONL record against the new schema.
2. Add verified transitive source-lineage/derived-view closure. The current Experience fixture has no parent-edge field, so this binary only groups direct root/family identities.
3. Add explicit semantic decoys, correction-between-retrieval-and-use and lifecycle closure.
4. Add exact subset enumeration for small evidence sets, sampled larger coalitions and decision-interaction estimates.
5. Add cluster-level summaries, confidence intervals and pre-registered primary contrasts.
6. Reproduce BM25, dense, sparse/dense RRF, intent-adaptive channels, context neighbours, causal-graph retrieval and state-first retrieval on the correct raw or structured data lanes.
7. Move evaluator truth to independent custody and run held-out external benchmarks through pinned adapters.

## 10. Current implementation and validation boundary

The follow-on branch is stacked directly on the inspected #13221 child-PR head and contains the following authored artifacts:

- `research/broca-mace-transfer-bench-v0-1/evaluator/src/bin/mace_counterfactual_evidence.rs`
- `data/schemas/mace_counterfactual_evidence_v0_1.schema.json`
- `docs/research/MACE_COUNTERFACTUAL_EVIDENCE_SUFFICIENCY_V0_1.md`

Example invocation after a Rust-enabled checkout:

```bash
MACE_SOURCE_REVISION="$(git rev-parse HEAD)" \
cargo run --manifest-path research/broca-mace-transfer-bench-v0-1/evaluator/Cargo.toml \
  --bin mace_counterfactual_evidence -- \
  64 mace_counterfactual_candidate.jsonl mace_counterfactual_evaluator.jsonl 12345 32
```

Arguments are cluster count, candidate JSONL path, evaluator JSONL path, public intervention seed and maximum source-root pairs. The runner writes separate streams. Evaluator output contains the frozen intervention manifest and hidden-outcome scores. Candidate output contains method, DecisionNeed/evidence digests, retrieved event IDs, chosen action, evidence-support IDs, typed-profile digest when available and resource counts, but not knockout labels, removed event IDs or evaluator utility. The candidate observation identifier allows the streams to be joined without placing intervention type in the candidate record.

### Important limits

- The root index groups equal nonzero `SourceRootDigest` values. A nonzero digest is an identity key, **not cryptographic authentication of its issuer**. The synthetic `source_family` integer is also a fixture identity, not an independently verified family credential.
- The current `Experience` structure has no explicit parent/derived-view edges. The implementation therefore does not yet compute transitive lineage closure; it groups records sharing the same direct root digest or family label.
- Unknown root/family values are counted separately and do not inflate known-identity counts. The code does not independently verify signatures, source authority, tenant permissions or external receipts.
- Matched-random controls preserve count within regime/objective/action/estimand strata, but do not yet guarantee equal source-root/family multiplicity, exact serialized bytes or equal verification work. They are stratified random controls, not fully dependency- and cost-matched placebo interventions. They are omitted when the candidate-visible pool cannot match a stratum; coverage is reported. The current byte measure is only `size_of::<Experience>()` per record and excludes serialization, allocator, index, sidecar, source and verification costs.
- Root-pair interventions are deterministically enumerated up to a caller-supplied cap. Higher-order coalition sampling, exact subset optimization, Shapley attribution, statistical aggregation and confidence intervals are not yet implemented.
- The audit now reruns both **exact-scope scan** and **typed-content HDC top-2 per action** on every full-evidence and knockout condition, then applies the same `Policy::CausalQualified` admission/decision rule. This separates the exact-evidence reference, the full typed-compression gap, the knockout gap within each method, and total regret versus exact full evidence. The first HDC index is rebuilt per condition; no incremental update/replay performance is claimed.
- Typed-content HDC remains structured feature retrieval over the synthetic Experience type, not natural-language or multimodal memory. The experiment reports candidate IDs and canonical view resolution; it does not yet report labelled retrieval recall/precision because the fixture has no independent per-query relevance annotation.
- DecisionNeedV1 is validated and used by the typed query adapter. The exact-scope method uses its declared scope and action alternatives. A general natural-language/task/environment parser is not implemented.
- The candidate/evaluator are still in one public repository. Separate JSONL files are a data-boundary test, not independent evaluator custody.
- The existing typed HDC remains structured typed-content retrieval; it is not natural-language understanding, a learned causal model or a production source verifier.

Abstention is an explicit zero-utility outside option only for the current synthetic benchmark. Every real application must define its own abstention cost/value and deadline policy.

**Validation:** Cargo, rustc and rustfmt are unavailable in this execution environment. The new binary, tests and JSON Schema are source-authored but unexecuted. No compilation, test pass, formatting pass, schema validation, runner result or hosted-check pass is claimed. The previously inspected PR workflows were skipped; skipped is not passed.

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
