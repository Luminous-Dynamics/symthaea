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

- **AML Cycle 2 official open-method results**, published November 19, 2026: Resurface 61.22 textual, InsightMem 63.33% coding resolved, and Khipu 43.26 multimodal. Separate commercial leaders are MemoraX at 69.26, 68.00% resolved and 53.68 respectively. These are different tracks and divisions, not a common score. The official rules require participant-hosted Add/Search endpoints; a repository-only or Docker-only submission is not sufficient.
- **Memora (ICML 2026)** reports 86.3% LLM-judge accuracy on LoCoMo and 87.4% on LongMemEval in its own setup, with up to 98% fewer context tokens than full-context inference. These are not directly comparable to AML or MACE.
- **LongMemEval-V2** reports AgentRunbook-C at 74.9% on its small split with 108.3-second query latency and 70.1% on medium with 139.9 seconds. Codex is 69.9%/177.2 seconds and 68.7%/185.8 seconds. These scores depend on its fixed reader and trajectory protocol.
- **AMA-Agent** reports 57.22% average accuracy on AMA-Bench and an 11.16-point improvement over the strongest baseline in that paper's protocol.
- **WorldMemArena** describes 400 multi-session multimodal tasks across lifelong evolution and agentic execution, with write/maintain/retrieve/use evaluation. Dataset revisions and sample counts should be pinned by release/commit before local comparison.
- **MemoryArena** finds that systems near saturation on long-context recall can still perform poorly on interdependent multi-session agent-environment tasks. This is why MACE needs decision utility, not only recall.
- **StateMemBench** contains 234 multi-session state-tracking scenarios. Its paper reports same-backbone current-state accuracy improvements from 0.205 to 0.363 on DeepSeek-V4-Flash and from 0.149 to 0.233 on Qwen-3.5-9B; these are setup-specific results, not a generic multiplier.
- **TimlMemBench** currently describes 208 reviewed bilingual samples, 1,566 sessions, 1,715 questions and 1,821 governance/activation cases. Its raw-native track is especially relevant because it tests ingestion and update behavior rather than giving every system an identical prebuilt memory pool.
- **FAMA** explicitly measures the cost of relying on obsolete or invalidated memories. **MCB** provides a small reuse-time decision study; its authors report limitations in corrected significance and naturalistic extension size, so treat it as a diagnostic, not a decisive league table.
- **CL-Bench** contains 1,899 tasks and 31,607 verification rubrics across four context-learning categories. Its central lesson for MACE is methodological: memory must outperform a strong naive in-context-learning control to show genuine experience gain.
- **Agentic Abstention** evaluates 28K+ instructions across WebShop, Terminal-Bench and QA. One reported held-out WebShop configuration improves timely abstention from 26.7% to 57.4%; this is specific to that model/scaffold/dataset.

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

1. Fix evaluator abstention semantics before interpreting any knockout or regret result.
2. Introduce a canonical EvidenceDependencyIndex over event IDs, verified root digests, source-family identity and derived-view lineage.
3. Add a frozen intervention manifest with event/root/family/coalition and matched-random knockout plans.
4. Add a candidate-input evidence-set digest that changes for every candidate-visible intervention while remaining independent of evaluator truth.
5. Add evaluator-only paired score records for regret, decision flips, abstention, hard-gate failure and support closure.
6. Add exact small-set coalition enumeration and deterministic sampled coalitions for larger sets.
7. Add a schema and regression tests for same-root aliases, common-cause attacks, complementary support, misleading support, supersession and source correction between retrieval and use.
8. Reproduce conventional KV, BM25, dense and hybrid retrieval under matched all-in budgets; add a byte/token-matched semantic placebo.
9. Run Rust formatting, compilation, focused tests, the intervention runner and schema validation on an exact commit head.
10. Move evaluator truth to independent custody and run held-out external benchmarks through pinned adapters.

## 10. Current implementation and validation boundary

This follow-on branch starts from the inspected child-PR head and adds the first evaluator hardening change: abstention is now scored as an explicit zero-utility outside option in the current synthetic benchmark, instead of being silently mapped to Action B. A regression test covers this distinction.

This is a synthetic evaluator convention only. It must be configurable per scenario before transfer to a domain where non-action has a nonzero cost or benefit.

The evidence-knockout engine, dependency-closure index, coalition sampler and separate intervention schema are specified here but have not yet been implemented. The existing typed HDC code is structured typed-content retrieval, not natural-language understanding or a learned causal model. The seven-arm runner and this branch's new regression test have not been executed in this environment: Cargo, rustc and rustfmt are unavailable. Hosted checks that are skipped are not passes.

No retrieval gain, decision-utility improvement, benchmark score, causal sufficiency result, production readiness or SOTA superiority is claimed.

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


---

## 11. MACE Evidence Dependency Knockout Audit v0.1 — first executable tranche (2026-11-22)

**Implementation:** research/broca-mace-transfer-bench-v0-1/evaluator/src/bin/mace_evidence_dependency_audit.rs  
**Record contract:** data/schemas/mace_evidence_dependency_audit_v0_1.schema.json  
**Branch:** research/mace-evidence-dependency-knockouts-v0-1

This follow-on turns part of the v0.1 intervention specification into an executable diagnostic. It does not modify the production allocator, the MACE causal policy or the typed HDC retrieval arm.

### Implemented

- A run-scoped MaceAuditDependencyIndex rejects duplicate event identities, missing predecessors, repeated/self references and cyclic dependency graphs.
- Event closure follows transitive derived descendants; source-root closure unifies records that share a root even when their family labels differ; source-family closure removes every member of that family and its descendants; lifecycle closure follows supersession links in both directions and then removes dependent descendants; two-root coalitions union root closures.
- A deterministic intervention planner generates full-evidence reference, event, root, family, lifecycle and bounded two-root-coalition knockouts.
- Matched random controls preserve the removed (action, in-memory payload-size) histogram, exclude the target's removed root/family identities and are retained only when their distinct root/family counts match. If an exact match cannot be constructed, the control is omitted rather than weakened.
- Each intervention is passed through the unchanged CausalQualified policy. Evaluator-only records score utility, absolute regret, regret added versus full evidence, decision flips, invalid evidence use, and remaining event/root/family counts.
- Candidate decisions and evaluator scores are written to separate JSONL files. Candidate records contain the digest of the actual candidate-visible evidence set and task DecisionNeed; they omit intervention kind/target and scenario-family labels. Evaluator rows carry the same candidate-input digest for offline pairing.
- The JSON Schema defines the manifest, candidate, evaluator, rejected-block and summary records.
- Authored unit tests cover transitive closure, same-root aliases with different family labels, family closures, supersession closure, preservation of a replacement under ordinary event knockout, root coalitions, unknown roots, duplicate IDs, missing links, cycles, matched-control determinism/identity balance, candidate-input digest changes, and candidate/evaluator separation (17 tests total).

### Architecture ownership and reuse

This index is deliberately a private, bounded evaluator projection, not a new general-purpose evidence graph, persistent store, universal independence flag or trust scalar. Compose with the established owners:

- [#10410](https://github.com/Luminous-Dynamics/symthaea/issues/10410) for provider-lineage/common-mode provenance profiles;
- [#4918](https://github.com/Luminous-Dynamics/symthaea/issues/4918) for claim-relative independence and shared failure-domain semantics;
- [#6283](https://github.com/Luminous-Dynamics/symthaea/issues/6283) for measurement validity and benchmark evidence-graph principles;
- [#6430](https://github.com/Luminous-Dynamics/symthaea/issues/6430) for Mycelix DKG reuse and deterministic bounded DAG projections.

The existing ingress EvidenceReceiptBundle already preserves lineage_digest, source family/root, acquisition mode, upstream source ID, source artifact digest, transformation digest, issuer and receipt digests. Those receipts are the right evidence base to extend. The current Experience projection and receipt bundle do not expose explicit event-level derived_from/supersedes references, so this binary must not infer them. The future MACE adapter should consume a verifier-issued sidecar projected from the existing lineage/DKG owners instead of introducing a second canonical graph.

### Critical lineage limitation

The current Experience projection contains event identity, source-family identity and source-root identity, but does not contain derived_from or supersedes references. The generated-world adapter therefore supplies zero dependency/lifecycle edges. On the generated data, event closure is a singleton and lifecycle-closure rows are not produced. Explicit synthetic graph edges are exercised by unit tests only.

This is intentional: the audit must not invent causal ancestry from shared roots or event order. To turn event/lifecycle closures into benchmark interventions, the next integration must export verified derived-view, correction and supersession lineage from receipt-preserving ingress/consolidation records into a candidate-visible dependency sidecar, and bind that sidecar into the candidate-input digest. A missing or unverifiable edge must not be treated as evidence of independence.

The root/family relationships exercised by the current generated fixtures are still synthetic. SourceRootDigest values are fixture identities, not independent Mycelix attestations. payload_bytes is size_of<Experience>, an in-memory lower bound; this first control does not match serialized sidecars, evidence receipts or token cost. A source-count match is not semantic exchangeability.

### Run and validate

~~~bash
MACE_SOURCE_REVISION="$(git rev-parse HEAD)" \
cargo run --release \
  --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml \
  -p broca-mace-evaluator \
  --bin mace_evidence_dependency_audit \
  -- 64 /tmp/mace-dependency-candidate.jsonl \
     /tmp/mace-dependency-evaluator.jsonl \
     /tmp/mace-dependency-summary.jsonl

cargo test \
  --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml \
  -p broca-mace-evaluator \
  --bin mace_evidence_dependency_audit
~~~

The candidate/evaluator JSONL streams are deliberately separate, but the evaluator source and labels remain public and co-located in this research workspace. This is not independent evaluator custody. The summary is descriptive and does not emit confirmatory confidence intervals; a later qualification must freeze a primary intervention contrast and cluster-level inference plan before opening held-out labels.

### Verification and claim boundary

At authoring time, no cargo, rustc or rustfmt executable is available in the execution environment. The Rust tests are authored but not executed; no compile, formatting, runner or CI pass is claimed. The JSON Schema text was parsed as JSON at authoring time, but no external JSON-Schema validator has been run. No utility gain, causal sufficiency result, external benchmark score, physical intervention authority or SOTA advantage is claimed.

The immediate next integration is a verifier-issued dependency sidecar from Mycelix/Symtropy/ingress, followed by schema validation, exact-head Cargo execution, hidden evaluator custody, and matched semantic/byte/token baselines.
