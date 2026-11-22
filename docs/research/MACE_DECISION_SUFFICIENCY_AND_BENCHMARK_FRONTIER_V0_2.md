# MACE Decision Sufficiency and Benchmark Frontier v0.2

**Programme:** Broca / Symthaea MACE  
**Snapshot:** 2026-11-22  
**Status:** Research definition, benchmark strategy and proposed qualification targets. This document reports no measured Symthaea score or SOTA result.

## 1. The field-level thesis

The memory problem should be reframed from:

> How much of the past can the system retrieve?

to:

> What is the smallest admissible representation of the past that preserves the information required to make a future decision correctly, and can the system prove that the representation remains applicable at the moment of use?

Call this **decision sufficiency**.

A memory is not successful merely because it recalls a true fact. It succeeds when it supports a better decision than the strongest matched alternative, with the necessary evidence, under the real resource and governance constraints.

This makes memory a task-conditioned information-compression problem, a causal intervention problem, and a freshness/authority problem simultaneously.

## 2. Formalizing decision sufficiency

Let:

- H be the evidence-complete interaction history available at the decision boundary;
- phi_B(H) be a memory representation produced under a declared all-in budget B;
- D be a preregistered decision family and target population/regime;
- pi_mem be the candidate policy that receives the compressed memory;
- pi_ref be a matched evidence-complete reference policy;
- L(a,y) be the declared loss of action a under evaluator outcome y.

Define the decision-sufficiency regret:

Δ_DS(B,D) = E[L(pi_mem(phi_B(H)), y) - L(pi_ref(H), y)].

The candidate is epsilon-decision-sufficient for D at budget B only when the preregistered upper confidence bound on Δ_DS is at most epsilon, and every hard integrity/authority/freshness gate passes.

This is deliberately conditional on a decision family. There is no reason to assume one compact memory is sufficient for every possible future task.

### 2.1 Use two references, not one

- **Evidence-complete reference:** the same frozen candidate/backbone sees the authorized, evidence-complete history. This approximates the decision information ceiling for that model and task.
- **Strongest practical memory reference:** the best reproduced conventional memory system (for example, full-context within budget, BM25/dense/hybrid retrieval, structured KV or a competitive memory architecture) under matched all-in costs.

The evidence-complete reference is a diagnostic ceiling, not a fair efficiency baseline when its context cost is much larger. A memory method should report both gaps.

### 2.2 The budget is a vector

A representation is not efficient if it hides the evidence sidecar or background work.

Record at least:
- retained source/evidence bytes, vector/index/codebook bytes, metadata, journals and tombstones;
- context tokens and model calls;
- verification and re-observation calls;
- build, update, correction, compaction and replay work;
- CPU/GPU time, p50/p95/p99 latency, allocations and concurrency behavior;
- monetary cost;
- privacy exposure/retention obligations;
- decision deadline and expected failure loss.

Hard authorization, privacy, scope, provenance and freshness constraints are feasibility gates, not terms that can be compensated for by utility.

## 3. Decision capsule as a sufficient-statistic candidate

A Decision Capsule is a typed, versioned candidate representation of a reusable decision relationship. It must preserve links to original evidence and cannot replace the source record.

A capsule's semantic content can include:
- decision objective and action alternatives;
- population/regime/preconditions/exclusions;
- action-to-outcome relationship and uncertainty;
- causal identification method and canonical estimand descriptor;
- observation horizon, delayed outcomes and common-cause limitations;
- epistemic status, conflicting evidence and explicit unknowns;
- source/revision/receipt references;
- bitemporal validity and supersession/retraction history;
- authority/consent/audience/purpose scope;
- reconstructibility class and resource profile.

The capsule can compress the decision representation without compressing away the proof. A screenshot, action trace, source record or signature remains an exact external evidence reference. Captions are retrieval indexes, not replacement authority.

## 4. Representation selection: persist, locator, snapshot, discard

The system should choose a representation for each candidate memory item:

1. **Persist knowledge:** for non-reconstructible procedures, causal relationships and qualified preferences.
2. **Store locator:** for mutable state that can be re-observed from a currently authorized source.
3. **Cache snapshot:** for state whose observation cost justifies a bounded, source-verified cache.
4. **Discard:** when retention or re-observation has non-positive expected decision value, or no admissible representation fits the hard constraints.

This representation factor must be tested on both mutable-state tasks and non-reconstructible procedure tasks. Otherwise the benchmark structurally favors one policy.

A locator is successful only when a source returns a canonical current-state receipt. A snapshot expiring at the moment of consumption is not valid. A retrieval-time lifecycle flag cannot override a later correction.

## 5. Query-conditioned typed HDC: what is now prototyped

The follow-on branch research/mace-typed-content-hdc-v0-1 adds TypedContentIndexV1 to the existing MACE HDC retrieval crate.

It encodes the present candidate-visible Experience structure:
- regime, objective and action;
- observed utility and measurement-error bound;
- causal-identification method and estimand digests;
- propensity and standardized-balance descriptors;
- intervention, pre-registration, reconciliation, trust, retraction, confounder-control and delayed-horizon flags.

Categorical features use deterministic role/value binding. Numeric values use deterministic random Fourier features for a one-dimensional Gaussian RBF kernel. The numeric bandwidths, dimension and seed are bound into a profile digest.

A SemanticExperienceQueryV1 uses the same vocabulary and can include optional numeric targets, method/estimand cues and required evidence-quality flags.

**Important scope boundary:** this is structured content retrieval over a typed evidence object. It is not natural-language understanding, and it does not prove that a vector has learned causal semantics. Event IDs, source-family/root identity, signatures, tenant/authority information and canonical receipt bytes remain exact sidecar data; none is approximated by a vector.

The current MACE policy Query exposes only regime/objective. The paired runner now includes typed-content HDC as a seventh arm using this candidate-visible scope/action plus the policy's categorical qualification profile (RCT method and explicit evidence-quality flags). Numeric overlap/balance limits remain exact post-retrieval MACE gates, not RBF point targets. A richer task-defined outcome/horizon/estimand profile still needs the candidate-visible DecisionNeed in #13222. Never populate target utility, causal estimand, hidden state or other query fields from evaluator truth.

### 5.1 Required metamorphic qualification

| Intervention | Expected result |
|---|---|
| Permute event IDs while preserving content | Content-based ranking remains invariant, except exact ties where the declared ID tie-break applies |
| Change a query-relevant candidate field but preserve its ID | Ranking changes when the query asks about that field |
| Change query action/regime/objective | Candidate ranking responds to the changed typed requirement |
| Move a numeric value closer to the query target | Expected similarity rises under the declared RBF bandwidth |
| Change dimension, seed, bandwidth or encoder schema | Profile digest changes |
| Mutate a canonical record after index construction | Resolver rejects a mismatched exact view digest |
| Retract/untrust/wrong-scope an otherwise similar item | Similarity may still retrieve it, but unchanged MACE gates must reject its use |

The last row is intentional: a candidate index is allowed to recall a dangerous record. The evidence policy is responsible for refusing it. Retrieval score is not an integrity verdict.

### 5.2 Next content-encoder improvement

The current candidate Experience schema is structured but contains no natural-language task description, visual payload, action sequence or explicit applicability text. Do not invent these fields. The next adapter should accept a separately typed, candidate-visible DecisionNeed and a versioned ExperienceView built from real source data, with exact evidence pointers held outside the vector.

If/when event streams are added, compare static typed HDC, zero-input LTC and input-driven LTC with the same canonical event-time schedule, irregular gaps, corrections and retractions. Do not call a zero-input state transition a temporal memory of an event sequence.

## 6. Causal evaluation: memory as an intervention

Keep distinct paired contrasts:

1. **Memory value:** candidate memory vs true empty memory.
2. **Content specificity:** relevant evidence vs genuinely irrelevant same-scope, token/byte-matched placebo.
3. **Substrate value:** typed HDC vs event-ID HDC and strongest reproduced conventional retrieval.
4. **Temporal contribution:** event-driven LTC vs static HDC and zero-input LTC.
5. **Lifecycle value:** post-correction vs pre-correction snapshot with current canonical revalidation.
6. **Representation value:** persist vs locator vs snapshot vs discard.
7. **Policy value:** cost-sensitive use/verify/clarify/abstain/probe vs fixed-threshold/prompt-only policy.

Within each world-cluster pair, freeze task, initial candidate state, actual observation, model/backbone, permissions, evaluator, deadline and all non-memory resources. Arm order can be randomized within pair blocks, but candidates must not learn from previous arms or evaluator feedback.

Bootstrap world-cluster means, not query rows. Report family/regime subgroup results, invalid/missing blocks, attrition and all hard-gate failures. Do not hide harmful memory dependence inside average utility.

## 7. External benchmark portfolio and current industry anchors

No single benchmark measures all these capabilities. Preserve official splits, evaluators and scores separately.

| Benchmark | What it tests | MACE use |
|---|---|---|
| Agent Memory Challenge Cycle 2 / future eligible cycle | Standard Add/Search capability in textual, coding and multimodal tracks | Official hosted Add/Search qualification and capability profile |
| LoCoMo / LongMemEval | Long-conversation recall, temporal questions and long-context QA | Retrieval/control lane; match exact model/judge/token accounting |
| MemoryAgentBench | Incremental retrieval, test-time learning, long-range understanding and selective forgetting | Multi-turn skill-acquisition and forgetting controls |
| MemoryArena | Interdependent multi-session Memory-Agent-Environment tasks | Downstream task use rather than recall-only scoring |
| AMA-Bench | Long agent trajectories, causal/objective information and tool-generated state | Causal/procedural transfer lane |
| WorldMemArena | Multimodal write, maintain, retrieve and use lifecycle | Preserve original frames, action traces and feedback; stage-wise fault localization |
| CL-Bench | Experience gain across sequential tasks | Compare against strong naive in-context learning |
| MCB (Remember, Verify, or Ask?) | Persist/use locally/verify/clarify decisions | Score action label and actual tool choice, including evidence-flip controls |
| Agentic Abstention | Timely stopping in interactive tasks | Measure excess actions after infeasibility becomes knowable |
| TimlMemBench / FAMA / StateMemBench | Governance, revocation, obsolete-memory penalties and current-vs-superseded state | Lifecycle and historical/current-state correctness |

### 7.1 Official AML Cycle 2 snapshot (published 2026-11-19)

| Track / division | Leader | Overall |
|---|---|---:|
| Textual / open methods | Resurface | 61.22/100 |
| Coding / open methods | InsightMem | 63.33% resolved |
| Multimodal / open methods | Khipu | 43.26/100 |
| Textual / commercial | MemoraX | 69.26/100 |
| Coding / commercial | MemoraX | 68.00% resolved |
| Multimodal / commercial | MemoraX | 53.68/100 |

These are separate track/division metrics. They must not be averaged together. AML participants must operate public Add/Search endpoints and pass a smoke test before formal evaluation; a repository-only runner is not a submission.

**Project stretch targets:** +5 absolute points over each relevant open-method Cycle 2 leader: Textual 66.22, Coding 68.33% resolved and Multimodal 48.26. These are goals, not predictions or achieved scores. Keep commercial-reference ambition separate.

### 7.2 Other strong comparison anchors

- **Microsoft Research Memora** reports 86.3% on LoCoMo and 87.4% on LongMemEval in its own ICML 2026 setup, with up to 98% fewer context tokens than full-context inference. Reproduce its backbone, judge, memory interface, retrieval settings and token accounting before any head-to-head claim.
- **AMA-Agent** reports 57.22% average accuracy on AMA-Bench, +11.16 percentage points over the strongest baseline in that paper's protocol. Its causality graph and tool-augmented retrieval are particularly relevant baselines.
- **WorldMemArena** comprises 400 multi-session multimodal tasks and finds that stronger writing/storage does not necessarily yield better task performance; it makes write, maintain, retrieve and use failures observable.
- **MemoryArena** couples memory with later environment actions rather than testing recall and acting in isolation.
- **MemoryAgentBench** covers four competencies: accurate retrieval, test-time learning, long-range understanding and selective forgetting.
- **StateMemBench** uses 234 multi-session state-tracking scenarios. Its paper reports StateMem improvements from 0.205 to 0.363 on DeepSeek-V4-Flash and 0.149 to 0.233 on Qwen-3.5-9B versus its strongest reported baselines; these are backbone/protocol-specific. This is a direct reason to treat supersession/current-state tracking as a first-class benchmark axis.
- **TimlMemBench** contains 208 reviewed bilingual samples, 1,566 sessions, 1,715 questions and 1,821 governance/activation cases. Its raw-native evaluation requires systems to ingest raw sessions, exposing lifecycle/scope/provenance failures that a prebuilt canonical memory pool can hide.
- **FAMA** (from the ACL 2026 Memora benchmark) explicitly penalizes reliance on obsolete or invalidated memories. It complements MACE's known-retraction and current-at-use controls.
- **CL-Bench** comprises 500 expert-created contexts, 1,899 tasks and 31,607 verification rubrics across six domains. Its evaluation finds that naive in-context learning can outperform dedicated memory systems; one public leaderboard snapshot reports a 27.9% top overall solve rate. MACE must therefore retain a strong ICL baseline and measure experience gain, not just memory-agent task accuracy.
- **MCB** uses 140 primary scenarios, 70 evidence-flip contrasts and a 36-item dialogue-derived extension. The paper reports verification recall exceeding clarification recall in all 12 label conditions, but only three contrasts surviving an exploratory Holm correction; a category oracle also obtains 0.800 accuracy. Treat this as a useful but limited diagnostic, not an enormous definitive leaderboard.
- **Agentic Abstention** studies 28K+ instructions across WebShop, Terminal-Bench and QA. Its NeurIPS 2026 site reports timely-abstention improvement from 26.7% to 57.4% for one CONVOLVE/Llama-3.3-70B WebShop held-out configuration. This is a setup-specific result.

### 7.3 MACE-specific promotion targets (proposed)

| Dimension | Proposed gate |
|---|---|
| Known-control invalid/stale/wrong-scope/wrong-estimand evidence use | Zero |
| Decision-support IDs unresolved or ambiguous | Zero |
| Paired downstream memory-value contrast | 95% cluster-bootstrap lower bound above zero |
| Decision-sufficiency regret gap vs evidence-complete reference | Upper confidence bound within a preregistered epsilon |
| HDC typed-content superiority | Positive matched contrast vs strongest reproduced conventional retrieval at equal all-in budgets |
| Worst-family success regression | No worse than 2 absolute percentage points |
| Memory efficiency | At least 25% lower all-in memory/context cost than full-history baseline with no more than 1 pp task-success loss on a preregistered lane |
| Timely abstention | At least 25% fewer excess post-infeasibility tool calls vs same-backbone prompt-only baseline, without >1 pp task-success regression |

The final two are aspirational project decision rules, not industry standards. Freeze the primary contrast and margins before held-out evaluation.

## 8. What would count as a real paradigm-level result?

The most distinctive result would not be “HDC retrieves more events.” It would be:

1. A compact capsule preserves the decision-relevant relation and exact evidence closure.
2. The system chooses persistence, locator, snapshot or discard based on calibrated future decision value and hard feasibility.
3. At use, the source/revision is revalidated at the actual bitemporal knowledge boundary.
4. The decision policy uses, verifies, clarifies, probes or abstains at the right time.
5. A paired intervention demonstrates better downstream utility and lower regret under equal all-in budgets.
6. The same design holds on independently evaluated text, coding, multimodal and interdependent agent-environment tasks.

This is falsifiable. If typed HDC cannot beat a simple structured KV/BM25/dense/hybrid baseline under matched budgets, do not use HDC. If storing everything beats selective representation under real total costs, revise the representation policy. If naive ICL wins, that is the correct benchmark result.

## 9. Immediate implementation sequence

1. Compile and test TypedContentIndexV1 and the seven-arm paired runner in the Rust workspace; run formatting and record exact-head receipts.
2. Validate ID-permutation, content-swap, query-swap, numeric-locality, profile-digest and decision-trace schema controls.
3. Add the richer candidate-visible DecisionNeed from an actual environment/task observation; do not derive it from evaluator labels.
4. Add an embedding/token/byte-matched relevance-negative placebo and competitive KV, BM25, dense and hybrid arms.
5. Add source correction between retrieval and consumption; query current canonical lifecycle on use.
6. Move evaluator/labels outside the candidate repository and run an independently sealed split.
7. Qualify through official AML Add/Search, MemoryArena, AMA-Bench, WorldMemArena and CL-Bench adapters separately.

## 10. Claim and validation boundary

The typed-content module is source-authored on a follow-on public branch. Its tests are authored but have not been executed. Cargo compilation, rustfmt, runtime RFF behavior, JSON receipt validation and downstream MACE utility evaluation remain outstanding. The current local environment has no cargo, rustc or rustfmt.

The typed index:
- returns candidate IDs and diagnostics only;
- uses an exact view digest as a consistency check, not source authentication;
- does not evaluate signatures or authority;
- does not independently prove causal identification;
- does not perform current-state source/revision verification;
- does not implement LTC event-stream evolution;
- does not provide a natural-language encoder;
- has no measured retrieval or decision-performance result.

No official benchmark score, production readiness or SOTA superiority is claimed.

## References

- Agent Memory Challenge Cycle 2 official results (2026-11-19): https://agentmemoryleaderboard.ai/results/cycle-2/
- AML evaluation/API flow: https://agentmemoryleaderboard.ai/
- Microsoft Research Memora (2026): https://www.microsoft.com/en-us/research/blog/memora-a-harmonic-memory-representation-balancing-abstraction-and-specificity/
- AMA-Bench (ICML 2026): https://proceedings.mlr.press/v306/zhao26bs.html
- MemoryArena (ICML 2026): https://proceedings.mlr.press/v306/he26am.html
- WorldMemArena (2026): https://arxiv.org/abs/2605.29341
- MemoryAgentBench (2025): https://arxiv.org/abs/2507.05257
- StateMemBench (2026): https://arxiv.org/abs/2608.19652
- TimlMemBench (Findings EMNLP 2026): https://aclanthology.org/2026.findings-emnlp.507/
- FAMA / From Recall to Forgetting (Findings ACL 2026): https://aclanthology.org/2026.findings-acl.1337/
- CL-Bench (2026): https://arxiv.org/abs/2606.05661
- Memory–Clarification Boundary (2026): https://aclanthology.org/2026.nora-1.2/
- Agentic Abstention (NeurIPS 2026): https://lhannnn.github.io/agentic-abstention/
