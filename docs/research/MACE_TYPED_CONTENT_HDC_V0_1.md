# MACE Typed-Content HDC Retrieval v0.1

**Status:** Source-authored representation prototype. Rust compilation, Rust tests and runtime measurements have not occurred.  
**Parent work:** MACE #13127 / #13141 and the existing causal-memory benchmark programme.  
**Branch:** research/mace-typed-content-hdc-v0-1

## 1. Why a second retrieval arm is needed

The original MACE event-ID HDC index is a legitimate candidate-index experiment, but it does not encode the content of an Experience. Its event codes are seeded from event IDs, its retrieval is pre-bucketed by scope/action, and its LTC profile evolves the bundled bucket vector with zero input.

Typed-content HDC is a separate experimental arm. It asks a narrower, testable question:

**Can an HDC representation built from typed candidate-visible experience fields retrieve experiences that resemble a structured decision query, independently of event identifiers?**

This is not yet natural-language semantic understanding, and it does not establish causal transfer. It is structured content similarity over the current MACE Experience schema.

## 2. Module boundary

The implementation is in:

- research/broca-mace-transfer-bench-v0-1/hdc-retrieval/src/typed_content.rs
- research/broca-mace-transfer-bench-v0-1/hdc-retrieval/src/lib.rs (module export)

Public API:
- TypedContentConfigV1
- SemanticExperienceQueryV1
- TypedContentIndexV1
- TypedContentError

The new index is separate from the event-ID CandidateIndex. It does not change existing retrieval defaults or MACE policy behavior.

## 3. Candidate-visible content encoded

The encoder uses the current Experience fields:

| Feature | Encoding |
|---|---|
| Regime, objective, action | Categorical role/value binding |
| Observed utility | Continuous random Fourier features |
| Measurement-error bound | Continuous random Fourier features |
| Identification method | Categorical role/value binding |
| Estimand digest | Exact digest as a categorical retrieval cue |
| Complete semantic-estimand digest | Exact digest as a categorical retrieval cue, plus completeness flag |
| Propensity min/max | Continuous random Fourier features after basis-point normalization |
| Standardized mean difference | Continuous random Fourier features after declared normalization |
| Interference controlled | Categorical boolean |
| Intervention / pre-registration / reconciliation | Categorical booleans |
| Source trusted / retracted | Categorical booleans |
| Confounders controlled / delayed horizon complete | Categorical booleans |

### 3.1 Exact identity is not vectorized

The following remain outside the lossy vector:
- event ID;
- source-family and upstream-root identity;
- source signature / receipt bytes;
- tenant, principal, capability, audience and purpose;
- canonical lifecycle authority.

Candidate hits retain the event ID and the existing candidate-visible Experience-view digest. The index retains exact Scope and Action keys and filters to exact target scope/action before approximate HDC ranking; wrong-scope/action records cannot crowd out valid top-k candidates. The consumer must resolve the ID against the canonical evidence registry, compare the exact view digest and re-run all MACE predicates. A vector match is never evidence of trust, provenance, causal identification or authority.

The estimand digest is included only as a candidate-ranking cue. Exact digest equality is still checked in the canonical record, and digest equality alone does not prove that the underlying semantic descriptor or receipt is authentic.

## 4. Encoding and numeric kernel

Categorical fields use deterministic role/value binding:

1. Derive a role hypervector from the frozen encoder seed and field tag.
2. Derive the categorical value hypervector from the same field tag and canonical category code.
3. Normalize both vectors to unit norm.
4. Bind role and value, then bundle the typed features.

Continuous fields use deterministic random Fourier features for the one-dimensional Gaussian RBF kernel:

k(x,y) = exp(-(x-y)^2 / (2 sigma^2))

Frequencies are generated from a deterministic normal stream and phases from a deterministic uniform stream. The RBF bandwidth sigma controls the numeric distance scale. The resulting value vector is normalized before role binding, so its contribution is not overwhelmed by unnormalized categorical vectors.

Role vectors and value vectors are unit normalized before binding. The complete candidate and query vectors are bundled and normalized. This makes each declared feature contribute on a comparable vector-norm basis.

The profile digest v1.1 binds the exact scope/action prefilter policy as well as:
- dimension;
- base seed;
- numeric-kernel version;
- utility bandwidth;
- measurement-error bandwidth;
- propensity bandwidth;
- balance bandwidth.

The source revision and the exact profile digest must be included in any experimental receipt. A digest is a configuration identifier, not a signature.

## 5. Query contract

SemanticExperienceQueryV1 contains mandatory:
- target regime;
- target objective;
- target action.

Optional query cues include:
- target observed utility;
- target measurement-error bound;
- target propensity min/max;
- target maximum standardized mean difference;
- identification method;
- exact estimand digest;
- target intervention / pre-registration / outcome-reconciliation / source-trust / retraction / interference-control / confounder-control / delayed-horizon values.

The convenience constructor `SemanticExperienceQueryV1::qualified_transfer` builds the current MACE policy's categorical qualification profile: RCT identification, intervention, pre-registration, reconciled outcome, trusted/non-retracted source, controlled interference/confounders and completed delayed horizon. It does not guess numeric overlap or balance requirements.

Optional numeric targets mean similarity to a declared reference value under the matching RBF bandwidth. They are not threshold predicates. Optional evidence-quality boolean targets are exact-match ranking cues, not admission gates; either true or false can be queried for positive controls and negative controls.

**Current integration boundary:** the paired MACE runner now includes `typed_content_hdc` as a seventh arm. It supplies only candidate-visible scope/action and the fixed categorical qualification profile derived from the MACE policy. This supports a structured evidence-retrieval comparison through the unchanged MACE decision gate, but it does not supply a task-defined target outcome distribution, semantic estimand or target horizon. The richer optional numeric/estimand fields remain unpopulated until #13222 defines a candidate-visible DecisionNeed. Never populate them from Truth/oracle state.

## 6. Retrieval and fail-closed behavior

The index returns:
- candidate event IDs;
- candidate-view digests;
- similarity and rank;
- best similarity and best-vs-second margin;
- profile digest;
- elapsed query time;
- vector/ID/digest/scope/action payload lower-bound bytes (including exact scope/action partition keys; allocator and map/runtime overhead remain excluded).

Regime/objective and action are also retained as exact index partition keys and used as **hard prefilters** before vector ranking. A high-similarity candidate from another regime, objective or action cannot occupy a top-k slot and crowd out an applicable candidate. This is recorded in the typed profile version/digest. The current payload lower bound includes these exact partition keys.

It does not return canonical Experience records or authorization.

Build rejects:
- empty memory;
- duplicate event IDs;
- dimensions outside the supported interval;
- non-finite numeric fields;
- negative measurement-error bounds;
- malformed propensity bounds;
- invalid bandwidth configuration.

Query rejects non-finite numeric targets, negative measurement-error targets and propensity targets outside the valid range.

A candidate consumer must:
1. resolve each ID to exactly one canonical Experience;
2. verify the candidate-view digest against the current record;
3. query canonical source lifecycle and authority state at the decision-use boundary;
4. run unchanged MACE provenance, source-root, causal-identification, estimand, scope, outcome, trust, retraction and delayed-horizon gates;
5. record the exact selected evidence set and decision trace.

An invalid or unresolved ID must fail closed. Similarity cannot override a MACE rejection.

## 7. Authored metamorphic tests

The module now contains ten authored tests for:
- ranking invariance under event-ID permutation when content is preserved;
- ranking changes when utility content changes but event IDs remain fixed;
- changed ranking when the target action/scope query changes;
- wrong-scope/wrong-action high-similarity distractors are removed by exact prefiltering before ranking;
- higher similarity for numeric values near the target than for distant values under the frozen kernel;
- profile digest sensitivity to dimension, seed, bandwidth and exact-filter policy;
- duplicate-ID and non-finite input rejection;
- query-time estimand/evidence-quality cues while preserving candidate-only authority;
- the paired-runner typed arm uses the MACE qualification profile, exact scope/action prefilters and emits a typed retrieval-profile digest;
- high-similarity wrong-scope/wrong-action items cannot displace in-scope candidates.

These tests are authored source, not executed passes. Cargo, rustc and rustfmt are unavailable in the present environment.

## 8. What this prototype proves—and does not prove

If compiled and the tests pass, the immediate claim would be limited to:
- deterministic typed-feature encoding;
- query/content role alignment;
- numeric locality under the configured RBF approximation;
- event-ID independence of content similarity;
- candidate-ID and view-digest return behavior.

It would not establish:
- natural-language semantic retrieval;
- retrieval from images, audio, video or spatial observations;
- temporal learning from event streams;
- HDC-LTC improvement over static typed HDC;
- a measured retrieval/decision utility lift or superiority in the paired runner;
- production source verification or authority;
- an official benchmark result or SOTA superiority.

The current implementation is static typed HDC. Input-driven LTC over irregular observation/event schedules is a later experiment and must reuse canonical time semantics.

## 9. Next engineering gates

1. Execute Cargo tests, rustfmt, JSON Schema validation and the seven-arm runner on an exact recorded source head.
2. Add deterministic retrieval-score receipts and validate their schemas.
3. Connect a candidate-visible DecisionNeed with task-defined outcome/horizon/estimand details; never derive them from evaluator truth.
4. Compare against exact scan, event-ID HDC, deterministic KV, BM25, dense and hybrid retrieval at equal all-in resource budgets.
5. Add a genuine same-scope, relevance-negative placebo matched on context, token and byte budgets.
6. Add source correction/retraction between retrieval and decision consumption; query canonical lifecycle again at use.
7. Move evaluator labels and custody outside candidate-visible code before independent qualification.

## 10. Benchmark interpretation and SOTA strategy

Typed HDC retrieval metrics should be reported separately from downstream MACE utility:

**Retrieval:** Recall@k, MRR, nDCG, relevant-vs-distractor margin, exact event-ID recovery, ID-permutation invariance, content-swap sensitivity, numeric locality, profile replay and candidate abstention.

**Decision:** paired utility and regret, false causal claims, wrong-scope/stale evidence use, harmful memory dependence, abstention/coverage and worst-family results.

**Cost:** total vector + exact sidecar + metadata bytes, index construction, query latency, CPU/GPU and verification costs. Do not report vector payload while excluding the source evidence needed to resolve it.

External qualification stays separate:
- AML Cycle 2 open-method leaders: Textual Resurface 61.22/100, Coding InsightMem 63.33% resolved, Multimodal Khipu 43.26/100.
- Commercial references: MemoraX Textual 69.26/100, Coding 68.00% resolved, Multimodal 53.68/100.
- Microsoft Research Memora reports 86.3% LoCoMo and 87.4% LongMemEval in its own ICML 2026 setup.
- AMA-Agent reports 57.22% average accuracy on AMA-Bench, +11.16 percentage points over its strongest baseline in that paper.
- WorldMemArena contains 400 multi-session multimodal tasks and diagnoses write/maintain/retrieve/use lifecycle failures.

These results are from different protocols and are not comparable to MACE synthetic retrieval or utility metrics. The near-term project targets remain +5 absolute points over the relevant AML open-method Cycle 2 track leaders, but only an official hosted Add/Search evaluation can support that claim.

References:
- https://agentmemoryleaderboard.ai/results/cycle-2/
- https://www.microsoft.com/en-us/research/blog/memora-a-harmonic-memory-representation-balancing-abstraction-and-specificity/
- https://proceedings.mlr.press/v306/zhao26bs.html
- https://arxiv.org/abs/2605.29341
- https://lhannnn.github.io/agentic-abstention/
