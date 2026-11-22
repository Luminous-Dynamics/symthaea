# MACE Paired Memory-to-Decision Runner v0.1

**Status:** First executable research tranche authored on the public MACE branch. It has not been compiled or run in the current environment. All emitted scores are explicitly diagnostic_only.

**Parent:** [#13141 — paired memory-to-decision intervention runner](https://github.com/Luminous-Dynamics/symthaea/issues/13141)

## 1. Purpose

This runner turns the existing generated MACE worlds into paired memory-access interventions. For each generated world-cluster/task block, it holds the task identity and current candidate input constant while changing the available memory/retrieval arm.

It reuses the existing generated-world source and broca-mace-policy qualification gates. It does not create a second evaluator, policy, source verifier, evidence registry, or memory store.

## 2. Current arms

| Arm | Current implementation | Interpretation |
|---|---|---|
| no_memory | Empty prior-memory slice; existing no-memory fallback | True empty-memory intervention, not the conservative no_memory label applied to a populated store |
| exact_scope_scan | All candidate-visible events matching the exact query scope/action | Candidate-visible structured ceiling, not an evaluator oracle |
| conventional_retrieval | Pinned deterministic event-ID/key-value ordering, top-2 per action | Reproducible baseline only; not yet BM25, dense, or hybrid retrieval |
| static_hdc | Existing HDC candidate index, top-2 per action | Tests the existing event-code/bucket index |
| hdc_ltc | Same index with the existing zero-input LTC evolution profile | Tests temporal state evolution in the existing index; does not establish query-conditioned semantic retrieval |
| typed_content_hdc | Typed Experience content index with a structured query for regime/objective/action and MACE qualification attributes | Query-conditioned structured-content HDC over the current Experience schema; not natural-language retrieval or input-driven LTC |
| semantic_placebo | Candidate-visible episode from a different task in the same generated world cluster and matching query scope | Structural sibling-task placebo only; it is not yet embedding-matched or exact token/byte matched |

The semantic_placebo arm is deliberately named to align with the v0.1 record schema, but the present fixture has no natural-language episode payload or semantic embedding with which to construct a genuine semantic placebo. Do not treat it as satisfying the final placebo design.

## 3. Evidence and leakage boundaries

- Candidate records contain decisions, retrieved IDs, decision-support IDs, candidate/input digests and resource observations. They do not contain evaluator utilities, optimal actions, causal labels or task-success labels.
- Evaluator outcomes are written to a separate JSONL stream.
- Event-ID HDC and typed-content HDC hits retain their index-time experience_view_digest. Before policy use, each hit is resolved against the canonical candidate-visible Experience record and its digest must match.
- Typed-content HDC uses the candidate-visible scope/action query plus MACE policy-defined method and categorical qualification cues. Numeric propensity/balance constraints remain exact post-retrieval MACE gates; they are not misrepresented as RBF point targets. The typed encoder profile digest is included in the candidate trace and profile field.
- Duplicate or unresolved event identities fail closed. A rejected lookup is recorded as an abstention; it is not counted as evidence use.
- The existing MACE is_qualified policy remains the authority for whether a resolved experience can support a decision. Similarity does not override evidence qualification.
- The candidate and evaluator are still compiled into one public executable and the fixture generator is public. The current runner is therefore **not an independent or hidden-evaluator challenge**.
- The schema's hidden_evaluator_digest compatibility field contains a digest of the public fixture generator. It does not denote secrecy, custody, or independent evaluation.
- outcome_receipt_sha256 is a deterministic hash of the synthetic evaluator outcome record, not a signed Mycelix or Symtropy receipt.

The input/state digests are deterministic identifiers for the synthetic fixture block. They are not hashes of a complete real-agent prompt, multimodal observation, or serialized production agent state.

### Candidate-visible DecisionNeed

Every generated task now carries a `DecisionNeedV1` that is separate from evaluator `Truth`. It binds:
- exact target scope and declared action alternatives;
- a complete semantic estimand with population, treatment/comparator, outcome definition/unit and follow-up horizon;
- the required identification method;
- a digest of the current MACE transfer-qualification profile (overlap, balance, interference and lifecycle requirements).

The generator creates this contract independently of hidden utility and optimal-action fields. The reduced candidate projection contains the need but not `Truth`. Candidate records carry `decision_need_sha256`, and the candidate-input digest also binds it. A negative-control test mutates hidden utilities/optimal action while asserting that the DecisionNeed and candidate-input digests remain unchanged.

The versioned qualification-profile digest now covers the actual hard gates: randomized design; known design/estimand receipts; complete descriptor and descriptor/digest equality; propensity overlap [500, 9,500] basis points; standardized mean difference ≤100,000 micros; known source family/root; unique event IDs; at least two independent source roots and two source families per source action group; at least two source events per action; at least one target-regime witness per action for invariant transport; both action alternatives; matched estimands across actions and (for transport) across regimes; intervention, pre-registration, reconciled outcome, trust/non-retraction, interference/confounder control, delayed-horizon completion; normalized utility/error in [0,1]. The digest is a configuration identity, not an issuer signature.

The typed HDC adapter constructs one query per declared action alternative via `SemanticExperienceQueryV1::from_decision_need`. The exact estimand digest is a candidate-ranking cue; canonical semantic descriptor equality, causal qualification and source validity are still checked by MACE after candidate resolution. The profile digest describes the authoritative MACE thresholds/flags; it does not replace their execution. This is currently a generated synthetic task contract, not yet a general task-language/observation parser.

## 4. Output streams

The runner writes three JSONL files:

1. mace_candidate_decisions.jsonl — one run manifest followed by candidate decisions.
2. mace_evaluator_scores.jsonl — evaluator-only synthetic outcomes.
3. mace_paired_summary.jsonl — paired contrast summary.

The records follow data/schemas/mace_memory_decision_intervention_v0_1.schema.json.

The summary contains seven contrasts:

- memory_vs_none: hdc_ltc versus no_memory.
- content_vs_placebo: hdc_ltc versus semantic_placebo.
- hdc_ltc_vs_conventional: hdc_ltc versus the pinned key-value baseline.
- hdc_ltc_vs_static_hdc: isolates the LTC-profile change from the static HDC profile.
- typed_content_hdc_vs_conventional: typed-content HDC versus the pinned key-value baseline.
- typed_content_hdc_vs_static_hdc: typed-content HDC versus the original event-ID HDC index.
- typed_content_hdc_vs_exact_scope_scan: typed-content top-k versus all candidate-visible exact-scope evidence. This is the decision-sufficiency gap against an evidence-complete reference, not an equal-cost practical baseline; interpret utility/regret loss together with saved bytes and latency.

corrected_vs_stale is intentionally omitted. The runner does not yet replay a real correction/retraction between retrieval and consumption. That requires the existing canonical lifecycle/source-revalidation owners and must not be faked with a cached flag.

## 5. Statistical procedure

The generated fixture currently produces 21 scenario families per world cluster. The default run uses 256 world clusters (5,376 pair blocks; 37,632 candidate/evaluator arm rows each across seven arms).

Arm execution order is pseudo-randomized within each pair block using a deterministic hash of the world-cluster/task identity. The permutation is replayable from the source revision and task identity; all arms still receive the same projected candidate task and no arm learns from another arm.

For each contrast:

1. Compute treatment-minus-baseline utility and regret within each pair block.
2. Average paired deltas within each world cluster.
3. Resample world-cluster means (not query rows) for 2,000 deterministic percentile-bootstrap replicates.
4. Report the two-sided 95% percentile interval, harmful-dependence rate, worst family task-success delta, p95 latency delta, and average resource-payload delta.

This is an initial bootstrap implementation, not a complete preregistered inferential pipeline. It does not yet implement multiplicity correction, missing-pair rejection diagnostics, sensitivity analysis, effective sample-size reporting, or hidden held-out splits. Those remain qualification requirements before a scientific superiority claim.

## 6. Resource accounting boundary

- HDC index_bytes is the existing implementation's vector/ID payload lower bound; it excludes allocator, tree, and runtime overhead.
- The key-value baseline's index bytes count stored u32 identifiers only; this also excludes container overhead.
- evidence_sidecar_bytes estimates the in-memory Rust Experience struct footprint. It is not a serialized database footprint.
- latency_ns measures the candidate arm's local elapsed wall time including its local index construction/retrieval path; it is not a controlled hardware benchmark.
- input_tokens and tool_calls are zero because this first tranche invokes deterministic Rust policies, not an LLM or external tool.
- No physical energy or neuromorphic efficiency claim is possible from these payload estimates or timing values.

Do not compare these estimates to vendor pricing, GPU-hours, joules, or production memory footprints without a matched measurement boundary.

## 7. Run

The runner requires an exact source commit supplied by the caller so the run manifest does not invent its own revision:

~~~~bash
MACE_SOURCE_REVISION="$(git rev-parse HEAD)" \
cargo run --release \
  --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml \
  -p broca-mace-evaluator \
  --bin mace_paired_intervention \
  -- 256 /tmp/mace-candidate-decisions.jsonl \
     /tmp/mace-evaluator-scores.jsonl \
     /tmp/mace-paired-summary.jsonl
~~~~

For a quick deterministic smoke-sized fixture, use 1 or 4 world clusters instead of 256. A small run is useful for wiring checks only and must not be used for an inferential claim.

The evaluator crate promotes its HDC retrieval and SHA-256 dependencies to normal dependencies because the new binary calls those packages directly. The generated stress source exposes only the scenario fields and generator required by this sibling binary.

## 8. Known limitations and next gates

1. Compile and run the runner; validate every output record against the JSON Schema, including the required decision_need_sha256 field.
2. Replace the structural sibling-task placebo with a genuinely irrelevant, same-scope, count/token/byte-matched placebo using task content that is safe for the same tenant.
3. Extend the current DecisionNeed contract only through task/environment-declared attributes and explicit versioned policy profiles; never derive target estimands/outcomes/horizons from evaluator truth.
4. Add a competitive conventional retrieval ladder: deterministic KV/episodic, BM25, dense and hybrid under equal total memory, context, verification, latency and compute budgets.
5. Keep exact-scope full scan as an upper-bound diagnostic, not a practical baseline.
6. Add full-history and salience-only arms only when their input and memory budgets are precisely matched.
7. Add evidence knockout and placebo substitution on a preregistered audit sample.
8. Compose post-correction and pre-correction snapshots with canonical source/revision revalidation at decision consumption; do not create another timestamp or lifecycle authority.
9. Add explicit candidate-side run validity, missing-pair counts, attrition-by-arm, cluster-weighting sensitivity and multiplicity correction.
10. Move hidden labels and evaluator operation to independent custody before any challenge-grade claim.
11. Extend to genuine visual/frame/action evidence and qualify against the official WorldMemArena and MemoryArena/AMA-Bench protocols without changing their scoring rules.

## 9. Claim boundary

A successful local run would establish only that this deterministic harness executes and emits paired diagnostic records on generated worlds. It would not establish:

- a measured HDC-LTC improvement;
- semantic retrieval quality;
- a production locator or fresh-at-use guarantee;
- a live Mycelix/Symtropy cryptographic integration;
- an official AML, MemoryArena, AMA-Bench, WorldMemArena, CL-Bench or LoCoMo score;
- independent evaluator validity or SOTA superiority.

A memory-value claim requires a preregistered held-out paired utility contrast whose cluster-bootstrap lower bound is above zero. An HDC-LTC superiority claim additionally requires a positive, matched comparison against the strongest reproduced conventional retrieval method at equal all-in budgets, with no hard integrity-gate failures.
