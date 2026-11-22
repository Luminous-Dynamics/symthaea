# Broca MACE Transfer Bench v0.1

**Status:** Deterministic synthetic research harness. Not the official MemoryArena or AMA-Bench runner, not a physical-world causal discovery claim, and not a measured SOTA result.

**Programme owner:** [Symthaea #9516 — causal experience transfer across interdependent agent sessions](https://github.com/Luminous-Dynamics/symthaea/issues/9516)

## Research question

Does memory acquired in Session A improve a dependent Session B action, while correctly refusing to transfer when the apparent lesson is confounded, out of scope, retracted, untrusted, unreconciled, or supported only by correlated repetitions?

This v0.1 slice isolates one narrow bridge: **qualified transition evidence → policy-relevant transfer**. It does not implement a general agent, train a causal model, or call an external model API.

## Run

The research workspace contains five separately scoped packages: oracle-blind policy, synthetic evaluator, receipt-gated ingress, proposal-only consolidation, and the provider-neutral AML Add/Search contract core.

```bash
cargo run --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml -p broca-mace-evaluator
cargo test --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml -p broca-mace-policy
cargo test --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml -p broca-mace-evaluator
cargo test --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml -p broca-mace-ingress
cargo test --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml -p broca-mace-consolidation
cargo test --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml -p broca-mace-consolidation --test raw_session_pipeline
cargo test --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml -p broca-aml-adapter
```

No external service, GPU, database or model API is required. The current verifier used by tests is synthetic, not a production Mycelix adapter.

## Decision-sufficient memory architecture

The broader research thesis and qualification ladder are documented in [MACE Decision-Sufficient Memory Architecture v0.1](../../docs/research/MACE_DECISION_SUFFICIENT_MEMORY_ARCHITECTURE_V0_1.md). It defines decision capsules, representation choice (persist/locator/snapshot/discard), use-time evidence revalidation, the HDC semantic/query-conditioned follow-on, and the external benchmark/industry comparison lanes. It is an integration contract over existing owners, not a second memory store or authority.

## Typed-content, query-conditioned HDC follow-on

The follow-on branch adds a separate typed-content index to the existing HDC retrieval crate: candidate Experience fields are encoded using deterministic role/value binding, continuous values use deterministic random Fourier features, and SemanticExperienceQueryV1 creates a structured query in the same field space. Exact event IDs and provenance remain sidecar references; all MACE admission gates still execute after canonical resolution.

This is a representation-level prototype only. The current MACE Query exposes scope, not the richer candidate-visible decision features required for end-to-end utility evaluation. Do not populate those richer query fields from evaluator truth. See [MACE Decision Sufficiency and Benchmark Frontier v0.2](../../docs/research/MACE_DECISION_SUFFICIENCY_AND_BENCHMARK_FRONTIER_V0_2.md) and [MACE typed-content HDC v0.1 protocol](../../docs/research/MACE_TYPED_CONTENT_HDC_V0_1.md).

## Paired memory-to-decision intervention runner

A separate executable now schedules seven paired intervention arms over the existing generated MACE worlds: true empty memory, exact-scope scan, deterministic key-value retrieval, static event-ID HDC, HDC-LTC, typed-content query-conditioned HDC, and a same-cluster sibling-task structural placebo. It writes candidate decisions, evaluator-only synthetic scores, and cluster-bootstrap summaries to separate JSONL streams.
The typed-content arm builds vectors from candidate-visible Experience fields and queries with the actual MACE scope/action plus the categorical causal-transfer qualification profile. Numeric overlap and balance limits remain exact post-retrieval MACE checks; richer outcome/horizon/estimand query requirements await the candidate-visible DecisionNeed contract in #13222. Typed profile digests are emitted in candidate records.

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

The structural placebo is not yet a text/embedding-level or exact token-matched placebo, and the deterministic key-value arm is not BM25/dense retrieval. The evaluator is public and co-located, so all outputs are diagnostic-only, not an independent benchmark result. See [MACE Paired Memory-to-Decision Runner v0.1](../../docs/research/MACE_PAIRED_MEMORY_DECISION_RUNNER_V0_1.md) for the record contract, resource-accounting caveats, and qualification gates.


## Build-graph boundary

- `policy/` is the candidate-policy crate. It contains candidate-visible experience/query types, policy logic and decision receipts. It has no dependency on the evaluator and no `OracleTruth` type.
- `evaluator/` owns scenarios, hidden target utilities, causal labels, scoring and reporting. It depends on the policy crate, not vice versa.
- `ingress/` verifies receipt interfaces, bitemporal eligibility and source/calibration/outcome lineage before projecting admitted interactions into candidate-visible experiences. It depends on policy, not evaluator.
- `consolidation/` consumes receipt-derived, action-/scope-specific evidence summaries and emits a proposal-only consolidation recommendation. It depends on ingress and policy; it does not write persistent memory or grant authority.

This is a stronger compile-time boundary than placing the oracle in another Rust type in the same source file. The evaluator source remains openly inspectable in this prototype; **it is not a secret or independently operated evaluator**. For a serious challenge, retain the evaluator in a separate controlled repository or service and expose only a versioned candidate API. The ingress verifier is also only an interface seam: tests use a mock verifier, not live Mycelix signatures or conductor state.

## Scenario families

The deterministic fixed corpus includes twenty-two two-session cases:

1. Identified interventional effect transfer.
2. Confounded observational association decoy.
3. Regime shift.
4. Objective shift.
5. Source retraction.
6. Tool success without reconciled external state.
7. Untrusted / poisoned source.
8. Incomplete delayed-consequence horizon.
9. Correlated duplicate echo from one source family.
10. Qualified negative experience.
11. Irrelevant-update control.
12. Under-supported candidate.
13. Positive control in a shifted regime.
14. Few-shot invariant-effect transport into a shifted regime.
15. Mechanism-break negative control after a regime shift.
16. Common-cause attack: two source-family labels share one upstream root.
17. Error-interval negative control: point estimates are close but measurement bounds exceed the equivalence margin.
18. Common-mode baseline shift with invariant treatment effect.
19. High-reward observational causal decoy.
20. Unsupported quasi-experimental causal decoy.
21. One-sided qualified evidence: the supported action is inferior, so active probing is required.
22. Estimand mismatch: apparently stable treatment effects target different populations/outcomes/horizons and must not transport.

## Compared policies

- `no_memory`: fixed conservative fallback.
- `frequency_only`: aggregate all observed outcomes, ignoring scope and evidence governance.
- `scope_aware`: restrict to exact regime and objective, but does not qualify source or causal evidence.
- `causal_qualified`: require exact scope, receipt-verified randomized causal design, pre-registration, reconciled outcome, trusted non-retracted source, confounder control, completed delayed horizon, unique event IDs, distinct source families and verified upstream roots. The design must bind known design/estimand digests, treatment-assignment bounds within [0.05, 0.95], maximum absolute standardized mean difference ≤0.10, and interference control. A comparative decision also requires qualified support for both actions; one-sided support abstains.
- `invariant_transport`: reuse prior-regime effects only with target-regime witnesses, method-qualified causal design, disjoint source families and roots, and worst-case source/target effect-interval difference within the fixed 0.12 normalized-utility margin.
- `active_probe`: try exact-scope and invariant transport first; otherwise emit a typed, bounded environment-interaction request.

The oracle is evaluator-only and is not linked as a dependency into the candidate-policy crate.

## Metrics (separate axes)

- Decision accuracy, mean utility and mean regret.
- Abstentions and the explicit conservative fallback used for utility/regret.
- Causal claims and causal overclaims.
- Invalid-evidence reuse and unsupported-A reuse.
- Evidence-bound decisions and evidence-reference counts.

## What v0.1 can and cannot establish

The harness regression-tests transparent policy rules against fixed and generated synthetic failure modes. The causal-qualified and invariant-transport policies are hand-authored, not learned causal models. A zero-overclaim or high-accuracy result on these twenty fixed cases is **not** evidence of causal discovery, real-world transfer, general intelligence or SOTA.

Remaining gaps:
- hidden external evaluator custody or sealed labels;
- randomized multi-seed worlds and source-/seed-/mechanism-held-out intervals;
- a production Mycelix/Symtropy verifier and direct consumption of #12731's full retention, forgetting, privacy-erasure and reversibility lifecycle;
- provider-native OpenAI/Anthropic session adapters wired end-to-end (the current provider capture and source-span validators are separate untrusted-input seams);
- official MemoryArena / AMA-Bench and AML Add/Search submissions under their exact rules;
- learned policies, model-provider ablations, calibration and full token/tool/CPU/GPU/energy/p95/p99 lifecycle-cost measurement.

## Next qualification gates

1. Move evaluator code and hidden labels to evaluator-controlled custody (#11816).
2. Generate source- and seed-held-out worlds with randomized task prevalence and confounders.
3. Connect #12731's broader consolidation lifecycle, including reversible forgetting, supersession and privacy erasure, to the MACE evidence pack; the current Session-A/B/C integration uses the MACE-specific proposal consumer and synthetic verifier.
4. Add causal-certificate mutation tests and replayable prediction-before-action receipts.
5. Reuse canonical source/claim/scope/action-effect contracts from #9075, #8265, #9016 and #9490.
6. Add official benchmark adapters only after dataset license and evaluator contracts are verified.
7. Measure paired transfer, worst-group regret, abstention coverage and full lifecycle cost under matched budgets (#8427).

## External comparison portfolio

- MemoryArena: interdependent multi-session agent–environment transfer.
- AMA-Bench: long-horizon state/action/observation/tool trajectories.
- TimlMemBench: raw-native lifecycle, revocation, scope and restraint.
- LongMemEval-V2: long-horizon retrieval and temporal question answering.
- ConsolidationBench: consolidation decision quality and governance.
- #8427: matched cross-benchmark quality/cost frontier.

Do not pool percentages across these tasks. Reproduce each official protocol separately and publish exact backbone, prompt, budget, data/evaluator revision and cost/latency.

## Related owners

- [#9516](https://github.com/Luminous-Dynamics/symthaea/issues/9516): causal experience transfer / MACE programme.
- [#9139](https://github.com/Luminous-Dynamics/symthaea/issues/9139): consolidation governance / #12731.
- [#8427](https://github.com/Luminous-Dynamics/symthaea/issues/8427): lifecycle cost-quality frontier.
- [#9075](https://github.com/Luminous-Dynamics/symthaea/issues/9075): bitemporal source-event ledger.
- [#8265](https://github.com/Luminous-Dynamics/symthaea/issues/8265): claim identity versus similarity.
- [#11816](https://github.com/Luminous-Dynamics/symthaea/issues/11816): evaluator and oracle independence.

## Generated stress tranche (MACE v0.14 research)

A second evaluator binary adds **256 deterministic clusters × 21 scenario families = 5,376 generated cases**. It rotates source-family and event identities and varies identifiable positive-control effects. Each cluster includes confounding, regime and objective shifts, source retraction, unreconciled tool success, untrusted poison, incomplete delayed outcomes, correlated duplicates, event-identity aliasing, a shared-common-root attack, an error-interval negative control, a common-mode baseline-shift positive control, qualified negative experience, few-shot invariant transport, a mechanism-break negative control, an unseen-target active-probe case, high-reward observational and unsupported quasi-experimental decoys, one-sided evidence requiring active probing, and estimand mismatch across regimes.

Run it independently:

```bash
cargo run --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml -p broca-mace-evaluator --bin mace_generated_stress
cargo test --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml -p broca-mace-evaluator --bin mace_generated_stress
```

The candidate gate requires unique event IDs, at least two distinct nonzero source-family IDs and at least two distinct nonzero verified 256-bit source-root digests for each action; source-family ID zero is rejected by both exact-scope and transfer qualification. MACE v0.14 requires both action alternatives to have qualified support and matching estimand identity and compares the treatment contrast mean(A)-mean(B) across regimes, using worst-case effect intervals including measurement bounds; the bounds remain untrusted until an ingress calibration receipt verifies them. A comparative choice also requires qualified evidence for both candidate actions; one-sided support is not treated as proof that the unsupported alternative is inferior. Different family labels cannot turn a shared upstream source into independent evidence. Root digests are supplied by the ingress verifier; the current verifier itself is still a mock. The runner reports aggregate metrics for all policies and per-family metrics for both exact-scope causal memory and invariant transport. It also reports paired utility lift versus no-memory with a deterministic **cluster-bootstrap 95% interval**, resampling world clusters rather than individual scenarios, and a Session-B / Session-C known-retraction recovery control.

This is a generated synthetic stress distribution—not an official MemoryArena/AMA-Bench adapter, learned-policy train/test split, sealed external oracle or real-world causal discovery. Source-family IDs vary to expose correlation/identity failures, but independence is not cryptographically attested. Bootstrap intervals quantify variation over these generated clusters only; they do not establish external validity. Session-C corrections are applied as observed source-state updates before the next decision.

## Invariant transport experiment (MACE v0.12 research)

The candidate crate adds an `invariant_transport` policy alongside exact-scope `causal_qualified`. It compares the treatment contrast `mean(A) - mean(B)` across regimes—not the absolute outcome level for each action—so a common-mode baseline shift does not invalidate an unchanged action effect. Both actions need target-regime witnesses; each source regime needs independently qualified evidence for both actions; source/target family and root digests must be disjoint. The worst-case difference between source and target treatment-effect intervals must fit within a fixed 0.12 normalized-utility equivalence margin.

The generated suite includes a few-shot invariant positive control, a common-mode baseline-shift positive control, and a mechanism-break negative control. Exact-scope policy should abstain on the positive few-shot case; invariant transport may transfer. It must abstain when treatment effects move beyond the tolerance, target action witnesses are missing, or source-family/root lineage overlaps.

This is a transparent empirical-invariance heuristic, not a general transportability proof. The 0.12 margin is a prototype constant, not empirically calibrated. For source regime s and target regime t, it requires `abs((mean_A_s - mean_B_s) - (mean_A_t - mean_B_t)) + err_A_s + err_B_s + err_A_t + err_B_t <= 0.12`. This is a deterministic worst-case interval-containment condition, not a statistical confidence interval. Root digests and measurement-error bounds must originate in receipt-verified source lineage and calibration; direct candidate-provided values are not trust anchors.

## Benchmark portfolio

See [MACE causal-memory benchmark portfolio](../../docs/research/MACE_CAUSAL_MEMORY_BENCHMARK_PORTFOLIO.md) for the current industry comparison, matched-protocol SOTA targets, paradigm shifts, and qualification gates. External benchmark scores remain separate; internal synthetic scores are not SOTA claims.

## Active probing / value-of-information experiment

The `active_probe` policy tries exact-scope evidence first, then invariant transport. Only if both are unavailable does it emit a typed `ProbeRequest` containing the target scope, both actions, replicate count and maximum normalized-utility cost. The evaluator validates that request before simulating four target-regime intervention observations (two independent source families per action), returns those observations to the candidate, and scores the second exact-scope decision.

A probe has a declared **0.12 normalized utility cost** (120,000 units on a 1,000,000-unit scale; this is not USD). The typed request specifies target scope, both actions, two replicates per action, and a maximum normalized-cost budget. The evaluator rejects wrong-scope, unsupported-action, unsupported-replicate and under-budget requests. Reports include probe requests, successful resolutions and normalized probe cost; utility/regret and cluster-bootstrap comparisons use **net utility after that cost**. A generated positive control has useful prior-regime experience but no target-regime witness, allowing active probing to be compared with abstention/no-memory without granting free information.

This is an environment-simulation result. The evaluator uses hidden scenario utilities to generate observations, but candidate policy code receives only the generated observations and query. The prototype does not yet model physical probe risk, safety constraints, experiment duration or real-world intervention cost.


Each experience now carries a claimed absolute measurement-error bound. Invariant transport requires:

`abs(mean_source - mean_target) + source_mean_error_bound + target_mean_error_bound <= 0.12`

This is a conservative deterministic interval-containment check, not a statistical confidence interval. Error bounds are candidate-visible claims in this prototype and need sensor/protocol attestation; future work should add calibrated uncertainty models, preregistered equivalence tests and independent target-regime witnesses.

## Provider-ecosystem findings (snapshot: 2026-11-22)

### OpenAI: durable memory is separate from session history

The OpenAI Agents SDK describes sandbox-agent memory as distinct from conversational `Session` history. It stores durable memory artifacts in a sandbox workspace, uses progressive disclosure (a small summary first, detailed rollout summaries only when needed), and describes a two-phase generation pipeline: extract raw lessons from runs, then consolidate them into a memory layout. It also supports separate memory layouts for different agents. The November 17 GPT-6.2 update added two-hour and eight-hour background Responses windows; these can support offline dreaming, but the job must pin an as-known evidence snapshot and cannot consume later corrections or grant its own promotion authority. See the [Agent memory documentation](https://openai.github.io/openai-agents-python/sandbox/memory/) and the [Agents SDK sandbox announcement](https://openai.com/index/the-next-evolution-of-the-agents-sdk/).

**Implication for Symthaea:** adopt the separation and progressive disclosure, but keep the canonical event ledger, provenance, corrections, confidence and authorization outside model-editable memory files. Let the model propose salient successes, strategies, constraints and failures; retain independent evidence and governance gates rather than imposing failure-only memory rules. Model-generated memory should be a derived view with evidence references, not the source of truth. Separate layouts should map to explicit Mycelix principals/domains, not only directory names.

### Anthropic: decouple session, harness and sandbox

Anthropic's Managed Agents describes three replaceable interfaces: an append-only session log, a harness that invokes the model and routes tool calls, and a sandbox for execution. It warns that harness assumptions can become stale as model capabilities change. Its November 6 CL-Bench database-task report shows Opus 5.5 at 0.838 with memory vs 0.655 without and Sonnet 5.5 at 0.833 vs 0.644 on a 30-question task. It also found that a “record only failures” rule reduced memory gains for stronger models; model-led salience followed by offline transcript-based correction performed better. See [Scaling Managed Agents](https://www.anthropic.com/research/managed-agents). Anthropic's earlier [long-running agent harness](https://www.anthropic.com/research/effective-harnesses-for-long-running-agents) uses an initializer and incremental per-session work that leaves durable artifacts for subsequent sessions.

**Implication for Luminous:** preserve model-provider and execution-provider neutrality. Treat OpenAI and Anthropic as interchangeable reasoning/harness adapters, not owners of canonical memory. Every model or SDK upgrade should rerun the same behavior contract and adversarial evaluation; do not preserve brittle prompt workarounds after a capability change.

### TimlMemBench: raw-native ingestion is part of the test

TimlMemBench reports 208 reviewed samples, 1,566 sessions, 1,715 questions and 1,821 governance/activation cases. Its raw-native evaluation requires systems to ingest raw sessions, rather than merely query a prebuilt canonical memory pool. It finds that retrieval accuracy can mask lifecycle, scope, provenance, activation and restraint failures. See the [ACL Anthology paper](https://aclanthology.org/2026.findings-emnlp.507/).

**Implication for MACE:** the workspace now includes receipt-gated structured raw-interaction envelopes, a proposal-only consolidation consumer, and an exact source-span validator for extractor-produced claims. The fixed/generated policy evaluators still use structured `Experience` fixtures; the span validator is not yet wired into an automatic natural-language transcript extraction and evidence-grounding pipeline. A raw-native benchmark must test transcript turns → source-span-bound untrusted claims → independent entailment and receipt verification → consolidation → Session-B decision → Session-C correction.

### Agent Memory Challenge Cycle 2

As of **November 19, 2026**, the official Cycle 2 results are published: open-method winners are Resurface **61.22/100** (Textual, 1/89), InsightMem **63.33% resolved** (Coding, 1/20), and Khipu **43.26/100** (Multimodal, 1/18). Commercial winners are MemoraX **69.26/100**, **68.00% resolved**, and **53.68/100**, respectively. There are nine separate track/division leaderboards and 289 track entries. Textual/Multimodal scores and Coding resolution rates are different measures and must not be combined. The Cycle 2 materials deadline (October 31) and evaluation close (November 4) have passed. The platform fixes data, Answer and Eval; participants implement Add/Search. See the [official competition page](https://agentmemoryleaderboard.ai/competition/) and [Textual / Closed-source Methods results](https://agentmemories.ai/leaderboard/industry/textual).

**Implication for Luminous:** the existing `broca-aml-adapter` is a protocol core, not a hosted service. A future open-method submission requires participant-hosted Add/Search endpoints, a fixed public commit/API version, operational capacity and review. Cycle 2's academic/open-method profile also specifies `text-embedding-v4` and `gpt-4o-mini` for embedding/LLM components; keep that benchmark-specific profile separate from Symthaea's native provider-neutral architecture. Target the next eligible cycle. MACE's synthetic causal-transfer score is a separate axis and must never be presented as an AML score.

### Cross-platform reference architecture

| Layer | Luminous responsibility | Provider relationship |
|---|---|---|
| Session/event log | Mycelix-backed, append-only, bitemporal event and provenance ledger | Import/export adapter; never canonical provider-only session storage |
| Memory consolidation | Symthaea dreaming / evidence-qualified summarization | Consume provider-generated summaries only as untrusted candidates |
| Durable memory view | Versioned, progressively disclosed memory packets with evidence references | Export into OpenAI memory files or Anthropic session/handoff artifacts |
| Reasoning/harness | Symthaea/Broca policy and model adapter | OpenAI, Anthropic or local model interchangeable |
| Execution sandbox | Symtropy/isolated execution provider | OpenAI/Anthropic or local sandbox interchangeable |
| Authority and revocation | Mycelix policies, capabilities, receipts and correction propagation | Provider tool calls are requests, not authority |
| Independent verification | SAM/evaluator and replayable evidence | Separate from candidate and model provider |

The advantage should be durable semantics and auditability across providers—not duplicating every provider feature.


### Mycelix provenance bridge

Mycelix's [FIN-SYS source profile](https://github.com/Luminous-Dynamics/mycelix/blob/main/docs/finance/FIN_SYS_SOURCE_PROFILE_V1.md) already states the relevant common-cause rule: ten publications reusing one BIS/IMF series remain one upstream provenance family. MACE's `source_root_id` is a synthetic proxy for this lineage, not proof of provenance. In production it must be derived from a verified source-ledger receipt and preserve the acquisition mechanism, original artifact digest, source family, upstream dataset/series, transformation lineage, and supersession links.

Likewise, `measurement_error_bound` is currently a candidate-visible claim. A production error bound must bind its instrument/method, units, calibration revision, operating range, validation evidence and validity interval. An agent must not increase its own trust by asserting a narrower bound.

## Raw-native Session-A ingress boundary (research-only)

The workspace adds `broca-mace-ingress` as its third package and `broca-mace-consolidation` as its fourth. The ingress crate has 36 deterministic mock-verifier tests covering receipt-gated single-event admission, bitemporal replay, authority status, lifecycle/retraction, calibration, lineage, receipt-bound outcome value/unit, batch idempotency and consolidation evidence-pack projection.

The adapter maps a raw interaction envelope into candidate-visible `Experience` only after:

- complete payload digest and canonical event identity are bound to a verifier-returned raw-event receipt;
- Mycelix-style authority status is `receipt_attested`, with canonical event hash and signed authority-receipt digest retained;
- client/source `occurred_at` is kept distinct from server-signed `received_at`; only `received_at` may satisfy `known_at`;
- source lineage binds source-family/root IDs, acquisition mode, upstream ID, artifact digest, transformation digest, issuer and an identified source-trust policy revision;
- measurement error binds to unit, instrument, method, calibration revision and validity interval;
- causal-assessment and externally reconciled outcome receipts are present;
- lifecycle is `Active` as known at the query time; `Retracted`, `Superseded(successor)` and unknown states fail closed.

Duplicate batch deliveries are re-verified. The same event ID is idempotent only when both payload digest and complete verified receipt bundle match; conflicting payload or receipt bundles reject the batch. Within a batch, one canonical upstream source ID cannot be assigned multiple provenance roots, preventing source-lineage splitting.

### Ingress-to-consolidation handoff

The ingress-issued `IngestedExperience` and `IngestBatch` types expose read-only accessors, so callers cannot construct an admitted result by setting their own trust booleans. Receipt authenticity and source trust are separate: `TrustedForScope`, `Untrusted` and `Unknown` come from a versioned trust-policy receipt. Validly signed but untrusted/unknown sources remain preserved as non-qualified evidence; they cannot count toward causal support. `build_consolidation_evidence_pack` creates a deterministic, sorted `ConsolidationEvidencePackV1` from selected admitted IDs. It preserves event/session/task identity, scope/action, outcomes, measurement bounds and full receipt bundles, while deriving distinct source-family and upstream-root counts from verifier-returned lineage **only for sources trusted under the identified policy**. The pack separately counts trusted, untrusted and unknown-trust events so a downstream consolidation policy can quarantine rather than silently inflate support. `summarize_for_scope` emits a global decision-neutral `ConsolidationEvidenceSummaryV1`. The new `summarize_action_for_scope` emits action-specific event, family, root and qualified-success counts, exact scope status, successful event IDs and worst measurement-error bound. It cannot pool successes from Action A into Action B. It emits no retain/compress/promote action or oracle score. A success requires trusted lineage, an externally verified successful outcome, intervention, pre-registration, confounder control, reconciliation and a completed delayed horizon.

This is an **input contract**, not a consolidation decision. The pack contains no proposed promotion action, oracle label or authorization grant. The raw-ingress tests now exercise Session-A interactions → receipt-preserving evidence pack → action-scoped support summaries → Session-B causal decision reconstructed only from the pack's immutable evidence references → Session-C bitemporal retraction → rebuilt pack → next decision. The MACE-specific proposal consumer now consumes action-specific summaries. The broader governed lifecycle policy in #12731 still does not consume this pack; privacy erasure, reversible storage transitions and retention/forgetting remain separately governed.

### Trust boundary

The `EvidenceVerifier` trait is an integration seam, **not cryptographic verification**. The production implementation must validate Mycelix canonical bytes, Ed25519 event signatures, issuer/key authorization, signed authority receipts, event-hash binding, registry revision, receipt-chain status, source lineage, calibration, outcome and lifecycle history. A valid signature establishes authenticity/authority relative to the trust policy; it does not establish causal identification or scientific truth.

Run the ingress checks:

```bash
cargo test --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml -p broca-mace-ingress
```


## Additional evaluation targets

The external portfolio also tracks StateMemBench (current-vs-superseded state), Memora/FAMA (obsolete-memory penalties), ConsolidationBench (retain/compress/promote/forget decisions and governance), and AgentMemoryBench (joint procedural/system and personal/factual memory). Their protocols and scores are distinct. In particular, ConsolidationBench reports that its internal consolidation-quality score did not predict external transfer in a later small study; MACE therefore scores consolidation governance and downstream transfer independently.

## Proposal-only consolidation consumer

The workspace now includes broca-mace-consolidation, a narrow consumer of receipt-derived ActionConsolidationEvidenceSummaryV1. It is separate from ingress: ingress validates evidence and reports support; consolidation evaluates whether the evidence is sufficient to propose a reusable procedure. It does not mutate persistent memory, erase source data, grant authority, or replace Mycelix lifecycle/privacy controls.

The default prototype gate mirrors the governed benchmark's three-support threshold: at least three qualified success events, three distinct source families, and three distinct upstream roots for the same action and exact scope; all trusted predictions must be pre-registered; every success must be receipt-reconciled, interventional, confounder-controlled and past its delayed horizon; the maximum attested measurement-error bound must be at most 0.05 normalized utility units. Untrusted or unknown-trust evidence is quarantined. Insufficient evidence remains provisional.

The proposal contains the exact action/scope, as-known coordinates, evidence event IDs, family/root counts, maximum measurement bound, all threshold parameters and policy revision. Invalid configurations (zero support thresholds or non-finite/out-of-range measurement ceilings) fail closed to provisional. It is a candidate recommendation, not an authorization or storage transaction.

The consolidation crate has 7 unit tests and 5 cross-crate tests in `consolidation/tests/raw_session_pipeline.rs`. They exercise raw Session-A ingestion, action-specific consolidation proposals, receipt-gated release, Session-B decisions, Session-C retraction/rebuild, correction of stale evidence, and the new common-mode-shift treatment-effect transfer from raw Session A into a shifted raw Session B. The test verifier is synthetic; no production Mycelix verifier is claimed.


## Raw-native ingestion and bitemporal replay (MACE v0.7 research)

The evaluator includes a provider-neutral JSONL raw-session ingestion path. It preserves original event lines, computes SHA-256 digests for the complete session and each event, rejects duplicate IDs and non-monotonic sequences, and reconstructs state at `(valid_at, known_at)`. A correction applies only if it was recorded by `known_at` and its effective `valid_from/valid_to` interval contains `valid_at`; absent `valid_from_ms`, its effective start defaults to `occurred_at_ms`. Corrections must target an existing `world_observation` and cannot predate the source in sequence, occurred time or recorded time.

The raw-JSONL outcome receipt binds the exact preserved observation-line digest, event/action identity, observation timestamp, utility value and unit. Any mismatch rejects the observation before it enters candidate memory; the verifier's value is not used to silently repair a conflicting raw event. Replay of another event's receipt, timestamp drift and unit substitution are explicit negative controls.

The raw event stream is not its own authority. Three externally verified receipt registries are required:
- **Execution receipt (Symtropy seam):** binds the exact action-plan event, action ID, action, scope, dispatch time, pre-registration, intervention mode, confounder control and delayed-horizon completion.
- **Outcome receipt:** binds the exact world-observation event and action ID, and supplies the reconciled observed utility **and unit**. The raw payload's claimed value must match the receipt exactly; candidate-supplied utility cannot override the independently reconciled value. The server-received authority receipt cannot predate the outcome observation or arrive after the query's `known_at`. A tool's self-reported success is never accepted as a world outcome.
- **Source receipt (Mycelix seam):** binds the source artifact digest to upstream source family/root, trust and calibrated measurement-error bound.

A fourth registry gates memory lifecycle. Only a verified `PromoteToProcedure` receipt whose source-event list contains the exact observation event can enter the transferable MACE experience set. `RetainRaw`, `CompressWithSourceLinks`, `PromoteToRule`, `Quarantine`, `Supersede`, `ForgetFromActiveIndex` and `EraseForPrivacy` remain distinct dispositions and are not silently treated as a promotable procedure. This is a candidate-ingestion gate; it does not perform physical erasure.

These registries are test seams for future Symtropy/Mycelix integrations, not cryptographic implementations or live connectors. The current JSONL envelope is provider-neutral; native OpenAI Agents SDK and Anthropic Managed Agents adapters remain future work.

The raw-ingestion module has 30 authored tests. Controls cover tool-success misattribution, forged action claims, tampered outcome payloads, missing execution/outcome/consolidation receipts, post-dispatch plans, late-arriving observations, retroactive and future-effective corrections, unknown/wrong correction targets, correction chronology, bitemporal retraction, unknown source artifacts, malformed/duplicate/out-of-order events, invalid error bounds, valid-time boundaries, more than 256 later events without raw-history truncation, and content-digest replay.

### Consolidation-producer status (2026-11-18)

Draft [PR #12731](https://github.com/Luminous-Dynamics/symthaea/pull/12731) now emits a candidate-side `ConsolidationReceiptProposal` binding the selected disposition and policy ID/version to the exact source event IDs, fixture payload digests, source families and upstream roots. Its structural evaluator checks exact binding, duplicate IDs and minimum independent support; it rejects the frequency-only promotion over correlated echoes.

This is a producer-side proposal with synthetic fixture digests, not a signed or cryptographically verified Mycelix receipt. MACE v0.7 in this PR still consumes injected test registries rather than PR #12731 output. The production adapter and independent verifier are specified in [#12846](https://github.com/Luminous-Dynamics/symthaea/issues/12846).

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

### Receipt-bound consolidation bridge (MACE v0.8)

The consolidation package now turns a governed procedure-promotion proposal into a canonical `ConsolidationReceiptProposalV1`. It binds the bitemporal snapshot, action/scope, policy digest and revision, principal/capability receipt, consolidation run, input episode-set digest, destination artifact identity/digest, retention policy, reversibility class, and the complete receipt bundles for every selected source event.

A separate `ConsolidationReceiptVerifierV1` interface must attest the exact canonical proposal digest before `verify_and_release_promoted_experiences` releases only the selected source-backed experiences to Session B. It rejects altered source revisions, changed output/policy identity, wrong authority scope, mismatched evidence-pack snapshots, and receipts not yet known at transfer time. The transfer must use the exact `known_at` snapshot bound by the proposal; a later snapshot requires a new evidence pack and promotion. Session C repeats the proposal/verification process over the surviving as-known evidence after retraction.

The integration test uses a fixture verifier only. It is not a live Mycelix signature verifier, production authority service, durable memory write, or proof of physical privacy erasure. The existing raw-JSONL evaluator registry remains a separate test seam and still needs to consume this exact proposal/receipt schema.

## Provider-shaped trace adapters (MACE v0.8 research)

The ingress crate contains capture-envelope adapters for Anthropic Managed Agents
persisted event history and OpenAI Agents SDK session items / RunItem-shaped
records. They preserve user/assistant messages, tool intents, tool outputs,
permissions, session status, handoffs and context-management events.

Each imported line is wrapped in a versioned envelope with session ID,
monotonic capture sequence and importer capture time. Anthropic native event
IDs are retained for reconnect deduplication; OpenAI input IDs are retained
where available, with capture sequence as the fallback identity. Changed
payloads under an existing native ID and reordered new records fail closed.

These are provider-shaped normalizers, not API clients or complete SDK parsers.
An importer must obtain persisted history and construct the capture envelope.
Anthropic stream-only event_start/event_delta previews are rejected because
they are not persisted records. Provider processed_at values remain opaque
metadata. For OpenAI session items without provider timestamps or stable IDs,
the importer supplies capture time and sequence; these are never relabelled
as underlying action occurrence time.

Provider messages, tool intents, tool outputs, permissions, handoffs and
session-idle events are trace evidence only. The normalized MACE projection
contains event digests and metadata, not transcript/tool-output bodies. It
uses event_type provider_trace_capture, which the causal ingester does not
promote to a world observation or candidate Experience. Execution, outcome,
source, calibration and consolidation receipts remain separate requirements.

Eleven provider-module tests cover classification, reconnect deduplication,
changed-payload collisions, preview rejection, session separation,
chronological order, unknown-event preservation and digest-only export. A raw
MACE integration test asserts provider traces alone produce zero Experiences.
These tests are authored, not runner-verified.


### Multi-agent thread ordering

The provider capture envelope accepts an optional thread_id (or stream_id). Sequence numbers are validated within each thread, not globally across concurrently executing subagents. The normalized event retains provider_sequence and provider_thread_id. MACE's raw JSONL requires one ingest sequence, so the exporter assigns a separate import-order sequence while preserving the provider's per-thread ordering metadata. This records a partial order without claiming a causal order between independent threads.

## Provider trace benchmark

See [MACE Provider-Native Trace Integrity and Outcome Grounding Benchmark v0.1](../../docs/research/MACE_PROVIDER_TRACE_INTEGRITY_BENCHMARK_V0_1.md) for the proposed trace-integrity, outcome-grounding, source-lineage, lifecycle, transfer and cross-provider evaluation tracks, baselines and qualification targets.

## Provider claim source spans

The ingress crate now includes a source-span validator for extractor-generated claims. Each proposal binds the original capture digest, provider event/thread identity, native event ID, JSON Pointer, UTF-8 byte range, exact span digest, assertion digest and extractor configuration. Validation returns an explicitly untrusted claim with the exact source text. It cannot become a candidate Experience, execution authority or world-outcome receipt; independent grounding and Mycelix/Symtropy receipts remain mandatory.

See [MACE Provider-Native Trace Integrity and Outcome Grounding Benchmark v0.1](../../docs/research/MACE_PROVIDER_TRACE_INTEGRITY_BENCHMARK_V0_1.md). This is an authored implementation and test tranche, not runner-verified.


## AML Add/Search adapter

See [MACE AML Add/Search Adapter v0.1](../../docs/research/MACE_AML_ADD_SEARCH_ADAPTER_V0_1.md) for the public protocol, Luminous memory projection and next-cycle qualification targets. The workspace includes a separate broca-aml-adapter Rust contract core with 15 authored tests. It is not a hosted service and has not been runner-verified. Hosted deployment, live Add/Search visibility and the next-cycle smoke path are tracked separately in [#13188](https://github.com/Luminous-Dynamics/symthaea/issues/13188).

## Current end-to-end integration inventory (2026-11-19)

The MACE branch contains six Rust packages: candidate policy, synthetic evaluator, receipt-gated ingress, proposal-only consolidation, provider-neutral AML Add/Search contract core, and HDC retrieval adapter. Current authored source counts are 25 policy tests, 15 fixed-evaluator tests, 14 generated-stress tests, 30 raw-JSONL evaluator tests, 36 ingress receipt/bitemporal tests, 11 provider-trace tests, 9 source-span claim tests, 9 consolidation-core tests, 29 re-observation tests, 13 portfolio-optimizer tests, 15 salience tests, 10 predictive-salience-learning tests, 11 memory-dose-response tests, 5 raw-session integration tests, 1 evaluator-level ingress→consolidation→MACE integration test, 14 AML adapter tests, 6 HDC retrieval unit tests and 5 HDC→MACE integration tests. These are source counts, **not executed test results**.

The new cross-package regression is `evaluator/tests/ingress_consolidation_transfer.rs`. It admits six raw Session-A interactions through `EvidenceVerifier`, builds the canonical `ConsolidationEvidencePackV1`, obtains action-specific procedure proposals for A and B, binds source/input/output digests into receipt proposals, releases experiences only after an independent fixture verifier accepts the proposal digests, and makes the Session-B causal-qualified choice. Session C retracts two A sources, rejects the retracted events at the later `known_at`, rebuilds the evidence pack, rejects replay of the stale consolidation receipt against the new snapshot, refuses to mint a replacement Action-A procedure from the one remaining qualified source, and verifies that the next candidate decision abstains and requests a bounded probe.

Run specifically with:
```bash
cargo test --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml -p broca-mace-evaluator --test ingress_consolidation_transfer
```
The verifier is deterministic fixture code: this validates crate composition and receipt contracts, not production cryptographic verification or physical causal inference.

The three evidence planes remain deliberately distinct:
- provider capture / source spans prove what bytes were captured and where a claim came from; they do not prove execution or truth;
- Symtropy execution and outcome receipts bind the proposed/dispatched action to the observed world event;
- Mycelix lineage, calibration, trust policy, lifecycle and bitemporal receipts govern source identity and whether the evidence was known/valid at the queried snapshot;
- Symthaea consolidation emits a proposal; an independent consolidation receipt verifier must attest the exact decision/input/output before the candidate can consume the promoted procedure.

The older raw-JSONL evaluator still uses test-only execution, outcome and source registries, but its consolidation gate now uses the same canonical `ConsolidationReceiptProposalV1`, `ConsolidationEvidencePackV1`, `VerifiedConsolidationReceiptV1` and `verify_and_release_promoted_experiences` contract as the cross-crate Session-A → B → C path. The adapter checks the exact (valid_at, known_at) snapshot and compares the released experience against the JSONL action, scope, utility, source-family/root digests, measurement bound, trust and reconciliation fields. The old `verified: bool` receipt is removed. Tests cover non-promotion dispositions, proposal-digest tampering, source-binding substitution after digest reissue, and replay at a later knowledge time.

This narrows the duplicate-schema gap; it does **not** make the JSONL fixture production-authoritative. Its execution/outcome/source registries remain synthetic test adapters, and no live Mycelix/Symtropy signature verification is claimed.

### Treatment-effect transport semantics (MACE v0.6)

Transport compares the causal contrast `mean(A) - mean(B)` between regimes. It does not require each action's absolute outcome mean to remain stable. This avoids rejecting a portable action effect when an environment-wide baseline shift moves both outcomes equally.

For source regime `s` and target regime `t`, the deterministic worst-case check is:

`abs((mean_A_s - mean_B_s) - (mean_A_t - mean_B_t)) + err_A_s + err_B_s + err_A_t + err_B_t <= 0.12`

The target must contain a witness for each action; every contributing source regime must contain independently qualified evidence for both actions. Root and family digests must be disjoint across source/target and across contributing source regimes. The aggregated effect is an equal-weight average of regime contrasts, avoiding raw-outcome pooling bias from unequal action counts. If its error interval crosses zero, the policy abstains.

This remains a deterministic prototype: measurement bounds and root digests require independent attestation, and the fixed equivalence margin is not a learned or statistically calibrated confidence level.


### Raw-native treatment-effect transport integration (MACE v0.6)

The consolidation integration suite now has a Session-A → Session-B transport control built from raw `RawInteractionV1` records. Session A is ingested at an explicit (valid_at, known_at) snapshot, grouped into action-/scope-specific evidence packs, promoted through proposal-only consolidation, and released only after an independent receipt verifier accepts the exact source bindings. Session B contributes one receipt-verified witness per action under a shifted regime with a +0.15 common-mode baseline shift. Exact-scope policy abstains; invariant transport uses the stable A-B treatment effect.

The fixture verifier is synthetic. This validates crate composition and receipt/evidence interfaces, not live Mycelix cryptographic verification or physical causal inference.


### Legacy JSONL consolidation gate (MACE v0.8)

The test-only JSONL adapter no longer accepts a `verified: bool` consolidation record. It now consumes a canonical promotion proposal, receipt-preserving evidence pack and verifier-issued receipt; it calls the shared canonical release function and checks full bitemporal snapshot equality before exposing the released experience. Additional equality checks bind the legacy JSONL observation to the canonical action, scope, outcome, source-family/root digests, trust and measurement error bound.

Four additional adversarial regressions cover non-promotion dispositions, output/proposal digest tampering, a substituted source event even when a matching proposal digest is supplied, and stale receipt replay against a later knowledge snapshot. This is still a synthetic fixture adapter: execution, outcome and source verifier registries are not live Mycelix/Symtropy verifiers.


## Provider run-manifest reproducibility

Provider traces can now be bound to a canonical run manifest containing model/release, API surface/version, harness revision, system-instruction digest, tool-schema digest, sandbox image, memory policy/state, runtime configuration, capture digest, run time window and retention mode. This enables matched tests for same-model/different-harness and different-model/same-harness.

The manifest is self-declared metadata: it proves consistency with the captured trace, not that the provider actually executed the declared model or harness. Provider/runtime signatures and independent attestation remain future work. It is not execution or outcome authority.

### Bitemporal and scope-safe predictive salience

The consolidation salience planner pins each candidate to the exact source scope and `(valid_at, known_at)` snapshot. Both coordinates are carried into the receipt-bound evidence digest and resulting surfacing-plan digest. Failure-only, recency-first, frequency-first and predicted-future-value selection all reject cross-scope evidence and evidence from a different knowledge snapshot; a model prediction cannot bypass the gate. Salience only chooses which already-admitted records to surface in context; it cannot promote evidence or grant action authority.

### Predictive-salience calibration benchmark (authored, not runner-verified)

A deterministic generated test evaluates 4,096 future-use forecasts over 256 task clusters with an equal 64-token context budget. It compares failure-only, recency-first, frequency-first and predicted-future-value selection. Forecasts are scored with Brier and ten-bin expected-calibration error; selectors are scored on hidden expected reuse value, one sampled realised outcome, value per token and regret against an oracle knapsack. Hidden future labels are held outside the selector inputs. This is a synthetic development/qualification fixture, not an AML or CL-Bench result. Future external-validity work must hold out task families, source roots, model versions and environment mechanisms.

### Independently qualified predictive-salience forecasts (MACE v0.10)

A raw `FutureUsePredictionV1` can no longer be placed directly into a memory-surfacing candidate. It must first pass `verify_future_use_prediction` and receive an opaque `VerifiedFutureUsePredictionV1` from a `SaliencePredictionVerifierV1`.

The calibration receipt binds the exact prediction and evidence digests, model/version/configuration, target scope, `(valid_at, known_at)` snapshot, training/calibration/held-out dataset digests, task-family and source-root split digests, cluster counts, Brier score versus a base-rate comparator, ECE, worst-group ECE, verifier identity and validity interval.

The current admission prototype requires at least 32 training clusters, 16 calibration clusters and 16 held-out clusters; calibration Brier must beat the base-rate Brier, ECE must be ≤0.05, worst-group ECE ≤0.10, and the verifier must attest task-family and source-root disjointness. The receipt must have been verified by the as-known time and remain valid at that snapshot. Surfacing-plan digests retain both the prediction digest and calibration-receipt digest.

This creates an important separation:
- model: proposes predicted future-use value;
- independent verifier: qualifies its calibration and held-out evidence;
- consolidation/salience: selects already-admitted memory under a token budget;
- Mycelix: must own the production trust roots, signed receipt verification, source-root lineage and correction/revocation semantics.

The current verifier is a synthetic fixture. There is no live Mycelix signature verification, production calibration registry or independently operated calibration service in this workspace. The interface is an assurance boundary, not a claim that cryptographic verification is implemented.

### Predictive-salience benchmark claim boundary

The 4,096-forecast development test is an **oracle-informed synthetic positive control**. Its generator uses a latent value to construct both hidden future-reuse probabilities and the candidate forecasts. It checks that a useful predictive signal can outperform fixed failure-only, recency-first and frequency-first heuristics under equal token budgets. It does **not** establish that a learned Symthaea model can infer that signal from real raw-session features, generalize to unseen tasks, or beat an external memory product.

The next credible experiment must:
1. train or fit the forecaster on a training split only;
2. calibrate on separate task clusters;
3. test on held-out task families, source roots, objectives, model versions and environment mechanisms;
4. include common-cause, drift, stale-memory, delayed-outcome and adversarially misleading-history cases;
5. compare Brier skill against base rate, ECE and worst-group ECE, selected realized utility, utility per token, regret, stale activation, and latency/cost;
6. include a blinded external evaluator and publish all split digests and run manifests.

No MACE salience SOTA result is claimed from the current synthetic positive control.


### Verified provider capture closure (MACE v0.11)

A capture digest binds the bytes that were supplied; it does **not** prove that the export is complete. The provider adapter now has a separate `ProviderCaptureClosureClaimV1` / `ProviderCaptureClosureVerifierV1` boundary. It checks the page-cursor chain, terminal-page marker, per-page digests, total raw event records, de-duplicated unique event count, duplicate count, run/session/provider identity and exact capture digest. A trusted verifier must then authenticate the export-closure receipt before yielding a `VerifiedProviderTraceBatchV1`.

The older `bind_provider_run_manifest` remains an internal-consistency binder only. It does not produce the verified-complete wrapper. The new type has private fields and can be obtained only through the closure-verification function.

The current verifier is a deterministic test seam, not a production OpenAI/Anthropic export signer or Mycelix receipt verifier. Where a provider API does not expose a signed export-completeness receipt, deployment must use an independently controlled collector/receipt authority or explicitly classify the capture as partial. A self-declared terminal-page flag is not proof by itself.

### AML Cycle 2 result-aware submission preflight

The official Cycle 2 results were published on November 19, 2026: open-source leaders are Resurface Textual 61.22/100, InsightMem Coding 63.33% resolved, and Khipu Multimodal 43.26/100. Commercial leaders are MemoraX Textual 69.26/100, Coding 68.00% resolved, and Multimodal 53.68/100. These are separate metrics and divisions.

The AML API guide also specifies that Open-Source Methods must use `text-embedding-v4` for embeddings and `gpt-4o-mini` for LLM-related components; rerankers are unrestricted. Commercial Products have no such model restrictions.

The `aml-adapter::submission` module now checks declared division, track, participant-hosted API mode, HTTPS Add/Search endpoints, secret-free URLs, source/runtime digests, and division-specific model usage. It returns a digest-bound declaration only: it does not prove endpoint reachability, runtime model identity, or a passed AML Smoke test. Nine authored preflight tests cover model policy, division differences, hosted endpoints, local/secret URLs and declaration digest changes.

## Learned future-use forecasting experiment

The consolidation crate now contains `salience_learning`, a deterministic logistic-regression baseline using seven observable features. It fits on training data, calibrates with a separate partition, and evaluates on held-out data. Split qualification requires disjoint cluster IDs, task-family IDs, upstream source-root digests and event IDs, plus one identical target-agent SHA-256 fingerprint across train, calibration and held-out partitions. The evaluator fixture uses 64 training, 32 calibration, and 32 held-out clusters; the evaluator owns the hidden reuse response function and future-reuse labels.

The report includes Brier score versus a **training-only** base-rate comparator, expected calibration error, worst task-family ECE, and split cardinalities. The split manifest carries separate dataset, task-family and source-root digests, and rejects overlap across clusters, families, roots or events. The trained model and calibrator can now emit a raw `FutureUsePredictionV1` bound to a combined model-and-calibrator pipeline digest. It remains untrusted: the existing `SaliencePredictionVerifierV1` must issue the independent calibration receipt before any prediction becomes eligible for surfacing. This is a learned synthetic baseline, not demonstrated utility lift, production forecasting, or a SOTA result.


For capability-level Cycle 2 anchors and +5-point targets (explicit recall, multi-hop, temporal, governance, care, workflow, privacy, streaming; coding new-feature/bug-fix; multimodal retrieval/relational/temporal/continual/personalization/governance), see the comparison table in [MACE_CAUSAL_MEMORY_BENCHMARK_PORTFOLIO.md](../../docs/research/MACE_CAUSAL_MEMORY_BENCHMARK_PORTFOLIO.md).
## Causal identification and estimand-consistency gate (MACE v0.14 research)

MACE no longer treats `intervention_confirmed` and `confounders_controlled` booleans as sufficient causal-identification metadata. Candidate-visible experiences carry a typed `CausalIdentificationV1`: method, receipt-bound design and estimand digests, treatment-assignment overlap bounds, covariate-balance bound and interference-control status.

Only a receipt-verified randomized design currently qualifies for causal transfer. The prototype gate requires known design/estimand digests, propensity bounds within [0.05, 0.95], maximum absolute standardized mean difference ≤0.10, and interference control. Observational and quasi-experimental histories remain admitted and retrievable as observations, but are not promoted into causal rules. The fixed and generated suites contain high-reward observational and unsupported quasi-experimental decoys; the exact-scope and transport paths share the same method gate.

These thresholds are prototype settings, not universal causal-inference standards. The injected `EvidenceVerifier` must bind the design receipt to the randomization mechanism, eligible population, preregistration, estimand, covariates and outcome horizon. Current implementations are fixtures; no live Mycelix signature or randomization audit is claimed. Quasi-experimental support requires separate diagnostics (e.g. DiD pre-trends, IV strength/exclusion, RD manipulation/continuity, and synthetic-control donor/placebo tests) rather than simply enabling a generic flag.

The canonical consolidation source-binding digest now includes all seven causal-design fields. A mutation test changes method, design digest, estimand digest, lower/upper assignment bounds, balance bound and interference status one at a time and requires the canonical source-binding digest to change.


The MACE v0.14 one-sided-evidence control is deliberately asymmetric: only Action B has two qualified independent sources, but the hidden target environment makes Action A better. The causal-qualified policy must abstain rather than infer that B is better from missing A evidence. Active probing then obtains target-scope observations for both actions, pays the declared probe cost, and can choose A from the new evidence.

MACE v0.14 additionally requires the receipt-bound estimand digest to remain consistent across all observations in each action group, between the two actions, and between source and target regimes. Different randomized designs may estimate the same declared estimand; a changed population, outcome, treatment contrast or horizon must carry a different digest and cannot be transported by the current policy. This compares canonical estimand identity; it does not prove that the verifier specified the population, outcome, treatment and horizon correctly.

**Semantic-estimand limitation:** the current v0.14 policy compares verified estimand digests; it does not independently decode or validate the semantic specification represented by those digests. The production Mycelix verifier must bind canonical population, treatment versions/contrast, outcome/unit, follow-up horizon, censoring/intercurrent-event strategy and target weighting into the signed assessment before transport is eligible.


### Model-conditioned memory dose response (MACE v0.16)

The new `memory_budget` module treats context size as a target-model-specific policy parameter rather than a global constant. A frozen candidate set of token budgets (including zero memory) is evaluated on paired calibration episodes from the same task clusters. The best mean net-utility budget is selected using calibration data only, with ties resolved toward the smaller budget. The selected budget is then qualified on a disjoint held-out set; a nonzero recommendation is rejected unless its paired cluster-bootstrap 95% lower bound is above zero.

Every trial binds the target-agent model digest, memory-pipeline digest, episode cluster, task family, upstream source root, token budget and realized net utility after declared costs. Calibration and held-out partitions must be disjoint by cluster, task family, source root and event. The resulting opaque `QualifiedModelMemoryBudgetV1` can be passed to the evidence-bound salience knapsack; the composed surfacing-plan digest binds model, pipeline, calibration, held-out qualification and selected memories.

This directly operationalises the November 19, 2026 IBM Research paper *How Much Should an Agent Remember?*: its AppWorld results report that useful memory quantity varies with target-model capability, and that a smaller selected subset can outperform a full guideline set for a weaker model. MACE's implementation is a deterministic calibration protocol, not a replication of the paper or evidence of an external benchmark gain.


### Receipt-qualified model-memory dose response (MACE v0.16)

The dose-response calibration does not accept plain utility rows as sufficient evidence. Each trial binds target-model and memory-pipeline digests, frozen environment and evaluation-protocol digests, task-family, upstream source root, cluster and exact episode identity, token budget, and net utility after declared costs.

Before fitting, `verify_memory_dose_trials` requires an injected `MemoryDoseTrialVerifierV1`. It binds the exact trial digest to a time-valid receipt and returns opaque verified trials. The statistical fitter only accepts those opaque records. Budget arms within a cluster must refer to the same episode, and calibration/held-out sets must be disjoint by cluster, family, root, event and episode, using matching environment and evaluation-protocol versions.

The current verifier is a deterministic test fixture, **not** a cryptographic verifier. A production adapter must authenticate an independently signed evaluator/run receipt and independently validate task outcome and utility-cost accounting, including token counts, latency and execution costs. Digest self-consistency alone is not authenticity.

The runtime selector rejects a budget for a different model fingerprint and binds the held-out dataset digest into the surfacing plan. A qualified zero-memory result produces a valid empty surfacing plan. The v0.16 future-use salience predictor is also target-model-conditioned: examples, training/calibration/held-out partitions, model, calibrator and raw prediction requests bind the target-agent fingerprint.


### Target-model-conditioned predictive salience (MACE v0.16)

The learned future-use model is trained, calibrated and evaluated against one explicit target-agent model digest. Every example carries that digest; mixed fingerprints are rejected within a split and across the train/calibration/held-out partitions. The split manifest, trained model, calibrator and raw-prediction pipeline bind the same target identity. A forecast names the agent it is intended for, and a model/input mismatch fails closed.

This is a qualification-boundary improvement, not a measured performance gain. The next experiment compares pooled, model-conditioned and hierarchical predictors under leave-one-model-family-out evaluation, using measured no-memory baselines and complete memory dose-response curves. Parameter count alone is not a capability metric.

Textual Add lifecycle entry points require the frozen word counter at reservation, visibility verification, completion and abandonment; the counter version is bound into idempotency state.

### Integrated memory-dose and salience qualification

The evaluator composes raw Session-A ingress evidence with per-model memory-dose calibration on one set of clusters, held-out budget qualification on disjoint clusters, and independently calibrated future-use salience. It verifies that matching target-model forecasts are surfaced within the qualified token budget, while forecasts for a different model are rejected. The evidence-digest helper is read-only and accepts only an already admitted evidence reference.

## HDC/HDC-LTC retrieval arm

The `broca-mace-hdc-retrieval` package supplies static ContinuousHV and HDC-LTC zero-input candidate indexes using the existing `symthaea-hdc-ltc` crate. It returns event IDs and ranking diagnostics only; consumers resolve IDs and rerun unchanged MACE qualification. No retrieval, utility or SOTA gain is claimed. See the [adapter README](hdc-retrieval/README.md) and [protocol](../../docs/research/HDC_LTC_MACE_TEMPORAL_RETRIEVAL_ARM_V0_1.md).

HDC retrieval hits now carry a candidate-visible Experience-view digest; canonical resolution rejects hits whose record has changed since index construction. This is a stale-snapshot check, not a source signature or trust attestation.

## Re-observation-aware representation

MACE now has a separate reobservation policy prototype alongside predictive salience. It chooses whether to persist a learned rule/procedure/preference, retain a verified locator for current state only when the locator receipt remains valid strictly beyond the decision deadline and calibrated availability justifies the expected failure penalty, cache a snapshot only through a verified expiry, or discard non-positive-value material. Locator value is risk-adjusted by calibrated availability probability and the consequence of failed re-observation. Stable facts can be retained without a transient-state expiry; the planner compares a locator against persistent retention where a source can be queried. The plan binds source/evidence identity, target-model prediction, exact `(valid_at, known_at)` snapshot, source-profile receipt, decision deadline, context cost, re-observation cost, calibrated availability probability and expected failure loss. A profile proved at an earlier knowledge time cannot be reused after the knowledge snapshot advances. The evaluator includes a raw Session-A ingress → calibrated future-use prediction → independently checked source-availability profile → representation-plan integration fixture.

See [MACE Re-observation-Aware Memory Benchmark v0.1](../../docs/research/MACE_REOBSERVATION_AWARE_MEMORY_BENCHMARK_V0_1.md). This is a candidate representation policy, not a production source adapter or measured performance gain.

The industry comparison distinguishes Microsoft Research's Memora memory-representation system (reported 86.3% LoCoMo / 87.4% LongMemEval in its own setup) from ACL 2026 Memora/FAMA. Any MACE comparison must reproduce the same backbone, judge, task split, memory interface and token accounting; no direct score comparison is claimed.

The re-observation benchmark now includes a WorldMemArena crosswalk: representation choices are evaluated separately at write/maintain/retrieve/use. External reference baselines also distinguish Microsoft Research Memora from the separate ACL Memora/FAMA work and from Mem0's vendor-maintained benchmark results; no cross-protocol score comparison is claimed.

Re-observation plans are now authority-scoped: a typed Mycelix-aligned scope binds principal, capability, audience, resource/tenant, purpose and policy revision. Source verification and decision planning must match that scope exactly; this is an injected verifier contract, not a replacement authorization system or a live capability/revocation check.


## Joint memory/re-observation portfolio (MACE v0.17 research)

The new `consolidation::portfolio` module consumes only representation plans reconstructed from a verified salience prediction and a verified re-observation profile. For one exact authority scope, task scope and `(valid_at, known_at)` snapshot, it chooses at most one representation per event under five shared limits:

- total context tokens;
- number of re-observation calls;
- aggregate re-observation latency work (not parallel wall-clock latency);
- estimated monetary/compute cost in the caller-declared micro-unit;
- expected failure-loss budget.

The re-observation module now enumerates individually valid alternatives for a stable fact (persist or locator), current state (locator or valid snapshot), and learned rules/procedures/preferences (persist only). The portfolio solver can therefore select a lower-value per-item option when the higher-value choice would consume scarce shared calls or tokens.

Two strategies are exposed: an exact deterministic branch-and-bound optimizer (up to 24 event groups and two million search nodes by default) and a greedy highest-individual-net-value baseline. The exact optimizer maximizes the sum of per-item net values already reduced by each option's own cost and expected failure loss, while separately enforcing aggregate resource/risk caps. It selects at most one representation per event. Ties prefer lower resource use, then canonical option digests. If the exact-search node limit is exceeded, the solver returns an error rather than publishing a partial or approximate plan.

The portfolio result is a digest-bound proposal, not a persistence operation, source-access grant, or action authorization. All options must use the same authority, scope and bitemporal snapshot. This first implementation sums latency work conservatively; it does not yet model parallel scheduling, shared query batching, privacy-exposure budgets or dynamic replanning.


### Trust-boundary tightening

Portfolio options are now admitted only when their representation-plan digest matches an option independently re-derived by `enumerate_memory_representation_options` from the same verified salience prediction, verified source profile and policy. A caller cannot mutate a plan's cost/disposition, recompute its digest, and present it as an eligible choice. The portfolio constructor still verifies the complete source/prediction/snapshot/authority binding before creating an option.

The exact optimizer assumes the supplied per-item `net_value_micros` are additive across distinct evidence payloads. It now rejects a byte-identical evidence digest being assigned multiple event IDs, but semantically overlapping facts with different payloads can still have interacting or diminishing value. This prototype is not a submodular/interaction-aware optimizer. A credible next version needs independently qualified marginal-value groups or a portfolio-level evaluator that measures the value of combinations against the same no-memory baseline; it must not simply sum correlated single-item predictions and call the result realized utility.


## Pareto resource frontier (MACE v0.18)

The consolidation crate now also exposes `optimize_memory_portfolio_frontier`. Instead of selecting only the maximum-net-value plan under one caller budget, it enumerates the bounded nondominated feasible plans across five dimensions: context tokens, re-observation calls, summed latency work, estimated cost and expected failure loss, while maximizing total net value. Dominated plans are removed; ties use canonical option digests. The frontier itself is digest-bound. It is bounded to 24 event groups, 2,000,000 search nodes and 128 nondominated plans by default. Exhaustion returns an error and never a partial frontier.

This is analogous in spirit—not an implementation reuse—to Sol-Atlas's bounded workforce active-evidence verifier, which keeps a Pareto frontier over staff time, cash (ZAR cents) and delay. MACE preserves its own memory-specific resource vector. The scalar exact optimizer and greedy baseline remain available; a caller can choose from the Pareto frontier under an explicit governance policy.

The frontier does not solve non-additive memory utility. It still assumes per-item net values add across distinct evidence payloads. Interaction-aware marginal value is tracked separately in #13184.


## Cross-system evidence and authority bridge

See [MACE cross-system evidence and authority bridge v0.1](../../docs/research/MACE_CROSS_SYSTEM_EVIDENCE_BRIDGE_V0_1.md) for the Mycelix signed-event/authority-receipt mapping, Sol-Atlas federated evidence/rights contract, Symtropy execution/outcome boundary and provider-neutral MACE projection. This is a contract and source mapping, not a live cryptographic connector.


## v0.18.1 semantic integrity hardening

The source-level canonicality checks now verify more than a recalculable digest:

- each selected item carries its own cost and expected-failure-loss values, allowing the plan to recompute all five aggregate resource totals and the summed net value;
- event IDs and option digests are unique within a selected plan;
- selected resources fit the declared budget;
- exact-search plans report the completed search-node count;
- Pareto frontiers verify each plan's digest and aggregate accounting, shared scope/as-known/authority/budget, strategy identity, deterministic sort order and pairwise nondominance;
- mutation controls rehash tampered values and require semantic validation to reject them.

This is internal consistency and semantic validation, **not** authenticity. A public digest is not a signature; production plans still require the appropriate Mycelix-signed receipt and policy/authority validation at use. The portfolio module now has **13 authored tests** by source count (not execution status).


## Portfolio interaction benchmark v0.3

A new evaluator binary, `mace_portfolio_interactions`, implements the first held-out diagnostic requested by [#13184 — portfolio-level marginal utility](https://github.com/Luminous-Dynamics/symthaea/issues/13184). It compares exact additive selection with a conservative pairwise-interaction selector fitted on historical pair observations and an evaluator-only oracle over complete selected sets.

The generator includes redundant, complementary, independent and conflicting item pairs, plus noisy relation labels and three separately reported evaluation domains: a same-family split-conformal holdout, a stable out-of-family holdout and a shifted-regime negative control. Training/calibration clusters and roots are disjoint; the conformal holdout shares the calibration task family but uses fresh cluster/root identities. The pairwise selector uses the calibrated lower bound of each interaction interval and abstains on unknown pair relations or an unsupported regime; it does not silently assume a missing interaction is zero. Both selectors obey the same token budget and one-representation-per-event constraint.

Run:
```bash
cargo run --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml -p broca-mace-evaluator --bin mace_portfolio_interactions
cargo test --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml -p broca-mace-evaluator --bin mace_portfolio_interactions
```

The current implementation is a synthetic pairwise diagnostic, not the production portfolio optimiser. Its relation labels and training observations are generated fixtures. The pairwise radius is a 90% split-conformal residual quantile. The same-family holdout is intended to satisfy the exchangeability premise; stable out-of-family and shifted-regime coverage are separate diagnostics. These intervals predict pairwise effects, not the mean. The test suite is authored but has not been executed in this environment.

**Verification:** 17 Rust tests are authored in the binary, but have not run here; no Cargo/Rust/rustfmt toolchain is available. The conformal coverage diagnostic uses evaluator-owned true relation classes, while candidate relation-label errors and unknowns are counted separately. The same-family holdout is the nominal 90% split-conformal domain; stable-family OOD and shifted-regime coverage are separate diagnostics. Source audits are not test passes.
