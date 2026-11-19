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

## Build-graph boundary

- `policy/` is the candidate-policy crate. It contains candidate-visible experience/query types, policy logic and decision receipts. It has no dependency on the evaluator and no `OracleTruth` type.
- `evaluator/` owns scenarios, hidden target utilities, causal labels, scoring and reporting. It depends on the policy crate, not vice versa.
- `ingress/` verifies receipt interfaces, bitemporal eligibility and source/calibration/outcome lineage before projecting admitted interactions into candidate-visible experiences. It depends on policy, not evaluator.
- `consolidation/` consumes receipt-derived, action-/scope-specific evidence summaries and emits a proposal-only consolidation recommendation. It depends on ingress and policy; it does not write persistent memory or grant authority.

This is a stronger compile-time boundary than placing the oracle in another Rust type in the same source file. The evaluator source remains openly inspectable in this prototype; **it is not a secret or independently operated evaluator**. For a serious challenge, retain the evaluator in a separate controlled repository or service and expose only a versioned candidate API. The ingress verifier is also only an interface seam: tests use a mock verifier, not live Mycelix signatures or conductor state.

## Scenario families

The deterministic fixed corpus includes eighteen two-session cases:

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

## Compared policies

- `no_memory`: fixed conservative fallback.
- `frequency_only`: aggregate all observed outcomes, ignoring scope and evidence governance.
- `scope_aware`: restrict to exact regime and objective, but does not qualify source or causal evidence.
- `causal_qualified`: require exact scope, intervention, pre-registration, reconciled outcome, trusted non-retracted source, confounder control, completed delayed horizon, unique event IDs, at least two nonzero source-family labels and at least two distinct verified upstream source-root digests. A comparative decision also requires qualified support for both candidate actions; one-sided support abstains.
- `invariant_transport`: reuse prior-regime effects only with target-regime witnesses, disjoint source families and roots, and worst-case source/target effect-interval difference within the fixed 0.12 normalized-utility margin.
- `active_probe`: try exact-scope and invariant transport first; otherwise emit a typed, bounded environment-interaction request.

The oracle is evaluator-only and is not linked as a dependency into the candidate-policy crate.

## Metrics (separate axes)

- Decision accuracy, mean utility and mean regret.
- Abstentions and the explicit conservative fallback used for utility/regret.
- Causal claims and causal overclaims.
- Invalid-evidence reuse and unsupported-A reuse.
- Evidence-bound decisions and evidence-reference counts.

## What v0.1 can and cannot establish

The harness can regression-test whether transparent policy rules reject these specific synthetic failure modes. The causal-qualified policy is deliberately hand-authored against visible fixture structure. A zero-overclaim or high-accuracy result on these seventeen fixed cases is **not** evidence of causal discovery, real-world transfer, general intelligence or SOTA.

It does not yet include:
- hidden external evaluator custody or sealed labels;
- randomized multi-seed worlds and source-/seed-held-out intervals;
- Session C delayed corrections or a full Session A → consolidation → Session B → correction chain;
- official MemoryArena / AMA-Bench adapters or licensed benchmark data;
- learned policies, model training or model-provider comparisons;
- token, tool-call, CPU/GPU, energy and p95/p99 lifecycle cost measurement.

## Next qualification gates

1. Move evaluator code and hidden labels to evaluator-controlled custody (#11816).
2. Generate source- and seed-held-out worlds with randomized task prevalence and confounders.
3. Add Session C corrections and connect the output of consolidation #9139 / PR #12731 to Session B transfer.
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

## Generated stress tranche (MACE v0.6 research)

A second evaluator binary adds **256 deterministic clusters × 17 scenario families = 4,352 generated cases**. It rotates source-family and event identities and varies identifiable positive-control effects. Each cluster includes confounding, regime and objective shifts, source retraction, unreconciled tool success, untrusted poison, incomplete delayed outcomes, correlated duplicates, event-identity aliasing, a shared-common-root attack, an error-interval negative control, qualified negative experience, few-shot invariant transport, a mechanism-break negative control, and an unseen-target case that triggers active probing.

Run it independently:

```bash
cargo run --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml -p broca-mace-evaluator --bin mace_generated_stress
cargo test --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml -p broca-mace-evaluator --bin mace_generated_stress
```

The candidate gate requires unique event IDs, at least two distinct nonzero source-family IDs and at least two distinct nonzero verified 256-bit source-root digests for each action; source-family ID zero is rejected by both exact-scope and transfer qualification. MACE v0.5 also requires both action alternatives to have qualified support and checks transport using worst-case effect intervals including candidate-supplied measurement bounds; the bounds remain untrusted until an ingress calibration receipt verifies them. A comparative choice also requires qualified evidence for both candidate actions; one-sided support is not treated as proof that the unsupported alternative is inferior. Different family labels cannot turn a shared upstream source into independent evidence. Root digests are supplied by the ingress verifier; the current verifier itself is still a mock. The runner reports aggregate metrics for all policies and per-family metrics for both exact-scope causal memory and invariant transport. It also reports paired utility lift versus no-memory with a deterministic **cluster-bootstrap 95% interval**, resampling world clusters rather than individual scenarios, and a Session-B / Session-C known-retraction recovery control.

This is a generated synthetic stress distribution—not an official MemoryArena/AMA-Bench adapter, learned-policy train/test split, sealed external oracle or real-world causal discovery. Source-family IDs vary to expose correlation/identity failures, but independence is not cryptographically attested. Bootstrap intervals quantify variation over these generated clusters only; they do not establish external validity. Session-C corrections are applied as observed source-state updates before the next decision.

## Invariant transport experiment (MACE v0.6 research)

The candidate crate adds an `invariant_transport` policy alongside exact-scope `causal_qualified`. It compares the treatment contrast `mean(A) - mean(B)` across regimes—not the absolute outcome level for each action—so a common-mode baseline shift does not invalidate an unchanged action effect. Both actions need target-regime witnesses; each source regime needs independently qualified evidence for both actions; source/target family and root digests must be disjoint. The worst-case difference between source and target treatment-effect intervals must fit within a fixed 0.12 normalized-utility equivalence margin.

The generated suite includes a few-shot invariant positive control, a common-mode baseline-shift positive control, and a mechanism-break negative control. Exact-scope policy should abstain on the positive few-shot case; invariant transport may transfer. It must abstain when treatment effects move beyond the tolerance, target action witnesses are missing, or source-family/root lineage overlaps.

This is a transparent empirical-invariance heuristic, not a general transportability proof. The 0.12 margin is a preregistered prototype constant, not empirically calibrated. Each observation carries a claimed absolute measurement-error bound; transport requires `abs(mean_source - mean_target) + mean_error_source + mean_error_target <= 0.12`. This is a deterministic worst-case interval-containment condition, not a statistical confidence interval. The bounds and root IDs remain candidate-visible claims and need sensor/protocol and source-ledger attestation.

## Benchmark portfolio

See [MACE causal-memory benchmark portfolio](../../docs/research/MACE_CAUSAL_MEMORY_BENCHMARK_PORTFOLIO.md) for the current industry comparison, matched-protocol SOTA targets, paradigm shifts, and qualification gates. External benchmark scores remain separate; internal synthetic scores are not SOTA claims.

## Active probing / value-of-information experiment

The `active_probe` policy tries exact-scope evidence first, then invariant transport. Only if both are unavailable does it emit a typed `ProbeRequest` containing the target scope, both actions, replicate count and maximum normalized-utility cost. The evaluator validates that request before simulating four target-regime intervention observations (two independent source families per action), returns those observations to the candidate, and scores the second exact-scope decision.

A probe has a declared **0.12 normalized utility cost** (120,000 units on a 1,000,000-unit scale; this is not USD). The typed request specifies target scope, both actions, two replicates per action, and a maximum normalized-cost budget. The evaluator rejects wrong-scope, unsupported-action, unsupported-replicate and under-budget requests. Reports include probe requests, successful resolutions and normalized probe cost; utility/regret and cluster-bootstrap comparisons use **net utility after that cost**. A generated positive control has useful prior-regime experience but no target-regime witness, allowing active probing to be compared with abstention/no-memory without granting free information.

This is an environment-simulation result. The evaluator uses hidden scenario utilities to generate observations, but candidate policy code receives only the generated observations and query. The prototype does not yet model physical probe risk, safety constraints, experiment duration or real-world intervention cost.


Each experience now carries a claimed absolute measurement-error bound. Invariant transport requires:

`abs(mean_source - mean_target) + source_mean_error_bound + target_mean_error_bound <= 0.12`

This is a conservative deterministic interval-containment check, not a statistical confidence interval. Error bounds are candidate-visible claims in this prototype and need sensor/protocol attestation; future work should add calibrated uncertainty models, preregistered equivalence tests and independent target-regime witnesses.

## Provider-ecosystem findings (snapshot: 2026-11-19)

### OpenAI: durable memory is separate from session history

The OpenAI Agents SDK describes sandbox-agent memory as distinct from conversational `Session` history. It stores durable memory artifacts in a sandbox workspace, uses progressive disclosure (a small summary first, detailed rollout summaries only when needed), and describes a two-phase generation pipeline: extract raw lessons from runs, then consolidate them into a memory layout. It also supports separate memory layouts for different agents. See the [Agent memory documentation](https://openai.github.io/openai-agents-python/sandbox/memory/) and the [Agents SDK sandbox announcement](https://openai.com/index/the-next-evolution-of-the-agents-sdk/).

**Implication for Symthaea:** adopt the separation and progressive disclosure, but keep the canonical event ledger, provenance, corrections, confidence and authorization outside model-editable memory files. Model-generated memory should be a derived view with evidence references, not the source of truth. Separate layouts should map to explicit Mycelix principals/domains, not only directory names.

### Anthropic: decouple session, harness and sandbox

Anthropic's Managed Agents describes three replaceable interfaces: an append-only session log, a harness that invokes the model and routes tool calls, and a sandbox for execution. It warns that harness assumptions can become stale as model capabilities change. See [Scaling Managed Agents](https://www.anthropic.com/research/managed-agents). Anthropic's earlier [long-running agent harness](https://www.anthropic.com/research/effective-harnesses-for-long-running-agents) uses an initializer and incremental per-session work that leaves durable artifacts for subsequent sessions.

**Implication for Luminous:** preserve model-provider and execution-provider neutrality. Treat OpenAI and Anthropic as interchangeable reasoning/harness adapters, not owners of canonical memory. Every model or SDK upgrade should rerun the same behavior contract and adversarial evaluation; do not preserve brittle prompt workarounds after a capability change.

### TimlMemBench: raw-native ingestion is part of the test

TimlMemBench reports 208 reviewed samples, 1,566 sessions, 1,715 questions and 1,821 governance/activation cases. Its raw-native evaluation requires systems to ingest raw sessions, rather than merely query a prebuilt canonical memory pool. It finds that retrieval accuracy can mask lifecycle, scope, provenance, activation and restraint failures. See the [ACL Anthology paper](https://aclanthology.org/2026.findings-emnlp.507/).

**Implication for MACE:** the workspace now includes receipt-gated structured raw-interaction envelopes, a proposal-only consolidation consumer, and an exact source-span validator for extractor-produced claims. The fixed/generated policy evaluators still use structured `Experience` fixtures; the span validator is not yet wired into an automatic natural-language transcript extraction and evidence-grounding pipeline. A raw-native benchmark must test transcript turns → source-span-bound untrusted claims → independent entailment and receipt verification → consolidation → Session-B decision → Session-C correction.

### Agent Memory Challenge Cycle 2

As checked on **November 19, 2026**, the official Cycle 2 page still labels leaderboard publication **expected November 19, 2026**; the page does not yet provide an official result table. The Cycle 2 submission deadline was October 31 and evaluation closed November 4, so this cycle is no longer open for new submissions. The platform standardizes Add/Search from the memory system and delegates Answer/Eval to the platform, with separate textual, coding and multimodal tracks. See the [official competition page](https://agentmemoryleaderboard.ai/competition/).

**Implication for Luminous:** prepare a provider-neutral Add/Search adapter and target the next eligible cycle. Do not claim a Cycle 2 rank or infer results from interim displays. The MACE synthetic causal-transfer score is a separate axis and must never be presented as an AML leaderboard score.

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

The consolidation crate has 7 unit tests and one cross-crate regression in `consolidation/tests/raw_session_pipeline.rs`. It exercises raw Session-A ingestion → receipt pack → action-specific consolidation proposals → Session-B causal decision → Session-C source retraction → rebuilt pack → abstention/active-probe request. The test verifier is synthetic; no production Mycelix verifier is claimed.


## Raw-native ingestion and bitemporal replay (MACE v0.7 research)

The evaluator includes a provider-neutral JSONL raw-session ingestion path. It preserves original event lines, computes SHA-256 digests for the complete session and each event, rejects duplicate IDs and non-monotonic sequences, and reconstructs state at `(valid_at, known_at)`. A correction applies only if it was recorded by `known_at` and its effective `valid_from/valid_to` interval contains `valid_at`; absent `valid_from_ms`, its effective start defaults to `occurred_at_ms`. Corrections must target an existing `world_observation` and cannot predate the source in sequence, occurred time or recorded time.

The raw-JSONL outcome receipt binds the exact preserved observation-line digest, event/action identity, observation timestamp, utility value and unit. Any mismatch rejects the observation before it enters candidate memory; the verifier's value is not used to silently repair a conflicting raw event. Replay of another event's receipt, timestamp drift and unit substitution are explicit negative controls.

The raw event stream is not its own authority. Three externally verified receipt registries are required:
- **Execution receipt (Symtropy seam):** binds the exact action-plan event, action ID, action, scope, dispatch time, pre-registration, intervention mode, confounder control and delayed-horizon completion.
- **Outcome receipt:** binds the exact world-observation event and action ID, and supplies the reconciled observed utility **and unit**. The raw payload's claimed value must match the receipt exactly; candidate-supplied utility cannot override the independently reconciled value. The server-received authority receipt cannot predate the outcome observation or arrive after the query's `known_at`. A tool's self-reported success is never accepted as a world outcome.
- **Source receipt (Mycelix seam):** binds the source artifact digest to upstream source family/root, trust and calibrated measurement-error bound.

A fourth registry gates memory lifecycle. Only a verified `PromoteToProcedure` receipt whose source-event list contains the exact observation event can enter the transferable MACE experience set. `RetainRaw`, `CompressWithSourceLinks`, `PromoteToRule`, `Quarantine`, `Supersede`, `ForgetFromActiveIndex` and `EraseForPrivacy` remain distinct dispositions and are not silently treated as a promotable procedure. This is a candidate-ingestion gate; it does not perform physical erasure.

These registries are test seams for future Symtropy/Mycelix integrations, not cryptographic implementations or live connectors. The current JSONL envelope is provider-neutral; native OpenAI Agents SDK and Anthropic Managed Agents adapters remain future work.

The raw-ingestion module has 26 authored tests. Controls cover tool-success misattribution, forged action claims, tampered outcome payloads, missing execution/outcome/consolidation receipts, post-dispatch plans, late-arriving observations, retroactive and future-effective corrections, unknown/wrong correction targets, correction chronology, bitemporal retraction, unknown source artifacts, malformed/duplicate/out-of-order events, invalid error bounds, valid-time boundaries, more than 256 later events without raw-history truncation, and content-digest replay.

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

See [MACE AML Add/Search Adapter v0.1](../../docs/research/MACE_AML_ADD_SEARCH_ADAPTER_V0_1.md) for the public protocol, Luminous memory projection and next-cycle qualification targets. The workspace includes a separate broca-aml-adapter Rust contract core with 15 authored tests. It is not a hosted service and has not been runner-verified.

## Current end-to-end integration inventory (2026-11-19)

The MACE branch now has five Rust packages: candidate policy, synthetic evaluator, receipt-gated ingress, proposal-only consolidation and AML Add/Search contract core. The current authored regression inventory is 19 policy tests, 26 raw-JSONL evaluator tests, 36 ingress receipt/bitemporal tests, 11 provider-trace tests, 9 source-span claim tests, 7 consolidation unit tests plus the raw-session integration test, and 14 AML adapter tests. These are source counts, **not executed test results**.

The three evidence planes remain deliberately distinct:
- provider capture / source spans prove what bytes were captured and where a claim came from; they do not prove execution or truth;
- Symtropy execution and outcome receipts bind the proposed/dispatched action to the observed world event;
- Mycelix lineage, calibration, trust policy, lifecycle and bitemporal receipts govern source identity and whether the evidence was known/valid at the queried snapshot;
- Symthaea consolidation emits a proposal; an independent consolidation receipt verifier must attest the exact decision/input/output before the candidate can consume the promoted procedure.

The older raw-JSONL evaluator now fails closed when the outcome receipt does not bind the exact preserved observation line, observed-at time, utility value and unit. The receipt registry remains a synthetic injected verifier seam. The canonical ingress/consolidation cross-crate path is stronger, but the raw-JSONL evaluator still uses a parallel test-only receipt schema; unifying it with the existing evidence receipt bundle and consolidation receipt types is the next integration task.

### Treatment-effect transport semantics (MACE v0.6)

Transport compares the causal contrast `mean(A) - mean(B)` between regimes. It does not require each action's absolute outcome mean to remain stable. This avoids rejecting a portable action effect when an environment-wide baseline shift moves both outcomes equally.

For source regime `s` and target regime `t`, the deterministic worst-case check is:

`abs((mean_A_s - mean_B_s) - (mean_A_t - mean_B_t)) + err_A_s + err_B_s + err_A_t + err_B_t <= 0.12`

The target must contain a witness for each action; every contributing source regime must contain independently qualified evidence for both actions. Root and family digests must be disjoint across source/target and across contributing source regimes. The aggregated effect is an equal-weight average of regime contrasts, avoiding raw-outcome pooling bias from unequal action counts. If its error interval crosses zero, the policy abstains.

This remains a deterministic prototype: measurement bounds and root digests require independent attestation, and the fixed equivalence margin is not a learned or statistically calibrated confidence level.


### Raw-native treatment-effect transport integration (MACE v0.6)

The consolidation integration suite now has a Session-A → Session-B transport control built from raw `RawInteractionV1` records. Session A is ingested at an explicit (valid_at, known_at) snapshot, grouped into action-/scope-specific evidence packs, promoted through proposal-only consolidation, and released only after an independent receipt verifier accepts the exact source bindings. Session B contributes one receipt-verified witness per action under a shifted regime with a +0.15 common-mode baseline shift. Exact-scope policy abstains; invariant transport uses the stable A-B treatment effect.

The fixture verifier is synthetic. This validates crate composition and receipt/evidence interfaces, not live Mycelix cryptographic verification or physical causal inference.
