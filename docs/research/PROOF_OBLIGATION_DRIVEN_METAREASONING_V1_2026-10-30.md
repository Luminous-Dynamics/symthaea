# Proof-Obligation-Driven Metareasoning (POD-Meta) V1

**Date:** 2026-10-30  
**Status:** research/integration protocol. This document adds no runtime behavior and reports no Symthaea benchmark results.

## Executive thesis

Symthaea already has two architectural directions that should be composed rather than duplicated:

- **Proof-carrying reasoning:** typed claims, premises, alternative justifications, derivation witnesses, source revisions, and independent verification.
- **Metareasoning:** choosing which internal cognitive operation is worth running next under uncertainty, risk, deadline, and resource limits.

The missing connection is a proof-obligation-driven scheduler:

> Select the next cognitive operation by asking which unresolved proof obligation it can discharge, how much decision risk that discharge is expected to remove, whether the operation is admissible, and what it costs.

This is more specific than selecting the most salient memory, the highest-confidence thought, or the next token. The scheduler should work on **semantic proof progress**, not the volume of generated reasoning text.

The core product is a sequence of independently checkable state transitions:

`query -> obligations -> candidate operations -> operation receipt -> proof-graph delta -> independent verification -> stop/continue`

A model may propose claims, evidence candidates, or next operations. It may not declare its own derivation verified, authenticate its own source, or grant itself action authority.

## Why this is a distinct next tranche

This proposal composes existing work and should not create a second general scheduler or proof engine:

| Existing owner | Already covers | POD-Meta contribution |
|---|---|---|
| #5468 COG-META-001 | Multi-timescale metareasoning architecture | Supply a proof-obligation state as a first-class scheduler input |
| #5469 / #5470 | Inventory and typed cognitive bids/decisions | Add a typed obligation-to-operation projection after the inventory and contract are qualified |
| #5471 | Shadow value-of-computation scheduler and finite stopping receipts | Measure marginal value per discharged obligation, rather than generic operation utility alone |
| #5475 / #5476 | Budget scaling and prospective VOC calibration | Calibrate by obligation and operation family; retain unknowns and counterfactual gaps |
| #5478 | Causal metareasoning ablations | Isolate the contribution of proof-aware scheduling |
| #9141 | Derivation witness extraction in Broca | Supply dependency-first support witnesses to the proof workspace |
| #9177 | Minimum-cost sufficient evidence packets | Reuse selected proof packets as possible operation inputs/results |
| #9075 / #9149 | Bitemporal evidence and governed retrieval integration | Preserve valid-time, known-time, lifecycle and scope constraints |
| `symthaea-legal-reasoning` | Typed inference, proof graph, conflicts, temporal rules | Initial deterministic oracle and formal proof substrate |

The scheduler must not replace `AttentionBid`, workspace competition, tool authority, or evidence authentication. A proof graph establishes what follows from its formalized premises; it does not establish that those premises are true or that their sources are authentic.

## Research update: what recent latent-reasoning work implies

The competitive comparison is now not simply textual CoT versus recurrent HDC.

- **Latent Recurrent Thoughts (EMNLP Findings 2026):** a task proposer and small recurrent reasoner refine continuous latents for a frozen language-model decoder. The paper reports outperforming non-thinking CoT prompting on the same backbone in its evaluated symbolic and language tasks, with answer supervision but no reasoning traces. This is a close architectural baseline for a Symthaea latent planner.
- **LOTUS (June 2026):** trains looped latent blocks with parallel gold CoT-step-token supervision. The authors report reaching explicit-CoT performance at 3B in tested settings and 2.5–6.9× lower thought-phase latency. The important lesson is that latent recurrence needs a task-relevant learning signal.
- **Google's Latent Reasoning with Supervised Thinking States (2026):** adds a recurrent pathway trained through teacher-forced supervised thinking states, avoiding a BPTT requirement in the described method. This is especially relevant to a Broca training path whose current CPU/GPU update rules do not provide temporal BPTT; it is a candidate comparator, not a ready-made drop-in for HDC/CfC.
- **Think-at-Hard (ICML 2026):** identifies overthinking in looped transformers and learns to selectively allocate latent iterations. The paper reports 3.8–4.4 percentage-point gains over always-iterate baselines while skipping iterations for 93% of tokens in its evaluated settings. This argues for measuring the value of a particular reasoning operation, not assuming that deeper recurrence is always beneficial.
- **TEMPO (EMNLP 2026):** measures temporal completeness as well as multi-hop retrieval. It contains 1,730 complex queries across 13 domains and 3,976 decomposed steps, with Temporal Coverage@k / Temporal Precision@k; its simplified leaderboard lists DiVeR at 32.0 NDCG@10 and 71.4% Temporal Coverage@10, while ReasonIR has 72.4% coverage in that snapshot. Query-to-step evaluation has a separate leaderboard, so do not conflate those figures. POAR should complement this suite rather than replace it: https://aclanthology.org/2026.emnlp-main.291/
- **TEMPS (EMNLP 2026):** a temporal-retrieval model that adds a temporal representation branch to a frozen semantic retriever. Its paper reports MRR gains across three temporal benchmarks and TS-Retriever R@1 of 25.39 versus the prior 19.92. Include a TEMPS-style temporal-ranker arm or its official checkpoint where license/compute allow; TEMPS is a method comparator, not the same dataset/metric as TEMPO: https://aclanthology.org/2026.emnlp-main.2657/
- **LongCoT (ICML 2026):** uses 2,500 expert-designed problems with dependency graphs spanning tens to hundreds of thousands of reasoning tokens. It directly stresses dependency preservation and recovery.
- **GRACE (EMNLP 2026):** provides step-level faithfulness labels separating inference mistakes from grounding mistakes.

These are published results in their own experimental settings. None is a head-to-head result against Symthaea, and none implies that a typed proof graph automatically wins.

POAR evaluation hygiene:
- Publish a Contamination Transparency Card at `research/pod-meta-lab/CONTAMINATION_TRANSPARENCY_CARD_V1.md`. It declares that public deterministic templates are exposed, third-party model training provenance is unknown, T1-T4 audits have not run, and the current fixture suite is not a clean public leaderboard set.
- Follow the GEM 2026 contamination taxonomy (T1 exact, T2 syntactic, T3 semantic, T4 task-level) and disclose corpus/cutoff, deduplication, decontamination, evidence, instruction-tuning/post-training exposure, prompts, few-shot settings, scoring, and run variance. No contamination detector reliably covers all tiers: https://aclanthology.org/2026.gem-main.50/
- Current graph generator stratifies chain depth (8/32/64) across development/calibration/test, uses split-disjoint source IDs and family IDs, and tests same-seed reproducibility. The 64-chain is held-out test difficulty, but rule/operator families remain largely shared; full family-held-out test generation is not yet done.
- Retraction mutation now explicitly tests both dependency invalidation and unaffected-claim stability.

References:
- Chen & Fu, Latent Recurrent Thoughts: https://aclanthology.org/2026.findings-emnlp.578/
- Fan et al., LOTUS: https://www.microsoft.com/en-us/research/publication/bridging-the-gap-between-latent-and-explicit-reasoning-with-looped-transformers/
- Google Research, Latent Reasoning with Supervised Thinking States: https://research.google/pubs/latent-reasoning-with-supervised-thinking-states/
- Fu et al., Think-at-Hard: https://proceedings.mlr.press/v306/fu26t.html
- Motwani et al., LongCoT: https://proceedings.mlr.press/v306/motwani26a.html
- Pham et al., GRACE: https://aclanthology.org/2026.emnlp-main.1582/
- Latent Recurrent Transformer: https://aclanthology.org/2026.findings-emnlp.2281/

## 1. The control object: an unresolved proof obligation

A proof obligation is not a vague confidence score. It is an exact, query-scoped requirement whose discharge can be independently evaluated.

Conceptual record (names are illustrative; do not add these types until #5469 identifies the correct owner and #5470 freezes the shared ABI):

```text
ProofObligationV1
├── obligation_id
├── decision/query identity
├── target claim and polarity
├── required support class
├── dependency / parent obligation IDs
├── scope and valid_at / known_at coordinates
├── admissible evidence policy/profile
├── state
├── verifier requirement
├── consequence if unresolved/incorrect
└── canonical obligation digest
```

State is typed, not a scalar:

```text
Open
CandidateEvidenceAvailable
SupportedUnderProfile
Contested
Contradicted
StaleOrRetracted
OutOfScope
Unauthorized
VerifierRejected
UnresolvableWithoutExternalInput
BudgetInsufficient
Abstained
```

Do not collapse these into `confidence = 0.7`. In particular:
- `SupportedUnderProfile` means the selected formal/evidence profile's obligations passed; it does not imply universal truth.
- `Contested` is not half-supported.
- `UnresolvableWithoutExternalInput` is not a reason to keep running the same internal operation.
- `BudgetInsufficient` must not become silent truncation or false completion.
- An obligation's consequence severity affects scheduling priority, not proof validity.

## 2. Obligation-to-operation routing

The proof workspace describes what remains unresolved. COG-META decides whether and when to spend resources on an operation that might resolve it.

| Obligation condition | Candidate cognitive operation | Required result receipt |
|---|---|---|
| Required premise missing | Retrieve / query specialist / seek evidence | Exact source revisions, spans, cost, and eligibility |
| Candidate premise lacks provenance | Inspect source / verify lineage | Source digest and provenance validation |
| Derivation fails independent check | Repair derivation / query rule pack / run verifier | Failed edge/reason and candidate corrected proof |
| Contradictory positive and negative support | Seek independent evidence / evaluate alternatives | Conflict preserved or explicit resolution under a named policy |
| Evidence is stale, superseded, or retracted | Refresh / re-query bitemporal ledger | Valid-time and known-time eligible revisions |
| Evidence falls outside scope or authority | Request clarification / alternate source | Explicit reason; no support admission |
| Alternative support exists | Compare proof packets / verify one or more paths | Selected and rejected support-set identities |
| No internal operation can resolve the obligation | Request external input / abstain / defer | Explicit stop reason |
| Required support already verified | Stop or proceed to dependent obligations | Verified proof slice and remaining obligation set |

One operation may support multiple obligations; one obligation may require multiple independent operations. Preserve that graph rather than forcing a single linear CoT.

### Cost-aware obligation selection

The scheduler may rank operations using an explicitly profiled estimate of:

- expected decision-risk reduction if the targeted obligation is discharged;
- likelihood the operation actually resolves it;
- uncertainty / calibration of that estimate;
- source independence and admissibility constraints;
- estimated work, wall time, energy, context tokens and specialist occupancy;
- deadline pressure and consequence of delay;
- whether an equivalent attempt has already run without semantic progress.

Do not create one universal scalar that mixes truth, urgency, moral value, compute cost and authority. Apply hard admissibility and mandatory-budget constraints first; then use a declared policy to shortlist or rank eligible operations.

A useful research quantity is:

`marginal value = expected downstream decision-risk reduction / expected resource cost`

but this is only an estimator. It must be calibrated prospectively against frozen outcomes; it is not a truth score and is not a theorem that the operation is worth performing.

## 3. Proof progress is the stopping signal

Token count, cycle count, another LLM response, a changed timestamp, and a recomputed confidence scalar are not semantic progress.

Progress must be bound to the proof/query state, for example:
- a previously open obligation gains an admissible source;
- a source revision is rejected as stale or unauthorized;
- a proof edge is independently verified or rejected;
- a contradiction is localized to a specific premise/rule;
- a new alternative proof is found and verified;
- a dependent claim becomes supported or contested;
- a counterfactual mutation correctly invalidates or preserves descendants;
- the system establishes that only external evidence or user clarification can resolve the uncertainty.

Define a repeat key from the operation type, target obligation, relevant evidence-state digest, rule/profile identity and input-context digest. Repeating the same operation against the same state without a new receipt is not progress.


### Receipt freshness and anti-replay invariant

An operation receipt is valid only against the exact current obligation/evidence state, not merely because its embedded verifier receipt repeats the same caller-supplied key. Every targeted obligation must carry a current state identity equal to the operation's input `progress_key`. A state-changing verifier result must carry a distinct `result_state_identity`; applying the result advances the targeted obligations to that identity. This makes a previously valid receipt inapplicable after a later correction, revocation, scope change, or other state transition. Missing state identities fail closed. The research lab uses opaque synthetic strings for this mechanism; they are not cryptographic digests or authenticated evidence. A production adapter must derive identities from canonical content and all relevant proof/evidence/profile dependencies, and must authenticate the verifier receipt independently.

Stop conditions must remain finite and typed:
- all required obligations discharged under the declared profile;
- no applicable operation can produce new admissible information;
- no candidate operation has positive *qualified* expected value;
- the remaining obligation requires external evidence or clarification;
- the decision is stable under the predeclared bounded alternatives;
- mandatory/compute/deadline budget reached.

Never promote an answer just because the budget expired.

## 4. Proof-obligation adaptive reasoning benchmark (POAR)

POAR measures whether proof-aware scheduling achieves better correctness/resource trade-offs than generic adaptive reasoning. It is a custom benchmark, not a substitute for LongCoT, GRACE, or official memory benchmarks.

The research crate now includes a public deterministic POAR-v1 conformance runner with 16 scheduler/receipt fixtures. In addition, research/pod-meta-lab/src/graph.rs adds an AND/OR proof-and-evidence graph suite and a separate recursive oracle. The planner-facing records expose evidence assertions, source families, independence groups, valid-time / known-time intervals, scope, authority, integrity, and lifecycle; the evaluator's source-truth labels remain private to GraphSuite. The initial generator has fixed templates (not stochastic natural-language generation), split-scoped source families and seed-specific source IDs, a late-correction pair, correlated-source echo, transitive retraction mutation, scope/authority mismatches, contested evidence, cycles, future-known leakage, 8/32/64-claim development/calibration/test dependency chains, false asserted content, and unverified-source cases. Retraction mutation checks both dependent-claim invalidation and unaffected-claim stability. Its mutation API can alter retraction state, asserted polarity, known/valid-time boundaries, scope, authority, integrity, and independence-group identity without changing the hidden oracle labels; paired mutation tests verify that the evaluator catches these changes. Independence thresholds are evaluated over each alternative's direct evidence requirements; dependent claims are independently evaluated under their own proof alternatives. The oracle returns per-case metrics rather than a composite score. It also scores a public ranked evidence list separately from proof citations using synthetic temporal evidence coverage@3 and precision@3; a correction fixture keeps recall complete while ranking a stale revision ahead of the current one reduces precision. A deterministic same-seed/same-split test checks reproduction. Development/calibration/test chain depths are stratified at 8/32/64 claims; source IDs and source-family IDs are disjoint between splits, and source IDs also change across seeds. A retraction control verifies invalidation of dependent claims and stability of an unrelated claim. These are synthetic conformance fixtures, not a trained-model benchmark; a confirmatory evaluation still needs an independently deployed evaluator, access-controlled oracle labels, held-out source/rule families, mutation tests, and matched model baselines.

### Deterministic task families

1. **Single missing premise:** one cheap retrieval completes a valid proof; more generation adds no support.
2. **Low-salience decisive source:** conventional top-k relevance misses the only required premise; proof-directed retrieval finds it.
3. **Deceptive plausible derivation:** one independent verifier pass catches a deliberately invalid edge.
4. **Overthinking trap:** the proof is already complete; further reasoning sometimes damages the answer or wastes budget.
5. **Contradictory evidence:** repeated retrieval cannot resolve the conflict; a different operation or abstention is required.
6. **Alternative proof:** two independent OR-of-AND support paths exist; one is stale or expensive while another remains valid.
7. **Late correction:** an old-valid-time correction arrives at a later known-time; historical as-known query must not leak it backward.
8. **Revocation cascade:** retracting a terminal source must invalidate dependent conclusions while preserving unrelated claims and alternative support.
9. **Scope/authority mismatch:** a semantically relevant source is not admissible for the target claim or principal.
10. **Budget boundary:** the exact sufficient packet exceeds the token/work budget by one unit; return `BudgetInsufficient`, not an incomplete proof labelled complete.
11. **Shared evidence echo:** multiple agents or memory summaries repeat the same source; support independence is not fabricated.
12. **No-information case:** internal reasoning cannot resolve a missing physical/empirical fact; stop and seek external observation.
13. **Long dependency chain:** local steps are tractable, but evidence dependencies must be retained over increasing chain depth.
14. **Null-benefit control:** the generic scheduler's additional complexity should not improve an already trivial/cached task.
15. **Stale receipt replay:** a valid proof receipt is followed by a newer revocation/correction; replaying the old receipt must be rejected against the current state identity, without mutating any obligation in a multi-target batch.
16. **Unchanged-state loop:** repeating the same operation against the same evidence-state identity must terminate with an explicit no-progress stop.

The task generator and independent oracle must keep hidden proof labels, hidden outcomes, and source-truth labels outside the runtime planner input. Hold out rule-template families and source families, not just random instances.

### Treatment arms

**Core deterministic arms**
- P0: no extra reasoning; direct query/answer.
- P1: fixed small budget.
- P2: fixed large budget.
- P3: generic COG-META scheduler without typed proof-obligation features.
- P4: proof-obligation-aware scheduler using fixed deterministic routing.
- P5: proof-obligation-aware scheduler plus prospectively calibrated value-of-computation estimates.
- P6: oracle operation selector (synthetic tasks only; diagnostic upper bound, never a deployable baseline).

**Language-mediated same-decoder arms**
- D0: direct answer.
- D1: explicit textual CoT.
- D2: explicit CoT + independent verifier.
- D3: structured proof planner + frozen decoder.
- D4: generic latent refinement + frozen decoder.
- D5: proof-obligation-driven latent refinement + proof workspace + same frozen decoder.
- D6: best reproducible matched latent baseline (LRT/LOTUS/Google supervised thinking states where weights, implementation, and licenses permit).

Freeze decoder, tokenizer, task inputs, prompt, context, output format, tool permissions, seed schedule and total inference budget across relevant arms. Compare native Broca generation separately; a Genesis-seeded native model and a pretrained LLM are not a same-decoder comparison.

### Primary metrics

Report independently:
- final-answer accuracy and correct abstention coverage;
- required-obligation recall and false-closure rate;
- proof node/edge precision, recall and F1;
- independent derivation acceptance;
- evidence support precision and provenance/authorization correctness;
- dependency retraction recall and unaffected-claim stability;
- source-family generalization and split leakage checks;
- AND/OR proof-path correctness, cycle rejection, and witness independence-group count;
- temporal evidence retrieval coverage@3 / precision@3, separately from final proof citations, across valid-time and known-time windows, including future-known leakage;
- scope, authority, source-integrity, revocation, and citation-completeness violations;
- stale-receipt replay rejection and transactional all-or-nothing behavior for mixed-currentness multi-obligation receipts;
- contradiction detection and resolution/abstention quality;
- semantic-progress efficiency (verified obligations discharged per resource unit);
- scheduler regret against the synthetic oracle;
- unnecessary-operation rate and overthinking regression rate;
- p50/p95 latency, decoder tokens, recurrent iterations, tool calls, CPU/GPU time, peak memory, energy and storage/retrieval cost;
- prospective VOC calibration (Brier/log loss for binary outcomes; MAE for cost/latency; calibration by operation family);
- risk-coverage curves and performance under task/source/profile shift.

Do not blend these into one intelligence score. Report each task family and proof profile separately.

### Proposed gates to freeze before confirmatory evaluation

These are hypotheses, not current Symthaea results:
- 100% of conclusions labelled `SupportedUnderProfile` pass the independent verifier in deterministic formal fixtures.
- 100% of state-changing receipts match the current obligation-state identity and advance to a distinct result-state identity; stale receipt replay never reverses a newer revocation/correction.
- 0 revoked, unauthorized, out-of-scope, or future-known-at evidence references in emitted accepted proof packets.
- 0 incomplete proof packets represented as complete.
- At least 99% dependency-retraction recall and 99% unaffected-claim stability in the synthetic mutation suite.
- Same-decoder answer accuracy no more than 2 absolute percentage points below explicit CoT, with the full cost/accuracy Pareto curve published.
- At matched accuracy and coverage, at least 20% lower all-in inference cost than the strongest eligible non-proof adaptive scheduler, with paired cluster-aware uncertainty excluding zero. This is a stretch promotion target, not a requirement to hide an inconclusive result.
- No positive promotion claim if any mandatory proof-integrity or authority invariant fails.

Tune operation-value predictors only on development/calibration splits. Bootstrap by source/rule-template family or independent task seed, not by near-duplicate generated instance. For stochastic policies, use at least eight disjoint seeds and preserve each run receipt.

## 5. Causal ablations

To prove that proof-obligation awareness causes the gain, not a stronger reader or more compute, compare:

| Treatment | Change |
|---|---|
| Full system | Obligation map + proof-aware routing + calibrated scheduler |
| Obligation lesion | Hide typed obligation features from scheduler but keep proof verifier |
| Routing permutation | Deliberately permute obligation-to-operation mapping |
| No-progress lesion | Record progress telemetry but disable its consumption by the stop policy |
| Generic scheduler | Use COG-META inputs without proof-obligation state |
| Random cost-matched | Select eligible operations randomly at matched cost |
| Cheap deterministic baseline | Fixed ordering, bounded attempts |
| Proof-verifier lesion | Preserve planner but disable independent proof checking (diagnostic negative control only) |

Hold exact candidate operations, budgets, evidence corpus, reader/decoder and downstream evaluator fixed. Where possible, use paired replay branches from identical state snapshots. A gain that only appears because the proof-aware arm receives more compute or better retrieval inputs is not a scheduler result.

## 6. External benchmark mapping

Use official task/evaluator versions and keep benchmark-specific scores separate:

- **LongCoT:** long dependency and state retention; POAR should not be presented as LongCoT replacement.
- **GRACE:** whether the final natural-language explanation is supported by the actual proof graph; separately label grounding versus inference errors.
- **FaithCoT-Bench (ICLR 2026):** add instance-level CoT unfaithfulness detection. Its release describes more than 1,000 expert-annotated trajectories from four models and four domains, with more than 300 unfaithful instances. This tests whether an apparently plausible CoT faithfully reflects the decision process; it complements rather than replaces GRACE's step-level grounding and inference labels: https://proceedings.iclr.cc/paper_files/paper/2026/hash/6c7154e394e24c69409256ccf8bf0804-Abstract-Conference.html
- **LongMemEval-V2:** use the official evaluator for long-horizon agent memory. It consumes long histories of multimodal web-agent trajectories and tests whether a memory system returns compact evidence for downstream answers; report answer accuracy and query latency: https://github.com/xiaowu0162/LongMemEval-V2
- **TimlMemBench:** retain canonical-pool and raw-native modes separately. It tests memory revocation, supersession, restriction, and activation/usage decisions; its published package reports 208 reviewed samples, 1,566 sessions, 1,715 questions, and 1,821 governance/activation cases: https://aclanthology.org/2026.findings-emnlp.507/
- **Faithfulness survey:** use *The Mirage of Explainability* for its taxonomy of post-hoc rationalization, sycophancy, internal alignment and external consistency. It motivates intervention-based trace tests; it is not itself a benchmark result: https://aclanthology.org/2026.emnlp-main.614/
- **Formal proof suites:** require acceptance by an independent kernel (for example, Lean) for theorem-proving tasks; keep theorem proving separate from source-grounded empirical reasoning.

### Benchmark comparability rule

Do not use a single composite leaderboard as the primary SOTA claim. Public leaderboards mix different model snapshots, benchmark coverage, evaluation protocols, and sometimes provisional versus source-verified evidence. Pin each benchmark's official dataset revision, evaluator, model/checkpoint, context and inference budgets, and evaluation date. Report a separate score and cost vector for each suite. Missing results are not zero, and scores from different benchmark definitions must not be averaged into one "reasoning" number.

Use leaderboard tables as discovery aids only. Promotion requires a reproducible run against the named official evaluator and a paired comparison under matched inputs and resources.
- **MemDerive / evolving-state tasks:** intermediate derivation and support graph correctness, including incremental updates.
- **MiniF2F / Lean or another formal checker:** formalizable theorem proof acceptance, kept separate from empirical source-grounded claims.

A custom synthetic win establishes only that the scheduler handles the declared proof/task fragment. Generalization claims require official external datasets with their own licenses, splits and evaluators.


## 6A. October 2026 frontier: what the scheduler must beat

The latest latent-reasoning literature changes the fair baseline. A POD-Meta comparison must include adaptive iteration, not only fixed-depth latent reasoning:

- **Think-at-Hard (ICML 2026):** selective latent iteration reports 3.8–4.4 percentage-point gains over always-iterate baselines at identical parameter counts while skipping iterations on 93% of tokens; its paper also reports higher gains with small additional parameter overhead. This makes per-token adaptive compute a mandatory comparator, not an optional future arm: https://proceedings.mlr.press/v306/fu26t.html
- **Latent Recurrent Thoughts (Findings EMNLP 2026):** a small recurrent reasoner proposes/refines continuous latent thoughts while a large decoder remains frozen; the paper reports improvements over prior frozen-decoder methods and non-thinking CoT under its matched settings: https://aclanthology.org/2026.findings-emnlp.578/
- **LOTUS (June 2026):** parallel supervision over latent thought blocks reports 2.5x–6.9x thought-phase latency reduction and closes the reported explicit-CoT gap at 3B in its evaluated tasks: https://www.microsoft.com/en-us/research/publication/bridging-the-gap-between-latent-and-explicit-reasoning-with-looped-transformers/
- **Decoding Self-Knowledge from Latent Reasoning Models (Findings EMNLP 2026):** its probe results distinguish problem-level from rollout-level self-knowledge; it reports useful selective prediction/adaptive routing, but weak trajectory ranking among rollouts of the same problem. This is a warning against treating a model's own confidence as proof that a particular reasoning trajectory or token is correct: https://aclanthology.org/2026.findings-emnlp.1187/
- **Latent Recurrent Transformer (Findings EMNLP 2026):** a separate paper from Latent Recurrent Thoughts. It reuses a prior token's high-level hidden state as cross-token recurrent memory while preserving the KV-cache interface and one forward per generated token; its reported experiments use about 2x ideal token compute in training and show 9% decoding-latency overhead over a standard Transformer. This is a relevant architecture comparator for temporal latent state, not a drop-in proof system: https://aclanthology.org/2026.findings-emnlp.2281/

These are paper-reported results under each paper's own datasets, model sizes and protocols. They are not directly comparable to POAR and are not Symthaea measurements. Reproduce the official settings before ranking against them.

### Updated matched arms and stretch targets

Add two arms to the language-mediated comparison where code, model weights, license, and compute permit:
- **D7: selective latent iteration** (Think-at-Hard-style token/step gating), with iteration decisions logged before outcomes;
- **D8: latent self-knowledge adaptive routing**, reporting problem-level selective prediction separately from within-problem rollout selection;
- **D9: Latent Recurrent Transformer-style cross-token recurrent memory**, matched by effective training/inference compute and a frozen decoder where available. Do not conflate this architecture with Latent Recurrent Thoughts; report latency overhead and KV-cache compatibility separately.

Add metrics for optional-iteration skip rate, iteration-induced regression rate (a correct pre-iteration result made incorrect by another loop), and decision-quality change per joule / wall-clock second. At matched answer accuracy, coverage and proof-integrity gates, target at least 90% optional-iteration avoidance on the predeclared easy/cached slice; treat this as a stretch target, not an expectation across hard cases. The key ablation is whether proof-obligation deltas improve *which* operation/iteration is chosen beyond latent confidence alone.

The distinctive hypothesis is **proof-carrying adaptive computation**: latent dynamics may propose the next state or operation, but a versioned obligation/evidence graph determines whether an iteration is useful, and an independent verifier determines whether the resulting claim can be promoted. This aims to combine selective compute with grounded state transitions rather than compete on hidden-state depth alone.

## 7. Implementation sequence and ownership

1. Preserve the existing COG-META prerequisites: finish/qualify #5469 inventory before #5470 chooses type ownership; do not bypass the dependency chain with speculative public types.
2. Implement a **research-only obligation adapter** from existing proof/query outputs to the existing shadow-scheduler input. Keep it behind a feature/test boundary; it cannot change production routing.
3. Keep the public POAR-v1 runner as a deterministic contract smoke suite; use the new AND/OR graph generator and private oracle as a first synthetic baseline, then harden it with graph-template mutation tests, source-family holdouts, and an evaluator process boundary before any confirmatory model comparison.
4. Add proof-directed operation routing in a standalone evaluation harness: obligation -> eligible operation set -> receipt -> proof delta.
5. Compare P3 generic COG-META against P4 fixed proof routing before fitting any value predictor.
6. Only then add P5 prospective VOC calibration using #5476, with prediction-before-outcome binding and explicit missing-counterfactual states.
7. Evaluate external tasks using LongCoT, GRACE, TimlMemBench and LongMemEval under matched decoders/costs.
8. Integrate latent HDC/CfC or Broca as a proposer after the proof/control loop demonstrates utility. Keep the existing Temporal Benchmark V2 prerequisite for any temporal-credit change.
9. Consider limited internal enforcement only through #5477 after shadow, calibration, stopping and rollback requirements have been independently qualified.

## 8. Failure conditions and claim ceiling

Stop or simplify the design if:
- generic COG-META matches proof-aware scheduling at equal cost;
- proof obligations add latency without improving accepted-case risk or correctness;
- operation-value forecasts do not calibrate out of sample;
- the task generator leaks target structure or admits shallow shortcuts;
- source-family shift destroys performance;
- the scheduler repeats the same operation on unchanged evidence;
- a proof/verifier or bitemporal failure can be overridden by a high latent score;
- proof-aware scheduling improves synthetic tasks but not held-out external tasks.

A successful result may support this narrow claim:

> Under declared task families, evidence policies, and resource budgets, proof-obligation-driven scheduling achieves a measured capability/resource improvement over specified generic and fixed-schedule baselines while preserving independent proof and authority invariants.

It does not establish that Symthaea beats frontier LLMs at general reasoning, that latent states expose their causal computation, that formalization equals truth, or that the system is conscious.
