# Broca lifecycle-aware action policy

**Policy version:** BROCA-LIFECYCLE-ACTION-V1  
**Status:** deterministic reference controller; Rust tests authored, not compiled/executed here.  
**Research comparator:** Vera-Life-Eval (GroundLM 2026, 8,500 questions / 41,203 versioned evidence items).

## Why action selection is a first-class memory capability

Retrieval, proof construction and answer generation are not the same operation. A memory can retrieve a passage that is lexically relevant and still be unsafe to use because it is stale, superseded, contradicted, outside the requested valid-time window, not yet known at the requested transaction-time cutoff, unauthorized, or insufficient to prove a required conclusion.

The controller therefore chooses an explicit action before answer generation:

- **ANSWER**: every required conclusion has current support and the exact packet passed independent sufficiency and optimality verification.
- **WARN_STALE**: the only evidence for at least one required conclusion is stale; the system must not present it as current.
- **EXPLAIN_CONFLICT**: current evidence remains contradictory and no accepted resolution selects a supported proof.
- **RETRIEVE_MORE**: retrieval has a declared remaining budget and more evidence may resolve missing support or conflict.
- **ABSTAIN**: required support is missing, untrusted, provenance-invalid, or cannot be proved within the completed search/budget.

This makes epistemic restraint and lifecycle handling measurable rather than hidden inside a prompt.

## Trust boundary

The controller does not infer claim states from natural-language text. The caller constructs one ClaimLifecycleLedgerEntry per required conclusion using the outputs of the trusted bitemporal resolver, source-span resolver, authorization evaluator, contradiction resolver and proof verifier.

A claim ledger entry records conclusion ID; lifecycle state; selected justification ID, if currently supported; and exact current, stale, conflicting and rejected evidence IDs. The policy validates that the evidence sets are disjoint and structurally consistent with the declared lifecycle state.

### Why VerifiedOptimal is not a public boolean

A public status field alone must not authorize answer generation. A caller could accidentally or deliberately set proof_status=VerifiedOptimal without running the independent verifier.

The implementation uses an opaque, non-serializable VerifiedLifecycleProofToken:

1. verify_proof_packet_for_lifecycle(request, result) invokes build_verified_proof_reader_context.
2. That adapter invokes the existing independent proof-budget verifier and validates the exact packet/context.
3. The token binds decision ID, principal, scope, valid_at, known_at, canonical request digest, packet digest, and selected proof/dependency set for every required conclusion.
4. decide_lifecycle_action accepts VerifiedOptimal only when this token matches the lifecycle claim ledger exactly.
5. The serializable lifecycle decision is an audit receipt, not an authority token. The final response gate must retain and present the original in-process token again through answer_is_permitted(Some(&token)). A deserialized ANSWER receipt without the token is rejected.

The token is not a digital signature or remote attestation. If this capability must cross a process/service boundary, replace it with an authenticated, signed verifier receipt and define the key-management / anti-replay protocol; do not simply serialize the current token.

## Deterministic policy order

The policy version defines a fixed precedence:

1. ANSWER only when all required claims are SupportedCurrent, the proof status is independently verified optimal, and the opaque proof token exactly binds every selected proof and dependency.
2. RETRIEVE_MORE when the retrieval controller explicitly reports a remaining call budget. This is preferred to a terminal action while more admissible evidence can still resolve the case.
3. ABSTAIN when any required claim is unauthorized or provenance-invalid.
4. EXPLAIN_CONFLICT when a required claim has unresolved current conflicting evidence.
5. WARN_STALE when only stale evidence can support a required claim.
6. ABSTAIN when a required claim remains missing or the proof packet has not passed independent verification.

The order is intentional and versioned. Any future policy change must receive a new policy version and a paired benchmark comparison; do not silently change the precedence.

## Lifecycle action benchmark plan

The primary lifecycle comparator is Vera-Life-Eval, from the paper *Lifecycle Faithfulness for Grounded Retrieval-Augmented Generation: Benchmarks, Evaluation, and the Vera-Life Controller* (GroundLM 2026, 29 October 2026).

The paper constructs 8,500 questions over 41,203 versioned evidence items:

| Split | Questions | Main failure mode |
|---|---:|---|
| Current | 1,500 | Current evidence should support a direct answer |
| Past | 1,500 | Answer must use the requested historical valid-time |
| Stale-only | 1,500 | No current support; stale evidence must not be presented as current |
| Update | 2,000 | Supersession and evolving facts |
| Conflict | 1,000 | Contradictory current evidence |
| Multi-hop | 1,000 | Lifecycle state must propagate through multiple premises |

The paper's action-conditioned LifeEM credits the correct decision, including WARN_STALE, EXPLAIN_CONFLICT, RETRIEVE_MORE and ABSTAIN, when those actions are the gold outcome. LifeEM is **not numerically comparable** to ordinary LoCoMo or LongMemEval answer accuracy.

### Published reference points

| Vera-Life-Eval reference | Reported result | How Broca should use it |
|---|---:|---|
| Vera-Life controller, Qwen3-8B, 1,500-question stratified subset | 62.3% LifeEM | Matched replication gate |
| Vera-Life controller, full 8,500 questions | 61.1% LifeEM | Full-set replication gate |
| Best prompt-only baseline, same generator | 37.4% LifeEM | Demonstrates lifecycle control beyond prompt-only behavior |
| Timestamp-only regime | 58.4% LifeEM | Tests richer lifecycle metadata beyond timestamps |
| Text-only regime | 50.4% LifeEM | Tests structured lifecycle relations |
| Lifecycle graph ablation | 31.7% LifeEM | Measures value of explicit lifecycle graph structure |
| Full-metadata stale rejection | 100% | Stale-only safety diagnostic |
| Full-metadata over-abstention | 0% | Avoid unnecessary refusal when evidence is actually sufficient |

Broca must reproduce the same generator, prompt, corpus, split, metadata regime and action-scoring protocol before claiming parity or superiority. Paper values are reference points, not Broca results.

### Prospective Broca targets

These are qualification targets, not measured results:

- Reproduce at least 62.3% LifeEM on the stratified split and 61.1% on the full set under the exact Qwen3-8B protocol.
- Achieve zero invalid ANSWER actions in deterministic conformance tests when required support is stale, retracted, unauthorized, cross-scope, provenance-invalid, unresolved-conflicted, or not independently verified.
- Match 100% stale rejection on the stale-only protocol while separately reporting false stale warnings on current and past questions.
- Match the paper's 0% over-abstention on the exact full-metadata split; additionally report answerable-abstention rate and calibration by split.
- Report LifeEM, ordinary answer accuracy, unsupported-assertion rate, citation/proof completeness and action confusion matrix separately.
- Do not tune action thresholds or precedence against hidden evaluation splits.

## Separate retrieval, admissibility and action metrics

A system must not hide a weak stage behind a strong aggregate:

| Stage | Metrics to publish | Failure examples |
|---|---|---|
| Candidate retrieval | strict recall-all@k, any-hit@k, fractional recall@k, MRR/nDCG, multi-session coverage | missed required source, stale candidate retrieved |
| Evidence admissibility | current-valid fraction, bitemporal reconstruction accuracy, scope leakage, authorization rejection, contradiction precision/recall | cross-scope or retracted evidence passed |
| Proof selection | proof sufficiency, independent source-family count, invalid certificate rate, exact packet tokens, optimality verifier acceptance | insufficient premise set, false optimality |
| Action controller | LifeEM, per-action precision/recall, stale rejection, conflict explanation accuracy, over-abstention, retrieval-more efficiency | answer when warning or abstention is required |
| Reader outcome | task answer accuracy, unsupported assertion rate, complete proof citation rate, calibration | fluent but unsupported response |
| Operations | p50/p95/p99 latency, tokens/query, cost/correct answer, storage/event, replay/invalidation cost | correct but too expensive or too slow |

A high-recall retriever can still return stale evidence. A sound verifier can still fail because retrieval missed a required premise. A capable reader can still produce an invalid answer if action policy is weak. Report all stages and their costs.

## Current implementation boundaries

The deterministic controller and claim-ledger contract are authored in memory_lifecycle_policy.rs. The current reference code does not yet extract every factual claim from arbitrary generated prose; render user-facing conflict/stale explanations; invoke retrieval-more autonomously; prove natural-language entailment from selected source spans; persist or remotely sign the opaque verifier capability; or measure LifeEM on Vera-Life-Eval.

The production generation loop is not yet wired to call this controller. No LifeEM score, benchmark parity, production qualification or SOTA claim is made by this reference implementation.

## References

- Vera-Life, *Lifecycle Faithfulness for Grounded Retrieval-Augmented Generation: Benchmarks, Evaluation, and the Vera-Life Controller*, GroundLM 2026: https://aclanthology.org/2026.groundlm-1.18/
- Vera-Life PDF: https://aclanthology.org/2026.groundlm-1.18.pdf
- LongMemEval-V2 protocol: https://xiaowu0162.github.io/longmemeval-v2/
- AgentMemory BM25 LongMemEval-S retrieval component report: https://www.agent-memory.dev/benchmarks

## Versioned synthetic conformance corpus

The file [lifecycle-action-fixtures-v1.json](lifecycle-action-fixtures-v1.json) contains 60 deterministic cases: five non-verified proof outcomes × six claim lifecycle states × two retrieval states (exhausted / continuation available). Each row carries the expected action. The Rust test suite reads this checked-in file using compile-time inclusion and asserts the policy output for every case. A separate real-proof integration test covers ANSWER, which is intentionally excluded from the synthetic table because it must be authorized by an actual independently verified packet and opaque token.

The fixture corpus is a policy-conformance suite, not a substitute for Vera-Life-Eval. It contains no natural-language answer generation, hidden evidence chronology or human annotation; it only makes the deterministic controller truth table reusable and versioned.

## Budget-aware RETRIEVE_MORE decisions

The lifecycle policy's retrieval-progress field can be populated by the separate [RETRIEVAL_VALUE_OF_INFORMATION.md](RETRIEVAL_VALUE_OF_INFORMATION.md) gate. It evaluates lower-bound correctness gain plus lifecycle-risk reduction against upper-bound retrieval harm, measured cost, remaining budget and a minimum net-value margin. Its SHA-256 decision receipt is carried into the lifecycle action digest. Early stopping preserves unused calls and is not reported as budget exhaustion.

Use the versioned [retrieval-value-fixtures-v1.json](retrieval-value-fixtures-v1.json) corpus to validate the deterministic utility gate. The estimate itself must come from a held-out-calibrated upstream estimator; the gate does not train or authenticate that estimator.

## Canonical VOI orchestration

The supported orchestration entry point is `decide_lifecycle_action_with_voi`. It takes the lifecycle claim ledger, the optional opaque proof-verifier capability, a calibrated `RetrievalValueDecision`, and the typed `EvidenceIndexSnapshotReceipt` used by the matching retrieval run. It derives unresolved claims itself and rejects estimates for a different principal, scope, `valid_at`, `known_at`, evidence-index digest or unresolved-claim set. It then embeds the VOI decision SHA into the lifecycle action receipt before selecting the action.

A `RETRIEVE_MORE` decision requires another snapshot-bound retrieval and a fresh value estimate after claim coverage is recomputed. A VOI `Stop` preserves unused call budget and leads to the applicable terminal action—typically `WARN_STALE`, `EXPLAIN_CONFLICT` or `ABSTAIN` when proof is missing. It never upgrades the claim ledger or independently authorizes `ANSWER`.
