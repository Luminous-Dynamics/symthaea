# MACE model-context exposure receipt v0.1

**Status:** protocol and JSON Schema proposal. This increment defines a harness-captured receipt contract; it does not yet implement a provider interceptor, signer, verifier, trusted runtime, or causal influence estimator. No benchmark result is claimed.

## Why this boundary is missing

The current memory-to-decision protocol distinguishes retrieved event IDs from decision-support event IDs and keeps evaluator scores outside the candidate stream. That is necessary, but it leaves an important gap:

- A retriever may return a record that the policy never admits.
- An admitted record may never be rendered into the final model request.
- A rendered record may be truncated, transformed, copied into a summary, or placed in a tool message.
- A model may report that it relied on an event that was not actually in its request.
- A model may be affected by a memory-derived summary without naming its parents.

A model-generated citation or a candidate-declared support list is not proof of what crossed the model boundary. The missing primitive is a receipt over the exact serialized request body, with byte-range mappings from transmitted spans back to canonical evidence and verified ancestry.

## Scope and relationship to existing owners

This is a narrow protocol seam that composes with, and does not replace:

- #13128: candidate/evaluator separation, intervention records and paired arms.
- #13141: the executable paired memory-to-decision runner.
- #13264: verified transitive evidence lineage and common-cause closure.
- #13284: decision sufficiency, cost/regret frontiers and active acquisition.
- #13319: poisoning, propagation, consequence and repair qualification.
- #13681 / #13683: experiment commitment and externally witnessed checkpoint qualification.

No new memory store, evidence registry, source authority, lifecycle controller, evaluator, witness service, or action authorization system is introduced.

## Receipt semantics

One receipt covers one model invocation. A multi-call agent trajectory emits one receipt per invocation, ordered by `invocation_sequence`, with every receipt bound to the same run, pair block, world cluster and query as appropriate.

The receipt binds:

1. The DecisionNeed and candidate policy digests.
2. The exact source-registry, lineage and lifecycle profiles used at admission.
3. Renderer, provider transport adapter and model revision identities.
4. The SHA-256 of the exact UTF-8 provider request body captured after serialization and immediately before network send.
5. Every memory-originated byte span in that body, mapped to a canonical evidence ID, payload digest, dependency closure, transformation receipt, source/authority/admissibility receipts and current-at-use revalidation receipt.
6. Explicit admission denials, so retrieval success cannot be confused with admissibility.
7. The provider response digest and the candidate decision-trace digest.
8. A signature from the configured context-observer identity, verified against an independently pinned observer profile.

The `span_start_byte` is inclusive and `span_end_byte` is exclusive, measured in UTF-8 bytes of the exact serialized request body whose digest is `provider_request_body_sha256`. The verifier must check bounds, recompute each span digest, reject overlaps that cannot be explained by the renderer's declared segment map, and ensure that every memory-origin span is enumerated exactly once.

For source evidence that is sent verbatim, the transformation receipt is an identity-transform receipt. For summaries, merged chunks, cached answers, tool projections or model-generated derivatives, the transformation receipt must bind the exact output digest and all required parent IDs/digests. Unknown ancestry is not an admissible substitute.

## Three statements that must never be conflated

- **Retrieved:** the memory subsystem returned an item.
- **Exposed:** the exact item or a derived span was transmitted in a model request under the recorded admission receipts.
- **Causally influential:** changing the availability/content of that evidence changes downstream outcomes under a controlled intervention.

This receipt establishes only the second statement, subject to the observer and transport trust assumptions. The first remains a retrieval diagnostic. The third must be estimated by #13141's randomized paired interventions and evaluator-only counterfactual replays. A model's `model_declared_support_event_ids` is retained as self-report only; it is not an authority or causal proof.

## Trust and limits

A valid Ed25519 signature proves that the configured signing key signed the receipt bytes. It does not, by itself, prove that the observer captured every request span or that the key is independent of the candidate process. The verifier must pin the observer profile outside the candidate-controlled memory/log/backup domain and qualify the capture implementation.

This v0.1 contract captures the client-side provider request. It cannot see hidden system prompts, provider-side prompt rewriting, internal retrieval, model routing, or transformations performed after the provider receives the request. Those require a provider-signed ingress receipt or a separately attested serving boundary; absent that evidence, the claim is explicitly limited to the transmitted client request.

The signature is Ed25519 over:

`SHA-256("MACE-CONTEXT-EXPOSURE-V1\\0" || JCS(receipt_without_signature))`

where `receipt_without_signature` is the complete receipt object with the top-level `signature` member removed. The `key_id` is resolved in the independently pinned observer profile; this schema does not define a new key registry or trust root.

## Required verifier checks

1. Strictly parse the receipt; reject duplicate/unknown fields, malformed digests, invalid signature encoding, duplicate exposure IDs and impossible byte ranges.
2. Verify the Ed25519 signature using a key authorized by the independently pinned observer profile.
3. Verify the observer profile, transport adapter, renderer, model and clock profile digests against the run manifest.
4. Recompute the request-body digest; validate each span hash and ensure full memory-origin span coverage.
5. Resolve every evidence ID to exactly one canonical source record and verify its payload digest.
6. Recompute/verify dependency closure and transformation receipts; reject cycles, dangling parents, ambiguous aliases and unverified roots.
7. Verify source authentication, authority, rights, scope, estimand and current lifecycle at the decision-consumption boundary.
8. Ensure every exposed span is admitted; a denial may not appear in the transmitted memory spans.
9. Bind the provider response and candidate decision trace to the same invocation and run.
10. Reject a missing receipt for any model invocation that consumed memory-derived content.

Cross-record invariants (unique invocation sequence, no missing calls, one-to-one receipt/decision-trace binding, exposure/denial disjointness, and paired-arm matching) are verifier responsibilities and are not expressible in JSON Schema alone.

## Negative controls

- Retrieved but not admitted; admitted but not rendered; rendered but removed before send.
- Truncated, duplicated, reordered, overlapping or incorrectly offset spans.
- A span hash that does not match the captured request body.
- A summary with a missing, swapped or retracted parent.
- Two evidence IDs that alias one source or share a common cause but are presented as independent.
- Correct but unauthorized content; wrong tenant, wrong DecisionNeed, wrong estimand or rights-restricted content.
- Correction or revocation between retrieval and actual model invocation.
- Stale cache, backup restoration, hidden memory injection through tool output or previous model output.
- Model-declared support that was not exposed; exposed evidence omitted from model-declared support.
- Missing, duplicated, replayed or cross-run receipts; altered request body; invalid/revoked observer key.
- Provider-side transformation without an ingress receipt: this must downgrade the claim, not silently pass.

## Integration and benchmark plan

**First implementation slice:** add a provider-neutral receipt type and verifier to the existing MACE adapter, with a fake transport that captures the exact serialized request bytes. Do not couple the first slice to a live commercial API.

**Second slice:** instrument one model boundary and demonstrate request-span coverage on deterministic fixtures. The observer must be outside the candidate memory process or explicitly labeled as same-process diagnostic.

**Third slice:** add paired intervention cases in which retrieved, admitted, exposed and causally influential are varied independently. Include semantically inert placebo spans and hidden evaluator-only utilities. Keep the current candidate policy/model fixed within each pair.

Report separately:
- retrieval precision/recall;
- admissibility violations among exposed spans;
- request-span coverage and receipt-verification rate;
- model-declared support precision/recall against exposed evidence;
- causal memory utility and harmful influence under paired ablation;
- regret, abstention, tail harm, token/byte/CPU/GPU/latency and all-in cost.

Promotion gates (targets, not results): 100% span coverage on deterministic fixtures; zero inadmissible exposed evidence; zero unverified ancestry counted as independent support; zero replayed/mismatched receipts; and positive held-out paired utility only when the cluster-level interval and pre-registered harm limits pass.

## Claim boundary

A context exposure receipt is not proof that a neural model attended to a particular token, understood a source, or relied on it in a causal sense. It proves, under a qualified capture boundary, which evidence-derived bytes were included in the request delivered by the client transport. Causal influence requires controlled intervention. External anchoring requires the separate MON-005 authenticated checkpoint and proof chain.
