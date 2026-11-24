# MACE Search Exposure Receipt v1

**Status:** implemented as a local service-side audit record in the hosted Add/Search prototype. Not an external signature, independent witness, tamper-proof log, or proof of decision causality.

## Purpose

A Search response is an exposure event. To measure whether memory influenced a later decision, MACE must first establish exactly what the reader could have seen. A retrieval log that records only the query or a count is insufficient: it must bind the ordered returned evidence and the exact response bytes.

The hosted service therefore creates a Search exposure receipt inside the same SQLite transaction as the Search snapshot. The receipt must commit before Search returns. If receipt persistence fails, Search fails closed and returns no candidate evidence.

## Receipt fields

The strict JSON payload is defined by `data/schemas/mace_search_trace_receipt_v1.schema.json`.

- `search_id`: unique identity for this retrieval invocation.
- `principal_id` and `tenant_id`: authenticated service boundary.
- `user_scope_sha256`: length-prefixed digest of principal, tenant and exact AML user_id.
- `query_sha256`: digest of the canonical serialized query content.
- `request_sha256`: digest of the canonical serialized Search request.
- `top_k`: requested result limit.
- `ordered_memory_ids_sha256`: digest of the exact ordered returned memory-ID array.
- `response_sha256`: digest of the serialized Search response.
- `retrieval_policy`: versioned ranking/tokenization identity.
- `returned_at`: service time when the response was formed.

The database separately stores the ordered memory-ID array, the receipt payload, and the receipt-payload digest. The receipt payload does not contain raw query text, raw AML user_id, message content or returned memory content.

## Evidence chain and limits

The current local chain is:

`Add transaction → committed source/projection → Search snapshot → receipt commit → handler returns Search response`

The receipt proves the service prepared and committed a digest record for the response before the handler returned. It does **not** prove the network delivered the response to the client, that the reader consumed it, or that it influenced a decision. A later evaluation can correlate a captured request/response with the internal receipt by recomputing the request, ordered-ID and response digests; the receipt alone is not proof of reader exposure.

It does **not** yet establish that a retrieved memory was admitted by Mycelix rights/authority policy, that the reader actually attended to it, that it changed the decision, or that it improved the outcome. Those require separate qualification, reader-use evidence, paired counterfactual intervention and outcome receipts.

The receipt uses SHA-256 commitments. A plain digest is not a signature, authenticity guarantee, anonymization mechanism or external witness. Low-entropy queries and user identifiers may be guessed from their digests. The current database and its audit rows share one rollback/admin domain; a privileged actor can rewrite both. T9/T10 and erasure-resurrection claims require a separately administered monotonic witness and independent verifier.

The Search receipt is internal and is not added to the strict AML response schema. The AML client still receives only the organizer-defined `data` array. An authorized offline audit/export path must be implemented before external evaluation can consume these receipts.

## Required follow-on

1. Bind the receipt to the exact source/candidate envelope and Mycelix qualification receipt.
2. Add a separate reader-use receipt identifying which returned IDs were passed into the answer/action context; do not infer use from retrieval alone.
3. Add a decision receipt that binds DecisionNeed, chosen action, accepted evidence IDs, abstention/probe status and the use-time qualification snapshot.
4. Add an independently verified outcome receipt.
5. Evaluate paired removal, substitution, correction and placebo interventions under #13397; the effect estimator must not trust model explanations as causal evidence.
6. Export receipts to an independent append-only witness domain, with key/retention/erasure policy and recovery tests.

## Claim boundary

The current receipt proves only that this service committed a digest record describing a Search response before returning it. It does not prove that the receipt is externally authentic, that the database has not been rolled back, or that evidence caused a later decision.
