# MACE Decision Evidence Exposure Link v0.1

**Status:** Research implementation on a child branch of the use-time revalidation work. This adds a typed observer-sidecar contract; it is not yet integrated into a live AMemGym agent adapter and carries no external benchmark result.

## Purpose

MACE has two complementary pieces:

- [#13390](https://github.com/Luminous-Dynamics/symthaea/pull/13390) defines candidate actions and an on-policy, profile-randomized total-effect evaluator.
- [#13967](https://github.com/Luminous-Dynamics/symthaea/pull/13967) adds a use-time evidence receipt that re-runs existing ingress verification at a bitemporal decision snapshot.

The missing join is the boundary between an evidence item that was retrieved and an item actually placed in a model request. A candidate-emitted list of evidence IDs is not a trusted record of retrieval, exposure, admissibility, or causal influence.

This increment adds `DecisionEvidenceUseLinkV1`. It binds one in-process use receipt to a retrieval-batch digest, exact serialized model-context digest and evidence byte span, and subsequent candidate-action digest. It separately records the task-specific A/B decision from the ingress policy and the candidate record's `action_type` (respond, ask_user, memory_mutation, tool_request or abstain); these are different concepts.

## Event chain and non-equivalences

Keep these sets and events distinct:

1. **Retrieved (R):** IDs returned by the memory backend.
2. **Admissible (A):** retrieved evidence that passed current source, lifecycle, scope, trust, lineage, purpose and bitemporal checks.
3. **Exposed (E):** exact evidence representation bytes actually included in the frozen model context.
4. **Cited (C):** evidence IDs the candidate chose to cite in its emitted action.
5. **Influential (I):** evidence whose controlled removal or substitution changes a preregistered decision/outcome contrast.

The new link joins A to E and then records the corresponding candidate action. It does not prove I. A citation is an agent claim, not an observation of causal influence. A context span proves byte inclusion in the hashed request, not that the model attended to or understood it.

Causal influence requires a paired counterfactual: run the same frozen episode with the evidence span removed or replaced by a token/semantic-matched placebo while holding remaining context, model, prompt, tools, deadline and randomization protocol fixed. Measure action and outcome differences. Estimate this **conditional exposure/ablation effect** separately from #13390's total on-policy policy effect, because the latter includes policy-induced changes to later disclosure and histories.

## Implementation contract

`link_verified_evidence_to_decision_v1`:

- requires the receipt's in-process verifier capability and valid canonical digest;
- checks event, evaluated decision, scope, purpose, consumer-policy and bitemporal-snapshot equality;
- parses the candidate-action envelope and checks schema version, record type, run, episode, sequence and action type against the exact bytes;
- fails closed on non-trusted sources, inactive lifecycle, incomplete ancestry when required, or invalid half-open validity;
- hashes the exact full model-context bytes and validates UTF-8 byte-span boundaries and the representation digest;
- binds the use receipt to its exact bitemporal query snapshot and requires model-context exposure to occur at or after that snapshot and binds run, episode, action sequence, retrieval batch, use-receipt digest, context digest, evidence representation digest, exact candidate-action bytes and times;
- emits an integrity-committed JSON representation marked `originating_process_only`.

The receipt and link digests are SHA-256 integrity commitments, not issuer signatures. Serialization does not preserve the private in-process verification capability. The current function remains a research seam: production must supply exact request bytes and spans from trusted adapter instrumentation, plus independently authenticate the observer and source transformation. It does not authorize tool calls or external effects; those require a separate authorization check at the side-effect boundary.

## Metrics for on-policy runs

Report separately by episode family, profile and policy arm:

- retrieval-to-admissible drop rate: rejected retrieved IDs / retrieved IDs;
- invalid or unreceipted exposure count and rate;
- stale, expired, revoked, wrong-scope or wrong-purpose exposure escape count (hard gate);
- citation support rate and unsupported citations (C minus A);
- admissible-but-unexposed evidence (A minus E);
- exposed-but-uncited evidence (E minus C);
- span/context/action link rejection reasons;
- paired utility and regret delta under evidence-span ablation, with cluster-aware uncertainty;
- model-context tokens/bytes, retrieval calls, verifier calls, re-observation latency, total lifecycle cost and p50/p95/p99 latency.

Integrity, privacy, consent and authority failures must not be combined into a weighted score that utility can offset. The desired adversarial escape target is zero. The performance target is positive paired decision utility with a cluster-bootstrap 95% lower confidence bound above zero, without worsening worst-group regret or user burden. These are prospective targets, not achieved results.

## Tests authored

The module adds deterministic tests for exact-span linking; rejection of serialized receipts without process-local custody; event/snapshot/policy mismatch; untrusted and expired sources; incomplete ancestry; context mutation; representation-span mutation; UTF-8 boundary errors; and action-before-exposure.

## Validation boundary and next integration

The focused ingress workflow covers the modified source path; a new strict Draft 2020-12 schema defines the link interchange. The authoring environment has no Rust toolchain, so no local compile, unit-test or Clippy success is claimed. The next integration is an adapter in the #13390 AMemGym candidate runtime that:

1. captures exact serialized provider request bytes before sending;
2. records each evidence span and transformation-output digest;
3. obtains a current use receipt before adding each evidence span;
4. writes one observer-sidecar link per exposed item;
5. links the candidate action only after it is emitted;
6. produces a matched ablation episode without leaking evaluator labels into the candidate.

External observer signatures, portable verification, production Mycelix authority and a sealed independent evaluator remain separate qualification requirements.


The candidate-action digest is checked against exact JSON bytes; the parsed record must identify the same schema version, record type, run, episode and sequence, and its action type is checked separately from the task-level A/B decision.
