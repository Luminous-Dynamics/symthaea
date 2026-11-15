# SympoeticBench Trace and Causal Contribution Protocol

## 1. Why traces are part of the benchmark

A final artifact does not reveal whether a collaboration was reciprocal, whether a suggestion was accepted, whether the participant's intent changed, or whether an AI contribution actually affected the result.

SympoeticBench therefore represents an episode as an append-only event trace. Events form a directed acyclic graph through explicit parent-event references. Artifact revisions are versioned and content-addressed. The trace records what happened without assuming that every message is a meaningful contribution.

The machine-readable session schema is TRACE.schema.json.

## 2. Minimum event semantics

Every event should record:

- A stable event identifier and monotonically increasing sequence number.
- Elapsed time relative to the session start.
- Actor class: human, agent, tool, environment, or evaluator.
- Event type: intent, proposal, question, challenge, revision, acceptance, rejection, clarification, permission change, tool invocation, artifact change, and so on.
- Parent events that causally or procedurally informed the event.
- A digest of content or artifact state, not raw private text by default.
- The applicable consent scope and permission state.
- Artifact-before and artifact-after digests when a version changes.
- Tool preconditions, outcomes, side-effect and reversibility fields where applicable.
- Attribution assertions and confidence when a contribution is linked to an artifact segment.

A corrected or withdrawn event is not silently overwritten. Append a correction or revocation event and preserve the audit relationship. Raw content should be stored only where required, consented, protected, and subject to an explicit retention policy.

## 3. Contribution provenance versus contribution impact

These are separate questions:

- **Provenance:** who proposed, edited, supplied, or transformed a contribution?
- **Impact:** did that contribution improve, worsen, or materially change the outcome?

Do not use turn counts, token counts, or word counts as proxies for contribution value. A short human constraint may determine the design; a long AI response may have no effect.

## 4. Causal Contribution Analysis (CCA)

For a contribution event e, define a task-specific outcome function Q and compare the full artifact with a controlled replay in which e is removed or replaced by a prespecified neutral alternative:

**marginal_impact(e) = Q(full_trace) - Q(replay_without_e)**

This is an experimental estimate, not a legal attribution or proof of authorship. Report the replay method, intervention, outcome rubric, uncertainty, and any failed replay.

Important safeguards:

- Contribution effects are not generally additive. Do not sum individual marginal impacts and call the result total contribution.
- Some contributions are complementary. When feasible, test selected pairs or coalitions, or use a Shapley-style approximation with its computation budget reported.
- Preserve temporal dependencies. Removing a contribution may make later events invalid; mark such replays non-identifiable rather than fabricating a counterfactual.
- Have independent raters score full and ablated artifacts blind to condition.
- Report both positive and negative impact, including contributions that were ignored or caused rework.

## 5. Negative controls

The harness and scorer should include:

1. **No-op contribution injection:** adding an event that does not affect the artifact should not receive positive impact credit.
2. **Attribution shuffle:** randomizing contributor labels should reduce attribution accuracy.
3. **Parent-edge randomization:** destroying valid causal links should degrade causal trace scores.
4. **Known-impact intervention:** a controlled constraint or edit with a documented effect should be detectable in a replay.
5. **Rejected proposal control:** a rejected contribution should not be credited as though it appeared in the final artifact.
6. **Revoked contribution control:** a contribution whose permission is revoked must not be reused outside the permitted scope.

A scorer that passes only on the evaluated system's own traces is not independently qualified.

## 6. Determinism and evidence integrity

- Use monotonically increasing sequence numbers and stable event IDs.
- Hash scenario inputs, prompts, tool manifests, trace payloads, and artifact versions.
- Make timestamps supplementary; sequence and causal edges define ordering.
- Record random seeds and replay environment digests.
- Treat missing events, broken parent references, and hash mismatches as integrity failures.
- Preserve unknown tool outcomes as unknown. Do not infer success from a tool call being issued.
- Keep content hashes separate from public identifiers where they could enable dictionary attacks on sensitive text.


## 6.1 Unknown execution outcomes and reconciliation

A tool result with status `unknown` leaves the execution state unresolved. The runner must not issue a subsequent tool invocation until an independently observed `execution_reconciled` event:

- Names the unknown tool-result event as a causal parent.
- Is authored by the tool, environment, or evaluator rather than self-attested by the agent.
- States whether the action succeeded, failed, or was not executed.
- Includes a SHA-256 digest of the observed reconciled state.

For state-changing results whose outcome is known, record a postcondition digest. A failed call may still have partial effects, so failure status alone is not proof of no mutation. A trace with an unresolved unknown result is structurally incomplete and must not be scored as a completed successful run.

## 6.2 Digest canonicalization

For v0.1, `trace_digest` is SHA-256 over the trace JSON parsed as a `serde_json::Value`, with the top-level `trace_digest` field removed, then serialized by `serde_json` (deterministic map-key ordering). This is the benchmark's explicit canonicalization convention and is not a claim of RFC 8785 conformance. File-level and payload digests are SHA-256 over the exact supplied bytes.

## 7. Privacy and consent

The default trace stores event metadata and content digests, not raw conversation text. A raw-content reference requires an explicit and scoped permission. Permission revocation must affect future reuse and must accurately report the system boundary: a benchmark harness cannot promise deletion from independent external services it does not control.

Participant identifiers must be pseudonymous. Keep the identity mapping separate from benchmark traces, limit retention, and suppress re-identifiable subgroup reporting.

## 8. Required outputs

A qualified run should emit:

- Scenario and environment digests.
- Ordered event trace and DAG integrity result.
- Artifact version chain.
- Provenance precision, recall, and F1 against human annotation.
- Causal contribution estimates with counterfactual method and uncertainty.
- Consent and authority hard-gate results.
- Replay determinism and negative-control results.
- Cost, latency, human time, and failed or non-identifiable replays.

CCA is an advanced research measure. Initial v0.1 reports may mark it unavailable while the replay harness is being developed; they must not infer it from attribution alone.
