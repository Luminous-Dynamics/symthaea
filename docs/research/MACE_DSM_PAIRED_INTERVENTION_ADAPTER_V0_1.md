# MACE DSM paired-intervention adapter contract v0.1

**Status:** Mapping specification only. No executable adapter is claimed.

## Purpose

Reuse the paired memory-to-decision intervention runner and cluster-aware inference work rather than build a second paired experiment engine. The current DSM reference evaluator has a separate episode/response contract. This document defines the safe field mapping and the scenario metadata needed before implementing a mechanical adapter.

Relevant owners:
- Paired intervention and resource receipts: https://github.com/Luminous-Dynamics/symthaea/pull/13869
- Paired baseline and cluster inference: https://github.com/Luminous-Dynamics/symthaea/pull/13836
- DSM parent episode contract: https://github.com/Luminous-Dynamics/symthaea/pull/13895
- DSM Rust reference evaluator: https://github.com/Luminous-Dynamics/symthaea/pull/13909

## 1. Candidate receipt mapping

The paired runner writes candidate receipts with schema mace_memory_decision_intervention_v0_2.

| Paired runner field | DSM response field | Rule |
|---|---|---|
| arm | arm_id | Must exactly match a pre-registered arm ID in the DSM arm manifest |
| decision | action | Map only through a versioned action-name mapping; reject unknown or lossy mappings |
| retrieved_event_ids | retrieved_evidence_ids | Preserve the full returned set, including inadmissible or subsequently rejected candidates |
| decision_support_event_ids | selected_evidence_ids | Preserve separately; every selected ID must appear in the retrieved set |
| latency_ns | cost.detailed_receipt.latency_ns | Copy exact nanoseconds; do not round into microseconds |
| phase_timings_ns.* | cost.detailed_receipt.phase_timings_ns.* | Copy all seven phase counters exactly |
| input_tokens | cost.detailed_receipt.input_tokens | Copy only when the source receipt reports it |
| model_calls | cost.detailed_receipt.model_calls | Copy only when the source receipt reports it |
| tool_calls | cost.detailed_receipt.tool_calls | Copy only when the source receipt reports it |
| external_tool_calls | cost.detailed_receipt.external_tool_calls | Copy only when the source receipt reports it |
| index_bytes | cost.detailed_receipt.index_bytes | Preserve the byte count |
| index_bytes_kind | cost.detailed_receipt.index_bytes_kind | Preserve exact measured/estimated/not-applicable qualifier |
| evidence_sidecar_bytes | cost.detailed_receipt.evidence_sidecar_bytes | Preserve the byte count |
| evidence_sidecar_bytes_kind | cost.detailed_receipt.evidence_sidecar_bytes_kind | Preserve exact measured/estimated/not-applicable qualifier |
| unmeasured_cost_stages | cost.detailed_receipt.unmeasured_cost_stages | Preserve each unmeasured stage; never infer zero |

The DSM candidate_sha256 is not the paired runner's candidate_input_sha256. The adapter must recompute the DSM domain-separated digest over the typed DSM candidate projection bound to episode_id, seed and family. It must then compute the DSM arm-binding digest using the resulting candidate digest and exact arm_id. Do not copy a different digest into either field.

The paired runner's decision_support_event_ids are policy support references. They do not, by themselves, prove that every selected event is valid under DSM's bitemporal, source-authority, scope and lifecycle rules. Run the DSM admission check over the source-event projection before scoring.

## 2. Cost fidelity and missingness

The paired runner labels some index/sidecar bytes as measured payload bytes and others as estimated Rust-struct or ID-vector sizes. DSM must preserve those labels. An estimated value must not be silently promoted to measured process memory.

The paired runner currently marks stages including ingestion, maintenance, database storage, network, GPU, LLM, external tools and durable write as unmeasured for this deterministic harness. Preserve this missingness. Do not translate an unmeasured stage to numeric zero.

DSM top-level summary cost fields (ingest_operations, query_operations, maintenance_operations, retained_bytes, returned_bytes, latency_us) remain nullable. The detailed receipt is the preferred lossless carrier for paired-runner timing and instrumentation. Aggregation must report the count of rows with a field present and the sum over only those rows.

No scorer-side timer or candidate-provided counter is an independently measured CPU, RSS, energy, database, network or dollar-cost result. Those require separate host-level measurement with a defined sampling and attribution protocol.

## 3. Identity and paired design mapping

The paired runner emits pair_block_id, world_cluster_id, query_id, run_id and source revision information in candidate and evaluator records. DSM v0.1 keys responses by episode_id × arm_id and does not yet have a first-class pair-block/cluster manifest.

Do not equate seed with an independent cluster. A seed can parameterize many episodes in the same generated world; multiple tasks in a world may be dependent. Before using #13836's cluster bootstrap, add an evaluator-private, digest-bound pair manifest mapping each episode_id to:
- pair_block_id;
- world_cluster_id / independent resampling cluster;
- query_id;
- generator and scenario digest;
- arm-manifest digest;
- run ID and exact source revision.

The candidate process must not receive evaluator truth or hidden outcome labels. Candidate-visible cluster fields may be published only when the benchmark protocol explicitly makes them available and the threat model accepts that disclosure.

A real no_memory arm must execute through the same policy and action environment with empty cross-session memory. A task-authored no_memory_action is only a convenience reference and is not a system baseline. If a true no_memory arm is present, declare its exact arm ID in no_memory_arm_id; the scorer then compares matched episode losses.

## 4. Required source-event projection (blocking prerequisite)

The current paired CandidateRun receipt contains event IDs, decisions and resource measurements. It does not carry all fields required to reconstruct DSM SourceEvent rows.

The DSM candidate source-event contract needs at least:
- stable event and source identifiers;
- source-family identity and canonical upstream-root digest (the paired `Experience.source_family` and `Experience.source_root_digest` fields can seed this mapping, while preserving their exact semantics);
- observation time and valid-from / valid-until timestamps;
- authenticated source authority tier;
- lifecycle relation (asserts, supersedes, retracts, contradicts, derived_from) and exact parent IDs;
- source claim / typed payload reference;
- uncertainty semantics and unit;
- event-level scope / authorization where applicable.

The paired runner's Experience has richer causal lineage such as source_root_digest, source_family, identification method, observed utility, measurement-error bound, intervention, pre-registration, outcome reconciliation, trusted/retracted state, confounder control and delayed-horizon completion. DSM v0.1 now has optional `source_family` and `source_root_sha256` fields and separately counts distinct source families and upstream roots; it must preserve the paired source-root digest exactly rather than hashing a display ID. These lineage values are identifiers, not proof of authenticated origin or statistical independence. However, the current Experience/candidate receipt still does not expose the complete DSM bitemporal/source-event lifecycle envelope. retrieved_event_ids alone cannot reconstruct it.

**Therefore the adapter must fail closed or produce NotComparable until a versioned candidate-visible event projection exists.** It must not invent observation times, valid-time intervals, parent edges or authority attestations from event IDs or evaluator-only Truth.

The appropriate next step is a versioned common scenario projection emitted by the generator and consumed by both the paired runner and DSM adapter. Keep source-event identity, causal lineage, bitemporal validity and role/scope metadata typed and independently committed.

## 5. Evaluator-private outcome mapping

The paired runner writes a separate evaluator_score record containing evaluator-only synthetic utility/regret, task success, invalid/stale/wrong-scope use, false causal claim count, decision change vs no memory, justification, harmful dependence, recovery count and outcome receipt digest.

Those values belong exclusively in the evaluator-private stream or a post-scoring report. They must never be included in the candidate prompt, response context, candidate digest input, or candidate-visible source projection.

The paired runner is itself labelled diagnostic-only because it uses public synthetic fixtures and a co-located evaluator. The DSM adapter inherits that qualification state. Converting its JSONL into DSM rows does not create evaluator independence or external validity.

## 6. Integration sequence

1. Stabilize the DSM response schema with separate retrieved and selected evidence IDs and nullable detailed cost receipts.
2. Define a common scenario projection from the paired generator containing the full candidate-visible source-event envelope without hidden Truth.
3. Add an evaluator-private pair manifest with independent cluster and pair-block mapping; reuse #13836's resampling code rather than duplicating it.
4. Implement the adapter as a pure, versioned transform with exact input/output digests and a NotComparable disposition for any missing mandatory source field.
5. Add round-trip fixtures for all eight paired arms, including empty-memory and structural placebo.
6. Add mutation tests for dropped retrieved IDs, selected IDs not in retrieved IDs, altered arm identity, stale/retracted event, wrong authority, scope changes, future observations, missing cost, mislabeled estimated bytes and hidden-truth leakage.
7. Carry paired source-family and source-root digests unchanged into the typed DSM event projection, and add an explicit source-root attestation/authority verifier before treating those roots as authentic or independent.
8. Qualify independent evaluator custody and host-measured resource accounting before any external or SOTA claim.

## Non-goals

- Do not create a second memory-policy runner.
- Do not copy evaluator-only score records into candidate-visible streams.
- Do not equate a public SHA-256 commitment with authenticity or provenance verification.
- Do not treat a deterministic policy baseline as a hosted LLM benchmark.
- Do not claim interoperability until a versioned source projection and exact-head adapter tests exist.
