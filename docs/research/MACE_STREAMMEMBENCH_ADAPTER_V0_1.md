# MACE StreamMemBench protocol adapter v0.1

**Status:** standalone Rust-side protocol and receipt validator with synthetic tests. It is not yet a registered StreamMemBench Python `MemorySystem`, has not run the official scorer, and does not claim a benchmark result.

## Official protocol mapping

The official StreamMemBench `MemorySystem` contract provides `reset_memories`, `add_stream_to_memory`, `generate_response`, `generate_response_with_feedback`, `store_interaction`, `retrieve_memory`, `inspect_memory_records`, `save_memories` and `diagnostics`. This crate defines the candidate input/receipt contract for a thin Python shim to map those methods to a Rust MACE host/runtime.

The required evaluation sequence is preserved:

1. Ingest each chronological stream segment, including segments without a scored evidence anchor.
2. Generate the initial answer from the initial user request.
3. Receive the benchmark feedback text. If the runner requests a revision, generate a revised answer to the same initial request with the prior answer and feedback.
4. Persist the evaluated interaction (request, initial answer, feedback and optional revised answer).
5. Generate the follow-up answer only after interaction persistence.

## Candidate label firewall

Candidate-facing commands accept only raw stream text, timestamped observations, user requests, feedback text, prior answers and ordinary interaction state. They do not have fields for `expected_behavior`, `evidence_statement`, `supporting_observations`, `source_span.raw_indices`, metric scores, judge rationales, `feedback_type`, or the four benchmark metrics. A recursive key scan rejects those labels even if they are nested in metadata.

The public StreamMemBench data is derived from EgoLife. Preserve the released access and license terms, use participant-scoped pseudonymous cluster hashes for evaluation receipts, and do not attempt participant re-identification.

## Evidence-governed sidecar

- Run binding pins the benchmark release, harness commit, dataset snapshot/split, language, candidate, memory config, answer model, generation config and participant-level independent unit.
- Every ingested segment contributes to a participant-local monotonic cursor chain. The validator checks contiguous sequence numbers, duplicate segments, day/segment order and cursor continuity.
- Each initial, revision and follow-up response receipt records exact request/answer/model-request/model-response digests, model/generation identity, token counts, latency and ordered evidence exposures.
- Each exposed evidence item binds its context bytes to its MACE use-time receipt, canonical #13973 `DecisionEvidenceUseLinkV1` commitment and source-lineage receipt.
- Interaction persistence is independently receipted and must occur before follow-up use; the trace rejects a missing/ambiguous persistence acknowledgement.
- The episode receipt is separate from the official scorer's result JSONL. It contains no `fidelity`, `initial_evidence_use`, `feedback_incorporation` or `followup_reuse` score fields.

## Statistical unit

The release contains 6 participants over 7 days, with many stream segments, evidence anchors and task pairs. Repeated tasks within one participant are not independent observations. For population-level uncertainty, participant is the default cluster unit; report all six cluster contributions and treat the first release run as descriptive/pilot rather than claiming population-wide confirmatory confidence from 16,214 task rows. A confirmatory result needs more independent participants or a justified external sampling design.

## Proposed matched experiment

Preserve the official scorer and run at least these arms under the same participant/day/timeline, answer model, prompt, token budget, API cost accounting and random seed:

- `rag_raw` and `rag_extracted` official baselines;
- MACE raw-stream evidence retrieval;
- MACE extracted/provenance-linked memory;
- MACE adaptive EVSI re-observation and evidence exposure;
- diagnostic ablations: feedback memory writes disabled, evidence-use gate disabled, provenance link removed, and follow-up retrieval disabled.

Report the four official metrics unchanged and separately. Add MACE sidecars for stale/unauthorized evidence, exposure-link coverage, provenance completeness, revision persistence, post-feedback correction, follow-up reuse, token/cost/latency and hard-gate violations. Never create a weighted aggregate that can trade a privacy/authority/erasure failure against higher recall.

## Run

```bash
cargo test --manifest-path research/mace-streammembench-adapter-v0-1/Cargo.toml
cargo clippy --manifest-path research/mace-streammembench-adapter-v0-1/Cargo.toml --all-targets -- -D warnings
```

These commands validate the independent Rust protocol kernel only. The next integration increment is a thin Python `MemorySystem` wrapper that forwards the official methods to this Rust host and emits the official result JSONL unchanged plus a MACE diagnostics JSONL sidecar.