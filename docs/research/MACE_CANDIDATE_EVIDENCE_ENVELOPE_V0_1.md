# MACE candidate evidence envelope v0.1

**Status:** research protocol and Rust reference validator; authored tests are not yet executed. This is an interchange contract, not a production authorization engine or proof of evidence truth.

## Purpose

Provide one backend-neutral boundary for BM25, dense retrieval, HDC, graph memory, episodic timelines, multimodal trajectory adapters and extracted-fact systems.

The envelope preserves the identity of the backend, the exact query scope and purpose, the original source digest, the payload digest, the modality, retrieval channel/rank, and any declared derivation parents. It does not coerce distinct memory representations into a synthetic common text field.

## Protocol record

Each JSONL line is a strict `mace_candidate_evidence_v0_1` / `candidate_evidence` record. The Draft 2020-12 schema is `data/schemas/mace_candidate_evidence_v0_1.schema.json`.

Required identity bindings:

- `run_id`, `candidate_id`, `backend_id`, `backend_revision_sha256`
- `query_scope`, `query_purpose`, `query_sha256`
- `evidence_id`, `source_id`, `source_sha256`, `payload_sha256`, `payload_ref`
- `modality`, `retrieval_channel`, one-based `retrieval_rank`
- `parent_evidence_ids`; derived evidence also requires `derivation_receipt_sha256`

The query text and payload are not copied into this manifest. `query_sha256` binds the exact query bytes. `payload_ref` is an opaque reference for a separately controlled resolver; this crate never opens a path, fetches a URL or interprets a payload reference as a command.

## Validation and normalization

The Rust reference parser:

1. Rejects unknown/missing fields, malformed digests, malformed identities and unsupported modalities.
2. Requires all rows in one batch to bind to the same run, candidate, backend revision, query digest, scope and purpose.
3. Rejects duplicate evidence IDs and duplicate ranks within the same retrieval channel.
4. Requires a derivation-receipt digest whenever parent IDs are declared; rejects self-parent and duplicate-parent references.
5. Sorts records deterministically by channel, rank and evidence ID.
6. Emits a SHA-256 candidate-set commitment over the normalized rows and common identity.

Scores are backend-native diagnostics. Scores from different channels are not comparable and must not be combined by this module. Fusion must be an explicit, pinned retrieval profile with its own identity.

## Critical trust boundary

A valid candidate envelope is **not qualified evidence**.

This module does not verify issuer signatures, source authority, legal rights, semantic support, lifecycle freshness, revocation, erasure, or the authenticity of a payload resolver. In particular:

- a matching digest is not an authenticated issuer;
- a derived item cannot acquire authority from its parent list;
- a retrieval rank is not an admissibility decision;
- a backend-supplied lifecycle flag must never self-authorize use;
- content-addressed references must be resolved through an allowlisted local adapter, not arbitrary network or filesystem access.

The later MACE qualification service must attach independently authenticated issuer, policy, rights, lifecycle-checkpoint and lineage receipts before evidence can be used in consequential decisions. Evaluator truth, benchmark labels and attack assignments remain outside the candidate process.

## Integration sequence

1. Adapt #13346 raw-session hits to this envelope while keeping verbatim source chunks retrievable through the controlled payload resolver.
2. Adapt #13347 LongMemEval-V2 state-window hits and screenshot references without flattening image evidence into text.
3. Add adapters for pinned dense, graph, episodic and extracted-fact baselines.
4. Add a distinct qualification-receipt protocol backed by the trusted issuer and lifecycle verifiers.
5. Extend the independent evaluator to freeze candidate envelopes before evaluator-only outcomes are opened.
6. Compare retrieval utility, evidence admissibility, causal influence, decision regret and all-in cost as separate endpoints.

## Reproduction

```sh
cargo test --manifest-path research/broca-mace-transfer-bench-v0-1/aml-adapter/Cargo.toml
cargo clippy --manifest-path research/broca-mace-transfer-bench-v0-1/aml-adapter/Cargo.toml --all-targets -- -D warnings
```

These commands are prescribed but have not been executed in the current authoring environment. External Draft 2020-12 meta-validation and benchmark runs remain pending.
