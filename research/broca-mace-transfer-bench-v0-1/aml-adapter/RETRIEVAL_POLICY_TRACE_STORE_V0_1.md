# MACE retrieval-policy trace store v0.1

Status: local filesystem durability and conformance adapter. Rust source and tests are authored but have not been compiled or executed. This is not an independent witness service, signing authority, or defense against whole-workspace rollback.

## Shared directory contract

The official LongMemEval-V2 Python plugin creates workspace_dir/retrieval_traces and passes that exact directory to the Rust sidecar through MACE_LMEV2_POLICY_TRACE_DIR. Both layers derive each filename from SHA256(UTF8(query_invocation_id)) followed by the suffix .trace.json.

The canonical JSON receipt schema is mace_retrieval_trace_receipt_v1 and includes a versioned Rust binary receipt digest. Rust and Python use the same file and invocation-derived name; they do not create independent competing copies.

The sidecar opens the directory before accepting requests, strictly loads existing final receipts into a fresh RetrievalPolicyMemoryV1, and refuses to start if a receipt is malformed, noncanonical, misnamed, tampered, over the size limit, or conflicts with an existing run/invocation binding.

## Atomic create-once publication

For each trace the Rust adapter:

1. Validates the typed receipt and recomputes its versioned binary SHA-256 commitment.
2. Serializes the JSON projection canonically.
3. Creates a unique temporary file in the same directory with create-new semantics.
4. Writes the complete JSON, fsyncs the file and closes it.
5. Publishes the final invocation-derived filename with a hard-link create-if-absent operation; an existing filename is never overwritten.
6. Removes the temporary name, fsyncs the directory on Unix, then reads back and compares the final bytes.
7. Restores/binds the receipt into the live in-memory policy store.

On Unix the directory uses mode 0700 and receipt files use mode 0600. Individual receipts are limited to 1 MiB and the store to 100,000 receipt files. Interrupted .pending files are non-authoritative and ignored during replay. Unknown final files and non-regular entries fail closed.

## Ambiguous publication outcomes

An error after publication may mean the final file exists even though the caller did not receive a successful acknowledgement. The Rust adapter labels persistence and post-publication binding errors as unknown outcomes. Python poisons/terminates the sidecar and refuses further operations until the saved receipt directory is reopened and reconciled. It does not silently restart an empty policy store or retry the same mutation.

If Python cannot persist the paired candidate-evidence receipt or verify the retrieval trace after the sidecar has returned, it also poisons the sidecar. The benchmark reader is not served a context when the evidence/trace pair could not be reconciled.

## Restore and replay checks

Rust strict JSON decoding checks exact schema version and known fields, operation and stop-reason variants, integer widths and booleans, per-operation evidence/outcome consistency, byte-accounting totals and completeness, the receipt SHA-256, typed trace invariants and invocation identity.

On load, filenames must match the SHA-256 of the opaque invocation ID and records are restored through the same in-memory invocation-binding rules. Python separately recomputes the receipt digest, checks query/run/domain/environment/tool-profile identity, pairs each trace with its candidate receipt, and verifies matching candidate-set digests. Both missing directories and orphan/unpaired receipts fail closed.

Rust serde_json canonical map ordering and Python sorted-key compact JSON are expected to produce identical bytes. The Python ingress test simulates the Rust-created canonical file and requires Python's create-once writer to accept the exact bytes rather than rewrite them. A fixed cross-language hash vector independently checks the binary receipt commitment.

## What this does not guarantee

The trace store shares the benchmark workspace and its backup/restore domain. Restoring the entire workspace can roll back candidate evidence and policy traces together. A privileged actor able to rewrite local files can also rewrite their hashes consistently.

The store does not sign or externally witness receipts, authenticate the benchmark publisher/tool provider/source IDs, prove that a tool actually returned the listed source IDs, prevent full workspace rollback, provide multi-host quorum/remote attestation/hardware monotonic counters, or persist promoted strategy notes yet.

Production requires a separate administrative/failure domain, authenticated trust anchor, independent backup/restore policy and source-observation receipts. The local adapter exists to make process restart, create-once semantics, canonical replay and audit boundaries testable.

## Reproduction

    cargo test --manifest-path research/broca-mace-transfer-bench-v0-1/aml-adapter/Cargo.toml
    cargo clippy --manifest-path research/broca-mace-transfer-bench-v0-1/aml-adapter/Cargo.toml --all-targets -- -D warnings

The retrieval-policy module and file-store tests are authored but unexecuted. No compile, CI, benchmark or security qualification is claimed.