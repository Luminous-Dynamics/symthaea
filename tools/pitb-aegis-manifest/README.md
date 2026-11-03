# PITB Aegis manifest verifier

A small, standalone Rust CLI for validating the *identity and byte integrity* of frozen Aegis task artifacts before they enter a PITB evaluation.

## Run

From this directory:

```sh
cargo test
cargo run -- <manifest.json> <artifact-root>
```

Dependency versions are pinned in Cargo.toml. Commit the generated Cargo.lock after the first dependency resolution, then use `--locked` in CI. The manifest is JSON and must conform to `schema/pitb-aegis-manifest-v0.1.schema.json`. The CLI uses strict Serde structs with unknown-field rejection and validates required metadata, digest syntax, unique category/path entries, relative artifact paths, symlink rejection, and SHA-256 equality for source Rust, translated Lean, and statement files. If a Lake manifest artifact is supplied, its digest must match the pinned toolchain identity.

## Important boundaries

This tool does **not**:
- download or authenticate the official Aegis release;
- establish that an artifact URI is controlled by the benchmark authors;
- prove that Rust and translated Lean have equivalent semantics;
- prove that a specification is non-vacuous or matches intended behavior;
- elaborate or kernel-check a Lean proof;
- independently replay a proof or validate a refutation;
- establish that benchmark labels are correct.

A passing result means only that the manifest metadata passes this implementation's checks and the referenced local bytes match their declared digests. Semantic correspondence, proof validity, provenance authentication, and benchmark-label validation require separate gates.

## Dataset provenance

Do not populate a production manifest with guessed release URLs, licenses, task counts, IDs, or digests. Acquire the official artifact, verify its published provenance, and pin the exact release first. Keep the 2,045 program-pair count, 11,533 candidate-specification count, and 2,000-task evaluation sample as separate fields/concepts.

## Security posture

The artifact root is canonicalized. Artifact paths must be relative and contain only normal path components; symlinks are rejected component-by-component. The verifier hashes bytes in a streaming fashion. The manifest itself is an input and is not authenticated by this CLI; preserve it in a signed or otherwise independently authenticated evaluation receipt when that capability is implemented.

## Exit behavior

- Exit 0: local manifest metadata and declared artifact digests pass.
- Non-zero: malformed manifest, invalid metadata, path traversal/symlink, missing artifact, read failure, or digest mismatch.

No CI or benchmark execution is implied by this documentation.
