# PITB SorryDB snapshot and source-integrity validator

A standalone Rust validator for reproducible task identities in SorryDB-style Lean proof obligations.

## Why this exists

SorryDB's task record contains a repository remote, branch, commit, Lean version, source path and coordinates for the open proof. The SorryDB database specification explicitly notes that any change to any file in the repository may affect the meaning or difficulty of every sorry in that repository. Therefore a task is not identified by a goal string alone: the complete repository commit and Lean toolchain are part of its identity.

The SorryDB documentation also says that `debug_info.goal` is for debugging and should not be relied upon as a client contract. This adapter therefore hashes the pretty-printed goal only as a diagnostic hint. It does not trust it as the authoritative elaborated goal.

## Commands

```sh
cargo fmt --check --manifest-path tools/pitb-sorrydb-integrity/Cargo.toml
cargo test --locked --manifest-path tools/pitb-sorrydb-integrity/Cargo.toml

# Run the adversarial controls and emit a schema-checkable report plus task identity:
cargo run --locked --manifest-path tools/pitb-sorrydb-integrity/Cargo.toml -- \
  --self-test-report /tmp/sorrydb-report.json /tmp/sorrydb-task.json

# Validate a locally available, byte-exact SorryDB snapshot:
cargo run --locked --manifest-path tools/pitb-sorrydb-integrity/Cargo.toml -- \
  --validate path/to/SorryDB_2601.json <upstream-git-blob-sha1> /tmp/sorrydb-report.json

# Confirm one task against exact objects in a local pinned repository checkout:
cargo run --locked --manifest-path tools/pitb-sorrydb-integrity/Cargo.toml -- \
  --verify-checkout /tmp/sorrydb-task.json /path/to/repository-checkout
```

## Guarantees

The snapshot validator:
- validates repository identity, Git SHA-1 commit, Lean release and safe relative source path;
- validates 1-based line and SorryDB's zero-based character-column coordinates;
- rejects empty/reversed ranges and duplicate canonical task identities;
- derives a stable PITB task ID from remote, branch, commit, Lean version, source path and coordinates;
- recomputes the Git blob SHA-1 from raw input bytes and also records raw SHA-256;
- records source record IDs only as provenance; they are not used as the canonical PITB identity;
- preserves the goal as a digest-only debugging hint;
- keeps `scoring_admission=false`.

The checkout verifier:
- checks `git rev-parse HEAD` against the task's pinned commit;
- checks the configured origin remote;
- reads `lean-toolchain` from `HEAD:lean-toolchain`, not from the working tree;
- reads the Lean source file from `HEAD:<path>`, not from the working tree;
- translates SorryDB's character columns to UTF-8 byte offsets and verifies that the exact span contains the `sorry` token.

A dirty worktree cannot change the source bytes being checked. The checkout receipt still sets `scoring_admission=false`: this source-level check does not run Lake/Lean or re-elaborate the goal.

## Trust boundary: three distinct proofs

PITB treats these as independent evidence layers:

1. **Snapshot identity** — the task really points to this repository commit, Lean toolchain and source range.
2. **Formal proof validity** — the original goal is reproduced in Lean, the proof replacement compiles, the kernel accepts it, exactly the intended sorry is removed, and all sibling sorries/goals are preserved.
3. **Specification integrity** — the formal proposition faithfully represents the intended mathematical statement, is not vacuous, has no dropped hypotheses, and uses only approved assumptions.

A kernel-checked proof establishes the encoded Lean proposition. It does not by itself establish that the proposition faithfully encodes the intended informal problem. The 2026 ICML benchmark audit demonstrates why both proof checking and semantic specification audits are required.

## SorryDB source pins

The pinned upstream SorryDB repository revision used for this adapter is:

- Repository: `SorryDB/SorryDB`
- Commit: `74b3b63f8ba890d675f362924b21ac55db410383`
- Full `SorryDB_2601.json` Git blob: `5fafddac467af6d30d8f5c8731d6edec9c273a47`
- `SorryDB_2601_1000_evaluation_split.json` Git blob: `af3b663429dc4689ee728d2fb5c7d8d5034476f7`

The snapshot files are not copied into this repository. The connector could resolve their Git blob identities but did not return the large file contents, so no full-snapshot validation or task-level score is claimed in this change. The validator is ready to run against an exact downloaded snapshot.

The source definitions used to ground the adapter are `doc/DATABASE.md`, `doc/STRATEGIES.md`, `sorrydb/database/sorry.py`, `sorrydb/utils/verify.py`, and `sorrydb/utils/verify_lean_interact.py` at the pinned SorryDB revision.

## Research direction

The next useful artifact is a separate, signed proof-verification receipt carrying the source-goal reproduction result, build/kernel status, exact-one-sorry removal, sibling-sorry preservation, proof artifact digest, checker version and approved-axiom policy. Semantic audit receipts should separately bind:
- non-vacuity / counterexample checks;
- hypothesis and assumption preservation;
- theorem statement correspondence to the source problem;
- axiom and unsafe declaration audit;
- independent human review where automated checks cannot establish intent.

No single aggregate pass rate should hide failures in any of these dimensions.
