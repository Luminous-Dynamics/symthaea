# PITB SorryDB benchmark integrity auditor

A standalone Rust validator for immutable SorryDB task snapshots and proof-replacement run receipts.

## Why this layer exists

SorryDB is a real-world Lean benchmark assembled from open proof obligations in active repositories. Its records identify the repository remote, branch, commit, Lean version, source path and sorry coordinates, along with a pretty-printed proof goal. The published pipeline rebuilds the original repository and checks proposed proof replacements in that environment.

PITB keeps this task-discovery value but adds a stricter admission rule for a **verified success**. Compiling a candidate is necessary, but does not independently establish that the exact source sorry was replaced, that the goal still corresponds to the intended target, that the proof is non-vacuous, or that the agent did not add axioms or other sorries.

## Pinned historical cohort

The external source manifest is `fixtures/sorrydb-2601-source-manifest.json`.

| Object | Immutable identity |
|---|---|
| SorryDB repository commit | `74b3b63f8ba890d675f362924b21ac55db410383` |
| Full snapshot Git blob | `5fafddac467af6d30d8f5c8731d6edec9c273a47` |
| 1,000-task evaluation split Git blob | `af3b663429dc4689ee728d2fb5c7d8d5034476f7` |
| License | Apache-2.0, blob `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64` |
| Published paper | [ICML 2026 / PMLR 306](https://proceedings.mlr.press/v306/letson26a.html) |

The paper reports 5,663 tasks from 78 repositories in the January 2026 snapshot and evaluates a 1,000-task slice selected for recency and repository diversity. The source manifest pins both task-data files by Git blob ID. The large dataset bytes are intentionally not copied into this adapter branch, so this project does not claim to have independently parsed the full 1,000-row split in CI.

## Stable task identity

The upstream SorryDB `id` remains a source identifier. PITB computes its own SHA-256 task fingerprint over:
- repository remote, branch, immutable commit and Lean version;
- source path and start/end coordinates;
- digest of the recorded proof goal.

This prevents a task from silently retaining identity when its target location, repository commit, Lean version or goal changes. It also avoids depending on undocumented or version-sensitive source ID derivation.

Task paths must be repository-relative and cannot contain traversal segments. Coordinates are 1-based and ordered. Repository commits must be full 40-character Git SHA-1 identifiers. When a source file digest is available, it is bound to all subsequent proof-run receipts.

## Verified-success contract

A proof run can be labeled `verified_success` only when all of these checks pass:

1. The immutable task fingerprint, repository commit, Lean version and source-file digest match.
2. The proposed replacement is inserted at the exact sorry range and that sorry is absent afterward.
3. The repository builds and Lean's kernel accepts the proof.
4. An independent semantic gate passes, including a non-vacuity check.
5. No new axioms and no new `sorry` placeholders are introduced.

The run must carry digests for the replacement patch, proof artifact, checker receipt, semantic-gate receipt and source-binding receipt. The current tool validates digest syntax and identity binding; it does not authenticate or independently reopen those receipt bytes. A surrounding verifier must do that before results are admitted to an external benchmark report.

Historical SorryDB records do not include a source-file SHA-256. Such records remain usable as reproducible task locators, but they cannot receive this stronger `verified_success` label until a checkout at the exact commit supplies and pins the source bytes.

## Run

```sh
cargo fmt --check --manifest-path tools/pitb-sorrydb-integrity/Cargo.toml
cargo test --locked --manifest-path tools/pitb-sorrydb-integrity/Cargo.toml
cargo run --locked --manifest-path tools/pitb-sorrydb-integrity/Cargo.toml -- --self-test

# Generate a synthetic report and validate its schema:
cargo run --locked --manifest-path tools/pitb-sorrydb-integrity/Cargo.toml -- \
  --self-test-report /tmp/sorrydb-integrity-report.json

# Audit a downloaded SorryDB JSON snapshot:
cargo run --locked --manifest-path tools/pitb-sorrydb-integrity/Cargo.toml -- \
  --tasks SorryDB_2601_1000_evaluation_split.json \
  --runs proof-runs.jsonl \
  --report /tmp/sorrydb-audit.json
```

The task file follows SorryDB's documented top-level `{"sorries":[...]}` shape. Proof runs are JSONL, one independently identified attempt per line.

## Beyond the published leaderboard

The SorryDB leaderboard currently reports a 57.2% Pass@1 result for its highest individual approach and 63.2% for the union of listed approaches. PITB's first target should be a paired reproduction on the exact same 1,000-task split, not an unpaired claim that a new model is superior.

A meaningful next milestone is:
- reproduce the strongest individual baseline with exact task-level receipts;
- beat 57.2% on the same fixed split with paired per-task comparison and repository-clustered confidence intervals;
- improve fresh hidden tasks from later SorryDB crawls to measure contamination resistance;
- report cost per verified proof, wall time, proof size, source/semantic audit rejection rate and maintainer acceptance separately from Pass@1;
- retain a failure taxonomy: proof-search failure, context/retrieval failure, environment/build failure, stale task, source mismatch, specification defect and semantic-gate rejection.

The benchmark's most important additional metric is not merely whether Lean accepts a term, but whether the exact intended open obligation was repaired faithfully in its original project context.
