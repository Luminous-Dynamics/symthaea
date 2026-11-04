# PITB SorryDB_2601 snapshot integrity validator

Standalone Rust validator for the frozen SorryDB_2601 snapshot and its 1,000-task evaluation split.

## Exact upstream pin

The pin is not a moving `master` reference. It records the SorryDB source repository commit, the full dataset Git blob and the evaluation split Git blob:

| Item | Pinned identity |
|---|---|
| Upstream repository | `SorryDB/SorryDB` |
| Repository commit | `74b3b63f8ba890d675f362924b21ac55db410383` |
| Full dataset | `data/SorryDB_2601/SorryDB_2601.json` |
| Full dataset Git blob | `5fafddac467af6d30d8f5c8731d6edec9c273a47` |
| Evaluation split | `data/SorryDB_2601/SorryDB_2601_1000_evaluation_split.json` |
| Evaluation split Git blob | `af3b663429dc4689ee728d2fb5c7d8d5034476f7` |
| Full task count | 2,601 |
| Evaluation task count | 1,000 |
| Source repository count | 78 |
| Upstream license | Apache-2.0 |

The snapshot pin lives at `snapshot/SorryDB_2601.pin.json`. The source files are not copied into this crate; users download them from the pinned upstream commit and pass their local paths to the validator. This keeps the repo small and makes source identity explicit.

## What is checked

For every SorryDB task, the validator checks:
- non-empty task ID;
- HTTPS repository URL, branch, 40-character repository commit and Lean version;
- safe relative `.lean` source path with no path traversal;
- valid start/end line and column ordering;
- non-empty proof goal and HTTPS source URL;
- blame and inclusion metadata;
- unique task IDs and unique full source identities;
- repository-registry membership when a registry is present.

**Coordinate convention:** SorryDB stores 1-based line numbers and zero-based columns. The validator explicitly accepts column zero. This convention is confirmed by the upstream `position_to_index` implementation used to replace and verify a sorry.

The validator also computes toolchain distributions, per-repository task counts, repeated normalized-goal groups, unique source locations and raw SHA-256 digests.

For the full/split relationship it checks that:
- the full and split inputs have the pinned Git blob IDs (using `git hash-object --stdin`);
- their task counts match the frozen 2,601 / 1,000 denominators;
- the full snapshot has the pinned 78 unique repositories;
- every evaluation task ID occurs in the full snapshot;
- each split task has the same source identity as its full-snapshot counterpart.

## Run

```sh
cargo fmt --check --manifest-path tools/pitb-sorrydb-integrity/Cargo.toml
cargo test --locked --manifest-path tools/pitb-sorrydb-integrity/Cargo.toml
cargo run --locked --manifest-path tools/pitb-sorrydb-integrity/Cargo.toml -- --self-test

# First obtain both JSON files from SorryDB at commit
# 74b3b63f8ba890d675f362924b21ac55db410383.
cargo run --locked --manifest-path tools/pitb-sorrydb-integrity/Cargo.toml -- \
  --verify \
  /path/to/SorryDB_2601.json \
  /path/to/SorryDB_2601_1000_evaluation_split.json \
  tools/pitb-sorrydb-integrity/snapshot/SorryDB_2601.pin.json \
  /tmp/sorrydb-2601-integrity-report.json
```

Git must be installed because the validator uses Git's own `hash-object` implementation to verify the exact upstream blob identity. It does not rely on a reimplemented Git hash.

## What this does not prove

A valid dataset record is not proof that the mathematical proposition is non-vacuous, nontrivial or faithful to the intended source. The tool does not clone 78 repositories, rebuild their Lean projects, compare the stored goal to a fresh Lean REPL goal, inspect hypotheses or axioms, or run a proposed proof.

The next assurance layer should issue a per-task *proof-substitution receipt* binding:
1. repository URL, branch, commit and exact source file blob;
2. original `sorry` coordinates and freshly reproduced Lean goal digest;
3. unchanged declaration statement/type digest before and after replacement;
4. candidate proof digest and exact modified source-file digest;
5. Lean toolchain, Lake manifest, dependency environment and build command;
6. baseline project build and post-substitution project build;
7. before/after `sorry` counts and target-specific kernel check;
8. axiom/declaration dependency diff and complete replay evidence.

Do not infer theorem correctness from a single `lake build` return code. The exact target must be resolved in the pinned source, its type must remain unchanged, the modified file/project must compile, and the target must no longer be a sorry. A downstream verifier must replay the source artifact and receipts.

## How this fits PITB

SorryDB measures whether agents can close real open obligations in ongoing Lean repositories. AutoformBench measures constructing missing formal infrastructure for advanced mathematical formalization. FVSpec measures formal verification tasks derived from real-world property-based tests. They are complementary tracks—not interchangeable task pools.

PITB adds a shared assurance/evaluation layer:
- frozen input and source provenance;
- source-to-goal identity and semantic audit;
- typed dependency / retrieval receipts;
- counterfactual minimal-support antichains;
- exact target proof substitution and replay;
- cost, latency, compute, abstention and maintainability.

The ambition is not to claim a single aggregate score across unlike tasks. It is to report a capability vector and a traceable evidence chain for every claimed success.
