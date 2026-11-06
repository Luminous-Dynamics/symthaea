# PITB deterministic project-disjoint benchmark split planner

Standalone Rust utility that partitions real-world Lean benchmark tasks by **source repository**, not by individual theorem row. It is intended for SorryDB first, and accepts the same `{ "sorries": [...] }` dataset envelope used by SorryDB.

## Why this matters

The frozen SorryDB_2601 paper evaluation has 1,000 tasks from 78 repositories. Every one of those 78 repositories also occurs in the 5,663-task full snapshot. The existing evaluation slice is therefore task-held-out, not project-disjoint: the same project and its supporting declarations can occur on both sides of a task-level split. This makes it unsuitable as the sole measure of generalization to unseen projects.

This planner creates repository-disjoint folds. All tasks from the same canonical remote are assigned together, even when the records refer to different branches or commits. It also reports repository/commit and repository/file cluster counts so uncertainty estimates can respect the actual dependency structure.

## Algorithm

1. Validate each upstream SorryDB task ID, immutable repository commit, Lean version, HTTPS remote and safe source path.
2. Canonicalize GitHub remotes to lowercase `https://host/owner/repository`, removing a terminal `.git`. The policy is deterministic and versioned; unknown URL shapes fail closed.
3. Group every task from a repository into one indivisible project cluster. Repository commit pairs and source-file clusters are counted separately.
4. Sort clusters by descending task count. Resolve equal-size ties using SHA-256 of `seed || NUL || canonical_repository_key`.
5. Assign each whole repository to the currently lightest fold (task count, then repo count, then a seed-bound fold tie key).
6. Emit a task-to-repository-to-fold ledger and verify exact once-only task/repository coverage.

This is a longest-processing-time greedy balance heuristic. The maximum fold load difference is bounded by the largest indivisible repository cluster; it does not split a large project to force a cosmetically balanced task count.

## Run

```sh
cargo fmt --check --manifest-path tools/pitb-grouped-benchmark-splits/Cargo.toml
cargo test --locked --manifest-path tools/pitb-grouped-benchmark-splits/Cargo.toml
cargo run --locked --manifest-path tools/pitb-grouped-benchmark-splits/Cargo.toml -- --self-test

# Five deterministic repository-disjoint folds for the pinned SorryDB_2601 evaluation split:
cargo run --locked --manifest-path tools/pitb-grouped-benchmark-splits/Cargo.toml -- \
  --plan /path/to/SorryDB_2601_1000_evaluation_split.json \
  /tmp/sorrydb-project-disjoint-folds.json \
  --folds 5 \
  --seed sorrydb-2601-project-folds-v1 \
  --source-blob-sha1 af3b663429dc4689ee728d2fb5c7d8d5034476f7
```

The optional `--source-blob-sha1` is verified by the planner against the exact Git blob object digest. The report always records the raw input SHA-256.

## Interpretation

Five folds over the SorryDB_2601 evaluation slice create a project-held-out cross-validation protocol **within that snapshot**. For each fold, all tasks from those repositories are held out together; the remaining folds are the complementary project groups. For a fair comparison:
- do not train or tune on the held-out fold's task labels;
- do not include held-out repository declarations in a cross-project retrieval index unless the experimental arm explicitly measures within-project retrieval;
- keep Mathlib/Lean library access, toolchain, model, scaffold, retry policy, wall-clock and cost budgets matched;
- report each fold separately, then aggregate with repository-level clustered confidence intervals;
- distinguish this *project-disjoint split* from temporal novelty and from model-pretraining contamination.

The planner does not claim these public tasks are unseen by a model. A future benchmark release should add a timestamped, prospective repository cohort (for example, repositories and commits entering SorryDB after a frozen cutoff) for stronger temporal evaluation.

## Synthetic controls

`--self-test` checks:
- canonical Git blob SHA-1 on the empty blob and `abc`;
- deterministic same-seed fold output;
- exact once-only task and project coverage;
- all tasks from URL aliases of the same project remain in one fold;
- balance bounded by the largest indivisible project cluster;
- duplicate upstream task IDs rejected;
- source Git blob pin mismatch rejected.

Synthetic tests exercise the split contract only; they do not measure solver performance.
