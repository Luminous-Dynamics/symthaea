# PITB SorryDB snapshot and split integrity auditor

A standalone Rust auditor for a frozen SorryDB full snapshot and evaluation split. This crate does not solve Lean goals; it establishes whether benchmark tasks and their source identities are reproducible enough to be scored.

## Frozen upstream identity

| Item | Pinned value |
|---|---|
| Repository | SorryDB/SorryDB |
| Repository commit | 74b3b63f8ba890d675f362924b21ac55db410383 |
| Full snapshot | data/SorryDB_2601/SorryDB_2601.json |
| Full snapshot Git blob SHA-1 | 5fafddac467af6d30d8f5c8731d6edec9c273a47 |
| Evaluation split | data/SorryDB_2601/SorryDB_2601_1000_evaluation_split.json |
| Evaluation split Git blob SHA-1 | af3b663429dc4689ee728d2fb5c7d8d5034476f7 |
| README-documented full size | 2,601 |
| Observed full size at pinned blob | 5,663 |
| Evaluation size | 1,000 |

**Release-integrity finding:** the README still describes the SorryDB_2601 full snapshot as 2,601 sorries, while the exact full-snapshot blob at the pinned repository commit contains 5,663 records. The audit reports this as documentation drift rather than silently changing the source count. The full dataset and split are not copied into this repository; CI checks them out from the pinned upstream commit and verifies their Git blob identities.

## What the pinned evaluation split establishes

The exact 1,000 evaluation records have 1,000 unique task IDs and 1,000 unique goal strings. All 1,000 occur in the full snapshot with exactly matching parsed task records. The split covers all 78 repositories, spans 107 repository-commit pairs and 339 repository/file clusters, ranges from 1 to 21 tasks per repository, and has up to 20 tasks in one repository/file cluster. It contains 14 distinct Lean toolchain strings and two records with start_column = 0.

The zero-based columns are retained verbatim. Do not convert them to one-based coordinates without checking the pinned source and SorryDB's coordinate convention. The tasks are clustered by project and file, so statistical comparisons should use repository/file-level clustered resampling, not an IID bootstrap over 1,000 rows.

## Audit checks

The Rust verifier:
1. Verifies the exact upstream Git blob SHA-1 of both raw JSON files.
2. Computes SHA-256 over each raw byte stream.
3. Parses a strict SorryDB record schema and rejects unknown fields.
4. Validates immutable repository commit SHA, Lean version, safe relative source path, line/column range, ISO-8601 metadata and exact commit-pinned source permalink.
5. Rejects duplicate task IDs and duplicate goal strings.
6. Requires every evaluation record to match the full snapshot record exactly, by task ID and every parsed field.
7. Reports repository/toolchain distributions, zero-based coordinates and source-file clustering.
8. Emits explicit warnings about stale documentation and source-reconstruction/semantic-audit requirements.

The auditor deliberately treats SorryDB's task ID as an opaque upstream identifier. A separate identity-versioning study is required before replacing it with a new canonical PITB task identity.

## Reproduce

    cargo fmt --check --manifest-path tools/pitb-sorrydb-audit/Cargo.toml
    cargo test --locked --manifest-path tools/pitb-sorrydb-audit/Cargo.toml
    cargo run --locked --manifest-path tools/pitb-sorrydb-audit/Cargo.toml -- --self-test

For a local checkout of the pinned upstream repository:

    cargo run --locked --manifest-path tools/pitb-sorrydb-audit/Cargo.toml -- \
      --verify \
      /path/to/SorryDB/data/SorryDB_2601/SorryDB_2601.json \
      /path/to/SorryDB/data/SorryDB_2601/SorryDB_2601_1000_evaluation_split.json \
      /tmp/sorrydb-audit-report.json \
      --pinned-source SorryDB/SorryDB@74b3b63f8ba890d675f362924b21ac55db410383

## Verify a task against committed source bytes

The auditor can independently verify one SorryDB row against a local repository checkout. It checks the immutable `repo.commit`, the configured `origin`, the committed `HEAD:lean-toolchain`, and the committed `HEAD:<source-path>` object. It converts the database's zero-based character columns to UTF-8 byte offsets and requires the exact source span to contain `sorry`.

```sh
cargo run --locked --manifest-path tools/pitb-sorrydb-audit/Cargo.toml -- \
  --verify-checkout /path/to/one-sorry-record.json \
  /path/to/repository-checkout \
  /tmp/source-checkout-receipt.json
```

This reads from Git objects, not working-tree files. A local uncommitted edit cannot change the bytes being verified. The resulting `pitb.sorrydb-source-checkout.v0` receipt includes the task ID, repository commit, Lean version, source coordinates and source-file SHA-256. It always sets `scoring_admission=false`: confirming the token is present does not reproduce the elaborated goal, build the project, or establish the informal specification.

The exact-head workflow runs an integration control that commits a synthetic Lean file, dirties the working copy, and verifies that the source checker still reads the original committed `sorry` span. It also validates the generated receipt against `sorrydb-source-checkout-v0.schema.json`.

## Replay the actual Lean goal, not just the source token

The auditor also supports `--replay-goal`. It first verifies the repository remote, commit, committed `lean-toolchain`, source-file Git object and exact `sorry` span. It then creates a temporary sibling Lean module from the committed source with `trace_state` inserted immediately before the recorded `sorry`, runs that module with `lake env lean` under the pinned project, and compares the live tactic state to the recorded `debug_info.goal` after whitespace normalization.

The replay receipt binds the task ID, repository and tree identity, source/goal/toolchain/Lake-manifest digests, exact replay command and output-log digest. It always sets `scoring_admission=false`: reproducing a goal is not a proof, independent kernel check or semantic specification review.

```sh
cargo run --locked --manifest-path tools/pitb-sorrydb-audit/Cargo.toml -- \
  --replay-goal /path/to/one-sorry-record.json \
  /path/to/exact-repository-checkout \
  /tmp/goal-replay-receipt.json \
  /tmp/goal-replay.log
```

The frozen integration task is SorryDB ID `7956fe8e4946a9ddf7098ee84c69669db5bc902715dad6e7cdf0db121aa5bfce`: `FKS.table_1_prop` in `PrimeNumberTheoremAnd/FioriKadiriSwidinsky.lean`, repository commit `ddcb6542f2b62883c8642d3db4ecc115afcce33e`, Lean `v4.26.0`. Its recorded coordinates are line 44, zero-based character columns 137–142. This is a nontrivial analytic-number-theory obligation, not a synthetic theorem. The workflow additionally pins the committed Lake manifest blob `c0405e3c9643b00b1ce6a8120516ca482e6fe4ec`, builds the upstream project with the pinned Lean toolchain, and requires the replayed goal to match.

## From task record to admissible proof result

A task is not independently reproduced merely because its JSON row contains a commit and a pretty-printed goal. For each scored task:

1. Clone the recorded repo.remote.
2. Checkout the immutable repo.commit, not the branch tip.
3. Install the exact repo.lean_version.
4. Rebuild the unmodified repository and capture the Lake manifest, repository tree and toolchain digests.
5. Locate the recorded source range without coordinate normalization.
6. Reproduce the sorry in the Lean environment and compare the live goal with debug_info.goal.
7. Run an independent kernel/checker on the proof replacement.
8. Separately audit that the formal statement expresses the intended property and is not vacuous or weakened.

SorryDB's database documentation warns that a change anywhere in a repository can affect a task's meaning. A branch name, current source tree or current Mathlib cache is not a substitute for the recorded repository commit and Lean version.

## Benchmark landscape and PITB's opportunity

The official SorryDB scoreboard reports 57.2% Pass@1 for Aleph Prover and 63.2% combined coverage for the listed approaches on the 1,000-task snapshot. Its paper emphasizes that available proof systems are complementary, not strictly ordered by a single capability. These numbers are baselines for a matched run, not performance claims for PITB.

Adjacent benchmarks measure different work:

- AutoformBench: 341 audited advanced-mathematics formalization tasks from 24 sources; the release reports GPT-6 Astra at 281/341 (82.4%) in its source-present condition. This requires building missing definitions, lemmas and formal infrastructure, so it is not directly comparable to filling an existing SorryDB proof hole.
- FVSpec: 9,415 Lean challenges translated from real-world Python property-based tests; it measures software-property formalization and proof generation, not only mathematical theorem completion.
- LeanGraphSearch: graph-based multi-hop premise retrieval reports up to 12% recall improvement over LeanSearch v2 on MathlibQR and improves complete-route coverage on MathlibMPR. PITB should test whether retrieval gains translate into verified exact-target completions and lower cost.
- ICML 2026 benchmark audit: a corpus-scale audit reports 4,833 findings, including 398 mechanically certified defects such as vacuity, counterexamples and unsound axioms. Kernel acceptance alone does not establish that a benchmark statement matches the intended source property.

The central PITB shift is to report **assurance-adjusted completion**, not raw proof success alone: exact source reconstruction, exact goal match, specification fidelity, non-vacuity/assumption sanity, independent kernel verification, and proof-maintenance validity. Failures at each stage remain separately visible; a failed semantic audit cannot be hidden by a passing kernel check.

## Recommended qualification targets

These are proposed targets, not achieved results.

| Metric | Pilot target | Strong target |
|---|---:|---:|
| Pinned task-byte reproducibility | 100% | 100% |
| Evaluation split exact membership | 100% | 100% |
| Source checkout/build/goal reproduction | 95% sampled tasks | 99% all scored tasks |
| Exact statement + source fidelity audit | 95% sampled tasks | 99% scored tasks |
| Vacuity/assumption sanity checks | 100% tasks | 100% plus human audit for flagged cases |
| Independent kernel acceptance | 100% claimed solutions | 100% plus second-checker replay |
| Assurance-adjusted Pass@1 | Beat 57.2% on matched budget | Exceed 63.2% combined public coverage on same frozen split |
| Cost per assurance-adjusted completion | Establish baseline | Reduce 25% without reducing verified coverage |
| Proof maintenance after upstream update | Establish baseline | 90% survive non-semantic updates or are repaired within budget |

A fair SOTA comparison must freeze model, scaffold, tool access, candidate pool, total wall/CPU/GPU budget, token budget, stopping policy, source visibility and verification path. Report Pass@1/Pass@k, unique solved tasks, cross-system union, cost per verified completion, abstention/coverage and confidence intervals clustered by repository and source file. Do not compare raw percentages across SorryDB, AutoformBench and FVSpec as if they measured the same task.

## Assurance-adjusted proof-run receipts

The snapshot auditor also accepts an optional JSONL file of proof attempts:

```sh
cargo run --locked --manifest-path tools/pitb-sorrydb-audit/Cargo.toml -- \
  --verify \
  /path/to/SorryDB_2601.json \
  /path/to/SorryDB_2601_1000_evaluation_split.json \
  /tmp/sorrydb-audit-report.json \
  --runs /path/to/proof-runs.jsonl \
  --pinned-source SorryDB/SorryDB@74b3b63f8ba890d675f362924b21ac55db410383
```

Each receipt binds the original SorryDB task ID to the exact repository commit, Lean version, source path and coordinate range, plus a SHA-256 of the recorded proof goal. It also carries digests for the source file, replacement patch, proposed proof, checker receipt, semantic-gate receipt and source-binding receipt.

A run is counted as `verified_success` only if all of the following are true:

- the proof was inserted and the exact recorded sorry range was replaced;
- the original sorry is absent after replacement, and the target proof term no longer depends on `sorryAx`;
- the target elaborates as a proposition (`Prop`);
- the declaration signature digest before and after the replacement is identical, preventing theorem weakening or target substitution;
- the project builds and Lean's kernel accepts the resulting proof;
- independent semantic and non-vacuity gates pass;
- source binding passes against the pinned repository commit and exact range;
- no new axioms and no new sorry placeholders were introduced.

The auditor rejects success receipts missing any gate, receipts for tasks outside the pinned 1,000-task evaluation split, mismatched commits/toolchains/coordinates/goals, invalid digests, duplicate run IDs, and infrastructure failures without a failure type. It reports successful attempts separately from unique tasks solved, with per-strategy attempt rate, unique-task coverage, mean successful wall time, total recorded cost and cost per unique verified task. Ten successful retries of one task remain one unique solved task.

The run verifier validates the receipt structure and identity bindings. It does not authenticate the referenced receipt bytes or independently rebuild the source repository. The separate checker/source-binding/semantic-gate evidence must be reopened and verified by the surrounding evaluator before using a receipt in a public score. In particular, a kernel pass alone is never enough for PITB's assurance-adjusted success label.

A synthetic run ledger is exercised by the exact-head workflow. Its JSONL bytes are SHA-256-bound by the report, and CI validates each row against `sorrydb-proof-run-v0.schema.json`. The same task may have multiple attempts, but run IDs must be unique. Per-strategy attempt counts and verified successes are reported separately; these are not deduplicated task-level leaderboard scores.


Historic SorryDB task rows do not contain a source-file SHA-256. The task snapshot auditor therefore continues to establish dataset membership and reproducibility inputs, while each proof-run receipt supplies the source digest and carries an independent source-binding receipt. This is not a substitute for actually checking out and hashing the source file at the recorded commit.

## Trust boundary

This crate verifies dataset bytes, record integrity and exact split membership. It does not download/build all 78 source repositories, prove that the sorry goal is the intended theorem, audit its assumptions for vacuity, or verify any proposed proof. Those are separate gates and must produce their own receipts before PITB counts a task as assurance-adjusted success.
