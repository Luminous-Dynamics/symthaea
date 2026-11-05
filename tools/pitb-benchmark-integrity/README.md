# PITB benchmark integrity: real-world task admission and proof outcomes

Standalone Rust validator for pinned real-world Lean tasks. The first upstream cohort is SorryDB_2601; the contract is reusable for FVSpec, AutoformBench and later Lean verification datasets.

## Paradigm shift

A benchmark row is not automatically a valid problem because it appears in a dataset. A Lean kernel can certify a proof of a proposition whose source translation is incomplete, vacuous, over-simplified or no longer matches the task. The task itself must first be reproduced at its exact source commit, and the resulting proof must be checked against that same target and environment.

PITB therefore separates four states:

1. **Listed** — the upstream dataset contains a task record.
2. **Reproduced** — exact repository commit, Lean toolchain, source location, build and live goal are independently re-established.
3. **Admitted** — the live goal fingerprint matches the pinned upstream goal and all reproduction gates pass.
4. **Verified success** — a proposed proof replaces the exact target placeholder, compiles, passes the Lean kernel and an independent semantic audit, preserves source integrity, and introduces no new axioms or unsafe declarations.

A listed task that cannot be reproduced is quarantined, not erased. Quarantined tasks remain visible in the denominator and the report separately presents solve rates over admitted tasks and over all listed tasks.

## Pinned upstream cohort: SorryDB_2601

The pin descriptor lives at `fixtures/sorrydb-2601-upstream-pin.json`.

| Artifact | Pinned identity |
|---|---|
| SorryDB repository commit | `74b3b63f8ba890d675f362924b21ac55db410383` |
| Evaluation split | `data/SorryDB_2601/SorryDB_2601_1000_evaluation_split.json` |
| Evaluation split Git blob | `af3b663429dc4689ee728d2fb5c7d8d5034476f7` |
| Evaluation tasks | 1,000 |
| Full `SorryDB_2601` Git blob | `5fafddac467af6d30d8f5c8731d6edec9c273a47` |
| Full snapshot tasks | 2,601 |
| Upstream repository root license | Apache-2.0 |

The official SorryDB paper describes a dynamic benchmark built from 78 real-world Lean projects and reports an experiment on a fixed 1,000-task snapshot. The source database format supplies repository remote/branch/commit, Lean version, sorry path and coordinates, plus a debugging goal. The SorryDB implementation derives its task ID from repository and location metadata and treats repository changes as potentially affecting all task meanings. PITB keeps the upstream IDs but also derives a task-record digest over the normalized pinned record. See the original [SorryDB paper](https://proceedings.mlr.press/v306/letson26a.html), [source repository](https://github.com/SorryDB/SorryDB) and [leaderboard](https://sorrydb.org/).

The repository license does not replace the licenses of individual source projects. This PR pins the upstream dataset but does not redistribute the 1,000 task records, source files, proof text or third-party project contents.

## Input contract

The CLI accepts one snapshot envelope and three JSONL ledgers.

### Snapshot envelope

Binds benchmark ID/version, snapshot ID, upstream repository commit, source evaluation-split path and Git blob ID, task count, repository license identifier, and SHA-256 of the exact normalized PITB task JSONL bytes.

### Normalized task row

Binds the upstream SorryDB task ID, exact source repository remote/branch/commit, Lean toolchain string, safe repository-relative path and 1-based start/end coordinates, plus SHA-256 of the exact debug-goal text.

Task identity is SHA-256 over the serialized normalized task record. The snapshot's task JSONL digest binds the full row sequence, including line order and newline bytes. Duplicate upstream task IDs and duplicate normalized task identities are rejected.

### Reproduction receipt

One receipt is required for every listed task, including failures. It binds:
- exact normalized task-record digest and source task ID;
- checkout commit and Lean toolchain;
- toolchain and environment digests;
- exact source-file digest;
- checkout, build and live-goal-capture receipt digests;
- build result, sorry-anchor presence and observed live-goal digest.

A task is admitted only when the checkout commit and Lean toolchain match, the repository builds, the expected sorry anchor is present, and the observed live goal matches the pinned database goal. Any mismatch produces a quarantine reason. A missing or duplicate reproduction receipt fails closed.

### Proof attempt outcome

One attempt binds a task record, reproduction receipt, prover/version/checkpoint, exact environment/toolchain, proof and patch digests, and verification receipt. A row is counted as `verified_success` only when all seven gates pass:

1. exact target reproduced;
2. source integrity preserved;
3. target placeholder removed;
4. Lean kernel accepted;
5. independent semantic audit passed;
6. no new axioms introduced;
7. no new unsafe declarations introduced.

A `verified_success` without both proof and patch digests is rejected. An attempt against a quarantined task is rejected. Reusing the same task for multiple independent prover attempts is allowed; attempt IDs must remain unique.

## Run

```sh
cargo fmt --check --manifest-path tools/pitb-benchmark-integrity/Cargo.toml
cargo test --locked --manifest-path tools/pitb-benchmark-integrity/Cargo.toml
cargo run --locked --manifest-path tools/pitb-benchmark-integrity/Cargo.toml -- --self-test

# Emit complete synthetic snapshot/task/reproduction/attempt ledgers:
cargo run --locked --manifest-path tools/pitb-benchmark-integrity/Cargo.toml -- \
  --self-test-fixture-dir /tmp/pitb-benchmark-integrity

# Verify an actual normalized benchmark cohort:
cargo run --locked --manifest-path tools/pitb-benchmark-integrity/Cargo.toml -- \
  --verify \
  snapshot.json tasks.jsonl reproductions.jsonl attempts.jsonl report.json
```

The checked-in workflow runs the CLI against a complete synthetic fixture directory, validates every input row and the generated report against strict JSON Schemas, and checks that one task is admitted, one is quarantined for a goal mismatch, and one verified attempt is scored only on the admitted task. These are contract tests, not a measured SorryDB run.

## Industry comparison and PITB's differentiation

| Benchmark/system | What it measures | Reported scope/results | PITB extension |
|---|---|---|---|
| SorryDB | Open proof obligations from active Lean projects | 1,000-task paper snapshot; 78 projects. The public leaderboard currently includes a 57.2% Pass@1 entry for Aleph Prover; this is not directly comparable to other rows without matching harness/budget. | Exact task admission, goal reproduction, proof-integrity and semantic gates; report both admission and solve denominators. |
| FVSpec | Real-world Python property-based tests translated into Lean challenges | 11,039 PBTs, 2,772 source tests lifted into 9,415 Lean challenges, 333 source repositories; the current dashboard reports mean 70% on easy and 49% on hard across sampled model evaluations. | Preserve source PBT → implementation → Lean specification correspondence and test semantic faithfulness, not just proof completion. |
| LeanSearch v2 | Global premise-set retrieval | On a 69-query research-math benchmark, reasoning mode reports 46.1% recovery of ground-truth premise groups within 10 candidates. | Capture candidate bundles, join them to exact task/environment identities, then measure verified route sufficiency and support-antichain recovery. |
| LeanGraphSearch | Multi-hop dependency graph retrieval | Reports up to +12% retrieval recall over LeanSearch v2 on MathlibQR, +3.7% complete annotated route coverage on MathlibMPR, and up to +10% downstream pass-rate gain over LeanSearch v2 on evaluated theorem benchmarks. | Test whether dependency evidence transfers to exact SorryDB/FVSpec tasks and whether routes survive counterfactual ablation. |
| Formal benchmark audit (ICML 2026) | Dataset/harness defect detection | Reports 4,833 audit findings including 398 mechanically certified issues such as counterexamples, vacuous theorems and unsound axioms across audited benchmark corpora. | Make source/goal reproduction, semantic review and negative controls first-class benchmark outputs. |

These are source-reported numbers on different cohorts, models and evaluation protocols. They are not a single head-to-head ranking. PITB should not claim to beat these systems until a locked, matched experiment is complete.

## Benchmarks and targets to pursue

**Integrity targets**
- 100% of scored tasks have immutable source commit, toolchain, source path/range and exact live-goal fingerprint.
- 100% of verified successes pass the Lean kernel, source-integrity, no-new-axiom, no-new-unsafe and independent semantic gates.
- 100% of quarantined tasks remain in published cohort accounting with explicit reasons.
- Zero silent denominator changes between benchmark snapshots.

**Capability targets**
- First reproduce the public SorryDB_2601 results under a recorded model, harness, K, compute and retry budget; do not compare a different setup with a leaderboard headline.
- Then target at least +5 percentage points over the strongest publicly reproducible matched baseline, with a paired confidence interval excluding zero and no increase in semantic defect rate.
- Separately report Pass@1, Pass@k, verified completion per wall-clock hour, cost per verified completion, build/reproduction coverage, and performance by repository, Lean version, mathematical domain and task difficulty.
- On FVSpec, report both proof completion and source-property faithfulness. Do not treat a proof of a translated property as verified software correctness until the translation has been audited.
- On retrieval, report Recall@K and nDCG only as diagnostics. Primary outcomes are complete support-route coverage, exact-target verified completion under a fixed prover budget, useful-novel-premise rate, and cost per verified completion.
- On support-antichain tasks, score route sufficiency, exact minimal-route recovery, common-support precision/recall and probability calibration, while preserving abstention coverage.

**Efficiency target**
Aim for at least 20% lower cost per verified completion at no more than 1 percentage point regression in verified solve rate, on the same task set, hardware/provider class and wall-clock budget. This is a qualification target, not a current measured result.

## Paradigm shifts

1. **From benchmark rows to reproducible task identities.** Every task is a pinned source commit + toolchain + source location + goal fingerprint, not merely a prompt and expected answer.
2. **From binary solve rates to task validity × proof validity.** Report admission, quarantine, proof correctness and semantic fidelity separately.
3. **From kernel-only to semantic assurance.** A kernel proves the formal proposition it receives; it does not prove that a translation captured the source intent. This is especially important for FVSpec and imported mathematics.
4. **From isolated proof attempts to support science.** Connect retrieval receipts and exhaustive support antichains to test which premise bundles are jointly sufficient, which alternatives exist and which declarations recur across all minimal routes.
5. **From leaderboard hill-climbing to sustainable engineering.** Report verified completion per cost/time and proof maintenance/impact, not just Pass@k.

## Trust boundary and limitations

This Rust validator checks JSON structure, identity joins, digest syntax, exact JSONL digest binding, source-location sanity and declared verification gates. It does **not** open or authenticate the bytes referenced by source-file, checkout, build, goal-capture, patch or verification receipt digests; it does not clone/build repositories or run Lean itself; it does not establish that a semantic auditor is independent or correct. Those artifacts must be opened, hash-checked and independently reproduced by the full benchmark harness before its report is accepted as cryptographic evidence.

This PR establishes the admission/scoring contract and synthetic CI tests. It does not run the SorryDB 1,000-task cohort, reproduce the leaderboard, or claim a solver improvement.
