# PITB SorryDB snapshot and semantic-integrity gate

A standalone Rust validator for frozen SorryDB datasets and exact per-task reproduction receipts.

## Why this is a separate benchmark gate

SorryDB is a live benchmark of open proof obligations taken from real Lean repositories. Its task identity includes the upstream repository, branch, commit, Lean version, source path and coordinates. A repository change can alter a task's meaning or available context even when the local theorem text does not change. PITB therefore treats the benchmark snapshot and the source checkout as part of the task—not as incidental metadata.

SorryDB's paper reports a 1,000-task evaluation snapshot drawn from 78 real-world projects. The official paper reports that approaches are complementary; its public results include a combined 35.7% result and a 30.3% single agentic result on that evaluation slice. These results are not directly comparable to isolated theorem benchmarks or formalization benchmarks; model, scaffold, source access, resource budgets and stopping policy matter.

## Frozen SorryDB-2601 identities

The checked-in provenance manifest pins:
- SorryDB source repository: `https://github.com/SorryDB/SorryDB`
- source repository commit: `74b3b63f8ba890d675f362924b21ac55db410383`
- full dataset Git blob: `5fafddac467af6d30d8f5c8731d6edec9c273a47`
- 1,000-task evaluation split Git blob: `af3b663429dc4689ee728d2fb5c7d8d5034476f7`
- full database task count: 2,601
- held-out evaluation task count: 1,000
- source license: Apache-2.0
- evaluation role: test

The data files are not duplicated in this crate. Supply the exact upstream JSON files when running the validator; it recomputes Git blob IDs with `git hash-object --stdin` and rejects mismatched bytes. The manifest is an identity record, not a claim that the tasks were reproduced.

## Run

```sh
cargo fmt --check --manifest-path tools/pitb-sorrydb-integrity/Cargo.toml
cargo test --locked --manifest-path tools/pitb-sorrydb-integrity/Cargo.toml
cargo run --locked --manifest-path tools/pitb-sorrydb-integrity/Cargo.toml -- --self-test

cargo run --locked --manifest-path tools/pitb-sorrydb-integrity/Cargo.toml -- \
  --verify \
  /data/SorryDB_2601.json \
  /data/SorryDB_2601_1000_evaluation_split.json \
  tools/pitb-sorrydb-integrity/fixtures/sorrydb-2601-provenance.json \
  /data/sorrydb-task-reproduction-receipts.jsonl \
  /reports/sorrydb-integrity-report.json
```

Each line of the receipt JSONL is one `pitb.sorrydb-task-reproduction.v0` record. Exactly one receipt must exist for every evaluation task.

## Reproduction receipt

A task receipt is bound to the exact SorryDB task ID, repository remote and commit, Lean version, source path and 1-based coordinates. It includes:
- SHA-256 for the exact dataset debug-goal text;
- source file and source-range digests;
- extracted elaborated-goal digest;
- Lean toolchain and optional Lake manifest digests;
- explicit checks for exact source-location reproduction, sorry marker, goal extraction, `Prop` target and clean project build;
- a goal-alignment result (`exact`, `normalized`, `reviewed`, `mismatch`), with a review receipt required for `reviewed`;
- independent semantic, vacuity and assumption audit states, each with its own evidence receipt when run.

The tool rejects unknown/duplicate tasks, missing receipts, duplicate receipts, source identity drift, an evaluation item absent from the full snapshot, a task record that differs between the full database and evaluation split, invalid source coordinates, a debug URL that does not bind the pinned commit/path, and Git blob byte mismatches.

A task is classified as reproducible only when source location, sorry marker, goal extraction, proposition type and full project build all pass and the extracted goal is not marked mismatched. Semantic/vacuity/assumption findings are reported independently. A reproduction pass is not a semantic-clean bill of health.

## Critical trust boundary

This crate checks task/snapshot identity, dataset bytes and digest syntax/binding of receipt fields. It does **not** itself clone 1,000 projects, run Lean, authenticate source/build log bytes, prove that a semantic audit was complete, or independently certify the author/reviewer of a receipt. A benchmark runner must open and authenticate the referenced evidence artifacts before leaderboard admission.

The gate deliberately sets `scheduler_admission=false`. A kernel-checked proof establishes the pinned Lean proposition; it does not establish that the proposition faithfully encodes the source repository's intended property.

## Semantic audit taxonomy

PITB should separately record:
- proposition and target sanity: target really has type `Prop`, no accidental `True`/tautology collapse, no unconstrained placeholders;
- vacuity: contradictory or impossible premises, unreachable hypotheses, empty-domain assumptions and hypotheses that make the theorem trivially true;
- assumptions: unexpected `sorryAx`, custom axioms, `admit`, unsafe trust escapes and unreviewed classical/oracle assumptions;
- translation/source fidelity: exact source range, imported declaration identities, and whether the benchmark goal is the same proposition the source project exposed;
- proof verification: independent Lean kernel replay at the original commit/toolchain, with no `sorry`, `admit` or new axioms in the submitted proof.

These signals are separate because recent benchmark audits have demonstrated mechanically certified counterexamples, vacuous theorems and unsound axioms, as well as semantic faults such as missing hypotheses and incorrect translations. Merely building a Lean file does not detect all such defects.

## Comparing against the field

Use a common frozen subset and identical resource ceilings:
- SorryDB: real open theorem completion and repository integration;
- AutoformBench: construction of missing definitions, intermediate lemmas and formalization infrastructure;
- FVSpec: real software-property formalization from property-based tests;
- DAPS / LeanGraphSearch / LeanSearch v2 / BM25: premise retrieval;
- no-retrieval and symbolic tactics: control baselines.

Primary outcomes:
1. verified exact-target completion after independent source reproduction;
2. semantically qualified completion after vacuity/assumption/specification audits;
3. cost and wall time per qualified completion;
4. held-out repository and time-split generalization;
5. proof maintenance under later repository changes;
6. useful minimal-support-route recovery under fixed target, environment and prover budget.

Secondary diagnostics: Pass@1/Pass@k, top-K recall, candidate diversity, declaration coverage, token use, compile time, generated declaration count, abstention calibration and human repair time. Retrieval scores from different systems must not be treated as calibrated probabilities.

## Synthetic controls

`--self-test` checks Git blob pinning, split membership, exact source identity, debug-goal digest, non-Prop rejection, mismatched goal rejection, duplicate receipts and the one-receipt-per-evaluation-task requirement. Synthetic tests exercise the contract; they are not a measured SorryDB evaluation.
