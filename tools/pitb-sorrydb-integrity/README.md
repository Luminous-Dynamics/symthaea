# PITB SorryDB snapshot integrity verifier

A standalone Rust verifier for the frozen SorryDB-2601 snapshot and its 1,000-task evaluation split.

## Pinned upstream cohort

| Item | Pinned identity |
|---|---|
| Benchmark | SorryDB-2601 |
| Upstream repository | `SorryDB/SorryDB` |
| Upstream repository commit | `74b3b63f8ba890d675f362924b21ac55db410383` |
| Full snapshot | `data/SorryDB_2601/SorryDB_2601.json` |
| Full snapshot Git blob | `5fafddac467af6d30d8f5c8731d6edec9c273a47` |
| Full snapshot size | 2,601 sorry tasks |
| Evaluation split | `data/SorryDB_2601/SorryDB_2601_1000_evaluation_split.json` |
| Evaluation split Git blob | `af3b663429dc4689ee728d2fb5c7d8d5034476f7` |
| Evaluation split size | 1,000 tasks |

The large upstream JSON assets are not duplicated in this repository. The upstream audit downloads the exact raw files from the pinned repository commit and checks their Git blob object IDs before parsing them.

## SorryDB task identity

The pinned SorryDB implementation constructs a stable ID as SHA-256 over Python's `json.dumps(sort_keys=True)` serialization of the Sorry dataclass, excluding the ID field and `metadata.inclusion_date`. The inclusion date is excluded so nightly re-indexing does not change identity. The canonical record includes the repository remote/branch/commit/Lean version, source coordinates, debug goal and blame metadata.

This verifier independently reproduces the Python serialization rules (including ASCII Unicode escaping and the default JSON separator spaces) and checks every record ID. It also validates:
- complete 40-character repository Git commits;
- recorded Lean toolchain version;
- normalized relative source paths and ordered source ranges;
- goal text and GitHub source URL bound to the same commit/path/line;
- timezone-aware ISO timestamps and blame metadata;
- unique task IDs, repository identities, goals and source locations;
- every evaluation-split task and repository/commit is present in the full snapshot.

## Run

```sh
cargo fmt --check --manifest-path tools/pitb-sorrydb-integrity/Cargo.toml
cargo test --locked --manifest-path tools/pitb-sorrydb-integrity/Cargo.toml
cargo run --locked --manifest-path tools/pitb-sorrydb-integrity/Cargo.toml -- --self-test

cargo run --locked --manifest-path tools/pitb-sorrydb-integrity/Cargo.toml -- \
  --validate \
  /path/to/SorryDB_2601.json \
  /path/to/SorryDB_2601_1000_evaluation_split.json \
  /tmp/sorrydb-integrity-report.json
```

The CLI rejects files whose computed Git blob IDs do not match the pinned release, whose record counts differ from 2,601/1,000, whose IDs do not recompute, or whose evaluation split is not a true subset.

## Integrity boundary

A passing report means:
- the exact pinned snapshot and split files were supplied;
- all task records are structurally valid;
- every SorryDB task ID recomputes under the pinned algorithm;
- the 1,000 evaluation tasks are members of the 2,601 snapshot.

It does **not** mean the source repositories were cloned/rebuilt or that the source coordinates still produce the recorded Lean goal. It does not establish that the formal proposition faithfully captures the intended mathematical claim, is non-vacuous, has sane hypotheses, or is free from inappropriate axioms. It does not validate generated proofs.

The report deliberately fixes these fields to false:
- `source_checkout_reproduction_verified`;
- `goal_to_source_semantic_fidelity_verified`;
- `proof_solution_integrity_verified`.

Those require independent Lean materialization, a source/goal correspondence receipt, non-vacuity and axiom audits, and exact-target kernel replay. This boundary matters: a kernel proof proves the formal proposition, not that the proposition faithfully represents the intended source problem.

## Published baseline and proposed qualification gates

SorryDB's project README reports the following results on its 1,000-task evaluation split: Trivial 2.1% Pass@1; tactics 8.4%; GPT-5.2 6.2% Pass@1 / 13.2% Pass@32; Gemini Flash 3 Agentic 30.3% Pass@1; combined 35.7% Pass@1. These are historical reference points, not matched 2026 model comparisons. A new PITB run must pin model, agent, source/tool access, total budget and stopping rules before claiming improvement.

Proposed qualification targets, not measured results:
- baseline replication within ±1.0 percentage point under the same runner, source visibility and budget;
- first competitive gate: at least 40.7% verified Pass@1 (407/1,000), +5 pp over the published 35.7% reference;
- stretch gate: at least 45.7% verified Pass@1 (457/1,000), +10 pp;
- cost gate: at least 20% lower cost per independently verified task with no more than 1 pp completion regression;
- zero false accepts: every counted success replaces only the registered sorry span, passes the exact target in the pinned Lean environment, and survives clean-checkout repository replay;
- report repo-clustered confidence intervals and per-repository results, not only a pooled percentage.

The static SorryDB score must be supplemented with a fresh temporal cohort and project/owner-disjoint evaluation. Static tasks can leak through public repositories and theorem text; a nightly update is not automatically a clean test split.

## Next layer: from proof filling to proof assurance

1. Materialize each repository at the exact recorded commit and Lean version; build the complete project.
2. Reproduce the exact sorry location and debug goal using a pinned Lean REPL or independent elaborator.
3. Capture the declaration type, statement AST digest, local context, imports, environment digest and axiom footprint.
4. Audit false/vacuous goals, missing or inconsistent hypotheses, accidental weakening and unsound axioms.
5. Replace exactly the sorry range; reject changes to the theorem statement, surrounding declarations or project configuration.
6. Independently replay the kernel proof and clean repository build; bind both to immutable receipts.
7. Feed proven targets into PITB retrieval/support-antichain experiments and then into proof-impact maintenance under later repository changes.

This turns SorryDB from “can an agent make the sorry disappear?” into “can it produce an intended, independently reproducible proof contribution that remains valid as the project evolves?”
