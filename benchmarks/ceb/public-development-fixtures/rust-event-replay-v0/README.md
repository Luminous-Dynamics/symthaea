# Public development fixture: deterministic event replay V0

**Status:** Source-only public development material. Not an admitted CEB task, not an executable evaluation fixture, not a sealed holdout, and not a performance result.

**Canonical family:** `issue-reproduction-debugging`  
**Candidate dossier:** [CEB Rust task/oracle candidates V0](../../../../docs/research/CEB_RUST_TASK_ORACLE_CANDIDATES_V0_2026-11-06.md)  
**Machine-readable requirement/oracle crosswalk:** [CEB Rust candidate oracle traceability V0](../../../../docs/research/CEB_RUST_CANDIDATE_ORACLE_TRACEABILITY_V0_2026-11-06.json)  
**Admission authority:** [#10609](https://github.com/Luminous-Dynamics/symthaea/pull/10609)  
**Execution-isolation prerequisite:** [#10064](https://github.com/Luminous-Dynamics/symthaea/issues/10064)

## Intended development objective

Provide a deterministic, reproducible example of a duplicate-delivery defect: the current processor appends an effect for every delivery and therefore applies the same event twice when a replay sends the same event identity again.

The intended repair must be derived from the full event-processing specification in the candidate dossier. This source-only fixture demonstrates only the **in-process duplicate-delivery subcase**. It does not implement or establish persistence, crash recovery, acknowledgement atomicity, concurrent processing, or restart safety.

## Workspace outline

- `event-contract`: event identity and result types.
- `event-store`: in-memory processor with the intentionally missing deduplication check.
- `replay-cli`: small consumer that delivers the same event twice to make the state transition visible by inspection.

## Integrity and safety boundary

- This fixture is committed to a public repository. It is exposed development material and can never become sealed or hidden holdout material.
- It has no CEB task ID, task manifest, repository slot, partition, difficulty assignment, environment pin, reference patch, grader digest, admission status or run receipt.
- The source has **not** been compiled, tested, benchmarked or executed. No Cargo commands were run.
- No candidate-controlled code, build scripts, procedural macros, tests or graders were executed.
- No reference solution or hidden test suite is included.
- The workspace is a starting point for static reviewer inspection. It must not be promoted to `fixture_ready` or admitted until the canonical #10609 requirements and #10064/#10369 gates are independently satisfied.

## Known authoring limitations

- No `Cargo.lock` is present because no dependency resolution was performed.
- The fixture is memory-only and cannot prove EVENT-04 through EVENT-07 persistence, crash, acknowledgement or ordering requirements in the candidate dossier.
- It is a deterministic reproduction scaffold, not a complete task or complete reference-oracle input. Any eventual task must separately model and qualify durable commit, acknowledgement, conflict, restart and concurrent replay behavior.
