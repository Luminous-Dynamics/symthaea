# Public development fixture: streaming top-k updates V0

**Status:** Source-only public development material. Not an admitted CEB task, not an executable evaluation fixture, not a sealed holdout, and not a performance result.

**Canonical family:** `performance-correctness`  
**Candidate dossier:** [CEB Rust task/oracle candidates V0](../../../../docs/research/CEB_RUST_TASK_ORACLE_CANDIDATES_V0_2026-11-06.md)  
**Machine-readable requirement/oracle crosswalk:** [CEB Rust candidate oracle traceability V0](../../../../docs/research/CEB_RUST_CANDIDATE_ORACLE_TRACEABILITY_V0_2026-11-06.json)  
**Admission authority:** [#10609](https://github.com/Luminous-Dynamics/symthaea/pull/10609)  
**Optimization-integrity controls:** [#10683](https://github.com/Luminous-Dynamics/symthaea/issues/10683) / [#10684](https://github.com/Luminous-Dynamics/symthaea/pull/10684)

## Intended development objective

Maintain exact top-k ranking as keys are updated. The current baseline pushes every version into a binary heap and does not invalidate older versions. A later update therefore leaves a stale copy of the same key eligible for the result set.

This fixture demonstrates only the **stale-entry semantic defect**. It does not establish an efficient update/removal algorithm, bounded compaction, memory ceiling, workload distribution, cold-start or unseen-input performance, or any measured speedup.

## Workspace outline

- `rank-contract`: item/key/score contract.
- `topk-core`: binary-heap baseline with deterministic ordering but no per-key update invalidation.
- `workload`: deterministic update sequence that shows why a stale old score can appear alongside the replacement score.

## Integrity and safety boundary

- This fixture is committed to a public repository. It is exposed development material and can never become sealed or hidden holdout material.
- It has no CEB task ID, task manifest, repository slot, partition, difficulty assignment, environment pin, reference patch, grader digest, admission status or run receipt.
- The source has **not** been compiled, tested, benchmarked or executed. No Cargo commands were run.
- No performance measurement, expert reference speedup or hardware profile is claimed.
- No candidate-controlled code, build scripts, procedural macros, tests or graders were executed.
- No reference solution or hidden test suite is included.
- The workspace is a starting point for static reviewer inspection. It must not be promoted to `fixture_ready` or admitted until the canonical #10609 requirements and #10064/#10369 gates are independently satisfied.

## Known authoring limitations

- No `Cargo.lock` is present because no dependency resolution was performed.
- The deterministic scenario exposes one semantic defect but does not yet implement the complete insert/update/remove/boundary/numeric-domain matrix in the dossier.
- No workload, CPU/memory limit, hardware, cache policy or measurement protocol has been specified. No speedup claim can be made from this fixture.
