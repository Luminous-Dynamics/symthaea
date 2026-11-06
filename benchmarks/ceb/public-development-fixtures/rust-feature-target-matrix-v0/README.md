# Public development fixture: Cargo feature/target matrix V0

**Status:** Source-only public development material. Not an admitted CEB task, not an executable evaluation fixture, not a sealed holdout, and not a performance result.

**Canonical family:** `feature-target-configuration`  
**Candidate dossier:** [CEB Rust task/oracle candidates V0](../../../../docs/research/CEB_RUST_TASK_ORACLE_CANDIDATES_V0_2026-11-06.md)  
**Machine-readable requirement/oracle crosswalk:** [CEB Rust candidate oracle traceability V0](../../../../docs/research/CEB_RUST_CANDIDATE_ORACLE_TRACEABILITY_V0_2026-11-06.json)  
**Admission authority:** [#10609](https://github.com/Luminous-Dynamics/symthaea/pull/10609)  
**Execution-isolation prerequisite:** [#10064](https://github.com/Luminous-Dynamics/symthaea/issues/10064)

## Intended development objective

Make the explicitly declared feature/target matrix coherent across a small Rust workspace. The baseline contains two independent configuration defects:

1. The client declares a `binary` feature but does not forward it to the protocol dependency, even though its source expects the dependency's binary API when the client feature is selected.
2. The platform adapter declares Unix and Windows source modules unconditionally, although each imports APIs available only on its respective target.

The eventual task must state exactly which feature sets and targets are supported. This fixture only exposes the source-level mismatch; it does not prove a particular cross-target toolchain can build.

## Workspace outline

- `protocol`: dependency with default JSON and optional binary APIs.
- `client`: forwards JSON but intentionally omits forwarding of its binary feature.
- `platform-adapter`: includes Unix and Windows modules unconditionally, creating an incorrect target boundary.

## Integrity and safety boundary

- This fixture is committed to a public repository. It is exposed development material and can never become sealed or hidden holdout material.
- It has no CEB task ID, task manifest, repository slot, partition, difficulty assignment, environment pin, reference patch, grader digest, admission status or run receipt.
- The source has **not** been compiled, tested, benchmarked or executed. No Cargo commands or cross-compilation commands were run.
- No candidate-controlled code, build scripts, procedural macros, tests or graders were executed.
- No reference solution or hidden test suite is included.
- The workspace is a starting point for static reviewer inspection. It must not be promoted to `fixture_ready` or admitted until the canonical #10609 requirements and #10064/#10369 gates are independently satisfied.

## Known authoring limitations

- No `Cargo.lock` is present because no dependency resolution was performed.
- The supported target list, Rust toolchain, target-specific build environment and exact feature matrix must be frozen before the task is specified.
- The fixture exposes two obvious source/configuration defects, but no reference patch, independent oracle or target matrix has been qualified.
