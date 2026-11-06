# Public development fixture: async trait boundary V0

**Status:** Source-only public development material. Not an admitted CEB task, not an executable evaluation fixture, not a sealed holdout, and not a performance result.

**Canonical family:** `rust-types-traits`  
**Candidate dossier:** [CEB Rust task/oracle candidates V0](../../../../docs/research/CEB_RUST_TASK_ORACLE_CANDIDATES_V0_2026-11-06.md)  
**Machine-readable requirement/oracle crosswalk:** [CEB Rust candidate oracle traceability V0](../../../../docs/research/CEB_RUST_CANDIDATE_ORACLE_TRACEABILITY_V0_2026-11-06.json)  
**Admission authority:** [#10609](https://github.com/Luminous-Dynamics/symthaea/pull/10609)  
**Execution-isolation prerequisite:** [#10064](https://github.com/Luminous-Dynamics/symthaea/issues/10064)

## Intended development objective

Make an async service trait usable across a required executor boundary without losing trait-object dispatch, typed associated error/result types, or the ability to borrow the service for the future's lifetime. The current fixture exposes a boxed future that is not declared `Send`; the proposed task would tighten the contract while preserving the public call shape.

The candidate dossier contains the normative TRAIT-01 through TRAIT-06 requirements. The traceability crosswalk includes both an explicit boxed-future alternative and a macro-based alternative as semantic designs; no macro implementation is included in this fixture.

## Workspace outline

- `domain-types`: event, receipt and typed service-error definitions.
- `async-contract`: object-capable service trait and borrowed boxed-future alias.
- `worker-adapter`: an erased trait-object consumer using the contract.

## Integrity and safety boundary

- This fixture is committed to a public repository. It is exposed development material and can never become sealed or hidden holdout material.
- It has no CEB task ID, task manifest, repository slot, partition, difficulty assignment, environment pin, reference patch, grader digest, admission status or run receipt.
- The source has **not** been compiled, tested, benchmarked or executed. No Cargo commands were run.
- No candidate-controlled code, build scripts, procedural macros, tests or graders were executed.
- No reference solution or hidden test suite is included.
- The workspace is a starting point for static reviewer inspection. It must not be promoted to `fixture_ready` or admitted until the canonical #10609 requirements and #10064/#10369 gates are independently satisfied.

## Known authoring limitations

- No `Cargo.lock` is present because no dependency resolution was performed.
- No runtime is included; the task's Send guarantee is a type-level API contract, not a runtime scheduling benchmark.
- Whether an async-trait macro dependency is permitted must be settled in the task specification before fixture freeze.
