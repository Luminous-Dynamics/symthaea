# Public development fixture: cross-crate API migration V0

**Status:** Source-only public development material. Not an admitted CEB task, not an executable evaluation fixture, not a sealed holdout, and not a performance result.

**Canonical family:** `multi-crate-api-regression`  
**Candidate dossier:** [CEB Rust task/oracle candidates V0](../../../../docs/research/CEB_RUST_TASK_ORACLE_CANDIDATES_V0_2026-11-06.md)  
**Machine-readable requirement/oracle crosswalk:** [CEB Rust candidate oracle traceability V0](../../../../docs/research/CEB_RUST_CANDIDATE_ORACLE_TRACEABILITY_V0_2026-11-06.json)  
**Admission authority:** [#10609](https://github.com/Luminous-Dynamics/symthaea/pull/10609)  
**Sampling authority:** [#10617](https://github.com/Luminous-Dynamics/symthaea/pull/10617)

## Intended development objective

Promote a crate-internal typed parser operation to the public API, migrate the CLI consumer to the typed result, and preserve the existing lossy legacy adapter for callers that still require it. Keep the storage adapter source-compatible with the legacy surface.

The intended contract is specified by the API-01 through API-06 requirements in the candidate dossier and the API-MIGRATION entries in the traceability crosswalk.

## Workspace outline

- `domain-types`: shared record and typed parse error definitions.
- `record-parser`: currently has an internal typed parser and a public legacy `Option` adapter.
- `storage-adapter`: intentionally continues using the legacy adapter.
- `cli`: currently collapses parse failure to a generic string error; the candidate task would migrate this consumer to the typed public API.

## Integrity and safety boundary

- This fixture is committed to a public repository. It is exposed development material and can never become sealed or hidden holdout material.
- It has no CEB task ID, task manifest, repository slot, partition, difficulty assignment, environment pin, reference patch, grader digest, admission status or run receipt.
- The source has **not been compiled, tested, benchmarked or executed**. No Cargo commands were run.
- No candidate-controlled code, build scripts, procedural macros, tests or graders were executed.
- No reference solution or hidden test suite is included.
- The workspace is a starting point for static reviewer inspection. It must not be promoted to `fixture_ready` or admitted until the canonical #10609 requirements and #10064/#10369 gates are independently satisfied.

## Known authoring limitations

- No `Cargo.lock` is present because no dependency resolution was performed.
- The baseline implementation intentionally exposes a lossy public adapter while keeping the typed parser operation crate-internal.
- A reviewer must decide whether the specification's exact compatibility promise is sufficiently clear before any task card is authored.
