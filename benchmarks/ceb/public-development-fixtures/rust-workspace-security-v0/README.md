# Public development fixture: workspace command/path boundary V0

**Status:** Source-only public development material. Not an admitted CEB task, not an executable evaluation fixture, not a sealed holdout, and not a security test result.

**Canonical family:** `security-correctness`  
**Candidate dossier:** [CEB Rust task/oracle candidates V0](../../../../docs/research/CEB_RUST_TASK_ORACLE_CANDIDATES_V0_2026-11-06.md)  
**Machine-readable requirement/oracle crosswalk:** [CEB Rust candidate oracle traceability V0](../../../../docs/research/CEB_RUST_CANDIDATE_ORACLE_TRACEABILITY_V0_2026-11-06.json)  
**Admission authority:** [#10609](https://github.com/Luminous-Dynamics/symthaea/pull/10609)  
**Execution-isolation prerequisite:** [#10064](https://github.com/Luminous-Dynamics/symthaea/issues/10064)

## Intended development objective

Separate untrusted workspace paths from command syntax and keep filesystem access within a declared workspace boundary. The baseline builds a shell command string by interpolating an untrusted path. The path helper rejects obvious absolute and parent-relative inputs but is only lexical; it does not resolve or constrain symlinks.

The intended task is to replace this with a narrow direct-argv command specification and a path-access policy whose guarantees match a frozen OS/filesystem threat model.

## Workspace outline

- `workspace-path`: lexical path helper that is explicitly not a symlink-safe authority boundary.
- `command-runner`: command description builder that intentionally interpolates a path into a shell string but does not execute it.
- `tool-cli`: consumer exposing the generated command description for static inspection.

## Integrity and safety boundary

- This fixture is committed to a public repository. It is exposed development material and can never become sealed or hidden holdout material.
- It has no CEB task ID, task manifest, repository slot, partition, difficulty assignment, environment pin, reference patch, grader digest, admission status or run receipt.
- The source has **not** been compiled, tested, benchmarked or executed. No Cargo commands were run.
- The shell command string is constructed as data only; no shell or process is invoked.
- No exploit payload, filesystem attack, candidate-controlled command, build script, test or grader was executed.
- No reference solution or hidden test suite is included.
- The workspace is a starting point for static reviewer inspection. It must not be promoted to `fixture_ready` or admitted until the canonical #10609 requirements and #10064/#10369 gates are independently satisfied.

## Known authoring limitations

- No `Cargo.lock` is present because no dependency resolution was performed.
- The lexical path helper does not canonicalize paths, open file descriptors, or defend against symlink races.
- No platform-specific `openat2`, descriptor-relative operation, Windows reparse-point policy, allowlisted executable policy, or direct process invocation is implemented.
- The fixture exposes only the shell-string/path design flaw; it does not establish a qualified security oracle or a complete threat model.
