# ADR-010: RSK Qualification Capsule and Evidence Receipts

**Date**: 2026-09-12  
**Status**: Proposed  
**Change Class**: A (Safety-Critical)

## Context

The focused RSK GitHub Actions lane has repeatedly remained queued while the repository-wide CI matrix is also queued. That is an infrastructure availability problem, not evidence that the RSK candidate passes or fails.

RSK needs a qualification path that can run locally, in CI, or in another controlled runner while producing comparable evidence bound to the exact source subject and toolchain.

The existing focused workflow also encoded its Rust commands directly in YAML. If local qualification and CI use separate command definitions, they can drift and produce incomparable evidence.

This ADR contains no physical replication mechanism, fabrication recipe, molecular design, biological implementation, or autonomous manufacturing path.

## Decision

Introduce `scripts/rsk_qualification.py` as the single executable definition of the focused Rust qualification gates for the RSK reference crates.

The harness:

1. discovers the enclosing Git worktree;
2. refuses to produce admissible evidence from a dirty worktree by default;
3. records exact Git commit and Git tree identities;
4. records **before and after** hashes of `Cargo.lock`, `rust-toolchain.toml`, the qualification script, qualification self-test, focused workflow, Class A detector, and production-admission gate document;
5. records Rust/Cargo/rustfmt/Clippy tool version output;
6. requires the observed `rustc` version to match the exact channel declared in `rust-toolchain.toml`;
7. executes the selected focused gates, using `--locked` for semantic Cargo commands;
8. preserves each command's combined output as a separate log;
9. hashes every command log;
10. re-checks the worktree after execution and rejects tool-induced repository mutation;
11. emits a canonical machine-readable receipt plus receipt SHA-256;
12. states explicitly that production admission remains `DENIED / NOT YET ELIGIBLE`.

The focused GitHub Actions workflow calls this same harness and uploads generated receipts/logs even when the qualification command fails.

## Qualification phases

The harness exposes three phases:

- `format`: `cargo fmt --check` for the two RSK crates;
- `core`: all-target `cargo check --locked`, `cargo test --locked`, and `cargo clippy --locked -D warnings` for both RSK crates;
- `all`: format plus core.

The workflow retains independent `format` and `core` jobs so formatting failure cannot hide compiler/test/Clippy evidence.

## Receipt identity

The receipt schema is:

```text
symthaea.rsk.qualification-receipt.v1
```

A receipt binds at least:

- qualification phase;
- Git commit;
- Git tree;
- branch name;
- pre-run and post-run clean/dirty state;
- selected tracked-input hashes before and after execution;
- whether selected inputs remained unchanged;
- expected Rust channel and whether the observed compiler matched it;
- tool version outputs;
- platform/Python metadata;
- optional GitHub Actions context;
- each executed command and exit code;
- start/finish timestamps and duration;
- per-command log SHA-256;
- overall qualification status;
- `admissible_evidence` boolean;
- explicit production-admission denial marker.

The receipt's `receipt_sha256` is computed over canonical sorted compact JSON before the digest field itself is added.

## Clean-worktree and post-run-integrity rule

A clean Git tree is required both **before and after** qualification for admissible evidence.

By default a dirty starting tree yields:

```text
qualification_status = blocked-dirty-worktree
admissible_evidence = false
```

If the worktree begins clean but a tool mutates tracked or otherwise visible repository state, or a selected evidence input hash changes, the run yields:

```text
qualification_status = fail-worktree-mutated
admissible_evidence = false
```

This matters especially for dependency resolution: semantic Cargo commands use `--locked`, and post-run integrity is independently checked so a tool cannot update `Cargo.lock` and still produce a clean qualification receipt.

`--allow-dirty` exists only for diagnostics. Even if all commands pass, the resulting receipt is:

```text
qualification_status = pass-dirty-diagnostic
admissible_evidence = false
```

Dirty diagnostic runs must never be promoted to release/admission evidence.

## Toolchain pin rule

Recording a tool version is not enough. The observed `rustc` version must match the exact `channel` declared in `rust-toolchain.toml`.

A command-successful run under another compiler yields:

```text
qualification_status = fail-toolchain-mismatch
admissible_evidence = false
```

and exits nonzero.

## Evidence does not equal production admission

A clean passing receipt proves only that the declared Rust qualification commands passed for the bound source/tool environment.

It does **not** prove:

- TLA+/formal properties;
- verified positive authority evidence;
- trusted time;
- durable replay/CAS/fork recovery;
- governance/recovery verifier correctness;
- physical containment;
- protected-branch enforcement;
- production admission.

The receipt therefore carries the explicit marker:

```text
production_admission = DENIED / NOT YET ELIGIBLE
```

until a future separately governed production-admission record changes the overall system status.

## Failure behavior

The harness runs all commands in the selected phase even if an earlier command fails so one execution can expose multiple independent defects.

Missing executables are recorded as explicit failed command results rather than silently skipped.

The script exits nonzero if any selected qualification command fails, the toolchain pin is wrong, or a clean worktree mutates during qualification.

Unexpected harness failure is itself a qualification failure; CI artifact upload is configured with `if: always()` so partial receipts/logs can still be retained where available.

## Class A self-protection

Because the qualification harness determines what evidence is collected, the harness **and its self-test** are Class A surfaces.

The following must remain Class A-protected:

- `scripts/rsk_qualification.py`;
- `scripts/test_rsk_qualification.py`;
- `.github/workflows/rsk-safety.yml`;
- `scripts/check-class-a-changes.sh`;
- the RSK authority/ledger crates;
- the governance charter.

The focused workflow self-check verifies that the generic Class A detector still contains all required RSK evidence/governance paths.

## Qualification harness self-test

The focused governance job runs `scripts/test_rsk_qualification.py` before Rust qualification.

The current candidate self-test uses only Python and Git and covers:

1. clean matching-toolchain command success -> `pass-clean` and admissible Rust-gate evidence;
2. wrong Rust version with otherwise successful fake commands -> `fail-toolchain-mismatch`;
3. dirty starting worktree -> `blocked-dirty-worktree` before qualification commands;
4. tool-induced `Cargo.lock` mutation -> `fail-worktree-mutated` and non-admissible evidence;
5. receipt SHA-256 recomputation from the canonical payload.

The five candidate self-tests were executed locally and passed 5/5 before this ADR update. That is useful implementation evidence, but it is not a substitute for independent CI on the exact candidate head.

## Alternatives considered

### Keep local commands documented only

Rejected. Documentation can drift from CI and does not produce machine-bound evidence.

### Keep CI YAML as the authoritative command list

Rejected. It makes local and independent-runner qualification a second implementation and provides no canonical receipt.

### Record but do not enforce the toolchain version

Rejected. A different compiler could produce a passing receipt despite not being the qualified toolchain.

### Check cleanliness only before execution

Rejected. Qualification tools must not be able to mutate tracked state and then claim they tested the committed subject.

### Treat a passing receipt as production admission

Rejected. Rust compilation/tests are one evidence class among many P0-P12 admission gates.

### Require GitHub Actions as the only valid executor

Rejected. Hosted-runner availability should not prevent independent qualification. Executor identity is recorded as evidence and can be restricted later by promotion policy.

## Verification

Candidate verification for this tranche includes:

- Python syntax compilation of the harness;
- help/argument parsing execution;
- temporary-Git-repository missing-tool failure exercise;
- 5/5 qualification-harness self-tests described above;
- receipt emission on failure;
- log-hash generation;
- exact Rust-pin mismatch rejection;
- pre-run dirty refusal;
- post-run `Cargo.lock` mutation rejection;
- workflow YAML parse check;
- workflow review showing CI invokes the same harness;
- workflow artifact upload on both success and failure;
- Class A detector coverage for the harness and self-test.

These checks are design/local execution evidence only until the exact candidate commit receives independent repository CI evidence.

## Consequences

### Positive

- local and CI Rust qualification cannot silently drift;
- queued GitHub runners no longer block independent evidence generation;
- failures are retained as evidence rather than lost in console-only logs;
- exact source/tool/input binding becomes explicit;
- dependency resolution cannot silently rewrite the qualified lockfile;
- a wrong compiler cannot produce admissible evidence;
- tool-induced repository mutation cannot masquerade as a clean qualification;
- dirty diagnostic runs cannot masquerade as qualification evidence;
- the evidence generator and its self-test are governance-protected.

### Residual risk

- receipts are integrity hashes, not yet signed transparency records;
- executor/platform trust is recorded but not yet cryptographically attested;
- command success still does not prove formal or production properties;
- GitHub-hosted runner backlog remains an independent infrastructure issue;
- a future production promotion process must define which executor identities/environments are acceptable.

## Related work

- #1326 — focused RSK CI and governance hardening
- #1335 — Class A production-admission umbrella blocker
- #1672 — formal refinement
- #1682 — exact build/runtime admission
- #1724 / #1726 / #1728 — current Rust semantic hardening stack
- #1762 — threat model and compromise budget
- #1766 — verified evidence boundary
- #1772 — governance/recovery boundary
- #1776 — trusted-time boundary
