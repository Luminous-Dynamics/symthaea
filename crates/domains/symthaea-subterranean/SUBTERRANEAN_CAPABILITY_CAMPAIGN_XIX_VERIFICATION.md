# Campaign XIX Verification Record

**Campaign:** Resource-conflict and multi-objective assurance
**Date:** 2026-07-20
**Baseline commit:** `64c9eaaae186bf8a41f9ad89cd5c2952be02cd20`
**Baseline tree:** `717d68d85e82823bda37babe855520ced35ca443`
**Verified code-and-protocol tree before this record:** `f5e6b788f616fed2a89b41499dd49d1c9431a9a9`

## Scope

Campaign XIX adds explicit resource budgets, protected-objective priority, starvation and service-fairness accounting, deterministic conflict arbitration, same-frame command restriction, operational evidence, checkpoint continuity, canonical requirements, traceability, counterfactual explanations, top-level certification integration, and a self-consistent reviewer bundle.

There are 27 campaign commits before this verification record. Including this record, the incremental delivery is patches 288–315 and the complete history is patches 1–315 after the initial import commit.

## Toolchain boundary

This sandbox does not contain `cargo`, `rustc`, `rustfmt`, Nix, or a resolvable network path from which a Rust toolchain can be installed. The standalone archive also omits the real workspace `symthaea-core`, `symthaea-fep`, and workspace dependency configuration.

No Rust compilation or test execution is claimed for Campaign XIX in this environment. The Rust tests and release contracts are included in the patch series, but authoritative execution remains a merge gate in the complete Rust 1.94 workspace.

## Executed source-level gates

### Structural Rust scan

A delimiter and lexical-state scanner processed all 150 Rust source files and found no unmatched braces, brackets, parentheses, strings, or block comments.

### Extended-structure completeness

Every literal construction of the following extended structures was inspected:

- `SafetyEvidenceRecord` includes `resource_conflict` or uses update syntax.
- `SubterraneanOperationalCheckpoint` includes `resource_conflict` or uses update syntax.

No incomplete literal was found.

### Registry and release cardinality

The source-level cardinality checks confirmed:

- 66 `RequirementId` variants and 66 ordered `RequirementId::ALL` entries.
- 16 `CertificationContract` variants and 16 ordered release-gate entries.
- 7 `ResourceConflictContract` variants and 7 ordered validation entries.
- All 5 `SUB-RES-*` requirements have canonical traceability links.

### Public API linkage

The exported API contains the supervisor, validation report, operational evidence snapshot, and self-consistent evidence bundle.

### Production-source marker audit

Before each `#[cfg(test)]` boundary in every changed Rust source, no production occurrence was found for:

- `unsafe {`
- `panic!`
- `todo!`
- `unimplemented!`
- `.unwrap()`
- `.expect()`

### Patch hygiene

`git diff --check` passes for Campaign XIX. The incremental Campaign XIX mail series applies without whitespace warnings. The complete historical series retains several documentation-only whitespace warnings inherited from earlier campaigns, but applies successfully and reproduces the exact final tree.

## Clean-room reconstruction

### Incremental path

The canonical v18 tree was archived into a fresh repository and patches 288–314 were applied. The resulting code-and-protocol tree was:

`f5e6b788f616fed2a89b41499dd49d1c9431a9a9`

### Complete path

The original uploaded `symthaea-subterranean.tar.gz` snapshot was initialized as one import commit. The verified prior 287-patch series and Campaign XIX patches 288–314 were then applied in order. The resulting tree was also:

`f5e6b788f616fed2a89b41499dd49d1c9431a9a9`

The prepared repository, incremental reconstruction, and complete reconstruction therefore matched exactly before this verification-record commit.

## Source metrics before this record

- 150 Rust source files.
- 51,543 Rust source lines.
- 407 test or ignore annotations.
- 188 tracked files.
- 27 Campaign XIX commits before this record.

## Hardening corrections made during review

1. **Throttled-work semantics:** the first integration treated `Throttled` as a full productive-work prohibition, despite the command envelope deliberately retaining bounded cutter and auger authority. Productive work is now allowed under `Nominal` and `Throttled`, but remains prohibited under `ReturnOnly` and `HoldForReview`.
2. **Counterfactual attribution:** resource conflict now has its own blocker identity and remediation text rather than being misclassified as policy governance or temporal uncertainty.
3. **Checkpoint release contract:** the checkpoint gate now preserves and compares non-default starvation, fairness, and restrictive-authority state through an actual operational checkpoint load.
4. **Release integration:** the seven contracts are part of the top-level certifiable-autonomy validator and cannot be omitted from the canonical release report.

## Unexecuted authoritative gates

Production acceptance still requires:

- Rust 1.94 full-workspace `cargo check`, Clippy, and tests.
- Real `symthaea-core` and `symthaea-fep` integration.
- Rustfmt verification using the workspace toolchain.
- Calibrated battery, thermal, timing, return-energy, and recovery-capacity models.
- Hardware-in-the-loop simultaneous-demand, exhaustion, starvation, and recovery campaigns.
- Independent review of stakeholder identities, objective classes, urgency policy, and fairness interpretation.
- Cryptographic build, evidence, and reviewer provenance adapters.
- Physical mission trials demonstrating that protected reserves remain conservative under model error.
