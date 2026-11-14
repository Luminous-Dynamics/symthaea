# AOBench Runtime Evidence Layer v0.1

**Date:** 2026-11-14  
**Status:** Rust source and unit fixtures committed; compilation and execution not yet verified  
**Package:** \`symthaea-organizational-closure\`  
**Tracking:** [AOBENCH-001 #11980](https://github.com/Luminous-Dynamics/symthaea/issues/11980) · [MEL-AUTO-001 #10098](https://github.com/Luminous-Dynamics/symthaea/issues/10098)

## 1. Purpose

This is the first executable slice of AOBench: distinguish a declared process graph from runtime evidence that component versions were produced, durably represented in an artifact store, and consumed by downstream processes.

It deliberately qualifies only **functional self-maintenance** under a frozen component/process contract. It does not report a biological autopoiesis score or claim organizational closure.

## 2. Package surface

The crate is a nested, independently locked Cargo workspace using exact \`serde\` and \`sha2\` versions. This keeps package qualification independent of unrelated root-workspace dev dependencies.

Core types:

- \`OrganizationSpec\`: frozen component and process declarations.
- \`ComponentSpec\` / \`ComponentClass\`: regenerative internal, immutable provisioned, external resource, constraint, or evidence artifact.
- \`ProcessSpec\`: required/optional component inputs and outputs, plus an external-dependency allow-list.
- \`ArtifactVersionId\`: exact component/version identity.
- \`ArtifactRef\`: version identity, digest, and producer origin.
- \`ArtifactStore\`: insert-once versioned bytes.
- \`ExecutionEvent\`: event sequence, process and executable version identities, exact input/output refs, external observations, policy identity, and result.
- \`ExecutionTrace\`: ordered events.
- \`VerificationReport\`: findings, successful execution count, verified outputs, and required-regenerative coverage.

Primary function:

\`\`\`rust
verify_trace(
    organization: &OrganizationSpec,
    trace: &ExecutionTrace,
    store: &ArtifactStore,
) -> VerificationReport
\`\`\`

The pass predicate is intentionally named \`qualifies_functional_self_maintenance()\`. It requires at least one required regenerative component, no verifier findings, and verified producer-plus-downstream-consumer coverage for every required regenerative component.

## 3. Checks performed

1. Component/process map keys match their declared IDs.
2. Required regenerative components have declared producers and consumers.
3. Successful events reference known process IDs and the frozen process version.
4. Event sequence is strictly increasing and execution IDs are unique.
5. Successful events satisfy required/allowed input and output contracts.
6. Input bytes exist and their SHA-256 digest matches the artifact reference.
7. Provisioned roots are restricted to immutable/constraint classes.
8. Produced inputs must match an exact earlier verified output version and producer event.
9. Output origin must name the current execution; output bytes must exist and hash to the declared digest.
10. Component output versions increase monotonically in the observed trace.
11. External observations must be declared, unique per event, and carry a digest when reported available.
12. An event cannot report success when a declared observed dependency is denied or unavailable.
13. Failed, abstained, and interrupted events cannot publish committed outputs.
14. A successful write must carry an authorization-policy identity. This records the policy version; it does not itself authorize the write.

## 4. Deterministic fixtures included

- Healthy producer followed by exact downstream consumption.
- Static-only recurrent declaration with no runtime events.
- Forged output digest.
- Interrupted execution attempting to expose an output.
- Undeclared external dependency.
- Available external dependency without a receipt digest.
- Duplicate artifact version insertion.
- Empty regenerative scope.

Expected acceptance: only the healthy fixture qualifies. The negative fixtures are designed to fail closed.

## 5. Trust boundaries and limitations

The code recomputes artifact digests, but its inputs still need a trustworthy source.

Not yet implemented:

- trusted/attested process execution logger;
- signature and authority verification for external receipts;
- authorization evaluation (the current policy digest is only an identity binding);
- durable transaction protocol and crash-consistent artifact commit;
- explicit parent-event causality beyond artifact producer lineage;
- maintenance-epoch recurrence graph;
- intervention randomization and cut-set analysis;
- assay-family adequacy and model-class distinguishability;
- full-run evidence manifest and reproducible report export.

An agent-generated event cannot prove that the declared executable ran. Production qualification must bind events to an independent runtime or supervisor, authenticate the executable identity, and verify immutable stored bytes outside the agent's write authority.

A digest proves content identity relative to the bytes supplied to the verifier; it does not prove source authenticity, correctness, or authorization.

## 6. Commands

From \`crates/domains/symthaea-organizational-closure/\`:

\`\`\`bash
cargo fmt --all -- --check
cargo check --locked --all-targets
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
\`\`\`

These commands have not yet been executed in this environment. Do not interpret source presence or authored fixtures as a pass.

## 7. Promotion rule

Keep this package research-qualified until exact-head formatting, check, tests, and strict Clippy pass. Before calling it a runtime self-maintenance benchmark, additionally qualify the trusted event source, durable commit boundary, and full set of positive/negative controls.

Even after those checks, its result remains evidence for bounded functional self-maintenance—not proof of biological autopoiesis, life, sentience or phenomenal consciousness.
