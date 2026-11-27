# Workspace stale core member exclusion — 21 November 2026

## Observed failure

The repository root Cargo workspace declares:

- \`members = ["xtask", "crates/core/*", "crates/domains/*", "crates/bridges/*"]\`
- an explicit \`exclude\` list for intentionally standalone, archived or incompatible crate paths.

A current \`cargo test -p symthaea-muse\` invocation fails before package selection with:

\`\`\`text
failed to load manifest for workspace member
.../crates/core/symthaea
referenced via crates/core/*

failed to read .../crates/core/symthaea/Cargo.toml
No such file or directory
\`\`\`

This is a wildcard discovery issue, not a Muse dependency or source-code failure. Cargo must enumerate workspace manifests before it can honor a package-scoped \`-p\` selection.

## Change

Add \`crates/core/symthaea\` to the root workspace's existing \`exclude\` list with a comment explaining that the path is a stale wildcard match without a Cargo.toml.

No workspace member glob is removed or narrowed. No default member is removed. No tests, lint gates, or package dependencies are bypassed. If a real crate is restored at that path in the future, its manifest can be reviewed and the exclusion intentionally removed.

## Exact-head validation

The dedicated workflow performs:

1. Exact PR-head checkout and SHA equality check.
2. Pinned Rust 1.96.0 installation.
3. \`cargo metadata --locked --no-deps --format-version 1\` over the actual monorepo.
4. A check that every enumerated workspace member resolves to a package record and an existing Cargo.toml.
5. \`cargo check --locked -p symthaea-muse --lib\`.
6. \`cargo test --locked -p symthaea-muse --lib --no-run\`.
7. Retention of the exact metadata, manifest hashes and source revision in a qualification artifact.

The final two commands validate the real package in the root workspace, not a staged copy.

## Relationship to Melothaea RMI

[RMI Track A PR #13070](https://github.com/Luminous-Dynamics/symthaea/pull/13070) has already passed isolated qualification of the exact RMI evaluator and runner: snapshot state isolation, all synthetic controls, 32 paired trials, fresh-process deterministic replay, and warnings-denied Clippy.

It remains draft pending a real-workspace package integration qualification. This exclusion is the minimal repository repair required to make that possible. The isolated pass is not re-labelled as a full package or monorepo pass.

## Lockfile reconciliation

After the wildcard scan and workspace metadata checks passed, the first real package check failed because the committed lockfile omitted one already-declared dependency edge for the existing workspace package `symthaea-epistemic-types`. A separate diagnostic run let Cargo compute the candidate lockfile and completed `cargo check -p symthaea-muse --lib` successfully. The only generated lockfile delta was:

```toml
[[package]]
name = "symthaea-epistemic-types"
version = "0.1.0"
dependencies = [
 "serde",
 "serde_json",
 "sha2 0.10.9",
]
```

This edge is now committed to `Cargo.lock`. The qualification job still uses `--locked`; the diagnostic run did not count as the final locked pass. The new exact-head run must demonstrate that the same real Muse package compiles with the committed lockfile and that the library tests can be test-compiled.
