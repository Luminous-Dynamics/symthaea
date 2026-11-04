# CEB Documentation Executability Audit V0 — 2026-11-04

**Status:** Static documentation-control implementation; exact-head CI validation pending. This document is not evidence that the scanner has passed its first repository-wide run.

## Why this is needed

Research documentation can accidentally turn an intended interface into an apparent implementation. A command block that names a future Cargo manifest is not proof that the manifest, binary target or executable exists. This is especially dangerous when later benchmark reports cite the command as though it had been run.

The two Rust repository-repair audit documents currently disclose that \`tools/rust-repair-benchmark-harness/Cargo.toml\` and the named binaries are absent on \`main\`, and that their command blocks are intended interfaces rather than runnable commands:

- [Independent fresh-image regrade audit](RUST_REPOSITORY_REPAIR_REGRADING_AUDIT_2026-10-24.md)
- [Paired repository-repair ablation audit](RUST_REPOSITORY_REPAIR_ABLATION_AUDIT_2026-10-24.md)

That disclosure is correct and must remain visible until the implementation is actually restored. By contrast, \`tools/beverage-benchmark-harness/Cargo.toml\` exists on \`main\`; its existence does not imply that every run or benchmark described by the beverage research document has been qualified.

## Implemented static check

The CI helper is [\`scripts/audit_doc_command_claims.py\`](../../scripts/audit_doc_command_claims.py). The CEB Contract Validation V0 workflow invokes its parser self-test and repository-wide audit.

The helper:

1. Enumerates tracked Markdown files from Git.
2. Reads fenced code blocks and identifies Cargo commands containing \`--manifest-path\`.
3. Resolves static relative paths from the repository root and checks that the referenced Cargo manifest exists.
4. When a command includes \`--bin NAME\`, parses the referenced manifest with Python's standard-library TOML reader and checks the explicit or conventional binary source path.
5. Allows an absent target only when the same document mentions the exact referenced path and clearly labels the implementation unavailable, not runnable, or design-only.
6. Reports dynamic manifest paths separately; it does not pretend those were verified.

The checker is deliberately offline and static. It does not invoke Cargo, build scripts, proc macros, tests, candidate code, graders or referenced commands. It does not modify files or produce benchmark scores.

## Claim boundary and limitations

This is a guard against a specific class of documentation error, not a complete shell interpreter or proof that a command succeeds.

- Relative paths are interpreted from the repository root. Commands that require another working directory need to state that explicitly and should be handled by a future version of the checker.
- Manifest paths containing variables, globs, backticks or absolute paths are classified as dynamic/out-of-scope rather than being marked verified.
- Binary source checks cover explicit \`[[bin]]\` paths and Cargo's conventional \`src/main.rs\`, \`src/bin/NAME.rs\`, and \`src/bin/NAME/main.rs\` layouts. More exotic generated or build-script-selected target layouts require manual review.
- The checker audits Cargo manifest and binary references only. It does not validate arbitrary shell scripts, Nix commands, container images, network endpoints, benchmark datasets, test semantics or runtime outcomes.
- A documented-unavailable result is not a failure, but it remains unavailable and must not be cited as executed evidence.
- A manifest or binary existing is not proof that it builds, passes tests, is secure, or has been run on the exact source head.

The next strengthening step is a broader typed command registry for script/Nix/container invocations, with a reviewer-owned exception file and exact commit binding. Do not infer that scope from this V0 helper.

## Acceptance and metrics

The first repository-wide CI run should report:

- number of tracked Markdown files inspected;
- number of static manifest references checked;
- number of explicitly documented unavailable references;
- number of dynamic/out-of-scope references;
- number and location of unannotated missing manifests or binary targets.

Required result: **zero unannotated missing manifests or binary targets**. Dynamic references remain an explicit coverage gap, not an implicit pass. The scanner's self-tests cover shell line continuations, dynamic paths and the required unavailable annotation.

No claim is made here about the result of that repository-wide run until the exact-head workflow reports it.
