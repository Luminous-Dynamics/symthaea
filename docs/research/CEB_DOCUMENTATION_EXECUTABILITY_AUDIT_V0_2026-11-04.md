# CEB Documentation Executability Audit V0 — 2026-11-05

**Status:** Static documentation-control implementation. The expanded Cargo manifest/package/binary audit passed on exact head `e3181846e3addb6a173af0ab140fa8196a8b5e42` in CEB Contract Validation V0 run #226.

## Why this is needed

Research documentation can accidentally turn an intended interface into an apparent implementation. A command block that names a Cargo package, manifest or binary target is not proof that it exists in the documented workspace. This is especially dangerous when benchmark reports cite a command as though it had been run.

The initial audit found concrete drift:
- Rust repository-repair regrade/ablation commands referenced a missing `tools/rust-repair-benchmark-harness/Cargo.toml`.
- The BASALT verifier README described a package whose manifest and Rust source were absent.
- The beverage benchmark standalone manifest existed, but its four declared binary source files were absent.
- The academic handoff and tools README named `symthaea-morphogenesis` and `symthaea-inspect` package commands although their manifests/source trees were not in the checkout.

The associated documents now identify these as design-only/unavailable interfaces. Those disclosures do not prove the referenced code works.

## Implemented static check

The CI helper is [`scripts/audit_doc_command_claims.py`](../../scripts/audit_doc_command_claims.py). The CEB Contract Validation V0 workflow invokes its parser/workspace self-tests and repository-wide audit.

The helper:
1. Enumerates tracked Markdown and Cargo manifest paths through Git; it does not trust untracked working-tree files as evidence.
2. Reads fenced code blocks and identifies Cargo commands with explicit `--manifest-path` or `-p` / `--package` selectors.
3. Resolves explicit relative manifest paths from the repository root and checks the manifest exists.
4. Resolves package selectors against the nearest Cargo workspace above the Markdown document, or the workspace containing an explicit manifest. It reads the workspace's declared `members` and `exclude` patterns rather than searching every Cargo package name indiscriminately.
5. For `--bin NAME`, parses the manifest and checks explicit `[[bin]]` source paths or Cargo's conventional binary source locations.
6. Splits shell commands only on control operators outside quotes and ignores flags following Cargo's `--` separator.
7. Allows a missing manifest/package/binary only when the same Markdown line names the exact path or selector (and binary target, when applicable) and clearly labels it unavailable, not runnable or design-only. An unrelated status note elsewhere in the document cannot waive the command.
8. Reports placeholders and unsupported/dynamic selectors separately; they are not counted as verified.

The checker is deliberately offline and static. It does not invoke Cargo, build scripts, proc macros, tests, candidate code, graders or referenced commands. It does not modify files or produce benchmark scores.

## Exact-head result

CEB Contract Validation V0 run **#226** passed on exact commit `82c3549b8309c69ecd079da57ea42f9a89ac2f12`:

- CEB contract: 10 tasks, 4 schemas, 39 negative controls, 10 repository configuration worlds and 7 conditional facts.
- Documentation audit self-test: passed.
- Tracked Markdown files inspected: **1,426**.
- Explicit `--manifest-path` references checked: **15**.
- `-p` / `--package` selectors resolved to workspace package manifests: **258**.
- Explicitly unavailable path/package/binary reference occurrences: **18**.
- Dynamic/out-of-scope references: **1**.
- Unannotated missing manifest, package or binary references: **0**.

These are **reference occurrences, not unique paths or unique package names**. A documented-unavailable occurrence remains unavailable; it is not a verified command. The single dynamic reference is not counted as passed.

The scan resolved vendor Binius package selectors against the vendor workspace, honoured workspace member globs and exclusions, and kept package names outside the selected workspace from being silently treated as valid. It also confirmed that quoted Nix-shell examples are not misclassified as Cargo package selectors.

## Separate beverage workflow result

A separate Beverage Benchmark Contract workflow on the same development branch fails at its first rustfmt step because `crates/domains/symthaea-culinary/src/bin/beverage_benchmark_receipt.rs` is absent. The beverage unit tests, schema validation, Clippy and receipt-generation steps are skipped. This is not a CEB contract failure; it confirms that the beverage CLI is not qualified by the current checkout.

## Claim boundary and limitations

This is a guard against a specific class of documentation error, not a complete shell interpreter or proof that a command succeeds.

- Explicit `--manifest-path` values are interpreted relative to the repository root. Package-only commands use the nearest tracked Cargo manifest above the Markdown document as the likely working workspace.
- The scanner does not interpret arbitrary `cd` / `pushd` state changes inside a code block. Commands whose effective working directory differs from the documented file's nearest workspace should use an explicit `--manifest-path` or be manually reviewed.
- Package matching supports exact package names in the selected workspace. Version-qualified specs such as `name@version` and placeholders such as `<crate>` are out-of-scope/dynamic, not verified.
- Workspace membership uses statically read `members` and `exclude` entries. Exotic Cargo workspace discovery or generated manifests require manual review.
- Binary source checks cover explicit `[[bin]]` paths and Cargo's conventional `src/main.rs`, `src/bin/NAME.rs` and `src/bin/NAME/main.rs` layouts. Generated or build-script-selected targets require manual review.
- It does not validate arbitrary shell scripts, Nix commands, container images, network endpoints, benchmark datasets, test semantics, executable permissions or runtime outcomes.
- A manifest, package or binary existing is not proof that it builds, passes tests, is secure, or has been run on the exact source head.
- Unavailable exceptions must keep the exact path/selector, optional binary target and unavailable reason on one Markdown line; wrapped notes should be rewritten as a single explicit annotation.

The next strengthening step is a typed command registry for script/Nix/container invocations, command-working-directory annotations, version-qualified package matching and a reviewer-owned exception file bound to exact commits. Do not infer that scope from this V0 helper.

## Acceptance and metrics

The CI audit reports:
- tracked Markdown files inspected;
- explicit manifest-reference occurrences checked;
- package-selector occurrences resolved to workspace package manifests;
- explicitly documented unavailable-reference occurrences;
- dynamic/out-of-scope references;
- unannotated missing manifests, packages or binary targets with source locations.

Required result: **zero unannotated missing manifests, packages or binary targets**. Dynamic references remain an explicit coverage gap, not an implicit pass. Self-tests cover shell line continuations, quoted control operators, command argument scopes, dynamic paths, exact unavailable annotations, workspace glob members, workspace exclusions and nearest-workspace selection.
