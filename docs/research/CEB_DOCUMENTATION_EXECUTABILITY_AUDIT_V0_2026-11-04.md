# CEB Documentation Executability Audit V0 — 2026-11-04

**Status:** Static documentation-control implementation. The first repository-wide run passed on the earlier exact head noted below; the current self-test/count-label refinement is awaiting its own exact-head workflow result.

## Why this is needed

Research documentation can accidentally turn an intended interface into an apparent implementation. A command block that names a future Cargo manifest is not proof that the manifest, binary target or executable exists. This is especially dangerous when later benchmark reports cite the command as though it had been run.

The Rust repository-repair audit documents and BASALT verifier README now explicitly identify their missing manifest paths as design-only/unavailable interfaces. The beverage research contract also now discloses that its standalone manifest exists but its four declared binary source files are absent. These disclosures are not proof that the referenced code works.

## Implemented static check

The CI helper is [scripts/audit_doc_command_claims.py](../../scripts/audit_doc_command_claims.py). The CEB Contract Validation V0 workflow invokes its parser self-test and repository-wide audit.

The helper:
1. Enumerates tracked Markdown files from Git.
2. Reads fenced code blocks and identifies Cargo commands containing --manifest-path.
3. Resolves static relative paths from the repository root and checks that the referenced Cargo manifest exists.
4. When a command includes --bin NAME, parses the referenced manifest with Python's standard-library TOML reader and checks the explicit or conventional binary source path.
5. Allows a missing manifest or binary only when the same document names the exact path and, for a binary, the exact target name, while clearly labelling it unavailable, not runnable or design-only.
6. Reports dynamic manifest paths separately; it does not pretend those were verified.

The checker is deliberately offline and static. It does not invoke Cargo, build scripts, proc macros, tests, candidate code, graders or referenced commands. It does not modify files or produce benchmark scores.

## First exact-head result

The first repository-wide CEB validation plus documentation audit passed on commit 36f7061f554a6593a1434a8b2e3c3b64e66bcfe3, workflow run 198:

- CEB contract: 10 tasks, 4 schemas, 39 negative controls, 10 repository configuration worlds and 7 conditional facts.
- Documentation parser self-test: passed.
- Tracked Markdown files inspected: 1,426.
- Static manifest references checked: 15.
- Explicitly unavailable references: 14.
- Dynamic/out-of-scope references encountered: 0.
- Unannotated missing manifest or binary references: 0.

The manifest/unavailable counts above are **reference occurrences, not distinct path counts**. A later self-test/count-label refinement is intended to make that distinction explicit in CI output; this document must be updated if its exact-head rerun changes the counts.

A separate Beverage Benchmark Contract run triggered by the beverage documentation correction failed at its first rustfmt step because the declared source file crates/domains/symthaea-culinary/src/bin/beverage_benchmark_receipt.rs is absent. The beverage unit tests, schema run, Clippy and receipt steps were skipped. This is not a CEB test failure and is not a test verdict; it confirms that the beverage CLI is not qualified by the current checkout.

## Claim boundary and limitations

This is a guard against a specific class of documentation error, not a complete shell interpreter or proof that a command succeeds.

- Relative paths are interpreted from the repository root. Commands requiring another working directory need to state that explicitly and should be handled by a future version.
- Manifest paths containing variables, globs, backticks or absolute paths are classified as dynamic/out-of-scope rather than being marked verified.
- V0 audits Cargo commands that name an explicit --manifest-path. Commands expressed only with -p/--package and no manifest path are not yet resolved to a package manifest; this remains a coverage gap and is disclosed rather than silently counted as checked.
- Binary source checks cover explicit [[bin]] paths and Cargo's conventional src/main.rs, src/bin/NAME.rs and src/bin/NAME/main.rs layouts. Generated or build-script-selected target layouts require manual review.
- The checker audits Cargo manifest and binary references only. It does not validate arbitrary shell scripts, Nix commands, container images, network endpoints, benchmark datasets, test semantics or runtime outcomes.
- A documented-unavailable result is not a failure, but it remains unavailable and must not be cited as executed evidence.
- A manifest or binary existing is not proof that it builds, passes tests, is secure, or has been run on the exact source head.

The next strengthening step is a typed command registry for script/Nix/container invocations, plus package-selector resolution and a reviewer-owned exception file bound to exact commits. Do not infer that scope from this V0 helper.

## Acceptance and metrics

The CI audit reports:
- number of tracked Markdown files inspected;
- number of static manifest-reference occurrences checked;
- number of explicitly documented unavailable-reference occurrences;
- number of dynamic/out-of-scope references;
- number and location of unannotated missing manifests or binary targets.

Required result: **zero unannotated missing manifests or binary targets**. Dynamic references remain an explicit coverage gap, not an implicit pass. Parser self-tests cover shell line continuations, distinct command argument scopes, dynamic paths and exact unavailable annotations for manifest and binary targets.
