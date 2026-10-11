# Research Qualification Harness v2

## Status

`SCI-INFRA-001A2` defines a runner-portable bootstrap **source qualification** theorem for research crates.

It deliberately separates:

```text
source identity
!= execution success
!= bootstrap lock reconciliation
!= final locked qualification
!= scientific correctness
!= replication / independence
```

The canonical executable is:

`scripts/qualify-research-crate.sh`

A successful run carries:

```text
authority=bootstrap-source-qualification-only
scientific_claim=NONE
```

It must not be interpreted as a scientific result or as final locked-build authority.

## Why v2

The original SCI-INFRA-001A draft established the right portability direction but left several assumptions in its adapter or caller rather than in the portable theorem itself.

Version 2 makes the direct/local interface fail closed on those assumptions and now also closes Git rename/object-mode ambiguity, lock-file type ambiguity, and evidence-publication ambiguity.

## Subject theorem

For a successful qualification run, the harness requires all of the following.

### Exact qualifier identity

- `--expected-head` is a full lowercase 40-hex Git commit identity.
- actual `HEAD` equals that identity exactly.
- `HEAD` has **exactly one parent**.
- that sole parent equals `--source-parent` exactly.

This is stronger than first-parent equality. Merge commits are not admissible qualifier subjects.

### Canonical direct-CLI identities

The portable interface independently validates:

- bounded program identity;
- full source/head commit identities;
- exact `x.y.z` Rust version identity;
- bounded Cargo package identities;
- canonical repository-relative source and qualifier paths;
- duplicate-free package/source/qualifier sets.

Paths reject absolute forms, `./`, empty components, repeated separators, `.` / `..` components, colon-delimiter ambiguity, unsupported characters, and excessive length.

These checks remain required even when a stricter manifest adapter exists above the harness.

## Qualifier-only scope

The sorted parent-to-head changed-path set must equal the declared `--qualifier-path` set exactly.

The scope command explicitly disables external diff helpers and rename detection:

```text
git diff --no-ext-diff --no-renames --name-only
```

This prevents local Git configuration or rename heuristics from changing the path theorem.

Each qualifier path must resolve at `HEAD` to a regular Git blob (`100644` or `100755`) and must check out as a regular non-symlink file.

No undeclared file, symlink, tree, or gitlink may be smuggled into the qualifier-only scope.

## Frozen source identity and object modes

Every declared source path must exist in both the source parent and qualifier head.

The exact Git mode, object type, and object identity are checked in both commits. Admissible source roots are only:

```text
100644 blob
100755 blob
040000 tree
```

Symlink blobs (`120000`) and gitlinks/submodules (`160000 commit`) are rejected.

For a declared tree, the harness recursively rejects symlink or gitlink descendants. Descendants must be regular `100644` or `100755` blobs.

The root Git object identity must be identical parent-to-head. For a directory, the tree identity transitively commits filenames, modes, and descendant object identities.

The checked-out source path must additionally be a non-symlink file/directory whose physical resolution remains inside the repository.

This is committed-source identity. It is distinct from the working-tree postflight theorem below.

## Toolchain identity

`rustc`, `cargo`, `rustfmt`, Clippy, `realpath`, `install`, and SHA-256 tooling must be available.

The parsed `rustc` and `cargo` version tokens must equal `--expected-rust` exactly. Substring matches are not accepted.

The receipt records the complete tool version strings and Rust host triple.

## Cargo.lock input theorem

The starting checkout must be clean and `Cargo.lock` must be:

- present;
- tracked by Git;
- a regular non-symlink working-tree file;
- a non-executable `100644` Git blob at `HEAD`.

A symlinked, untracked, executable, or special-file lock is not an admissible bootstrap subject.

## Evidence sink isolation

`--output-dir` is interpreted as a fresh external evidence directory.

The parent directory must already exist as a non-symlink directory. The final evidence directory itself must not exist before qualification and is created with mode `0700` outside the subject repository tree.

Thus:

```text
subject checkout = execution input
fresh external directory = evidence sink
```

Evidence is first constructed in a private scratch directory. The three public evidence files are then installed as regular files and moved into place with destination-as-file semantics. The final external directory must contain exactly:

```text
Cargo.lock.generated
Cargo.lock.patch
receipt.txt
```

The harness rejects symlink evidence outputs and bounds generated lock/patch/receipt sizes.

This hardens evidence publication against accidental path aliasing. It is **not** a hostile same-UID sandbox theorem; deliberately malicious build code that persists background processes requires a stronger isolated execution substrate.

## Bootstrap lock theorem

The harness preserves the original lock, runs `cargo check` for every declared package, and treats the resulting lock as a bootstrap reconciliation candidate.

The candidate must produce a non-empty `Cargo.lock` patch.

Additivity is checked numerically with Git `--numstat` over the lock:

```text
additions > 0
deletions == 0
path == Cargo.lock
```

A replacement of an existing lock line necessarily contributes a deletion and therefore fails. This is stronger and less syntax-dependent than scanning patch prefixes alone.

The generated lock must contain every declared package and both the generated lock and patch must stay within bounded evidence sizes.

The harness retains:

- `Cargo.lock.generated`;
- `Cargo.lock.patch`;
- SHA-256 of both artifacts.

The original checkout lock is restored before success returns.

A bootstrap candidate still requires later inspection/replay and final `--locked` qualification. Additive lock materialization is not equivalent to a final dependency lock theorem.

## Executable gates

For every declared package, the harness requires:

1. `cargo check -p <package>`;
2. package-scoped `cargo fmt --check`;
3. `cargo test -p <package>`;
4. `cargo clippy -p <package> --all-targets -- -D warnings`.

After the bootstrap checks, any working-tree mutation other than the temporary `Cargo.lock` delta fails immediately.

## Postflight immutability

After tests and Clippy:

- declared source paths must have no working-tree or index diff from `HEAD`;
- external diff helpers are disabled for those checks;
- the global repository status may contain only the expected temporary `Cargo.lock` modification;
- after lock restoration, the repository must be fully clean.

This working-tree theorem is distinct from the parent/head Git-object identity theorem.

## Receipt v2

The line-oriented receipt begins with:

```text
schema=symthaea.research-bootstrap-receipt.v2
authority=bootstrap-source-qualification-only
scientific_claim=NONE
```

Direct input validation prevents newline/delimiter ambiguity in caller-controlled identity fields.

Receipt keys use disjoint semantic namespaces. Tool/environment identity fields use names such as:

```text
rustc_version=...
cargo_version=...
rustfmt_version=...
clippy_version=...
rust_host_triple=...
```

Theorem outcomes use explicit gate names:

```text
scope_gate=PASS
source_immutable_gate=PASS
lock_additive_gate=PASS
cargo_check_gate=PASS
rustfmt_gate=PASS
cargo_test_gate=PASS
clippy_gate=PASS
postflight_worktree_immutable_gate=PASS
```

A key that denotes tool identity is therefore never reused as a gate result. Consumers may enforce singleton semantics for all scalar keys and allow repetition only for the declared list fields `package`, `source_path`, `source_object`, and `qualifier_path`.

The receipt binds at least:

- program;
- source parent;
- qualifier head;
- single-parent count;
- SHA-256 of the executing harness bytes;
- Rust/Cargo/rustfmt/Clippy versions;
- Rust host triple;
- generated lock and lock-patch SHA-256;
- package identities;
- frozen source paths and Git object identities;
- qualifier paths;
- gate PASS markers.

## Local self-test

The exact script can run its input-boundary tests without Cargo or GitHub Actions:

```bash
bash -n scripts/qualify-research-crate.sh
scripts/qualify-research-crate.sh --self-test
```

The current SCI-INFRA-001A2 bytes were syntax-checked and self-tested before freeze, then matched against GitHub's exact Git blob identity.

This self-test validates parsing/canonicalization boundaries only. It does not establish a full qualification run against a research crate.

## Relationship to trusted scheduler admission

This harness is intentionally scheduler-independent.

A GitHub `pull_request_target` adapter must not execute candidate/base-controlled copies of this harness merely because a child PR changes only a manifest. Trusted scheduler policy and candidate source identity are separate trust domains.

The hardened admission successor uses a dual-checkout and fresh-runner sealing design:

```text
trusted default-branch policy checkout
  parser + subject guard + this harness + sealer
             |
             | verifies / executes against
             v
candidate checkout
  exact qualifier head
  exact sole source parent
  data-only manifest
  frozen research source
```

The adapter binding receipt must bind trusted policy separately from candidate identity, and final authentication must not rely on candidate-controlled helper bytes.

## Nonclaims

SCI-INFRA-001A2 does not establish:

- hostile same-UID build-code sandboxing;
- canonical program scope for a scheduler manifest;
- final `Cargo.lock` acceptance;
- reproducible binaries;
- supply-chain provenance beyond the recorded bootstrap environment;
- GitHub scheduler trust;
- merge enforcement;
- scientific truth, novelty, causality, replication, or independence.

Those require separate evidence lines.