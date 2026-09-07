# RCA Canonical Lineage Trusted Recovery

## Scope

This recovery harness exists only for draft PR #578, `fix(epistemics): derive canonical evidence lineage generation`, while GitHub-hosted runner assignment is severely backlogged.

It is a **correctness/reproducibility recovery path**, not a new epistemic authority and not performance evidence.

Frozen target generation:

```text
PR:                  #578
target commit:       d3a47c4e3ff7857772693910a90398dbba7133ec
target tree:         af59a9987073cf5983256a2ab05bc57b063856f6
exact base:          4c0e29a34ef4a8e33953eb5615ae18121571edd6
canonical workflow:  0da83cfddf6f9fa4b3e4a755158d87a6a396a75b
witness workflow:    c8493aa4486e6dc3069afc82aa25dd4d435f49b4
hosted Rust:         1.94.0
```

If any of these identities changes, this recovery harness is stale and must not be used as evidence for the new generation.

## Why #578 is the first RCA recovery target

The shadow-disposition prerequisite stack is ordered:

```text
#578 canonical evidence-lineage generation
  ↓
#531 preregistered disposition policy
  ↓
#555 effective policy
  ↓
#582 exact cross-artifact preflight
  ↓
#585 canonical-lineage-bound preflight
  ↓
#588 preregistered evaluation surface
```

Recovering downstream children before #578 has executable evidence would spend scarce runner capacity without establishing their foundation.

Only the earliest blocked prerequisite receives a trusted recovery workflow in this tranche.

## Frozen target diff

The recovery workflow verifies that the complete diff from the exact base to #578 is exactly:

```text
.github/workflows/rca-canonical-evidence-lineage-identity.yml
.github/workflows/rca-independent-evidence-set-witness.yml
crates/core/symthaea-epistemic-governance/CANONICAL_EVIDENCE_LINEAGE_IDENTITY.md
crates/core/symthaea-epistemic-governance/INDEPENDENT_EVIDENCE_SET_WITNESS.md
crates/core/symthaea-epistemic-governance/src/evidence_set_witness.rs
crates/core/symthaea-epistemic-governance/src/lib.rs
crates/core/symthaea-epistemic-governance/src/lineage_identity.rs
```

This excludes root Cargo/Nix/toolchain changes, build scripts, live cognition, RCA bridge/runtime code, and unrelated governance files.

## Trust split

The workflow uses two independent directories:

```text
trusted harness
    = exact main commit that defines the recovery workflow

frozen source
    = exact #578 commit/tree
```

The target source never supplies runner policy or recovery shell code.

The harness is fetched from `GITHUB_SHA` on trusted `main`. The target and its exact base are fetched separately from public Git and authenticated by:

- exact target commit;
- exact target tree;
- base-is-ancestor check;
- exact hosted workflow blob for both focused gates;
- exact seven-path diff allowlist.

The workflow declares `permissions: {}` and does not use `actions/checkout`, third-party JavaScript Actions, or a caller-supplied SHA/command.

## Hosted gate reproduction

The recovery harness reproduces the semantic/static checks from both exact #578 hosted workflows.

### Canonical lineage identity

It freezes:

- legacy `graph_id` excluded from canonical governance identity;
- complete semantic graph commitment fields;
- canonical node/parent ordering;
- stable derivation-kind tags;
- no JSON-byte, debug-format, `DefaultHasher`, or Rust `Hash` identity semantics;
- all adversarial identity tests named by the hosted gate;
- no canonical-belief/workspace/action/self-improvement/disposition authority.

### Independent evidence-set witness

It freezes:

- evidence-item identity distinct from ancestry-root multiplicity;
- canonical complete lineage-generation binding;
- pairwise-independent-set issuance from the validated graph only;
- no caller-authored roots, pair status, independence count, or lineage identity;
- no count shortcut API;
- serializer-independent witness identity;
- Serialize-only issued witness boundary;
- all adversarial witness tests named by the hosted gate;
- no downstream belief/workspace/action/disposition authority.

### Dependency direction

The harness requires the normal dependency tree for `symthaea-epistemic-governance` to remain free of RCA runtime/bridge dependencies.

## Compiler and dependency parity

Both hosted #578 workflows explicitly install Rust `1.94.0`.

The recovery harness therefore does **not** use the repository's newer ambient `rust-toolchain.toml` version as a substitute. It creates a temporary recovery toolchain descriptor under `RUNNER_TEMP` and asks the trusted harness's pinned `nix/ci-rust-shell.nix` to resolve Rust `1.94.0` through the target flake's pinned `rust-overlay`.

The shell asserts:

```text
rustc --version == 1.94.0
```

before running qualification commands.

Recovery additionally uses Cargo `--locked` for metadata, dependency-tree, tests, and Clippy. This is intentionally stricter than the hosted workflow: dependency resolution may not mutate or reinterpret the committed lock graph.

The Rust gate is:

```text
cargo tree --locked -p symthaea-epistemic-governance -e normal
cargo fmt -p symthaea-epistemic-governance -- --check
cargo metadata --locked --format-version 1
cargo test --locked -p symthaea-epistemic-governance --lib lineage_identity
cargo test --locked -p symthaea-epistemic-governance --lib evidence_set_witness
cargo test --locked -p symthaea-epistemic-governance
cargo clippy --locked -p symthaea-epistemic-governance --all-targets -- -D warnings
```

## Execution precondition

Do not dispatch this recovery workflow merely because the runner is online.

The main-only `Self-hosted NixOS Runner Smoke` must first succeed on the same trusted CPU capability after the recovery infrastructure exists on `main`.

Only then may:

```text
Trusted Recovery - RCA Canonical Lineage
```

be dispatched.

## Attestation

On success, the recovery workflow emits a deterministic text manifest containing:

- GitHub run/attempt;
- runner identity/OS/architecture;
- trusted harness commit;
- target commit/tree/base;
- both hosted workflow blobs;
- Rust 1.94.0 toolchain identity;
- pinned nixpkgs revision;
- `flake.lock` and `Cargo.lock` SHA-256;
- exact gate description;
- locked dependency mode;
- correctness/reproducibility-only evidence scope.

The manifest SHA-256 is printed and summarized in `GITHUB_STEP_SUMMARY`.

No artifact upload Action is required.

## Evidence boundary

A PASS means only:

```text
this exact #578 tree
+ these exact hosted gate semantics
+ Rust 1.94.0
+ this pinned dependency graph
+ this qualified trusted CPU substrate
→ executed successfully
```

It does **not** mean:

- #578 is merged;
- downstream RCA PRs are qualified;
- the shadow-disposition engine may be implemented;
- canonical belief or action authority exists;
- trusted CPU timing is comparable to GitHub-hosted timing;
- queued hosted runs should be rewritten as PASS.

Hosted and trusted-recovery evidence remain separately attributable execution records.
