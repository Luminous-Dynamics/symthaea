# Applying Symthaea Music Theory Patch Series 17

## Expected base

Apply this series to Patch Series 16 with Git tree:

```text
f9f0c8252ca819be2994ee16823c2e14a16c1968
```

Confirm the tree before applying:

```bash
git rev-parse HEAD^{tree}
```

## Apply the mail series

Extract the patch archive and run from the `symthaea-music-theory` repository:

```bash
git am --3way patches/*.patch
```

When the numbered patch files are in the current directory:

```bash
git am --3way ./*.patch
```

After application, compare `git rev-parse HEAD^{tree}` with the final tree
recorded in the Series-17 release summary.

## Verify

Run the canonical project checks:

```bash
cargo fmt --all -- --check
cargo check --all-targets
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets
```

Run the repository's normal Nix verification lane.

Useful publication-tool smoke checks:

```bash
cargo run --example evidence_publication_policy -- --help
cargo run --example evidence_publication_delegation -- --help
cargo run --example evidence_publication_catalog -- --help
cargo run --example evidence_third_party_audit_package -- --help
```

Exercise the complete publication path with the real external delegation
verifier. The catalog `publish` command now requires that verifier and refuses
to mutate the catalog when authentication fails.

## Revert

Before sharing the commits, reset to the Series-16 tip. After sharing, revert
the Series-17 commits in reverse order rather than rewriting shared history.

## Important trust limits

- A structurally valid signed delegation is not authenticated until the
  external verifier succeeds.
- A catalog SHA-256 is tamper-evident but does not establish who controls the
  catalog authority.
- A status proof is valid only for its exact packaged catalog head.
- A third-party audit package does not prove legal compliance, transparency-log
  freshness, or deletion from uncontrolled copies.
