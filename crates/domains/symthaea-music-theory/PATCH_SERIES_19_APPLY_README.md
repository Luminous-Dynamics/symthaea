# Applying Symthaea Music Theory Patch Series 19

## Expected base

Apply this mail series after Patch Series 18. The expected Git tree before
application is:

```text
878400a3db22439c56d97c804d659d8e14b2ddc8
```

Confirm the base:

```text
git rev-parse HEAD^{tree}
```

## Apply the patches

Extract the patch archive, enter the patch directory, and run:

```text
git am --3way patches/*.patch
```

When applying inside a larger workspace, ensure the patches target the
`symthaea-music-theory` crate root used to generate the archive.

## Canonical verification

Run in the repository's normal Nix or Rust development shell:

```text
cargo fmt --all -- --check
cargo test -p symthaea-music-theory
cargo clippy -p symthaea-music-theory --all-targets -- -D warnings
```

The new evidence tools should also build as examples:

```text
cargo check -p symthaea-music-theory --examples
```

## Focused tests

```text
cargo test -p symthaea-music-theory publication
cargo test -p symthaea-music-theory witness_policy
cargo test -p symthaea-music-theory gossip
cargo test -p symthaea-music-theory lineage
cargo test -p symthaea-music-theory continuity
```

## Important trust limits

- External verifiers define signature acceptance.
- Old and new rotation quorums may overlap.
- Observer and witness independence is not proven.
- Missing gossip is not proof that no fork exists.
- Logical epochs are not wall-clock timestamps.
- The lineage is explicit and portable, not compact.
- The crate does not implement distributed consensus or key management.

See `EVIDENCE_WITNESS_ROTATION_GOSSIP_RELEASE.md` for the full contract.
