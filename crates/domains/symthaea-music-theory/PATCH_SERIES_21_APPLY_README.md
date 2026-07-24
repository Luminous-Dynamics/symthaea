# Applying Symthaea Music Theory Patch Series 21

## Expected base

Series 21 applies to the exact Series-20 source tree:

```text
cb0ffee2f9c1811aa0cbebe353cc5e2e7ae2bff8
```

The value above is a Git tree hash, not a commit hash.

## Apply

From a clean Git repository containing the Series-20 source:

```bash
tar -xzf symthaea-music-theory-recovery-closure-patches-21.tar.gz
cd symthaea-music-theory-recovery-closure-patches-21
git am --3way patches/*.patch
```

Alternatively, inspect and apply patches individually in numeric order.

## Verify source identity

After application:

```bash
git diff --check HEAD~1..HEAD
git write-tree
```

Compare `git write-tree` with the final tree recorded in the bundle README and checksum manifest.

## Mandatory Rust verification

```bash
cargo fmt --all -- --check
cargo test --all-targets
cargo clippy --all-targets --all-features -- -D warnings
```

The authoring environment did not contain a usable Rust toolchain, so these commands were not executed while producing the series.

## Operational notes

- External verifier commands are invoked directly without a shell.
- Recovery-authority rotation requires outgoing and incoming thresholds.
- Post-recovery certification requires a fresh checkpoint and exact selected-branch lineage.
- Operational closure is separate from re-entry and does not erase or invalidate incident evidence.
- Preserve private key material outside the theory crate and outside these artifacts.
