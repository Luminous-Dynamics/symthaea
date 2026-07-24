# Applying Symthaea Music Theory Patch Series 21

## Expected base

Replace `<SERIES_20_FINAL_TREE>` only after independently verifying the exact Patch Series 20 source snapshot.

```text
git rev-parse HEAD^{tree}
# <SERIES_20_FINAL_TREE>
```

## Apply

```text
sha256sum -c SHA256SUMS
git am --3way patches/*.patch
```

Do not apply the `.patch-plan.md` files in this authoring kit. They are specifications for producing the real mail series once the exact base is available.

## Canonical verification

```text
cargo fmt --all -- --check
cargo check --all-targets --all-features
cargo test --all-targets --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo check -p symthaea-music-theory --examples
```
