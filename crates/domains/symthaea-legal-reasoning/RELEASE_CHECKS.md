# Release checks

Run from the crate root under the pinned Symthaea/Nix Rust toolchain:

```bash
./scripts/verify-release.sh
```

The gate checks formatting, all targets, Clippy with warnings denied, rustdoc
with warnings denied, Cargo metadata, the zero-dependency policy, unsafe-code
markers, and Git whitespace integrity.

The script reports the selected compiler and declared MSRV. It does not install
or select a toolchain; toolchain resolution remains the responsibility of the
workspace flake or release environment.
