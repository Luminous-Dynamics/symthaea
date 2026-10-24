<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Visual display corpus verifier

This isolated Rust package validates the conventional-display workload manifest in the Symthaea research program.

## Run from the repository root

```sh
cargo fmt --manifest-path tools/visual-display-corpus-verify/Cargo.toml -- --check
cargo clippy --manifest-path tools/visual-display-corpus-verify/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path tools/visual-display-corpus-verify/Cargo.toml
cargo run --manifest-path tools/visual-display-corpus-verify/Cargo.toml
```

Use `--manifest PATH` and `--schema PATH` to validate a copy or a mutation fixture.

## What it checks

- JSON Schema Draft 2020-12 validation, with no remote or file reference resolution enabled.
- The v0.1 freeze: 32 cases and schema count bounds must agree.
- Stable, unique case IDs; unique and resolvable reference IDs; and supported evidence lanes.
- Lane/evidence matching (software mathematics is not a physical measurement).
- Empty/missing required metadata, self-promotion away from `UNQUALIFIED`, duplicate metrics/failure modes, and corpus/schema drift.
- SHA-256 digests of the exact raw manifest and schema bytes, plus per-lane case counts.

A successful report is `CONTRACT_VALID_UNQUALIFIED`. It means only that the manifest conforms to this frozen contract. It is **not** a display benchmark PASS, HDR/ClearMR certification, measured power saving, physical luminance, motion clarity, accessibility outcome, or human-perception result.

The crate uses exact versions for direct dependencies. A generated Cargo.lock should be committed once the pinned toolchain can resolve/build the package in a controlled environment; this authoring session cannot claim that compile/test evidence. The focused CI workflow is intended to establish exact-head compilation and test results without the full Symthaea matrix.
