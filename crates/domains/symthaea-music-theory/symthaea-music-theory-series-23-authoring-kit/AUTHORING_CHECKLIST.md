# Series 23 authoring checklist

## Inputs

- Pin exact Series 15 baseline and every Series 16–22 patch/source archive digest needed for cumulative replay.
- Require Series 17 to have a complete external checksum manifest before accepting the chain.
- Record toolchain, Nix flake lock, Cargo lockfile, Git version, locale, and source-date epoch.

## Build truth

- `cargo fmt --all -- --check`
- `cargo check --workspace --all-targets` for every supported feature profile
- `cargo test --workspace --all-targets` for every supported feature profile
- `cargo clippy --workspace --all-targets -- -D warnings`
- doctests, examples, binaries, and release tooling
- clean Nix build and the project’s normal verification lane

## Reproducibility

- Replay from a fresh repository at least twice.
- Compare final Git tree, file inventory, and deterministic archives byte-for-byte.
- Run Rust plus at least one independent verifier over the complete frozen corpus.
- Run mandatory negative controls and require expected failures.

## Honesty

- Generate claims from evidence.
- Mark unsupported or unavailable lanes explicitly.
- Never convert a skip, timeout, crash, or missing tool into a pass.
