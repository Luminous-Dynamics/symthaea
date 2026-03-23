# Contributing

## Code Guidelines

- **Rust stable** — no nightly features required
- **`nix develop` first** — the flake provides mold linker and sccache
- **No custom CARGO_TARGET_DIR** — use the project's default `target/`
- **Feature-gate new subsystems** — defaults are empty for minimal compilation
- **CognitiveSubsystem trait** — new managers must implement immutable-borrow process()
- **Co-prime intervals** — new manager tick intervals must be co-prime with all existing ones
- **Named constants with citations** — thresholds must cite scientific literature in comments

## Pull Request Process

1. Ensure `cargo test --lib` passes with default features
2. Ensure `cargo clippy` produces no warnings
3. Add tests for new functionality
4. Update documentation if public API changes
5. Submit PR to the standalone repo (not the monorepo) for CI
