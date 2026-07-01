# Contributing

## Development Environment

```bash
# Clone and enter the development shell
git clone https://github.com/Luminous-Dynamics/symthaea.git
cd symthaea
nix develop  # Required — provides mold linker, sccache, build tools
```

The `nix develop` shell is **required**. Without it, `cargo build` will fail with `cannot find 'ld'` because `.cargo/config.toml` requires the `mold` linker.

## Code Guidelines

- **Rust stable** — no nightly features required
- **No custom CARGO_TARGET_DIR** — use the project's default `target/`. sccache handles caching via `~/.cargo/config.toml`
- **Feature-gate new subsystems** — all defaults are empty for minimal compilation
- **CognitiveSubsystem trait** — new managers must implement immutable-borrow `process()`. The trait's signature makes it impossible for a manager to mutate shared cognitive state.
- **Co-prime intervals** — new manager tick intervals must be co-prime with all existing ones to prevent phase-locking. Check existing intervals in `src/cognitive_loop/managers/mod.rs`.
- **Named constants with citations** — thresholds in `src/cognitive_loop/thresholds/` must cite scientific literature in comments (e.g., `// Schultz 1997`)
- **NaN/Inf guards** — all feedback pathways must check `.is_finite()` before applying
- **No aspirational tests** — only test what exists

## Testing

```bash
# Unit tests (fast, default features)
cargo test --lib

# Full test suite (all features)
cargo test --all-features

# Specific sub-crate
cargo test -p symthaea-broca --lib

# Property tests
cargo test --lib proptest
```

## Pull Request Process

1. Ensure `cargo test --lib` passes with default features
2. Ensure `cargo clippy` produces no warnings
3. Ensure `cargo fmt --check` passes
4. Add tests for new functionality
5. Update documentation if public API changes
6. Submit PR to the **standalone** repo (not the monorepo) for CI

The standalone repo runs CI via `symthaea-ci.yml`: fmt, clippy, test, docs, 49-feature matrix, and 52 sub-crate verification.

## Architecture Principles

- **Edit, don't duplicate** — one implementation per feature
- **Fix the flake, don't hack** — if the Nix flake doesn't provide something, fix the flake
- **Right complexity from start** — no hacks, no "we'll fix it later"
- **Consciousness-first** — every subsystem should interact with the consciousness pipeline
