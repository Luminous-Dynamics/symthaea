# Contributing to Mycelix

Thank you for your interest in contributing to Mycelix, a fractal CivOS built on Holochain.

## Prerequisites

- **Rust** 1.82 or later
- **Holochain** 0.6.0
- **HDK** 0.6.0
- **WASM target**: `rustup target add wasm32-unknown-unknown`
- **Nix** (recommended): A `flake.nix` is provided for reproducible environments. Run `nix develop` to enter the shell.

## Getting Started

```bash
git clone https://github.com/Luminous-Dynamics/mycelix.git
cd mycelix
nix develop  # recommended
```

## Building

All cluster DNAs compile to WASM:

```bash
cargo build --release --target wasm32-unknown-unknown
```

Use the justfile for individual clusters:

```bash
just build-commons
just build-civic
just build-identity
just build-governance
```

## Testing

**Unit tests** run directly with cargo:

```bash
cargo test
just test-identity
just test-commons
```

**Integration tests** (sweettest) require a running Holochain conductor. See each cluster's `tests/` directory for details.

### getrandom WASM Gotcha

WASM builds require the custom getrandom backend. Set this flag before building or testing WASM targets:

```bash
export CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS='--cfg getrandom_backend="custom"'
```

All zomes in this project already configure this, but you must set it in your environment for builds to succeed.

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Make your changes, keeping commits focused and well-described.
3. Ensure all tests pass: `cargo test` for unit tests, sweettest for integration tests.
4. Run formatting and linting:
   ```bash
   cargo fmt --check
   cargo clippy -- -D warnings
   ```
5. Submit a pull request against `main` with a clear description of the change.

## Code Style

- **Formatting**: `cargo fmt` (default rustfmt settings).
- **Linting**: `cargo clippy` with no warnings. All clusters enforce `#![deny(unsafe_code)]`.
- **Numeric safety**: All f64/f32 fields in zome entry types must include `is_finite()` validation guards.
- **No unsafe code**: `#![deny(unsafe_code)]` is enforced across all crates.

## License

Mycelix is licensed under Apache-2.0. By submitting a pull request, you agree that your contributions will be licensed under the same terms.
