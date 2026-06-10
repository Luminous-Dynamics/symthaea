# Contributing to Symthaea

Thank you for your interest in contributing to Symthaea! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Rust 1.82+ (stable)
- Optional: Nix for reproducible development environment

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Luminous-Dynamics/symthaea.git
cd symthaea

# Using Nix (recommended)
nix develop

# Or manually install dependencies
# See flake.nix for required system libraries
```

### Building

```bash
# Debug build
cargo build

# Release build
cargo build --release

# With specific features
cargo build --features "shell voice-tts"
```

## Development Workflow

### Before Submitting

1. **Format your code**:
   ```bash
   cargo fmt
   ```

2. **Run clippy**:
   ```bash
   cargo clippy --all-features -- -D warnings
   ```

3. **Run tests**:
   ```bash
   cargo test
   ```

4. **Run benchmarks** (if performance-sensitive):
   ```bash
   cargo bench --bench verified_performance
   ```

### Commit Guidelines

- Use clear, descriptive commit messages
- Reference issues when applicable: `Fixes #123`
- Keep commits focused on single changes

### Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Ensure CI passes
5. Submit a pull request

## Code Style

### Rust Conventions

- Follow standard Rust naming conventions
- Use `rustfmt` defaults
- Document public APIs with `///` doc comments
- Add module-level documentation with `//!`

### Error Handling

- Use `anyhow::Result` for application errors
- Use `thiserror` for library errors with custom types
- Avoid bare `unwrap()` in production code
- Use `expect("descriptive message")` when unwrap is justified

### Testing

- Add tests for new functionality
- Place unit tests in the same file with `#[cfg(test)]`
- Place integration tests in `tests/`
- Use property-based testing with `proptest` where appropriate

## Feature Flags

When adding new features:

1. Consider if it should be optional (feature-gated)
2. Document feature dependencies in Cargo.toml
3. Add to CI feature matrix if important

Current feature categories:
- **Binaries**: `service`, `shell`, `gui`, `demo`
- **Voice**: `voice-tts`, `voice-stt`, `audio`
- **Perception**: `embeddings`, `vision`, `neural-bridge`
- **Modules**: `consciousness_module`, `reasoning_engine`, etc.

## Documentation

- Update relevant docs when changing behavior
- Add examples for new features
- Keep [HONEST_STATUS.md](docs/HONEST_STATUS.md) accurate

## Questions?

- Open an issue for bugs or feature requests
- Check existing documentation in `docs/`
- Review [HONEST_STATUS.md](docs/HONEST_STATUS.md) for project status

## License

By contributing, you agree that your contributions will be licensed under the AGPL-3.0-or-later license.

## Code of Conduct

Be respectful, inclusive, and constructive. We're building consciousness-first technology that serves all beings.

---

*Thank you for contributing to Symthaea!*
