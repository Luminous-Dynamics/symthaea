# Installation

> **Status:** draft — fuller platform-specific notes coming in Phase 0 of the [roadmap](../roadmap.md).

## Rust toolchain

Symtropy tracks **Rust stable** and maintains a minimum supported Rust version (MSRV). Check `rust-toolchain.toml` at the repo root.

```bash
rustup update stable
```

## Via crates.io

```toml
[dependencies]
symtropy-math = "0.2"              # Apache-2.0 OR MIT
symtropy-physics = "0.2"           # Apache-2.0 OR MIT
symtropy-bevy = "0.2"              # Apache-2.0 OR MIT (Bevy plugin)

# Optional: research layer (AGPL-3.0-or-later)
symtropy-consciousness-physics = "0.1"

nalgebra = "0.34"
```

## From source

```bash
git clone https://github.com/luminous-dynamics/symtropy
cd symtropy
cargo build --release
```

## Platform notes

- **Linux** — First-class target. X11 is currently default; Wayland support being re-enabled in Phase 0.
- **macOS** — Builds; audit in progress. Gamepad and audio path need validation.
- **Windows** — Builds; audit in progress.
- **WASM** — Target in Phase 4; requires Bevy WASM setup.

## Dev dependencies for contributing

```bash
cargo install mdbook         # to build this book
cargo install cargo-criterion # for benchmarks
```

## Running tests

```bash
cargo test -p symtropy-math -p symtropy-physics -p symtropy-consciousness-physics --lib
```

## Running the book

```bash
mdbook serve book/           # live preview at http://localhost:3000
```
