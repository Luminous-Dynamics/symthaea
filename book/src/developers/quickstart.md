# Developer Quick Start

## Prerequisites

- **NixOS** or Nix package manager (the flake provides mold linker, sccache, and all build tools)
- **Rust** (stable, provided by flake)
- ~4 GB disk for build artifacts (sccache handles caching)

## Build & Run

```bash
# Enter the development environment
nix develop

# Build with default features (minimal kernel)
cargo build --release

# Run tests
cargo test --lib                    # Unit tests (default features)
cargo test --all-features           # Full test suite

# Run specific sub-crate tests
cargo test -p symthaea-core --lib   # Core HDC/LTC/Phi tests
cargo test -p symthaea-broca --lib  # Language pipeline tests
```

## Project Structure

```
symthaea/
├── src/                          # Main crate (~5,584 tests)
│   ├── symthaea.rs               # Public facade (8-phase pipeline)
│   ├── cognitive_loop/           # Core loop + 20+ managers
│   └── consciousness/            # Sub-crate integrations
├── symthaea-core/                # HDC, CfC, IIT, substrate (~4,031 tests)
├── crates/                       # 63 sub-crates
│   ├── symthaea-broca/           # Language pipeline (43-channel encoder)
│   ├── symthaea-neuromodulators/  # 9-transmitter bath
│   ├── symthaea-harmonies/       # Eight Harmonies value framework
│   ├── symthaea-psych-bench/     # 141 cognitive benchmarks
│   ├── symthaea-soma/            # Mobile embodiment
│   ├── symthaea-web/             # Web portal (Leptos 0.8)
│   ├── symthaea-pulse/           # Telemetry dashboard
│   └── ...                       # 55 more specialized crates
├── tests/                        # Integration tests (~1,731)
├── examples/                     # 293 example files (50 benchmarks)
└── papers/                       # 27 research papers + book
```

The workspace has **65 members** and **~120 feature flags** (all disabled by default).
