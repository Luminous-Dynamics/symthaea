# Changelog

All notable changes to Symthaea HLB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security
- Hardened the service daemon and benchmark API control planes around bearer-authenticated access, private result handling, audit logging, and read-only remote execution policy.
- Added split hardened CI coverage for lib, daemon, API, and Nix deployment paths.
- Added Nix deployment checks for both service-module evaluation and VM-backed socket-activation smoke coverage.

### Changed
- Froze the current daemon protocol/auth surface as the default compatibility boundary and documented reserved-but-unimplemented daemon verbs explicitly.

## [0.5.0] - 2026-02-03

### Added
- **Genesis deterministic seeding**: SHAKE-256 sponge from constitution phrase, all 14+ modules migrated
- **BPTT gradient fix**: Proper back-propagation through output projection, 23% error reduction in 30 epochs
- **Adaptive HDC dimensionality**: Start at 2048, scale to 16384 based on prediction error
- **Stability regimes**: Crystallized/Plastic/Fluid dynamics with decrystallization after 100 idle cycles
- **Compositionality engine**: Sequential/Parallel/Fallback operators wired into MAGI and REPL
- **Neural bridge in cognitive loop**: BGE-M3 embedding -> HDC projection -> CfC temporal processing
- **Streaming Ollama output**: Tokens appear as they arrive in REPL
- **Discovery from crystallization**: Regime transitions trigger neighbor exploration
- **Multi-scale temporal prediction**: Hierarchical CfC world model differentiates timescales
- **Full pipeline demo**: 4-phase example (learning, composition, adaptation, consolidation)

### Performance (release build)
- CfC inference: 34us/step (30K steps/sec)
- BPTT training: 5ms/step (200 steps/sec)
- HDC-LTC (2048-dim): 2.2ms/step
- HDC-LTC (16384-dim): 17ms/step
- Neural bridge cache: 6.5M x speedup

### Fixed
- BPTT gradient chain (was targeting hidden state, not output projection)
- 3 plasma physics test failures
- f32::MAX loss in REPL (dimension mismatch)
- OOM during linking (switched to lld)
- 16 critical Dependabot vulnerabilities

### Testing
- 3388 tests passing, 0 failures (release mode)
- Full suite runs in ~7 minutes

## [0.2.0] - 2026-01-26

### Added
- **Swarm Intelligence**: Hybrid Iroh + Holochain architecture for P2P consciousness
  - Iroh 0.95 integration for real-time tensor streaming
  - Holochain Cortex layer for trust and identity verification
  - Hyperfeel module for synthetic mirror neurons
  - Local swarm simulation for testing
- **Negative Prototypes**: Active disbelief system to prevent hallucination overconfidence
  - Gravity wells around known fictional/mythological topics
  - Fixes the "Dunning-Kruger Trap" in knowledge seeding
- **Nix Packaging**: Full `nix build` support
  - Reproducible builds via flake.nix
  - Cross-platform compatibility

### Fixed
- 37 compiler warnings silenced with proper `#[allow(dead_code)]` annotations
- Removed mycelix-sdk path dependency for Nix sandbox compatibility
- CI workflows updated to use non-deprecated GitHub Actions

### Changed
- Upgraded to Iroh 0.95 unified API (iroh-net deprecated)
- Regenerated Cargo.lock with 1,136 packages

### Tests
- 827 library tests passing
- 45 swarm-specific tests
- Full integration test suite

## [0.1.0] - 2025-12-01

### Added
- Initial release of Symthaea HLB (Holographic Liquid Brain)
- HDC (Hyperdimensional Computing) with 16,384-dimensional vectors
- 33 topology generators for consciousness modeling
- LTC (Liquid Time-Constant) neural networks
- Integrated Information Theory (IIT) Phi calculation
- ConsciousnessGraph with autopoietic loops
- MultiBandState oscillatory router (delta, theta, alpha, beta, gamma)
- PyPhi integration for IIT 3.0 validation
- Speech-to-text with symthaea-stt crate
- Symthaea-sentinel for zero-shot audio pattern recognition
- NixOS integration and language parsing
- Embeddings via Qwen3/BGE models
- TUI shell interface
- Service binary with Unix socket and TCP support

[0.5.0]: https://github.com/Luminous-Dynamics/symthaea-hlb/compare/v0.2.0...v0.5.0
[0.2.0]: https://github.com/Luminous-Dynamics/symthaea-hlb/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Luminous-Dynamics/symthaea-hlb/releases/tag/v0.1.0
