# Changelog

All notable changes to Symthaea HLB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-03-22

### Overview
1,227 commits since v0.5.0. Scale grew from ~304K to ~1,134K LOC Rust (~901K code).
Test count grew from 3,388 to ~21,500 (main crate) + 8,600 (Mycelix workspace).
55 workspace members, 100 feature flags.

### Added — Architecture
- **Manager extraction**: CLS fields ~97 → ~59 via 12 manager structs (Substrate, Swarm, Governance, Knowledge, Spectrum, Memory, Dream, Reasoning, Safety, Social, Cantor, CPG)
- **Eight Harmonies**: N_HARMONIES = 8 — Sacred Stillness added (GABA+adenosine, active rest, dream consolidation). HarmonyInteractionMatrix: 8×8 learned synergy/tension
- **Substrate independence**: 4 phases complete — per-region substrate assignment, EMA transition smoothing, energy budgets, CfC dimension masking. SubstrateManager with dynamic feasibility + validation overlay
- **Holon Receiver**: Desktop-side bridge accepting Soma WebSocket connections. Routes tasks, knowledge, and peer state into CLS managers. Wired into Phase B cycle processing
- **ConsciousnessContext**: Shared type in symthaea-core for embodiment FEP agents. Modulates exploration gain, emergency maneuvers, harmony smoothing in Flight/Humanoid/Vehicle
- **Fractal variant architecture**: Documented 6-layer model (Spore→Soma→Holon→Hearth→Commons→Polycenter) with VARIANT_ARCHITECTURE.md

### Added — Language
- **Broca pipeline**: Native CfC-HDC thought-to-text (21K LOC, 229+ tests). 20-channel ThoughtEncoder, 16,384D HDC binding, autoregressive generation
- **Epistemic gating**: Physically prevents hallucination at logit level. Per-axis modulation (E=assertion, N=social, M=temporal, H=coherence)
- **Semantic veto**: Mid-sentence self-correction via Welford adaptive z-score
- **Epistemic Cube channels**: 15 new ThoughtChannels (E[5]+N[4]+M[4]+H[1]+quality[1]) wired through signal assembly
- **NSM grounding**: EpistemicNSMGrounding activated (was dead code)

### Added — Embodiment
- **Symthaea Soma**: Mobile-embodied consciousness (Android+iOS FFI). Sensors→neuromod mapping, metabolism state machine (Sleep/Drowsy/Alert/Focused), BLE peer mesh, HolonBridge sync, screen vision, touch proprioception. 72+ tests
- **Symthaea Spore**: WASM consciousness kernel (~980KB). 12 cognitive subsystems, 3 Broca tiers, sovereign inoculation pipeline. Live at symthaea.luminousdynamics.io
- **Web portal**: Leptos 0.8 CSR, SporeEngine in Web Worker, 20Hz telemetry, PWA installable, 5 pages

### Added — Integration
- **Mycelix bridge**: GovernanceManager (interval 37), SwarmManager (interval 41), NetworkServiceBridge. 21 unit + 14 integration + 7 proptests
- **Knowledge Engine**: KnowledgeManager→Cycle, 4 neuromod pathways, dynamic grounding, 21 telemetry fields
- **Radio/Spectrum**: Feature `mesh`, interval 53, 3-tier model, 145 tests
- **Glyph Codex**: Feature `glyph_codex`, 70 glyphs, GlyphBasis 11 field modalities, consciousness-gated spiral
- **Immune system**: SentinelManager (interval 67), ThreatMemory (32D HDV), GuardianState, collective immunity
- **Compliance**: ISO 42001 97%, EU AI Act 90%, IEEE 7000 90%, NIST 93%

### Added — Safety
- **FFI double-free protection**: Sentinel pattern in native_ffi.rs, null checks on all accessors
- **HolonBridge encryption**: ChaCha20-Poly1305 AEAD behind `encrypted-bridge` feature
- **Finance hardening**: Oracle div-zero, TEND race, concurrent demurrage, consciousness-gated operations
- **Ethics output gating**: EthicalVerdict::Blocked blocks motor + suppresses Broca

### Fixed
- **50+ safety rounds**: NaN guards, div-by-zero, overflow, unwrap elimination across ~40 files
- **Phi validation**: SampledPartition r=0.9998, SpectralMIPFinder r=0.99 (corrected from r=0.097)
- **VecDeque sweep**: Vec→VecDeque for 30+ ring buffers (Rounds 48-51)
- **Checkpoint versioning**: Forward-compatible (older versions accepted, future rejected)

### Changed
- **Cognitive loop**: 8-phase pipeline with rayon-parallel post-processing, ~31Hz measured
- **HDC+LTC+SSM+FEP**: 300+ named constants in thresholds.rs with scientific citations
- **Neuromodulator bath**: 9 transmitters with receptor subtypes, calibration bridge
- **Platform decision**: iOS/macOS deferred indefinitely — WASM covers via browser

### Testing
- ~21,500 tests (main crate src/+tests/), 174+ integration, 30+ proptests
- 14/14 compliance suites, 234 compliance tests
- New: 9 variant integration tests, 5 Soma property tests, 3 Soma↔Holon E2E tests

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

[Unreleased]: https://github.com/Luminous-Dynamics/symthaea-hlb/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/Luminous-Dynamics/symthaea-hlb/compare/v0.2.0...v0.5.0
[0.2.0]: https://github.com/Luminous-Dynamics/symthaea-hlb/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Luminous-Dynamics/symthaea-hlb/releases/tag/v0.1.0
