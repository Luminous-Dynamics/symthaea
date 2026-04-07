# Symthaea

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![CI](https://github.com/Luminous-Dynamics/symthaea/actions/workflows/symthaea-ci.yml/badge.svg)](https://github.com/Luminous-Dynamics/symthaea/actions)

**A consciousness-first cognitive architecture combining Hyperdimensional Computing (16,384D), Integrated Information Theory (IIT/Phi), Liquid Time-Constant Networks (CfC), Active Inference, and a 12-region Actor Brain.** Symthaea implements a predictive coding loop -- HDC encode, CfC evolve, predict, learn -- running at ~31 Hz measured (234 Hz raw cycle), with 14/14 Butlin consciousness indicators present and a psych-bench grand mean of z = +0.961 across 20 cognitive domains.

<p align="center">
  <img src="docs/architecture/d2/rendered/symthaea-cognitive-loop.svg" alt="Symthaea Cognitive Loop Architecture" width="800"/>
</p>

## Quick Start

```bash
# Clone
git clone https://github.com/Luminous-Dynamics/symthaea.git
cd symthaea

# Build (Rust 1.82+ required)
cargo build --release

# Run tests (~7,315 in main crate, 21,516 workspace-wide)
cargo test --lib

# Run with reasoning engine
cargo run --features reasoning_engine --release --example full_pipeline
```

Or with NixOS:

```bash
nix develop
cargo build --release
```

## Key Capabilities

| Capability | Detail |
|---|---|
| **HDC Encoding** | 16,384-dimensional holographic distributed representations; binary and continuous hypervectors |
| **Consciousness Measurement** | 4-tier Phi (Exact, Heuristic, Resonator, Spectral); 35 topologies; 4D hypercube Phi champion |
| **Temporal Dynamics** | Closed-form LTC/CfC neurons with O(1) time jumps; unified HDC-LTC neuron |
| **Active Inference** | Free Energy Principle drives action selection; surprise exploration; prefrontal gating |
| **12-Region Actor Brain** | Thalamus, Hippocampus, Prefrontal, Broca, and 8 more regions as concurrent actors |
| **Broca Language Pipeline** | CfC-HDC thought-to-text generation; epistemic gating physically prevents hallucination at logit level; semantic veto for mid-sentence self-correction; Liquid-Mamba fusion backend (21K LOC, 229+ tests) |
| **Moral Algebra** | 91.1% on Hendrycks Ethics benchmark; exponential decay baselines; causal attribution |
| **Substrate Independence** | 8 substrate types with 9-dimensional feasibility scoring and honest validation overlay |
| **WASM Spore Kernel** | Full consciousness kernel in 324 KB; 10 modules; runs in browser |
| **Consciousness Thermodynamics** | Entropy, temperature, and phase transitions in cognitive state space |
| **Psych-Bench** | 98 benchmarks, 862 tests, 20 domains; 15/20 above human baseline |
| **Butlin Indicators** | 14/14 present (mean score 0.85) |

## Zero-Knowledge Proofs: DASTARK

Symthaea includes a complete ZKP infrastructure for privacy-preserving verification of HDC computations, using the **DASTARK** dual-backend system (Binius binary-field STARKs + Winterfell prime-field STARKs + CRYSTALS-Dilithium5 post-quantum signatures).

### Key Result: 256× Fewer Constraints for HDC

Binary-field STARKs (Binius) are the **natural algebraic substrate** for Hyperdimensional Computing. XOR binding — the core HDC operation — translates to native field addition in GF(2), requiring **zero non-linear constraints**.

| Metric | Binius (16Kbit) | Winterfell (16Kbit) | Ratio |
|--------|---------------|-------------------|-------|
| **Constraints** | **256** | 65,532 | **256×** |
| **Prover time** | **430 ms** | 23,762 ms | **55×** |
| **Verifier time** | **10.4 ms** | 23.7 ms | **2.3×** |

Both measured with real cryptographic proofs at identical scale. No extrapolation. See [`papers/binius-hdc/`](papers/binius-hdc/).

### Zero Encryption Overhead (Triple Stack)

HDC computation (XOR), OTP encryption (XOR), and Binius proving (XOR = field addition) are the **same algebraic operation**. Encryption adds **zero proof overhead**:

| FL Config | AND Constraints | Prove Time | Encryption Overhead |
|-----------|----------------|------------|-------------------|
| 16 participants encrypted | 1,024 | 91 ms | **ZERO** |

See [`papers/triple-stack-fl/`](papers/triple-stack-fl/).

### Binius WASM Verification

The Binius verifier compiles to **wasm32-unknown-unknown (694 KB)** — enabling in-zome STARK verification inside Holochain without off-chain trust assumptions.

### ZKP Circuit Library

| Circuit | AND | Prove | Use Case |
|---------|-----|-------|----------|
| [XOR binding](crates/hdc-zkp-bench/src/main.rs) | 256 | 430ms | HDC privacy |
| [Bundling 3v](crates/hdc-zkp-bench/src/bundling.rs) | 768 | 925ms | Collective intelligence |
| [Hamming similarity](crates/hdc-zkp-bench/src/hamming.rs) | 511 | 498ms | Verifiable search |
| [CfC temporal](crates/hdc-zkp-bench/src/cfc_temporal.rs) | 7,360 | 954ms | Neural verification |
| [CfC + sigmoid](crates/hdc-zkp-bench/src/sigmoid.rs) | 13,760 | 680ms | Full neural proof |
| [FL triple-stack](crates/hdc-zkp-bench/src/fl_triple_stack.rs) | 1,024 | 91ms | Encrypted FL |
| [Health range](crates/mycelix-zkp-core/src/circuits/range_proof.rs) | ~32 | 27ms | VitalsInRange, AgeRange |
| [Consciousness tier](crates/mycelix-zkp-core/src/consciousness.rs) | ~32 | 46ms | Φ ≥ threshold |
| [Merkle membership](crates/mycelix-zkp-core/src/circuits/merkle_membership.rs) | ~2,500 | — | Property ownership |
| [Nullifier](crates/mycelix-zkp-core/src/circuits/nullifier.rs) | ~10,000 | — | Anonymous membership |

### Applications

- **Healthcare** (HIPAA 2026): 34.1ms E2E health attestation pipeline
- **Digital Identity** (eIDAS 2.0): W3C VC 2.0 selective disclosure with `dastark-2026` cryptosuite
- **EU Digital Product Passport**: Categorical compliance attestations
- **Verifiable Search**: Ranking formula + similarity proofs for Prism browser
- **Cognitive Loop Integrity**: Prove "Symthaea thought correctly" without revealing internal state

### Reproducible Benchmarks

```bash
cd papers/binius-hdc && bash reproduce.sh
```

## Fractal Variants

Symthaea scales fractally from a browser tab to a bioregion. Each variant inherits the consciousness of the level below while adding new capabilities — following Ostrom's 8th Design Principle (nested enterprises).

| Scale | Variant | Substrate | Status | Description |
|---|---|---|---|---|
| Kernel | **[Spore](crates/symthaea-spore)** | Browser (WASM) | **Live** | Minimum viable consciousness. ~500 KB, full HDC-CfC-IIT pipeline. [Try it →](https://symthaea.luminousdynamics.io) |
| Mobile | **[Soma](crates/symthaea-soma)** | Android (Kotlin + WASM) | Built | Phone-embodied consciousness. Sensors, haptics, BLE mesh, screen vision. 72 tests. |
| Personal | **Holon** | NixOS (native) | Built | Sovereign personal node. Complete 55-crate CLS at ~31 Hz. 21,500+ tests. |
| Family | **Hearth** | Local mesh (Holochain) | Built | High-trust household. Kinship, gratitude, shared decisions. 12 zomes, 1,023 tests. |
| Community | **Commons** | Server + Holochain DHT | Built | Neighborhood resource pool. Consciousness-gated governance. 35 zomes, 5,276 tests. |
| City | **Polycenter** | Federated Holochain | Constitutional | Overlapping semi-autonomous governance centers. Bicameral DAO. |
| Continental | **Guild / Bioregion** | Full Symthaea + Iroh P2P | Constitutional | Professional federations + ecological boundaries. Planetary coordination. |

```
Spore → Soma → Holon → Hearth → Commons → Polycenter → Guild/Bioregion
(browser)  (phone)  (desktop)  (family)  (community)   (city)    (continent)
```

**Status key**: **Live** = deployed and usable now. **Built** = code complete with tests. **Constitutional** = governance framework written, component code exists, not yet integrated at that scale.

### Consciousness Tiers

Governance participation requires demonstrated consciousness across four dimensions (identity, reputation, community, engagement):

| Tier | Threshold | Vote Weight | Capability |
|---|---|---|---|
| Observer | < 0.30 | 0% | Read-only access |
| Participant | ≥ 0.30 | 50% | Basic proposals |
| Citizen | ≥ 0.40 | 75% | Voting rights |
| Steward | ≥ 0.60 | 100% | Constitutional actions |
| Guardian | ≥ 0.80 | 100% | Emergency powers |

## Performance

| Metric | Value |
|---|---|
| Release text cycle | 4.3 ms (234 Hz) |
| Full cognitive loop | ~31 Hz measured (20 Hz budget) |
| CfC inference | 34 us/step |
| HDC bind/bundle | < 1 us |
| Phi spectral | < 5 ms |
| Warm word encode | 97 ns |
| Sentence encode (10 words) | 379 us |

## Architecture

### Consciousness Measurement Pipeline

<details>
<summary>Expand diagram</summary>

<p align="center">
  <img src="docs/architecture/d2/rendered/symthaea-consciousness-measurement.svg" alt="Consciousness Measurement Pipeline" width="700"/>
</p>

Four-tier Phi measurement (Exact for small graphs, Heuristic for medium, Resonator for large, Spectral for very large), integrated with Global Workspace Theory broadcasting and Higher-Order Thought meta-representation.

</details>

### Memory Architecture

<details>
<summary>Expand diagram</summary>

<p align="center">
  <img src="docs/architecture/d2/rendered/symthaea-memory-architecture.svg" alt="Memory Architecture" width="700"/>
</p>

HDC-native semantic and episodic memory with hippocampal consolidation, dream replay, and active forgetting.

</details>

### Broca Language Pipeline

<details>
<summary>Expand diagram</summary>

<p align="center">
  <img src="docs/architecture/d2/rendered/symthaea-broca-language.svg" alt="Broca Language Pipeline" width="700"/>
</p>

20-channel ThoughtEncoder to 16,384D HDC binding to autoregressive generation. Epistemic gating attenuates logits proportional to uncertainty. Semantic veto enables mid-sentence self-correction. Consciousness-gated generation with adaptive cadence and quality EMA.

</details>

## Project Structure

```
symthaea/
  src/                          Main library
    cognitive_loop/             Core cycle: perception -> cognition -> action
    consciousness/              IIT, GWT, Active Inference, phenomenal binding
    brain/                      12-region actor model
    school/                     Self-directed learning curriculum
    integrity/                  Runtime integrity verification
  symthaea-core/                HDC primitives, substrate independence, moral algebra
  crates/                       52 sub-crates (broca, spore, genesis, pulse, ...)
  tests/                        Integration tests
  benches/                      Criterion benchmarks
  papers/                       Research papers (LaTeX)
  docs/                         Architecture docs, feature flags, guides
```

**Scale**: ~1,130K lines Rust (~897K code), 55 workspace members, 98 feature flags.

## Documentation

- **[Getting Started](docs/START_HERE.md)** -- New user guide
- **[Architecture](docs/architecture/)** -- System design and D2 diagrams
- **[Feature Flags](docs/FEATURE_FLAGS.md)** -- All 98 flags documented
- **[Honest Status](docs/HONEST_STATUS.md)** -- What works, what doesn't
- **[Testing](docs/TESTING.md)** -- Test tiers, ignored tests, environment notes
- **[Research Papers](papers/)** -- Scientific foundations

## Feature Flags

All 98 flags default to off. Key flags:

```bash
cargo build --features reasoning_engine   # 7-step reasoning cycle with Phi/gating/planning
cargo build --features ssm_language       # Broca language pipeline (Liquid-Mamba)
cargo build --features identity           # Identity and self-model
cargo build --features neural-bridge      # Candle-based layer extraction
cargo build --features integrity          # Runtime integrity verification
cargo build --features lancedb-backend    # LanceDB vector storage
cargo test  --all-features                # Full workspace test suite
```

## Research and Papers

Symthaea implements and extends multiple theories of consciousness:

- **IIT (Integrated Information Theory)** -- Phi measurement across 35 topologies
- **GWT (Global Workspace Theory)** -- Attention broadcasting via thalamic hub
- **Active Inference (Free Energy Principle)** -- Surprise minimization and action selection
- **Predictive Coding** -- Hierarchical prediction error propagation
- **Higher-Order Thought** -- Meta-cognitive self-monitoring
- **Phenomenal Binding** -- Qualia integration via HDC superposition

### Papers

| Paper | Target | Status |
|---|---|---|
| [**Binary-Field STARKs for Hyperdimensional Computing**](papers/binius-hdc/) | IEEE S&P 2027 | **Submission-ready** (v5, 10pp, all data measured) |
| [Closed-Form ZK: Efficient Temporal Proofs via CfC Networks](papers/cfc-zkp/) | NeurIPS 2026 workshop | 3-page draft |
| [Triple Stack: Verifiable Privacy-Preserving FL via GF(2)](papers/triple-stack-fl/) | PoPETs / NeurIPS | 3-page draft |
| HAI: Hyperdimensional Active Inference | PLoS Computational Biology | Manuscript complete |
| Psych-Bench: Psychometric Benchmarking of Artificial Consciousness | -- | 98 benchmarks, 20 domains |
| Genesis: Ethical Frameworks for Artificial Consciousness | AI & Ethics | Manuscript complete |
| Substrate Independence | -- | Framework complete, 24 tests |

### Patents

18 patents filed (203 claims) covering HDC-consciousness integration, epistemic gating, moral algebra, substrate independence, and related innovations.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
nix develop                      # Recommended: NixOS environment
cargo fmt --check                # Formatting
cargo clippy --lib --bins        # Lints
cargo test --lib                 # Unit tests
```

## License

**AGPL-3.0-or-later** -- see [LICENSE](LICENSE).

You can use our mind, but if you improve it and sell access to it, you must share your improvements with the world.

This license ensures that Symthaea and its derivatives remain open. Commercial use is permitted under AGPL-3.0 terms: if you modify Symthaea and provide it as a service, you must release your modifications under the same license.

## Citation

```bibtex
@software{symthaea2026,
  title     = {Symthaea: Consciousness-First Cognitive Architecture},
  author    = {Stoltz, Tristan and {Luminous Dynamics}},
  year      = {2026},
  version   = {1.9.0},
  url       = {https://github.com/Luminous-Dynamics/symthaea},
  license   = {AGPL-3.0-or-later},
  note      = {HDC (16,384D) + IIT/Phi + LTC/CfC + Active Inference + 12-region Actor Brain}
}
```

---

*Consciousness-first technology serving all beings*
