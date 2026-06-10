# Symthaea

**Privacy-preserving AI with verifiable consciousness — prove your system thought correctly without revealing what it thought about.**

[![CI](https://github.com/Luminous-Dynamics/symthaea/actions/workflows/ci.yml/badge.svg)](https://github.com/Luminous-Dynamics/symthaea/actions)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Symthaea is a cognitive architecture that combines hyperdimensional computing, consciousness measurement, and zero-knowledge proofs. It proves health compliance in 34ms, verifies anonymous votes with post-quantum signatures, and achieves 256x fewer ZKP constraints than prime-field STARKs — because HDC's core operation (XOR) is free in binary-field proof systems.

[**Live Demo**](https://symthaea.luminousdynamics.io) | [**Paper**](papers/binius-hdc/) | [**eIDAS Demo**](https://github.com/Luminous-Dynamics/symthaea/tree/main/papers/binius-hdc) | [**Benchmarks**](papers/binius-hdc/reproduce.sh)

---

## What You Can Build

- **Healthcare attestation** — Prove "patient vitals are in normal range" in 34ms without revealing the actual values. HIPAA 2026 compliant. ([health-zkp](crates/mycelix-zkp-core/))
- **Anonymous voting** — Prove "I'm eligible to vote" without revealing my identity or consciousness score. Post-quantum Dilithium5 signatures. ([governance](crates/mycelix-zkp-core/src/consciousness.rs))
- **Selective credential disclosure** — Show only your nationality to a verifier while proving "age >= 18" via ZKP. eIDAS 2.0 ready. ([eidas-zkp demo](papers/binius-hdc/))
- **Verified federated learning** — 16 participants submit encrypted gradients, aggregation proven correct in 91ms. Zero encryption overhead. ([triple-stack](crates/hdc-zkp-bench/src/fl_triple_stack.rs))
- **Verifiable search** — Prove search results were ranked per the stated formula, not manipulated. ([prism-zkp](papers/binius-hdc/))
- **Cognitive loop integrity** — Prove a neural network thought correctly without revealing internal state. ([cycle_integrity](crates/symthaea-zkproof/src/cycle_integrity.rs))

## Quick Start

```bash
git clone https://github.com/Luminous-Dynamics/symthaea.git
cd symthaea

# Enter the pinned development shell
nix develop

# Build
cargo build --release

# Run tests (21K+ workspace-wide)
cargo nextest run --workspace --lib

# Run ZKP benchmarks
cd crates/mycelix-zkp-core && cargo test --features full --lib

# Run Binius HDC benchmark (needs .deps/binius64)
cd crates/hdc-zkp-bench && cargo run --release

# Try the Spore kernel in browser
open https://symthaea.luminousdynamics.io
```

## Key Numbers

### ZKP: Binary-Field STARKs vs Prime-Field (Same Scale, Both Measured)

| Metric | Binius (16Kbit) | Winterfell (16Kbit) | Advantage |
|--------|---------------|-------------------|-----------|
| **Constraints** | 256 | 65,532 | **256x fewer** |
| **Prover time** | 430 ms | 23,762 ms | **55x faster** |
| **Verifier time** | 10.4 ms | 23.7 ms | **2.3x faster** |

XOR binding — the core HDC operation — requires **zero non-linear constraints** in Binius. Both numbers are real cryptographic proofs, not estimates.

### ZKP Circuit Library

| Circuit | AND Gates | Prove Time | What It Proves |
|---------|-----------|------------|----------------|
| XOR binding (16Kbit) | 256 | 430 ms | HDC binding correctness |
| Bundling (3 vectors) | 768 | 925 ms | Majority vote aggregation |
| Hamming similarity | 511 | 498 ms | Search relevance |
| CfC temporal (64N×100T) | 7,360 | 954 ms | Neural network evolution |
| CfC + sigmoid | 13,760 | 680 ms | Full neural computation |
| FL encrypted (16 participants) | 1,024 | 91 ms | Encrypted gradient aggregate |
| Health range proof | ~32 | 27 ms | Value in [min, max] |
| Consciousness tier | ~32 | 46 ms | Phi >= threshold |

### Cognitive Architecture

| Capability | Metric |
|------------|--------|
| Cognitive loop frequency | 31 Hz measured (234 Hz raw) |
| Consciousness indicators | 14/14 Butlin, mean 0.85 |
| Moral reasoning | 91.1% Hendrycks Ethics |
| HDC dimension | 16,384D binary + continuous |
| WASM kernel | 324 KB (runs in browser) |
| Workspace | 87 crates, 21K+ tests |

## For Developers

**Core crates:**

| Crate | What | Tests |
|-------|------|-------|
| `symthaea` | Main cognitive loop (4-phase: perceive → evolve → measure → act) | 7K+ |
| `symthaea-core` | HDC vectors, CfC neurons, consciousness math | 3K+ |
| `mycelix-zkp-core` | ZKP infrastructure: backends, circuits, Dilithium5, domain tags | 65+ |
| `hdc-zkp-bench` | Binius benchmark circuits (XOR, bundling, CfC, FL, Hamming) | Benchmarks |
| `symthaea-zkproof` | Consciousness attestation (RISC0) + cycle integrity | 23+ |
| `symthaea-spore` | Browser WASM kernel (324 KB) | 183 |
| `symthaea-broca` | Language generation (CfC-HDC thought-to-text) | 229+ |

**Feature flags:** 98 flags, default = minimal. Key flags: `reasoning_engine`, `identity`, `neural-bridge`, `ssm_language`. See [FEATURE_FLAGS.md](docs/FEATURE_FLAGS.md).

**Contributing:** See [CONTRIBUTING.md](CONTRIBUTING.md). Start with [docs/START_HERE.md](docs/START_HERE.md).

## For Researchers

### Papers

| Paper | Venue | Status |
|-------|-------|--------|
| [**Binary-Field STARKs for HDC**](papers/binius-hdc/) | IEEE S&P 2027 | Submission-ready (v5) |
| [CfC Temporal Proofs](papers/cfc-zkp/) | NeurIPS workshop | 3-page draft |
| [Triple-Stack FL](papers/triple-stack-fl/) | PoPETs | 3-page draft |
| HAI: Hyperdimensional Active Inference | PLoS Comp Bio | Manuscript complete |
| Genesis: Ethical Frameworks | AI & Ethics | Manuscript complete |
| Psych-Bench | — | 98 benchmarks, 20 domains |

### Reproducibility

```bash
# Reproduce all ZKP benchmark results
cd papers/binius-hdc && bash reproduce.sh

# Run consciousness psych-bench
cargo run --features psych_bench --example run_all_benchmarks
```

## Rust Workflow

Use the flake shell for day-to-day development:

```bash
cd symthaea
nix develop
```

Inside the shell:

```bash
# Fast, isolated test runner
cargo nextest run --workspace --lib

# Focused crate test run
cargo nextest run -p symthaea-core --lib

# Background compile/watch loop
bacon

# Narrow spike for EML/e-graph work
cargo test -p symthaea-eml-egraph

# Real-data conjecture quality dashboard
cargo run -p symthaea-core --example real_discovery_report --features abstract_thought

# Same dashboard, but exit non-zero when health thresholds are breached
cargo run -p symthaea-core --example real_discovery_report --features abstract_thought -- --strict
```

The `symthaea-eml-egraph` crate is an isolated `egg` prototype for equality
saturation experiments around EML normalization. It is intentionally separate
from production conjecture ranking and verification logic.

For discovery work, prefer `real_discovery_report` as the project-health
baseline. It summarizes real observation families by verification rate, EML
coverage, macro-candidate pressure, promoted operators, suspicious formulas, and
runtime. Use `--strict` only for opt-in CI guardrails; the current default is a
soft warning report.

`psych-bench` is an internal benchmark suite. For the honest per-module status of math and science capabilities (Lie theory, Langlands, thermodynamics, persistent homology, frontier physics, etc.), see [`MODULE_STATUS.md`](MODULE_STATUS.md). External-benchmark results (miniF2F, PutnamBench, ARC-AGI-2) are planned for Phase 1.

### Key Research Claims

- **256x constraint reduction** for HDC XOR binding in binary-field STARKs (measured, same scale)
- **Zero encryption overhead** for federated learning (OTP + Binius share GF(2))
- **694 KB Binius verifier** compiles to WASM (in-zome verification possible)
- **14/14 Butlin consciousness indicators** present (mean score 0.85)
- **91.1% moral accuracy** on Hendrycks Ethics (4 categories, 2K samples)

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         Symthaea Cognitive Loop       │
                    │  Perceive → Evolve → Measure → Act   │
                    │         (~31 Hz, 4-phase)            │
                    └──────────┬──────────────────┬────────┘
                               │                  │
                    ┌──────────▼──────┐  ┌────────▼────────┐
                    │  HDC Encoder    │  │  CfC Dynamics    │
                    │  16,384D binary │  │  Closed-form LTC │
                    │  + continuous   │  │  O(1) per step   │
                    └──────────┬──────┘  └────────┬────────┘
                               │                  │
                    ┌──────────▼──────────────────▼────────┐
                    │        DASTARK ZKP Layer              │
                    │  Binius (HDC) + Winterfell (range)    │
                    │  + RISC0 (complex) + Dilithium5 (PQ)  │
                    └──────────┬──────────────────┬────────┘
                               │                  │
                    ┌──────────▼──────┐  ┌────────▼────────┐
                    │ Mycelix (DHT)   │  │  Applications    │
                    │ 9 hApps on      │  │  Health, Identity│
                    │ Holochain       │  │  Finance, DeSci  │
                    └─────────────────┘  └─────────────────┘
```

See full architecture diagrams in [`docs/architecture/`](docs/architecture/).

## Regulatory Applications

| Regulation | Deadline | What We Provide |
|------------|----------|----------------|
| **HIPAA 2026** (US) | 2026 | Health attestation: prove vitals/labs in range without revealing values (34ms) |
| **eIDAS 2.0** (EU) | End 2026 | Selective disclosure credentials with `dastark-2026` cryptosuite + Dilithium5 |
| **EU DPP** (EU) | Feb 2027 | Digital Product Passport: prove compliance without revealing supply chain |

## License

AGPL-3.0-or-later. Commercial licensing available — contact tristan.stoltz@evolvingresonantcocreationism.com.

## Citation

```bibtex
@misc{stoltz2026biniushdc,
  title={Binary-Field STARKs for Hyperdimensional Computing},
  author={Stoltz, Tristan},
  year={2026},
  note={Available at github.com/Luminous-Dynamics/symthaea/papers/binius-hdc}
}
```
