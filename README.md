# Symthaea

**Experimental cognitive architecture for testing verifiable state transitions, representation mechanisms, and consciousness-relevant computational indicators — paired with a zero-knowledge proof stack for privacy-preserving attestation.**

[![CI](https://github.com/Luminous-Dynamics/symthaea/actions/workflows/ci.yml/badge.svg)](https://github.com/Luminous-Dynamics/symthaea/actions)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Symthaea combines hyperdimensional computing (HDC), closed-form liquid neural dynamics (CfC), consciousness-relevant measurement (IIT-inspired Φ, Butlin-indicator probes), and zero-knowledge proofs. **It does not claim to prove consciousness, and nothing below is a regulatory compliance certification.** What it does have: working ZK circuits over HDC's binary-field structure, measured against a prime-field baseline, and an internal research program that documents its own retractions — see the Evidence table below before trusting any specific number.

[**Live Demo**](https://symthaea.luminousdynamics.io) | [**Paper**](papers/theory-foundations/binius-hdc/) | [**Benchmarks**](papers/theory-foundations/binius-hdc/reproduce.sh) | [**Historical Status Audit (2026-02-05)**](docs/HONEST_STATUS.md)

---

> **Documentation freshness (2026-11-26).** This is a large, actively changing research workspace. Older package, test, feature-flag, and artifact-size totals are intentionally not presented as current facts. To inventory the exact checkout you are using, run `cargo metadata --no-deps --format-version 1` and the relevant test commands below. Benchmark results are scoped to their stated harness and machine; rerun the linked reproduction scripts before relying on them externally. The linked Honest Status document is a historical audit, not a current workspace inventory.

## Evidence Table

| Claim | Current evidence | Limitation |
|---|---|---|
| HDC XOR binding with Binius | Standalone `hdc-zkp-bench` constructs and runs a Binius prover and verifier over 16,384-bit vectors | This is an internally reproduced benchmark, not a third-party audit or a blanket ZK system guarantee |
| Prime-field comparison | The standalone benchmark includes an analytical constraint/time model for a prime-field comparator | It does **not** run a Winterfell prover; do not describe these figures as a measured head-to-head |
| HDC bundling and CfC experiments | Separate Binius circuit experiments live in `crates/hdc-zkp-bench/src/` | The source documents scope limits; for example, Hamming/popcount and nonlinear activation handling are experimental approximations |
| Federated aggregation | A Binius prototype exercises proof of an encrypted-gradient aggregate computation | This is not a complete secure aggregation protocol, key-management system, or production FL deployment |
| Cycle integrity | `symthaea-zkproof` contains phase/chain commitment structures and an optional proof-bytes field | The current helper initializes proof bytes empty; do not treat it as an end-to-end STARK/zkVM proof pipeline |
| Consciousness-related indicators | Internal re-check retained 4/12 Butlin probe rows, with one narrowly scoped internal causal-wiring result | Not evidence of consciousness, sentience, or demonstrated functional capacity |
| Moral reasoning | Internal Hendrycks Ethics re-run dated 2026-07-15: 56.2% overall, with the `virtue` category at 76.5% | Earlier 91.1%/94.5%/84.7% figures were retracted as leakage-inflated; this is not an external leaderboard result |
| Cognitive-loop frequency | Internal telemetry reports approximately 31 Hz on one reference machine | Not a real-time guarantee or a portable performance claim |

The repository contains different levels of evidence: measured Binius proof experiments, analytical comparison models, structural commitment code, and educational cryptographic simulations. They are deliberately not presented as interchangeable. None of these results establishes general intelligence, sentience, legal/medical/financial compliance, or production security.

## What The ZKP Stack Demonstrates

- **Measured Binius HDC experiments** — the standalone benchmark exercises XOR binding and additional bundling, Hamming-style, CfC, and aggregation circuits. See [`hdc-zkp-bench`](crates/hdc-zkp-bench/). Each circuit has its own implementation limits; the benchmark source is the authority for what is actually constrained.
- **Encrypted aggregation prototype** — [`fl_triple_stack.rs`](crates/hdc-zkp-bench/src/fl_triple_stack.rs) and [`triple_stack.rs`](crates/hdc-zkp-bench/src/triple_stack.rs) exercise Binius proofs about ciphertext computation. They do not provide a complete production OTP key lifecycle or end-to-end federated-learning service.
- **Cycle-integrity commitments** — [`cycle_integrity.rs`](crates/core/symthaea-zkproof/src/cycle_integrity.rs) represents commitments across cognitive-cycle phases. Its current helper leaves `proof_bytes` empty; a commitment chain is not itself a completed STARK proof.
- **Post-quantum signing compatibility subset** — [`stubs/mycelix-zkp-core`](stubs/mycelix-zkp-core/) contains a standalone subset with a feature-gated Dilithium module and shared types. The module's own documentation says the private STARK backends are not vendored into this standalone repository.
- **GIS proof experiments** — [`src/mycelix/gis/zk_proofs.rs`](src/mycelix/gis/zk_proofs.rs) explicitly describes its elliptic-curve operations as educational simulations and recommends established cryptographic libraries for production. It must not be represented as a production Bulletproofs implementation.

## Quick Start

```bash
git clone https://github.com/Luminous-Dynamics/symthaea.git
cd symthaea

# Enter the pinned development shell
nix develop

# Build
cargo build --release

# Run workspace library tests (coverage depends on the selected workspace members)
cargo nextest run --workspace --lib

# Run the standalone Binius HDC benchmark
cargo run --manifest-path crates/hdc-zkp-bench/Cargo.toml --release

# Try the Spore kernel in browser
open https://symthaea.luminousdynamics.io
```

## Measured and modelled results

### 16,384-bit HDC XOR benchmark

| Metric | Binius | Prime-field comparator |
|---|---:|---:|
| Constraint count | 256 (measured) | 65,532 (analytical model) |
| Prover time | 430 ms (measured) | 23,762 ms (analytical model) |
| Verifier time | 10.4 ms (measured) | 23.7 ms (analytical model) |

The Binius column comes from an actual prover/verifier execution in the standalone benchmark. The comparator figures are generated from a constraint-count and per-constraint timing model; the standalone `hdc-zkp-bench` package has no Winterfell dependency and does not run a Winterfell prover. Consequently, these values are **not** a measured same-machine Binius-versus-Winterfell comparison.

### Binius circuit experiments

| Circuit experiment | AND constraints | Prover time | Interpretation |
|---|---:|---:|---|
| XOR binding (16Kbit) | 256 | 430 ms | Binius proof of the XOR relation |
| Bundling (3 vectors) | 768 | 925 ms | Majority-of-three circuit |
| Hamming-style similarity | 511 | 498 ms | Experimental; source comments describe a simplified approximation, not a complete bit-level popcount proof |
| CfC temporal (64 neurons × 100 steps) | 7,360 | 954 ms | State-update circuit using witness-supplied activation values |
| CfC + sigmoid | 13,760 | 680 ms | Experimental activation/circuit approximation |
| Encrypted FL aggregate (16 participants) | 1,024 | 91 ms | Prototype proof for aggregate computation; not a complete secure-FL protocol |

Times and counts are historical internal measurements from the benchmark harness, not performance guarantees. Re-run the standalone command on the target commit and hardware before publishing comparisons.

### Cognitive architecture and internal indicators

| Capability | Reported observation | Scope |
|---|---|---|
| Cognitive-loop frequency | Approximately 31 Hz (234 Hz raw) | Internal measurement on one reference machine |
| Butlin indicator probes | 4/12 rows retained after construct-validity review; one has limited internal causal-wiring support | Not evidence of consciousness |
| Moral reasoning (Hendrycks Ethics) | 56.2% overall; virtue 76.5% | Internal held-out re-run dated 2026-07-15; an earlier higher result was retracted |
| HDC dimension | 16,384D binary and continuous representations | Architecture/configuration detail |
| Browser kernel | WASM target exists | Output size depends on build profile and features |
| Workspace | Multi-tier Rust workspace | Inventory varies by commit and feature selection |

## For Developers

**Core crates:**

| Crate | What | Tests |
|-------|------|-------|
| `symthaea` | Main cognitive loop (4-phase: perceive → evolve → measure → act) | See crate-local tests |
| `symthaea-core` | HDC vectors, CfC neurons, consciousness math | See crate-local tests |
| `mycelix-zkp-core` standalone subset | Shared types and feature-gated Dilithium support; full STARK backends are not vendored here ([source](stubs/mycelix-zkp-core/)) | See crate-local tests |
| `hdc-zkp-bench` | Standalone Binius benchmark experiments (XOR, bundling, CfC, FL, Hamming-style) ([source](crates/hdc-zkp-bench/)) | Separate package |
| `symthaea-zkproof` | Cycle computation and commitment structures; not a complete end-to-end proof backend ([source](crates/core/symthaea-zkproof/)) | See crate-local tests |
| `symthaea-spore` | Browser WASM kernel | See crate-local tests |
| `symthaea-broca` | Language generation and CfC-HDC experiments | See crate-local tests |

**Feature flags:** Feature availability varies by crate and is evolving. The root manifest's defaults are intentionally minimal. Key flags: `reasoning_engine`, `identity`, `neural-bridge`, `ssm_language`. See [FEATURE_FLAGS.md](docs/FEATURE_FLAGS.md).

**Contributing:** See [CONTRIBUTING.md](CONTRIBUTING.md). Start with [docs/START_HERE.md](docs/START_HERE.md).

## For Researchers

### Papers

| Paper | Venue | Status |
|-------|-------|--------|
| [**Binary-Field STARKs for HDC**](papers/theory-foundations/binius-hdc/) | IEEE S&P 2027 | Submission-ready (v5) |
| [CfC Temporal Proofs](papers/theory-foundations/cfc-zkp/) | NeurIPS workshop | 3-page draft |
| [Triple-Stack FL](papers/evaluation/triple-stack-fl/) | PoPETs | 3-page draft |
| HAI: Hyperdimensional Active Inference | PLoS Comp Bio | Manuscript complete |
| Genesis: Ethical Frameworks | AI & Ethics | Manuscript complete |
| Psych-Bench | — | 98 benchmarks, 20 domains |

### Reproducibility

```bash
# Run the Binius benchmark package included in this public repository
cargo run --manifest-path crates/hdc-zkp-bench/Cargo.toml --release

# Run the internal consciousness psych-bench
cargo run --features psych_bench --example run_all_benchmarks
```

The paper-level [`reproduce.sh`](papers/theory-foundations/binius-hdc/reproduce.sh) is a historical multi-backend script. It currently refers to `crates/mycelix-zkp-core` and `mycelix-health/crates/health-zkp` sibling paths that are not present in this standalone public repository. It therefore does **not** reproduce the complete historical table from a clean standalone clone. The direct manifest command above is the supported way to run the public Binius benchmark package.
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

`psych-bench` is an internal benchmark suite. For the honest per-module status of math and science capabilities (Lie theory, Langlands, thermodynamics, persistent homology, frontier physics, etc.), see [`MODULE_WIRING_STATUS.md`](docs/MODULE_WIRING_STATUS.md). External-benchmark results (miniF2F, PutnamBench, ARC-AGI-2) are planned for Phase 1.

### Key Research Claims

- **Binius**: a measured prover/verifier implementation exists for the HDC XOR-binding benchmark.
- **Prime-field comparison**: the standalone benchmark's comparison numbers are analytical estimates, not a measured Winterfell run.
- **Other circuit experiments**: bundling, Hamming-style, CfC, and encrypted aggregation circuits are prototype experiments with the limits documented in their source.
- **Cycle integrity**: current code provides commitment structures; the helper does not generate a populated STARK proof.
- **Consciousness indicators and moral reasoning**: internal exploratory measurements only, with the limitations and retracted results recorded above.
## Architecture

Symthaea's active cognitive loop is a Rust application/workspace with modular HDC representations, temporal dynamics, memory, perception, reasoning, and action interfaces. The ZKP-related work is a set of separate experimental components, not a single integrated proof layer.

```text
                 ┌─────────────────────────────┐
                 │   Symthaea cognitive loop   │
                 │ perceive / evolve / measure│
                 │            / act            │
                 └──────────────┬──────────────┘
                                │
              ┌─────────────────┼──────────────────┐
              ▼                 ▼                  ▼
       HDC + CfC/LTC      Binius experiments   Integrity records
        core/runtime       hdc-zkp-bench       symthaea-zkproof
              │                 │                  │
              └─────────────────┴──────────────────┘
                                │
                  Mycelix / app integration surfaces
                  (independently feature-gated and
                   maturity-qualified by component)
```

See the current [architecture documentation](docs/architecture/) and [module wiring status](docs/MODULE_WIRING_STATUS.md).
## Regulated-domain applicability

The current public repository does not provide a verified HIPAA health-attestation pipeline, an eIDAS-certified selective-disclosure credential system, or a Digital Product Passport implementation. The standalone `mycelix-zkp-core` subset omits private STARK backends. The GIS proof module explicitly uses simulated field operations for educational purposes and is not production cryptography.

No code in this repository constitutes HIPAA, GDPR, 42 CFR Part 2, eIDAS 2.0, EU DPP, medical-device, or other regulatory certification. Any regulated deployment would require separate cryptographic review, threat modelling, interoperability testing, legal assessment, and independent audits.

## License

AGPL-3.0-or-later. Commercial licensing available — contact tristan.stoltz@evolvingresonantcocreationism.com.

## Citation

```bibtex
@misc{stoltz2026biniushdc,
  title={Binary-Field STARKs for Hyperdimensional Computing},
  author={Stoltz, Tristan},
  year={2026},
  note={Available at github.com/Luminous-Dynamics/symthaea/papers/theory-foundations/binius-hdc}
}
```
