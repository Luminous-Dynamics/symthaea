# Symthaea

**Experimental cognitive architecture for testing verifiable state transitions, representation mechanisms, and consciousness-relevant computational indicators, with multiple proof-system integrations. The bundled Binius64 HDC harness demonstrates computational integrity only; it does not provide zero-knowledge input privacy.**

[![CI](https://github.com/Luminous-Dynamics/symthaea/actions/workflows/ci.yml/badge.svg)](https://github.com/Luminous-Dynamics/symthaea/actions)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Symthaea combines hyperdimensional computing (HDC), closed-form liquid neural dynamics (CfC), consciousness-relevant measurement (IIT-inspired Φ, Butlin-indicator probes), and several proof-system integrations. **It does not claim to prove consciousness, and nothing below is a regulatory compliance certification.** The vendored `vendor/binius64/ARCHITECTURE.md` describes the standard `binius-prover`/`binius-verifier` API present in this checkout as non-zero-knowledge; this is a snapshot-specific limitation, not a claim that all current Binius64 releases lack ZK. Current upstream has separate `ZKProver`/`ZKVerifier` wrappers ([prover source](https://github.com/binius-zk/binius64/blob/main/crates/prover/src/zk_config.rs), [verifier source](https://github.com/binius-zk/binius64/blob/main/crates/verifier/src/zk_config.rs)), but those modules are absent from this vendored tree and are not used in the public HDC harness. The bundled harness therefore makes no confidentiality claim for witness inputs. The public benchmark's Winterfell comparison is an analytic estimate, not a measured same-scale proof.

[**Live Demo**](https://symthaea.luminousdynamics.io) | [**Paper**](papers/theory-foundations/binius-hdc/) | [**Benchmarks**](papers/theory-foundations/binius-hdc/reproduce.sh) | [**Honest Status**](docs/HONEST_STATUS.md)

---

## Evidence Table

| Claim | Status | Evidence | Limitations |
|-------|--------|----------|--------------|
| HDC XOR binding (16Kbit) | Binius64 computational-integrity proof is generated and verified by the public harness | `crates/hdc-zkp-bench/src/main.rs`; vendored `vendor/binius64/ARCHITECTURE.md` labels this backend non-zero-knowledge | Witness-input confidentiality is not guaranteed; the public artifact also does not establish a measured cross-backend ratio |
| Health-value range proof (~34ms historical claim) | Not reproduced by the current public standalone benchmark | Historical paper/readme claim only; `crates/mycelix-zkp-core/` is not a path in this standalone repository | No fresh proof execution or independent review in this branch; not a HIPAA compliance certification |
| Selective-disclosure age≥18 proof | Described as an eIDAS-shaped research demo; not independently reproduced in this audit | Research paper/demo description only | Not an eIDAS-certified wallet, credential, issuer, or trust service |
| Butlin consciousness-indicator probes | Structural implementation, partially audited | Internal construct-validity re-check found only 4/12 probe rows survive; of those, one (`AE-2`) has real empirical support to date, and that result is explicitly scoped as "causally-supported internal wiring evidence," not functional capacity and not evidence of consciousness | **Not evidence the system is conscious.** Most of the original 12-row matrix did not hold up to its own re-audit |
| Moral-reasoning benchmark (Hendrycks Ethics) | Internal benchmark, re-run and corrected | Canonical 2026-07-15 re-run on held-out test CSVs: 56.2% overall; only the `virtue` category (76.5%) is meaningfully above chance | An earlier 91.1%/94.5%/84.7% figure was **retracted as leakage-inflated** — if you see those numbers anywhere else in this repo's history, they are superseded |
| Cognitive loop runs at ~31 Hz measured | Reproduced internally | Loop telemetry | Measured on one reference machine; not a real-time guarantee |

None of this is a claim of general intelligence, sentience, or legal/medical/financial compliance. Where a component is a real, working, reproducible cryptographic or engineering result, we say so and show the command. Where a result was wrong, we say that too, in place, rather than deleting the history.

## What the public proof harness demonstrates

- **Packed-word XOR integrity:** a Binius64 circuit generates and verifies a proof that the public output equals the XOR of witness input words. The backend is not zero-knowledge; this does not establish input confidentiality or a complete encryption protocol.
- **Hamming distance:** the correction branch now constrains a proper population count of XOR words and sums them into a public distance. Rust tests are authored but pending branch build/CI.
- **Strict bitwise majority:** the correction branch uses bit-sliced counters and a per-lane threshold comparison; ties resolve to zero. Tests are authored but not yet executed on this branch.
- **Temporal recurrence:** current demos prove a packed-bit recurrence with witness-supplied gating words, not numerical fixed-point CfC dynamics.

The public standalone benchmark does **not** currently provide a zero-knowledge Binius64 proof, nor reproduce the cited ~34ms health-attestation pipeline, an end-to-end age credential, a correct encrypted numeric-gradient aggregate, or a measured same-scale Winterfell XOR proof. Those application-level claims must remain unverified until their source paths, semantics, and measurements are independently reproducible.

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

# Run the public Binius proof-circuit tests
cargo test --manifest-path crates/hdc-zkp-bench/Cargo.toml

# Run the public Binius benchmark suite
cargo run --release --manifest-path crates/hdc-zkp-bench/Cargo.toml

# Try the Spore kernel in browser
open https://symthaea.luminousdynamics.io
```

## Key Numbers

### Publicly reproducible Binius evidence

The public harness generates and verifies a Binius proof for packed-word XOR. XOR is linear in this representation and contributes no nonlinear AND/MUL constraints. Equality checks still define the public statement, but this benchmark's AND/MUL statistic is not a count of all circuit structure.

An earlier internal run recorded approximately 430 ms proving time, 10.4 ms verification time, and a 75 KB proof for the 16,384-bit XOR demo. These are historical reference values, not fresh measurements from this correction branch. An older 256-count used different constraint accounting and must not be read as the current nonlinear AND/MUL gate count.

The public code computes the Winterfell comparator with an analytic model and does not construct or run a Winterfell prover. **No measured same-scale speedup or constraint-reduction ratio is claimed by this public artifact.**

The previous Hamming result is superseded: its circuit summed/truncated XOR-word values rather than population counts. The previous 8- and 16-input aggregation results are superseded: those paths reduced inputs with AND rather than majority. The previous federated-learning claim is withdrawn because the circuit did not constrain OTP encryption or numeric gradient aggregation. The temporal proof demo is a bitwise recurrence with witness-supplied gates, not a proof of numerical fixed-point CfC dynamics.

### Cognitive Architecture

| Capability | Metric | Status |
|------------|--------|--------|
| Cognitive loop frequency | 31 Hz measured (234 Hz raw) | Reproduced internally |
| Butlin indicator probes | 4/12 rows survive construct-validity re-check; 1 has empirical support so far, scope-limited | Not evidence of consciousness — see Evidence table |
| Moral reasoning (Hendrycks Ethics) | 56.2% overall (virtue: 76.5%) | Corrected 2026-07-15; supersedes an earlier retracted 91.1% figure |
| HDC dimension | 16,384D binary + continuous | — |
| WASM kernel | 324 KB (runs in browser) | — |
| Workspace | 87 crates, 21K+ tests | — |

## For Developers

**Core crates:**

| Crate | What | Tests |
|-------|------|-------|
| `symthaea` | Main cognitive loop (4-phase: perceive → evolve → measure → act) | 7K+ |
| `symthaea-core` | HDC vectors, CfC neurons, consciousness math | 3K+ |
| `mycelix-zkp-core` | ZKP infrastructure: backends, circuits, Dilithium5, domain tags | 65+ |
| `hdc-zkp-bench` | Binius benchmark circuits (XOR, bundling, CfC, FL, Hamming) | Benchmarks |
| `symthaea-zkproof` | Computation attestation (RISC0) + cycle-integrity proof demo; not proof of consciousness | 23+ |
| `symthaea-spore` | Browser WASM kernel (324 KB) | 183 |
| `symthaea-broca` | Language generation (CfC-HDC thought-to-text) | 229+ |

**Feature flags:** 98 flags, default = minimal. Key flags: `reasoning_engine`, `identity`, `neural-bridge`, `ssm_language`. See [FEATURE_FLAGS.md](docs/FEATURE_FLAGS.md).

**Contributing:** See [CONTRIBUTING.md](CONTRIBUTING.md). Start with [docs/START_HERE.md](docs/START_HERE.md).

## For Researchers

### Papers

| Paper | Venue | Status |
|-------|-------|--------|
| [**Binary-Field STARKs for HDC**](papers/theory-foundations/binius-hdc/) | Intended venue: IEEE S&P 2027 | Audit revision required; cross-backend and application claims are being re-scoped |
| [CfC Temporal Proofs](papers/cfc-zkp/) | NeurIPS workshop | 3-page draft |
| [Triple-Stack FL](papers/triple-stack-fl/) | PoPETs | 3-page draft |
| HAI: Hyperdimensional Active Inference | PLoS Comp Bio | Manuscript complete |
| Genesis: Ethical Frameworks | AI & Ethics | Manuscript complete |
| Psych-Bench | — | 98 benchmarks, 20 domains |

### Reproducibility

```bash
# Reproduce all ZKP benchmark results
bash papers/theory-foundations/binius-hdc/reproduce.sh

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

- **Binary-field alignment:** XOR is linear in the Binius representation; a measured same-scale Winterfell speedup or constraint-reduction ratio is not established by the current public harness.
- **Opaque-word majority:** corrected circuits encode strict bitwise majority, but do not implement or prove one-time-pad encryption or numeric federated-gradient aggregation.
- **WASM verifier size and in-zome deployment:** historically reported, not independently verified by this audit branch.
- **4/12 Butlin indicator probes** survive an internal construct-validity re-check; only 1 has empirical support to date, explicitly scoped as internal causal-wiring evidence, not consciousness evidence
- **56.2% moral accuracy** on Hendrycks Ethics (canonical re-run; an earlier 91.1% figure was retracted as leakage-inflated)

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
                    │        PROPOSED PROOF LAYER              │
                    │  Binius integrity + future ZK backend    │
                    │  + RISC0 (complex) + Dilithium5 (PQ)  │
                    └──────────┬──────────────────┬────────┘
                               │                  │
                    ┌──────────▼──────┐  ┌────────▼────────┐
                    │ Mycelix (DHT)   │  │  Applications    │
                    │ 9 hApps on      │  │  Health, Identity│
                    │ Holochain       │  │  Finance, DeSci  │
                    └─────────────────┘  └─────────────────┘
```

Architecture diagrams are conceptual; they are not evidence that every pictured integration is currently deployed. See [`docs/architecture/`](docs/architecture/) for design material.

## Cryptographic Primitives Relevant to Regulated Domains

**None of the following is a compliance certification.** These rows describe proposed application patterns, not end-to-end capabilities established by the current public proof harness. No accredited assessment, legal review, or regulatory audit has been performed.

| Regulation | What the primitive demonstrates | What it is NOT |
|------------|----------------------------------|-----------------|
| **HIPAA** (US) | Intended application: prove a range predicate without revealing the value; not reproduced by the current public standalone benchmark | No HIPAA-compliant system or covered-entity certification is established |
| **eIDAS 2.0** (EU) | Intended application: selective-disclosure credential; not independently reproduced by the current public standalone source | No certified wallet, issuer, or trust-service provider is established |
| **EU DPP** (EU) | Proposed application: prove a compliance predicate without exposing full supply-chain data | No Digital Product Passport implementation or conformity assessment is established |

If you are evaluating this for an actual regulated deployment, treat everything here as a research prototype and get your own legal/compliance review.

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
