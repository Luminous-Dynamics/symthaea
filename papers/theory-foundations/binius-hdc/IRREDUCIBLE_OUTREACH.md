# Contribution Proposal: Domain-Specific Binius Circuits for Healthcare, Identity, and Federated Learning

## To: Irreducible Team (Binius64)

### Who We Are

Luminous Dynamics — building Mycelix, a decentralized platform on Holochain with 11 clusters (health, governance, identity, finance, etc.) and Symthaea, a consciousness research system. We've been using Binius64 extensively and have results to share.

### What We've Built (All Open Source)

**9 domain-specific Binius circuits** — the only non-infrastructure Binius circuits we're aware of:

| Circuit | AND Constraints | Prove Time | Use Case |
|---------|----------------|------------|----------|
| HDC XOR binding (16Kbit) | 256 | 430 ms | Privacy-preserving HDC |
| Majority bundling (3 vecs) | 768 | 925 ms | Collective intelligence |
| Majority bundling (8 vecs) | 2,048 | 635 ms | Federated aggregation |
| Hamming similarity | 511 | 498 ms | Verifiable search |
| CfC temporal (64N×100T) | 7,360 | 954 ms | Neural network verification |
| CfC with sigmoid | 13,760 | 680 ms | Full neural computation |
| Encrypted XOR binding | 256 | 153 ms | Zero-overhead FHE |
| FL 3 participants | 192 | 58 ms | Federated learning |
| FL 16 participants | 1,024 | 91 ms | Federated learning |

### Key Results

1. **256× constraint reduction** vs Winterfell prime-field STARK (same-scale, both measured at 16,384 bits)
2. **55× faster proving** (Binius 430ms vs Winterfell 23.8s)
3. **Zero encryption overhead** — OTP encryption adds 0 AND constraints (triple-stack FL)
4. **694 KB WASM** — Binius verifier compiles to wasm32-unknown-unknown
5. **34.1 ms E2E pipeline** — Winterfell health attestation with Dilithium5 PQ signatures

### The Paper

"Binary-Field STARKs for Hyperdimensional Computing" — 10-page paper with all data measured. Available on our public repo. Targeting IEEE S&P 2027.

### What We'd Like to Discuss

1. **Circuit contributions**: We'd like to upstream our domain-specific circuits as Binius64 examples. This gives the Binius ecosystem real-world application circuits beyond infrastructure (zkVM, state proofs).

2. **WASM deployment**: We've confirmed binius-verifier compiles to wasm32 (694KB). Are there known issues with running verification in WASM runtimes like wasmer (used by Holochain)?

3. **Optimization guidance**: Our circuits use `bxor`, `band`, `iadd_32`, and `assert_eq`. Are there patterns we're missing that could reduce the structural AND overhead (currently 256 AND for 256 assert_eq calls)?

4. **Binius for healthcare/identity**: The EU eIDAS 2.0 mandate (end 2026) and HIPAA 2026 create demand for privacy-preserving credential verification. Binius's binary-field efficiency makes it ideal for categorical health attestations (boolean compliance checks are XOR-native).

### Repository

Public: `github.com/Luminous-Dynamics/symthaea`
- Benchmarks: `crates/hdc-zkp-bench/src/`
- ZKP core: `crates/mycelix-zkp-core/src/`
- Paper: `papers/binius-hdc/`

### Contact

Tristan Stoltz — tristan.stoltz@evolvingresonantcocreationism.com
Luminous Dynamics — github.com/Luminous-Dynamics
