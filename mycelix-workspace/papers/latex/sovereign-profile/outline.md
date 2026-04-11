# 8D Sovereign Profile: Multi-Dimensional Civic Identity for Post-State Governance

## Target Venue
IEEE S&P, USENIX Security, or ACM CCS (security/identity track)

## Status
OUTLINE (April 2026)

## Abstract (Draft)

We present the 8-Dimensional Sovereign Profile, a governance credential system that replaces single-axis identity verification (Proof of Humanity, Worldcoin iris scan, Gitcoin Passport) with eight physically-grounded dimensions measured from independent source clusters. Each dimension---epistemic integrity, thermodynamic yield, network resilience, economic velocity, civic participation, stewardship, semantic resonance, and domain competence---is measured directly from verifiable on-chain activity rather than self-reports or financial proxies. Constitutional invariants (no single dimension exceeding 50% weight, mandatory exponential decay with bounded lambda, AI agents capped at Steward tier) prevent dimensional capture and permanent oligarchy. Privacy is architecturally enforced through zero-knowledge STARK proofs (~1ms prove, ~0.3ms verify, 7.5KB proof) that demonstrate threshold compliance without revealing raw dimensions. Deployed across 141+ Holochain zomes in 16 cluster DNAs with 87 unit tests, the system has been validated through 300-year governance simulations showing zero Guardian vetoes attempted (deterrence effect) and 100% community survival across all seeds.

## Paper Structure

### 1. Introduction
- The single-axis identity problem (PoH, Worldcoin, Gitcoin = one dimension)
- Why multi-dimensional credentials resist Sybil attacks better
- The 8D sovereign profile as a holographic governance geometry

### 2. The Eight Dimensions
- Each dimension with source cluster, measurement method, saturation level
- Why these eight and not four (attack vectors closed by expansion)
- Table: Dimension | Source | Measures | Saturation | Sybil Cost

### 3. Constitutional Envelope
- 16 hardcoded invariants (DHT validation enforcement)
- 7 unamendable core rights
- Decay bounds (lambda_min/max), weight caps, grace periods
- Why lambda=0 is constitutionally impossible

### 4. Privacy Architecture
- ZKP circuit design (Winterfell STARK 0.13.1)
- Commitment scheme: SHA-256(SOVEREIGN:v1:{did}:{score}:{lambda}:{elapsed})
- Proof of threshold without dimension disclosure
- Comparison to selective disclosure credentials (Verifiable Credentials, BBS+)

### 5. HDC Encoding
- 16,384-bit BinaryHV per profile
- Dimension-weighted bit allocation
- Monotonicity guarantee
- Tier derivation via popcount

### 6. Security Analysis
- Sybil cost analysis (8D vs 1D vs 4D)
- Collusion resistance via independent source clusters
- Dimension capture prevention (50% weight cap)
- Time-based attacks (decay prevents credential fossilization)

### 7. Evaluation
- 87 unit tests across decay, HDC, collectors modules
- 300-year governance simulation (5 seeds, 50 agents)
- Migration report: 74 zome coordinators, 0 regressions
- Comparison table vs Worldcoin, Gitcoin Passport, PoH, BrightID

### 8. Related Work
- Proof of Humanity (Kleros)
- Worldcoin (iris biometrics)
- Gitcoin Passport (stamp aggregation)
- BrightID (social graph)
- Verifiable Credentials (W3C)
- Soulbound Tokens (Weyl, Ohlhaver, Buterin 2022)

### 9. Conclusion
- 8D sovereign profiles as a new primitive for post-state governance
- Open-source (AGPL-3.0), published crate on crates.io

## Key Code References
- `crates/sovereign-profile/src/lib.rs` — Core types
- `crates/sovereign-profile/src/decay.rs` — Constitutional decay (28 tests)
- `crates/sovereign-profile/src/hdc.rs` — HDC encoding (17 tests)
- `crates/mycelix-bridge-common/src/constitutional_envelope.rs` — Invariants
- `crates/mycelix-bridge-common/src/sovereign_gate.rs` — gate_civic()
- `crates/mycelix-zkp-core/src/consciousness.rs` — STARK circuits
