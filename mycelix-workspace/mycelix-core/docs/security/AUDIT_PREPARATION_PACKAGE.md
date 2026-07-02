# Security Audit Preparation Package

**Mycelix Protocol**
**Version**: 1.0
**Prepared For**: External Security Auditors
**Last Updated**: 2026-01-18

---

## Executive Summary

This document provides auditors with a comprehensive overview of the Mycelix protocol, its architecture, and the specific components requiring security review. The audit scope covers smart contracts, zero-knowledge circuits, consensus mechanisms, and Holochain zomes.

**Estimated Audit Duration**: 6-8 weeks
**Recommended Budget**: $150,000 - $300,000

---

## Table of Contents

1. [Protocol Overview](#1-protocol-overview)
2. [Audit Scope](#2-audit-scope)
3. [Architecture](#3-architecture)
4. [Critical Components](#4-critical-components)
5. [Key Files Reference](#5-key-files-reference)
6. [Known Issues and Limitations](#6-known-issues-and-limitations)
7. [Security Assumptions](#7-security-assumptions)
8. [Test Coverage](#8-test-coverage)
9. [Build and Test Instructions](#9-build-and-test-instructions)
10. [Contact Information](#10-contact-information)

---

## 1. Protocol Overview

### 1.1 What is Mycelix?

Mycelix is a decentralized federated learning (FL) protocol that enables privacy-preserving machine learning across a network of participants. The protocol combines:

- **Holochain**: Agent-centric distributed data storage
- **Ethereum**: Settlement layer for payments and anchoring
- **Zero-Knowledge Proofs**: Verification without revealing data
- **Byzantine Fault Tolerance**: 45% fault tolerance via MATL trust layer

### 1.2 Core Innovations

| Innovation | Description | Security Relevance |
|------------|-------------|-------------------|
| **MATL Trust Layer** | Multi-factor trust scoring (PoGQ, TCDM, Entropy) | Sybil resistance, Byzantine detection |
| **K-Vector ZKP** | 8-dimensional trust proofs via STARK | Privacy of trust metrics |
| **Feldman DKG** | Distributed key generation for threshold signing | Key security, ceremony integrity |
| **RB-BFT Consensus** | Reputation-weighted Byzantine consensus | 45% tolerance claim |

### 1.3 Token Economics

- **Token**: MYC (utility token)
- **Uses**: Compute credits, governance, staking, rewards
- **Distribution**: Network rewards, no pre-sale to US persons

---

## 2. Audit Scope

### 2.1 In-Scope Components

| Priority | Component | Language | LOC | Risk Level |
|----------|-----------|----------|-----|------------|
| **P0** | Smart Contracts | Solidity | 3,135 | CRITICAL |
| **P0** | ZK Circuits (kvector-zkp) | Rust | 1,031 | CRITICAL |
| **P1** | Consensus (rb-bft-consensus) | Rust | 2,184 | HIGH |
| **P1** | DKG (feldman-dkg) | Rust | 2,169 | HIGH |
| **P1** | Trust Layer (matl-bridge) | Rust | 1,863 | HIGH |
| **P2** | FL Aggregator | Rust | 22,396 | MEDIUM |
| **P2** | Holochain Zomes | Rust | 8,922 | MEDIUM |

### 2.2 Out-of-Scope

- Frontend applications
- Documentation
- Test fixtures
- CI/CD infrastructure
- Third-party dependencies (audit separately if concerned)

### 2.3 Audit Objectives

1. **Smart Contract Security**
   - Reentrancy vulnerabilities
   - Access control bypass
   - Integer overflow/underflow
   - Economic exploits (front-running, MEV)
   - Upgrade mechanism safety

2. **Cryptographic Security**
   - ZK circuit soundness
   - Key generation security
   - Signature verification
   - Random number generation

3. **Protocol Security**
   - Byzantine fault tolerance claims
   - Sybil attack resistance
   - Economic incentive alignment
   - Game-theoretic attacks

4. **Implementation Security**
   - Memory safety (Rust)
   - Panic conditions
   - Resource exhaustion
   - Input validation

---

## 3. Architecture

### 3.1 System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Mycelix Protocol                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐   │
│  │ Participant │   │ Participant │   │   Participant       │   │
│  │   Node      │   │   Node      │   │   Node              │   │
│  └──────┬──────┘   └──────┬──────┘   └──────────┬──────────┘   │
│         │                 │                      │              │
│         └─────────────────┼──────────────────────┘              │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Holochain DHT Layer                     │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │ FL Zome      │  │ PoGQ Zome    │  │ Bridge Zome  │   │   │
│  │  │ (coordinator)│  │ (validation) │  │ (cross-chain)│   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Core Libraries (Rust)                   │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────────────┐   │   │
│  │  │ kvector-zkp│ │ feldman-dkg│ │ rb-bft-consensus   │   │   │
│  │  │ (ZK proofs)│ │ (key gen)  │ │ (Byzantine fault)  │   │   │
│  │  └────────────┘ └────────────┘ └────────────────────┘   │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────────────┐   │   │
│  │  │matl-bridge │ │fl-aggregator│ │ differential-privacy│   │  │
│  │  │ (trust)    │ │ (ML ops)   │ │ (privacy)          │   │   │
│  │  └────────────┘ └────────────┘ └────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Ethereum (Settlement)                   │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐     │   │
│  │  │MycelixRegistry│ │PaymentRouter │ │ReputationAnchor│    │   │
│  │  │ (registry)   │ │ (payments)   │ │ (proofs)     │     │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Trust Flow

```
Local Training → Gradient Encryption → DHT Storage
       ↓
PoGQ Calculation → Trust Score → ZK Proof Generation
       ↓
Aggregation (Byzantine-resistant) → Model Update
       ↓
Ethereum Anchor → Payment Distribution
```

### 3.3 Security Boundaries

| Boundary | Trust Level | Validation |
|----------|-------------|------------|
| User Input | Untrusted | Full validation |
| Peer Messages | Semi-trusted | Signature + PoGQ |
| Smart Contracts | Trusted (audited) | On-chain verification |
| Local Computation | Trusted | N/A |

---

## 4. Critical Components

### 4.1 Smart Contracts (CRITICAL)

#### MycelixRegistry.sol (621 LOC)
**Path**: `contracts/src/MycelixRegistry.sol`

**Purpose**: Central registry for nodes, models, and protocol parameters.

**Critical Functions**:
```solidity
function registerNode(bytes32 nodeId, bytes calldata pubKey) external;
function updateNodeStatus(bytes32 nodeId, NodeStatus status) external;
function setProtocolParameter(bytes32 key, uint256 value) external onlyGovernance;
```

**Security Concerns**:
- Access control for admin functions
- Node registration spam prevention
- Parameter bounds validation

#### PaymentRouter.sol (644 LOC)
**Path**: `contracts/src/PaymentRouter.sol`

**Purpose**: Routes payments to FL contributors based on reputation.

**Critical Functions**:
```solidity
function routePayment(bytes32 roundId, address[] calldata recipients, uint256[] calldata amounts) external;
function createEscrow(bytes32 escrowId, uint256 amount) external;
function releaseEscrow(bytes32 escrowId, address recipient) external;
```

**Security Concerns**:
- Reentrancy on payment distribution
- Escrow release authorization
- Payment splitting correctness

#### ReputationAnchor.sol (409 LOC)
**Path**: `contracts/src/ReputationAnchor.sol`

**Purpose**: Anchors trust score Merkle roots on-chain.

**Critical Functions**:
```solidity
function storeReputationRoot(bytes32 root, uint256 epoch) external;
function verifyAndRecord(bytes32 leaf, bytes32[] calldata proof, uint256 score) external;
```

**Security Concerns**:
- Merkle proof verification
- Epoch management
- Root update authorization

#### ContributionRegistry.sol (743 LOC)
**Path**: `contracts/src/ContributionRegistry.sol`

**Purpose**: Records FL contributions and computes rewards.

**Security Concerns**:
- Batch operation gas limits
- Contribution deduplication
- Reward calculation overflow

#### ModelRegistry.sol (718 LOC)
**Path**: `contracts/src/ModelRegistry.sol`

**Purpose**: Registers and manages ML model versions.

**Security Concerns**:
- Model hash collision
- Version upgrade path
- Access control for deprecation

### 4.2 Zero-Knowledge Circuits (CRITICAL)

#### kvector-zkp (1,031 LOC)
**Path**: `libs/kvector-zkp/src/`

**Purpose**: Proves trust score validity without revealing components.

**Key Files**:
- `air.rs` - Algebraic Intermediate Representation
- `prover.rs` - STARK proof generation
- `verifier.rs` - Proof verification
- `constraints.rs` - Circuit constraints

**Cryptographic Properties Required**:
- Soundness: Invalid proofs rejected
- Completeness: Valid proofs accepted
- Zero-knowledge: No information leakage

**Security Concerns**:
- Constraint soundness
- Field element overflow
- Random oracle security
- Fiat-Shamir transformation

### 4.3 Consensus (HIGH)

#### rb-bft-consensus (2,184 LOC)
**Path**: `libs/rb-bft-consensus/src/`

**Purpose**: Reputation-weighted Byzantine fault tolerant consensus.

**Key Files**:
- `consensus.rs` - Main consensus logic
- `reputation.rs` - Reputation weighting
- `voting.rs` - Vote aggregation
- `slashing.rs` - Misbehavior penalties

**Security Claims**:
- Tolerates up to 45% Byzantine nodes (by reputation weight)
- Reputation decay prevents accumulation attacks
- Slashing deters misbehavior

**Security Concerns**:
- Long-range attacks
- Nothing-at-stake
- Reputation manipulation
- Vote equivocation detection

### 4.4 Distributed Key Generation (HIGH)

#### feldman-dkg (2,169 LOC)
**Path**: `libs/feldman-dkg/src/`

**Purpose**: Generates threshold signing keys without trusted dealer.

**Key Files**:
- `ceremony.rs` - DKG ceremony orchestration
- `share.rs` - Secret share generation
- `verification.rs` - Feldman commitment verification
- `reconstruction.rs` - Key reconstruction

**Security Properties**:
- No single party learns the secret
- t-of-n threshold for reconstruction
- Verifiable secret sharing

**Security Concerns**:
- Malicious dealer detection
- Share verification completeness
- Ceremony abortion handling
- Timeout attacks

### 4.5 Trust Layer (HIGH)

#### matl-bridge (1,863 LOC)
**Path**: `libs/matl-bridge/src/`

**Purpose**: Multi-factor Adaptive Trust Layer.

**Components**:
- `pogq.rs` - Proof of Gradient Quality
- `tcdm.rs` - Trust-weighted Cosine Distance Median
- `entropy.rs` - Behavioral diversity measurement

**Trust Formula**:
```
T = 0.4 × PoGQ + 0.3 × TCDM + 0.3 × Entropy
```

**Security Concerns**:
- Gaming individual metrics
- Collusion attacks
- Trust manipulation over time

---

## 5. Key Files Reference

### 5.1 Smart Contracts

| File | LOC | Description | Priority |
|------|-----|-------------|----------|
| `contracts/src/MycelixRegistry.sol` | 621 | Node/model registry | P0 |
| `contracts/src/PaymentRouter.sol` | 644 | Payment routing | P0 |
| `contracts/src/ReputationAnchor.sol` | 409 | Trust anchoring | P0 |
| `contracts/src/ContributionRegistry.sol` | 743 | Contribution tracking | P1 |
| `contracts/src/ModelRegistry.sol` | 718 | Model management | P1 |

### 5.2 Cryptographic Libraries

| File | LOC | Description | Priority |
|------|-----|-------------|----------|
| `libs/kvector-zkp/src/air.rs` | 280 | ZK circuit definition | P0 |
| `libs/kvector-zkp/src/prover.rs` | 250 | Proof generation | P0 |
| `libs/kvector-zkp/src/verifier.rs` | 180 | Proof verification | P0 |
| `libs/feldman-dkg/src/ceremony.rs` | 450 | DKG ceremony | P0 |
| `libs/feldman-dkg/src/share.rs` | 380 | Secret sharing | P0 |

### 5.3 Consensus and Trust

| File | LOC | Description | Priority |
|------|-----|-------------|----------|
| `libs/rb-bft-consensus/src/consensus.rs` | 520 | BFT consensus | P1 |
| `libs/rb-bft-consensus/src/slashing.rs` | 340 | Slashing logic | P1 |
| `libs/matl-bridge/src/pogq.rs` | 411 | PoGQ calculation | P1 |
| `libs/matl-bridge/src/tcdm.rs` | 353 | TCDM calculation | P1 |

### 5.4 Holochain Zomes

| File | LOC | Description | Priority |
|------|-----|-------------|----------|
| `zomes/federated_learning/coordinator/src/*.rs` | 5,103 | FL coordination | P2 |
| `zomes/pogq_validation/coordinator/src/*.rs` | 519 | PoGQ validation | P2 |
| `zomes/bridge/coordinator/src/*.rs` | 1,012 | Cross-chain bridge | P2 |

---

## 6. Known Issues and Limitations

### 6.1 Documented Limitations

| Issue | Component | Severity | Notes |
|-------|-----------|----------|-------|
| GPU backends incomplete | fl-aggregator | Low | CUDA/Metal/OpenCL stubbed |
| Testnet only | All contracts | Info | Not deployed to mainnet |
| Winterfell v0.10 | kvector-zkp | Info | Consider upgrade to v0.13 |

### 6.2 Areas of Concern (Self-Assessment)

| Area | Concern | Mitigation Needed |
|------|---------|-------------------|
| PaymentRouter reentrancy | CEI pattern used, verify completeness | Audit confirmation |
| ZK scaling edge cases | Large values may overflow | Bounds checking review |
| DKG ceremony timeouts | Potential liveness issues | Timeout handling review |
| Trust score manipulation | Long-term gaming possible | Economic analysis |

### 6.3 Previous Findings (Internal Review)

All findings from internal security review have been remediated:
- H-01: ZK Proof Scaling - Fixed with bounds checking
- H-02: Abstention Attack - Fixed with participation tracking
- M-01: PaymentRouter Reentrancy - Fixed with CEI + mutex
- M-02: ContributionRegistry Pagination - Fixed
- M-03: Slashing Time Bounds - Fixed
- M-04: DKG Ceremony Timeouts - Fixed
- M-05: Degenerate Input Detection - Fixed

---

## 7. Security Assumptions

### 7.1 Trust Assumptions

| Assumption | Justification | If Violated |
|------------|---------------|-------------|
| Ethereum is secure | Largest PoS network | Settlement failures |
| Holochain DHT is available | Agent-centric design | Liveness issues |
| <45% Byzantine by reputation | Economic incentives | Consensus failure |
| Winterfell STARK is sound | Peer-reviewed library | ZK soundness loss |

### 7.2 Cryptographic Assumptions

| Assumption | Primitive | Strength |
|------------|-----------|----------|
| Discrete log hard | secp256k1, ed25519 | 128-bit |
| SHA-3 collision resistant | SHA3-256 | 128-bit |
| BLAKE3 preimage resistant | BLAKE3 | 128-bit |
| Random oracle model | Fiat-Shamir | Theoretical |

### 7.3 Economic Assumptions

| Assumption | Mechanism | Attack Cost |
|------------|-----------|-------------|
| Sybil expensive | Staking requirement | Stake × validators |
| Slashing deters | 0.5% stake per violation | Reputation loss |
| Honest majority (by stake) | PoS-like incentives | >50% stake |

---

## 8. Test Coverage

### 8.1 Smart Contracts

```bash
cd contracts
forge coverage
```

| Contract | Line Coverage | Branch Coverage | Function Coverage |
|----------|---------------|-----------------|-------------------|
| MycelixRegistry | 92% | 85% | 100% |
| PaymentRouter | 95% | 88% | 100% |
| ReputationAnchor | 90% | 82% | 100% |
| ContributionRegistry | 88% | 80% | 100% |
| ModelRegistry | 91% | 84% | 100% |

**Total Tests**: 204 (202 unit + 2 fuzz)

### 8.2 Rust Libraries

```bash
cargo tarpaulin --workspace --out Html
```

| Library | Line Coverage | Notes |
|---------|---------------|-------|
| kvector-zkp | 85% | ZK circuits |
| feldman-dkg | 82% | DKG ceremony |
| rb-bft-consensus | 78% | Consensus |
| matl-bridge | 80% | Trust layer |
| fl-aggregator | 75% | Large codebase |

**Total Tests**: 336+ across workspace

### 8.3 Integration Tests

| Test Suite | Location | Tests |
|------------|----------|-------|
| DKG Ceremony | `tests/integration/src/dkg_ceremony.rs` | 12 |
| Trust Lifecycle | `tests/integration/src/trust_lifecycle.rs` | 15 |
| Consensus | `tests/integration/src/consensus_integration.rs` | 20 |
| Byzantine | `tests/integration/src/ecosystem_stress.rs` | 18 |
| E2E Pipeline | `tests/integration/src/e2e_pipeline.rs` | 10 |

---

## 9. Build and Test Instructions

### 9.1 Prerequisites

```bash
# Rust (stable)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable

# Foundry (Solidity)
curl -L https://foundry.paradigm.xyz | bash
foundryup

# Nix (optional, for reproducible builds)
sh <(curl -L https://nixos.org/nix/install) --daemon
```

### 9.2 Build

```bash
# Clone repository
git clone https://github.com/Luminous-Dynamics/Mycelix-Core.git
cd Mycelix-Core

# Build Rust libraries
cargo build --workspace --release

# Build smart contracts
cd contracts
forge build
```

### 9.3 Run Tests

```bash
# Rust tests
cargo test --workspace

# Smart contract tests
cd contracts
forge test -vvv

# Integration tests
cd tests/integration
cargo test --release

# Specific library tests
cargo test -p kvector-zkp
cargo test -p feldman-dkg
cargo test -p rb-bft-consensus
```

### 9.4 Static Analysis

```bash
# Rust
cargo clippy --workspace -- -D warnings
cargo audit

# Solidity
cd contracts
forge fmt --check
slither .
```

---

## 10. Contact Information

### 10.1 Primary Contacts

| Role | Name | Email | Response Time |
|------|------|-------|---------------|
| Security Lead | [REDACTED] | security@luminous-dynamics.io | <4 hours |
| Technical Lead | [REDACTED] | tech@luminous-dynamics.io | <8 hours |
| Protocol Lead | [REDACTED] | protocol@luminous-dynamics.io | <24 hours |

### 10.2 Communication Channels

- **Secure Email**: Use PGP (keys at keybase.io/mycelix)
- **Signal**: Available for sensitive discussions
- **GitHub Issues**: For non-sensitive findings
- **Weekly Sync**: Thursdays 15:00 UTC

### 10.3 Vulnerability Disclosure

Please follow responsible disclosure:
1. Report to security@luminous-dynamics.io
2. Allow 90 days for remediation
3. Coordinate public disclosure

---

## Appendix A: Threat Model

### A.1 Attacker Profiles

| Profile | Capabilities | Goals |
|---------|-------------|-------|
| Malicious Node | Single node, moderate stake | Steal rewards, manipulate model |
| Colluding Group | <45% stake, coordination | Break consensus, double-spend |
| External Attacker | No stake, network access | DoS, exploit vulnerabilities |
| Insider | Code access, no runtime | Backdoor, information leak |

### A.2 Attack Surfaces

| Surface | Threats | Mitigations |
|---------|---------|-------------|
| Smart Contracts | Reentrancy, access control | Audits, formal verification |
| P2P Network | Eclipse, DoS | Peer diversity, rate limiting |
| ZK Proofs | Soundness attacks | Library audits, testing |
| Consensus | Byzantine attacks | 45% threshold, slashing |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| MATL | Multi-factor Adaptive Trust Layer |
| PoGQ | Proof of Gradient Quality |
| TCDM | Trust-weighted Cosine Distance Median |
| DKG | Distributed Key Generation |
| RB-BFT | Reputation-based Byzantine Fault Tolerance |
| STARK | Scalable Transparent ARgument of Knowledge |
| FL | Federated Learning |

---

*End of Audit Preparation Package*
