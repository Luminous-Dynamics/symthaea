# Mycelix Protocol: System Architecture

> **Roadmap-Aware Architecture**: This document describes both our current production implementation (Phase 1) and future evolution (Phase 2+), providing a clear path from pragmatic deployment to advanced privacy-preserving infrastructure.

## Architectural Design Principles

This architecture is guided by core principles that shape every technical decision:

1. **Sovereignty First**: The system empowers users with ultimate ownership and control over their identity and data. This is reflected in our choice of Holochain's agent-centric model.

2. **Defense in Depth**: Security never relies on a single mechanism. Every critical component is secured by multiple, redundant layers—economic incentives, cryptographic proofs, and operational monitoring.

3. **Pragmatic Decentralization**: Use the right tool for the right job. We employ agent-centric P2P systems (Holochain) for scalability and sovereignty, and blockchains (Layer-2) for finality and settlement.

4. **Evolvability**: The architecture is designed to be upgraded over time. We build on proven foundations today (Merkle Proofs, Byzantine-resistant aggregation) while maintaining a clear path to advanced cryptography tomorrow (Zero-Knowledge Proofs, ZK-Rollups).

## Overview

The Mycelix Framework employs a hybrid distributed ledger architecture that combines the scalability and user sovereignty of Holochain with the finality and interoperability of Layer-2 blockchain technology.

## Architectural Layers

```
┌─────────────────────────────────────────────────────────────┐
│                   Industry Adapters Layer                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐         │
│  │  Federated   │ │  Healthcare  │ │   Energy     │  ...    │
│  │  Learning    │ │    Data      │ │    Grid      │         │
│  └──────────────┘ └──────────────┘ └──────────────┘         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Meta-Core Services                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐         │
│  │  Reputation  │ │   Identity   │ │   Currency   │         │
│  │   System     │ │   Management │ │   Exchange   │         │
│  └──────────────┘ └──────────────┘ └──────────────┘         │
└─────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│   Holochain (P2P Layer)  │  │  Layer-2 (Settlement)    │
│                          │  │                          │
│  - Agent-centric DHT     │  │  - EVM Smart Contracts   │
│  - Source Chains         │  │  - Global Settlement     │
│  - Local Validation      │  │  - DeFi Integration      │
└──────────────────────────┘  └──────────────────────────┘
              │                           │
              └─────────────┬─────────────┘
                            ▼
              ┌──────────────────────────┐
              │  Cross-Chain Bridge      │
              │                          │
              │  Phase 1: Merkle + Val   │
              │  Phase 2: ZK-Bridge      │
              └──────────────────────────┘
```

## Core Components

### 1. Industry Adapter Layer

**Purpose**: Encapsulate industry-specific logic while leveraging universal infrastructure.

**Key Features:**
- Self-contained Holochain DNAs
- Standardized interfaces to Meta-Core
- Independent scaling per adapter
- Parallel development by domain experts

**Example Adapters:**
- **Federated Learning**: Byzantine-resistant ML training with Proof of Gradient Quality (PoGQ)
- **Healthcare Data**: Privacy-preserving medical research with verifiable eligibility
- **Energy Grid**: Decentralized grid optimization with agent-centric coordination

### 2. Meta-Core Services

**Reputation System:**
```python
class ReputationScore:
    """
    Dynamic reputation with time decay and appeals.

    Components:
    - Base Score: Accumulated from contributions
    - Decay Function: e^(-λt) time-based reduction
    - Appeal Mechanism: Jury-based dispute resolution
    """

    def calculate_current_score(self, last_update: int) -> float:
        time_elapsed = current_timestamp() - last_update
        decay = math.exp(-DECAY_RATE * time_elapsed)
        return self.base_score * decay
```

**Phase 2 Enhancement**: Integration of Zero-Knowledge Proofs for **Verifiable Eligibility** will allow users to generate proofs about their reputation (e.g., "My score in this domain is above 4.5") without revealing their exact score or identity, providing state-of-the-art **Provable Privacy**.

**Identity Management:**
- Self-Sovereign Identity (SSI) model
- Verifiable Credentials (VCs) in private wallets
- Selective disclosure of information
- Integration with Holochain agent model

**Currency Exchange:**
- Automated Market Maker (AMM) or Order Book
- Cross-chain bridging between Holochain and L2
- Protocol fees to DAO treasury
- MEV mitigation through private mempool

### 3. Holochain P2P Layer

**Agent-Centric Architecture:**

```rust
// Each agent maintains their own source chain
pub struct SourceChain {
    entries: Vec<Entry>,
    headers: Vec<Header>,
    signatures: Vec<Signature>,
}

// Shared validation rules in DNA
pub fn validate_entry(entry: Entry, context: ValidationContext) -> ExternResult<ValidateCallbackResult> {
    // Industry-specific validation logic
    // All peers enforce the same rules
    Ok(ValidateCallbackResult::Valid)
}
```

**DHT Architecture:**
- Data sharded across peers based on content hash
- Each peer stores subset of network data
- Gossip protocol for data synchronization
- Intrinsic data integrity through hashing

**Scalability Model:**
- Linear scaling with number of participants
- No global consensus bottleneck
- Each adapter has own DHT (sharded by industry)
- Resource contribution scales with users

### 4. Layer-2 Settlement

**Smart Contract Architecture:**

```solidity
// Core bridge contract
contract MycelixCurrencyExchange {
    // Merkle root of Holochain escrows
    bytes32 public currentMerkleRoot;

    // Validator registry
    IBridgeValidatorRegistry public validatorRegistry;

    // Execute swap with Merkle proof
    function executeSwap(
        address recipient,
        uint256 amount,
        bytes32[] calldata merkleProof,
        bytes[] calldata validatorSignatures
    ) external {
        // 1. Verify validator signatures
        require(
            validatorRegistry.verifySignatures(validatorSignatures),
            "Invalid signatures"
        );

        // 2. Verify Merkle proof
        require(
            verifyMerkleProof(merkleProof, recipient, amount),
            "Invalid Merkle proof"
        );

        // 3. Execute swap
        _executeSwapLogic(recipient, amount);
    }
}
```

**Why Layer-2:**
- 100x lower gas fees than Ethereum L1
- ~1000x higher throughput
- Inherits L1 security through fraud/validity proofs
- EVM compatibility for easy integration

**L2 Selection Criteria:**

| Solution | Security | Finality | Cost | EVM Compat | Status |
|----------|----------|----------|------|------------|--------|
| Optimistic Rollup | L1-inherited | 7 days | Low | High | ⚠️ Slow withdrawal |
| ZK-Rollup | L1-inherited | Fast | Medium | Growing | 🔮 Phase 2 upgrade |
| **Polygon PoS** | **Separate** | **Fast** | **Very Low** | **100%** | **✅ Phase 1** |

### 5. Cross-Chain Bridge

The Mycelix Bridge is designed with a **phased, evolving architecture** to balance immediate feasibility with long-term security and scalability.

#### Phase 1: Verifiable Bridge (Current Implementation)

The initial production version of the bridge uses a defense-in-depth model based on a staked Validator Network and on-chain Merkle Proof verification. This provides a high degree of security and is buildable with today's battle-tested technologies.

**Architecture:**

```
Holochain DNA                    Bridge Validators              L2 Smart Contract
     │                                   │                            │
     │  1. User escrows tokens          │                            │
     ├─────────────────────────────────►│                            │
     │                                   │  2. Observe & validate    │
     │                                   ├───────────────────────────►│
     │                                   │  3. Sign message          │
     │                                   │                            │
     │                                   │  4. Post Merkle root      │
     │                                   │  + signatures             │
     │                                   ├───────────────────────────►│
     │                                   │                            │
     │                                   │  5. User submits proof    │
     ◄───────────────────────────────────┼────────────────────────────┤
     │  6. Swap executed                │                            │
```

**Security Layers:**

**Layer 1 - Economic Security:**
- Validators stake capital (e.g., $100K USDC)
- Slashing for provable misbehavior
- Fraud proofs submitted by watchdogs
- Bounty rewards for successful fraud detection

**Layer 2 - Cryptographic Security:**
- On-chain Merkle proof verification
- Smart contract validates state directly
- Validators are untrusted messengers
- Mathematical guarantee, not trust

**Layer 3 - Operational Resilience:**
- Heartbeat checks (iAmAlive() calls)
- Automatic jailing of inactive validators
- State reconciliation on startup
- Byzantine fault tolerance (BFT)

#### Phase 2: ZK-Bridge (Future Upgrade)

As outlined in our research on Zero-Knowledge Proofs, the long-term roadmap is to upgrade the bridge to a full **ZK-Rollup architecture**.

**Mechanism**: A decentralized network of "Provers" will generate a single ZK-SNARK or ZK-STARK that attests to the validity of all Holochain state transitions since the last on-chain update.

**Benefits:**

1. **Trustlessness**: Eliminates reliance on the economic honesty of the validator set, replacing it with pure mathematical proof.

2. **Scalability**: Compresses thousands of off-chain transactions into a single, cheap on-chain verification (~300K gas regardless of transaction count).

3. **Privacy**: Can obscure transactional details, preserving user privacy while maintaining verifiability.

**Implementation Path:**
- Phase 1 deployment validates architecture and economics
- Parallel R&D on ZK-proof generation (Circom circuits, witness generation)
- Gradual migration: Phase 1 validators → Phase 2 provers
- Backward compatibility maintained during transition

This upgrade represents the ultimate evolution of the bridge to a state of **maximal security and efficiency**.

## Data Flow Examples

### Example 1: Federated Learning Contribution

**Layer-by-Layer Execution:**

```
User Layer (Training Node):
  └─► Trains model on local data
  └─► Computes gradient update
  └─► Submits to local Holochain agent

P2P Layer (Holochain DNA):
  └─► Creates entry in source chain
  └─► Publishes gradient hash to DHT
  └─► Triggers VRF-based validator selection

Validation Layer (Validator Nodes):
  └─► Download gradient from DHT
  └─► Validate using private test dataset
  └─► Compute Proof of Gradient Quality (PoGQ) score
  └─► Submit validation result to DHT

Aggregation Layer (Hierarchical):
  └─► Cluster aggregators combine local gradients
  └─► Global aggregation produces model update
  └─► Byzantine-resistant algorithm (Krum, Multi-Krum) filters attacks

Meta-Core Layer (Reputation System):
  └─► High-quality contributions → increased reputation
  └─► Low-quality contributions → decreased reputation
  └─► Reputation score updated in DHT

Settlement Layer (Optional - L2 Smart Contract):
  └─► Issue Mycelix Credits on Holochain (immediate)
  └─► Option to bridge to L2 for liquidity (via bridge)
  └─► Trade on currency exchange for other tokens
```

### Example 2: Cross-Chain Currency Swap (Phase 1)

**Layer-by-Layer Execution:**

```
User Layer (Holochain):
  └─► Calls escrow() in Mycelix Credits DNA
  └─► Locks 1000 MYCX tokens
  └─► Entry added to local source chain

P2P Layer (Holochain):
  └─► Escrow entry published to DHT
  └─► Validators observe via WebSocket subscription

Bridge Layer (Validators):
  └─► Observe escrow event
  └─► Construct Merkle tree of all pending escrows
  └─► Reach consensus on Merkle root hash
  └─► Each validator signs message with root
  └─► Post aggregate signature to L2 smart contract

Settlement Layer (L2 Smart Contract):
  └─► Validates validator signatures (quorum check)
  └─► Stores new Merkle root on-chain
  └─► Emits event confirming update

User Layer (L2):
  └─► Constructs Merkle proof (client-side, using DHT data)
  └─► Calls executeSwap() with proof + validator signatures

Settlement Layer (L2 Smart Contract):
  └─► Verifies signatures (economic security)
  └─► Verifies Merkle proof (cryptographic security)
  └─► Executes AMM swap logic
  └─► Transfers L2 tokens to user

Result:
  └─► User now has L2 tokens (can trade on DeFi, transfer, etc.)
  └─► Holochain escrow remains locked (security guarantee)
```

## Scalability Analysis

### Holochain Scalability

**Horizontal Scaling:**
- Each new user adds compute/storage resources
- DHT sharding distributes data load
- No global state to synchronize
- Industry adapters scale independently

**Performance:**
- Thousands of transactions per second per adapter
- Sub-second confirmation times
- Minimal hardware requirements

### Bridge Scalability

**Phase 1 (Merkle Proofs - Current):**
- Batch multiple escrows into single Merkle tree
- Proof size: O(log n) where n = number of escrows
- Gas cost: ~100K gas per proof verification
- **Practical limit**: ~1000 swaps/day before costs become prohibitive

**Phase 2 (ZK-Rollup - Future):**
- Single ZK-SNARK proves entire Holochain state transition
- Verify thousands of transactions with one proof (~300K gas)
- 100x cost reduction vs Phase 1
- **Practical limit**: ~100,000 swaps/day
- Roadmap: Phase 11+ implementation

### Overall System

**Bottleneck Analysis:**
1. ✅ **Holochain**: Scales horizontally, no bottleneck
2. ⚠️ **Bridge (Phase 1)**: Batching helps, but still costly at scale
3. ✅ **Bridge (Phase 2)**: ZK-proofs eliminate bottleneck
4. ⚠️ **L2 Contract**: Standard blockchain limits (~30 TPS for complex operations)
5. ✅ **Solution**: Most activity stays on Holochain, only settlement crosses bridge

**Result:** System scales to millions of users with Phase 1 architecture, tens of millions with Phase 2 ZK-Bridge.

## Security Model

### Threat Model

**Assumptions:**
- Up to f < n/3 Byzantine validators (Phase 1)
- Network partitions possible
- Smart contract bugs possible
- Users may be malicious

**Guarantees:**
- Byzantine-resistant aggregation (Federated Learning adapter)
- Economic + cryptographic bridge security (Phase 1)
- Mathematical guarantee of correctness (Phase 2 ZK-proofs)
- No single point of failure
- Transparent, auditable operations

### Attack Mitigation

| Attack Vector | Phase 1 Mitigation | Phase 2 Enhancement |
|---------------|-------------------|-------------------|
| Validator collusion | Economic slashing + Merkle proofs | Eliminated (no validators needed) |
| Sybil attacks | Reputation-gated onboarding | ZK-proofs of reputation without revealing identity |
| Data poisoning | PoGQ validation + reputation | ZK-verified eligibility criteria |
| Front-running (MEV) | Private mempool submission | Inherent in ZK-proof privacy |
| Smart contract bugs | Audits + emergency pause | Same + ZK-verifier contract audits |
| Network partitions | State reconciliation | Same (DHT resilience) |

## Governance Integration

**DAO Structure:**
```
┌─────────────────────────────────────┐
│           DAO Governance            │
│                                     │
│  Token Voting: Economic stake       │
│  Reputation Voting: Contribution    │
│  Council: Cross-domain coordination │
└─────────────────────────────────────┘
          │              │
    ┌─────┴──────┐  ┌───┴────────┐
    │ Meta-Core  │  │  Industry  │
    │ Upgrades   │  │  Adapter   │
    │            │  │  Proposals │
    └────────────┘  └────────────┘
```

**Proposal Types:**
1. Meta-core parameter changes
2. Treasury allocation
3. New adapter whitelisting
4. Emergency actions
5. Bridge Phase 2 migration approval

## Technology Stack

**Holochain:**
- Rust-based core
- WASM for zome code
- WebSocket connections
- Conductor for app hosting

**Layer-2:**
- Solidity 0.8+
- Hardhat/Foundry for development
- OpenZeppelin libraries
- Chainlink VRF for randomness

**Bridge Validators (Phase 1):**
- Python/Rust for service
- WebSocket to Holochain
- Web3.py for Ethereum
- PostgreSQL for state

**ZK-Provers (Phase 2):**
- Circom for circuit design
- snarkjs/rapidsnark for proving
- WASM for client-side proof generation
- Bulletproofs for range proofs

**Frontend:**
- TypeScript/React
- Holochain client
- wagmi/viem for Ethereum
- RainbowKit for wallets

## Deployment Architecture

### Development
```
Local Holochain Conductor
  └─► Mycelix Credits DNA
Local Hardhat Node
  └─► Bridge Contracts
Mock Validator Service
```

### Testnet
```
Holochain Testnet
  └─► Multiple DNAs
Polygon Mumbai
  └─► Deployed Contracts
Decentralized Validators (3-5)
PostgreSQL (per validator)
```

### Production
```
Holochain Mainnet
  └─► Production DNAs
Polygon PoS Mainnet (Phase 1)
  └─► Audited Contracts
Validator Network (20-50)
  └─► Geographic distribution
  └─► Independent operators
DAO Treasury Multi-sig
```

## Monitoring & Observability

**Key Metrics:**
- Holochain DHT health
- Validator uptime (Phase 1)
- Bridge latency
- Slashing events (Phase 1)
- ZK-proof generation time (Phase 2)
- DAO proposal activity
- Reputation distribution

**Alerting:**
- Validator downtime (Phase 1)
- Failed Merkle proofs (Phase 1)
- Failed ZK-proof verification (Phase 2)
- Unusual gas costs
- Suspicious reputation spikes
- Treasury threshold events

---

## Glossary of Terms

- **Agent-Centric**: Architecture where each user maintains their own data chain, rather than contributing to a global ledger.
- **Byzantine Fault Tolerance (BFT)**: System's ability to continue operating correctly even when some nodes behave maliciously.
- **DHT (Distributed Hash Table)**: Decentralized storage where data is sharded across network peers based on content hash.
- **Layer-2 (L2)**: Blockchain scaling solution that processes transactions off the main chain (Layer-1) while inheriting its security.
- **Merkle Proof**: Cryptographic proof that a piece of data exists in a Merkle tree, enabling efficient verification.
- **PoGQ (Proof of Gradient Quality)**: Novel mechanism for validating machine learning gradients against a private test dataset.
- **VRF (Verifiable Random Function)**: Cryptographic function that generates provably random outputs.
- **Zero-Knowledge Proof (ZKP)**: Cryptographic proof that a statement is true without revealing any information beyond the statement's validity.
- **Zome**: A module in a Holochain DNA that contains the application logic and validation rules.
- **zk-SNARK**: Zero-Knowledge Succinct Non-Interactive Argument of Knowledge; a specific type of ZKP with small proof size and fast verification.

---

## Related Documentation

- [Hybrid DLT Design](./HYBRID_DLT.md)
- [Federated Learning Adapter](./FL_ADAPTER.md)
- [Currency Exchange Bridge](./CURRENCY_EXCHANGE_BRIDGE.md)
- [Phase 2 ZK-Bridge Research](../../research/ZKP_HEALTHCARE_PROTOTYPE.md)
- [Security Model](../07-security/OVERVIEW.md)
- [Governance Structure](../08-governance/DAO_STRUCTURE.md)

---

*"The Trust Layer is the innovation. Storage is just configuration. Merkle Proofs get us to production today. Zero-Knowledge Proofs get us to perfection tomorrow."*

**Current Status**: Phase 1 architecture complete and ready for deployment. Phase 2 R&D active.
