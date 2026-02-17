# Mycelix Protocol: Ethereum Ecosystem Integration

This diagram shows how the Mycelix Protocol integrates with the Ethereum ecosystem to provide decentralized identity, reputation, and governance infrastructure.

```mermaid
graph TB
    subgraph "Ethereum Ecosystem"
        subgraph "Layer 1: Ethereum Mainnet"
            ENS[ENS<br/>Name Resolution]
            ERC20[ERC-20<br/>Governance Token]
        end

        subgraph "Layer 2: Polygon/Optimism"
            DID_REGISTRY[DID Registry<br/>ERC-1056]
            VC_REGISTRY[VC Registry<br/>ERC-725]
            REP_CONTRACT[Reputation Contract<br/>Merkle Anchors]
            DAO_GOV[DAO Governance<br/>Snapshot/Aragon]
        end
    end

    subgraph "Mycelix Protocol"
        subgraph "Holochain Layer"
            USER_AGENT[User Agent<br/>Source Chain]
            DHT[Distributed Hash Table<br/>Gradient Storage]
            VAL_NETWORK[Validator Network<br/>PoGQ Validation]
        end

        subgraph "Cross-Chain Bridge"
            BRIDGE_VAL[Bridge Validators<br/>Staked Nodes]
            MERKLE_GEN[Merkle Proof<br/>Generator]
            STATE_SYNC[State<br/>Synchronizer]
        end

        subgraph "Core Services"
            DID_SERVICE[DID Service<br/>Create/Resolve]
            VC_SERVICE[VC Service<br/>Issue/Verify]
            REP_SERVICE[Reputation Service<br/>PoGQ Scoring]
        end
    end

    subgraph "External Integrations"
        GITCOIN[Gitcoin Passport<br/>Proof of Personhood]
        SNAPSHOT[Snapshot<br/>Off-Chain Voting]
        COMPOUND[Compound/Aave<br/>DeFi Lending]
        ARAGON[Aragon<br/>DAO Framework]
    end

    subgraph "Users & Applications"
        USERS[End Users<br/>Wallet Integration]
        DAOS[DAOs<br/>Governance]
        DEFI[DeFi dApps<br/>Reputation Scoring]
        FL_APPS[FL Applications<br/>Healthcare, Finance]
    end

    %% User interactions
    USERS --> DID_SERVICE
    USERS --> VC_SERVICE
    DAOS --> REP_SERVICE
    DEFI --> REP_SERVICE
    FL_APPS --> VAL_NETWORK

    %% Service to DHT
    DID_SERVICE --> USER_AGENT
    VC_SERVICE --> DHT
    REP_SERVICE --> VAL_NETWORK

    %% DHT to Bridge
    USER_AGENT --> BRIDGE_VAL
    DHT --> MERKLE_GEN
    VAL_NETWORK --> STATE_SYNC

    %% Bridge to L2
    BRIDGE_VAL --> DID_REGISTRY
    MERKLE_GEN --> REP_CONTRACT
    STATE_SYNC --> VC_REGISTRY

    %% L2 to L1
    DID_REGISTRY -.-> ENS
    REP_CONTRACT -.-> ERC20

    %% L2 to DAO
    REP_CONTRACT --> DAO_GOV

    %% External integrations
    DID_REGISTRY --> GITCOIN
    REP_CONTRACT --> SNAPSHOT
    REP_CONTRACT --> COMPOUND
    DAO_GOV --> ARAGON

    %% Styling
    classDef ethereum fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef holochain fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef bridge fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef services fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    classDef external fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef users fill:#fff9c4,stroke:#f9a825,stroke-width:2px

    class ENS,ERC20,DID_REGISTRY,VC_REGISTRY,REP_CONTRACT,DAO_GOV ethereum
    class USER_AGENT,DHT,VAL_NETWORK holochain
    class BRIDGE_VAL,MERKLE_GEN,STATE_SYNC bridge
    class DID_SERVICE,VC_SERVICE,REP_SERVICE services
    class GITCOIN,SNAPSHOT,COMPOUND,ARAGON external
    class USERS,DAOS,DEFI,FL_APPS users
```

## Integration Points

### 1. Identity Layer (W3C DIDs + Ethereum)

```mermaid
sequenceDiagram
    participant User
    participant Mycelix
    participant Holochain
    participant Polygon
    participant ENS

    User->>Mycelix: Create DID
    Mycelix->>Holochain: Generate DID document
    Holochain-->>Mycelix: DID: did:mycelix:abc123
    Mycelix->>Polygon: Register DID on-chain
    Polygon-->>Mycelix: Transaction hash
    Mycelix->>ENS: Link DID to ENS name
    ENS-->>Mycelix: user.mycelix.eth
    Mycelix-->>User: DID + ENS name
```

### 2. Reputation Bridging (Holochain → Ethereum L2)

```mermaid
sequenceDiagram
    participant Validators
    participant Holochain
    participant Bridge
    participant Polygon
    participant DAO

    Validators->>Holochain: Submit PoGQ validations
    Holochain->>Holochain: Aggregate scores
    Holochain->>Bridge: Request state commitment
    Bridge->>Bridge: Generate Merkle proof
    Bridge->>Polygon: Submit proof + reputation root
    Polygon->>Polygon: Verify proof
    Polygon-->>DAO: Updated reputation available
    DAO->>DAO: Use reputation for voting
```

### 3. DeFi Integration (Reputation-Based Lending)

```mermaid
sequenceDiagram
    participant User
    participant Mycelix
    participant Polygon
    participant Compound

    User->>Mycelix: Request reputation score
    Mycelix->>Polygon: Query reputation contract
    Polygon-->>Mycelix: Score: 87.3/100
    User->>Compound: Request loan
    Compound->>Polygon: Check reputation
    Polygon-->>Compound: Verified: 87.3/100
    Compound->>User: Approve loan (lower collateral)
```

## Smart Contract Architecture

### Reputation Contract (Polygon)

```solidity
// Simplified example
contract MycelixReputation {
    struct ReputationAnchor {
        bytes32 merkleRoot;      // Root of reputation Merkle tree
        uint256 timestamp;        // Block timestamp
        address[] validators;     // Bridge validators who signed
        bytes[] signatures;       // Validator signatures
    }

    mapping(address => uint256) public reputationScores;
    ReputationAnchor[] public anchors;

    function submitAnchor(
        bytes32 _merkleRoot,
        bytes[] memory _signatures
    ) external onlyBridge {
        // Verify validator signatures
        // Store anchor
        // Emit event for indexing
    }

    function verifyReputation(
        address _user,
        uint256 _score,
        bytes32[] memory _proof
    ) public view returns (bool) {
        // Verify Merkle proof against latest anchor
    }
}
```

## Gas Cost Analysis

### Phase 1 (Merkle Proofs)

| Operation | Gas Cost | USD Cost (100 gwei) |
|-----------|----------|-------------------|
| Submit reputation anchor | ~80,000 | ~$3-6 |
| Verify reputation (read) | ~30,000 | ~$1-2 |
| Register DID | ~50,000 | ~$2-4 |
| Bridge 1,000 users | ~150,000 | ~$6-12 |

**Optimization**: Batch multiple operations into single transaction

### Phase 2 (ZK-Rollup - Estimated)

| Operation | Gas Cost | USD Cost (100 gwei) |
|-----------|----------|-------------------|
| Submit ZK proof (1,000 tx) | ~300,000 | ~$12-24 |
| Verify single reputation | ~5,000 | ~$0.20-0.40 |
| **Cost per user** | ~300 | **~$0.01-0.02** |

**Improvement**: 100x reduction in cost per user

## Ecosystem Integration Roadmap

### Phase 1 (Current)
- ✅ ERC-1056 DID registry
- ✅ Merkle proof bridge
- ✅ Basic reputation anchoring

### Phase 2 (Q1-Q2 2026)
- Snapshot strategy for reputation-weighted voting
- Gitcoin Passport integration for Sybil resistance
- ENS resolution for DIDs
- Compound/Aave integration for reputation-based lending

### Phase 3 (Q3-Q4 2026)
- ZK-Rollup upgrade for scalability
- Cross-chain bridge (Ethereum ↔ Polygon ↔ Optimism)
- DAO treasury management
- Aragon plugin for reputation governance

## Ethereum Ecosystem Benefits

### For DAOs
- **Sybil Resistance**: Reputation makes fake identities costly
- **Merit-Based Governance**: Weight votes by contribution quality
- **Progressive Decentralization**: Gradually transfer control to contributors

### For DeFi
- **Under-Collateralized Lending**: Use reputation as collateral
- **Risk Assessment**: Better credit scoring for lending protocols
- **Fraud Prevention**: Detect and penalize malicious actors

### For dApps
- **Portable Reputation**: Users carry reputation across applications
- **Reduced Onboarding Friction**: Leverage existing reputation
- **Interoperability**: W3C standards + Ethereum standards

## Security Considerations

### Bridge Security Layers

1. **Economic Security**: Validators stake 100K USDC, slashed for misbehavior
2. **Cryptographic Security**: Merkle proofs verified on-chain
3. **Operational Security**: 7/10 multi-sig for emergency pause
4. **Fraud Proofs**: Anyone can challenge invalid state transitions (7-day window)

### Attack Vectors & Mitigations

| Attack | Mitigation |
|--------|-----------|
| Validator collusion | Slashing + reputation penalties |
| Merkle proof forgery | On-chain verification + fraud proofs |
| Front-running | Commit-reveal scheme for sensitive operations |
| MEV extraction | Flashbots integration + transaction ordering rules |
| Bridge manipulation | Multi-sig control + time locks |

---

**Export Instructions**:
1. View on GitHub (renders automatically)
2. Export to PNG: Use [Mermaid Live Editor](https://mermaid.live/)
3. Export to SVG: Use `mmdc` CLI tool
4. Include in grant application to show Ethereum alignment
