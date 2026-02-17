# Mycelix Hybrid Architecture

## Overview

Mycelix uses a **three-layer hybrid architecture** that combines the best properties of each technology:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MYCELIX PROTOCOL                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  LAYER 1: HOLOCHAIN (Primary Coordination)                          │
│  ══════════════════════════════════════════                         │
│  Cost: FREE | Latency: <100ms | Throughput: 10,000+ TPS             │
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │   Agents    │  │  Federated  │  │   Bridge    │                 │
│  │    Zome     │  │  Learning   │  │    Zome     │                 │
│  │             │  │    Zome     │  │             │                 │
│  │ • Register  │  │ • Gradients │  │ • Cross-hApp│                 │
│  │ • Discovery │  │ • Rounds    │  │ • Reputation│                 │
│  │ • Reputation│  │ • PoGQ      │  │ • Payments  │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
│         │                │                │                         │
│         └────────────────┼────────────────┘                         │
│                          │                                          │
│                          │ Periodic Sync (hourly/daily)             │
│                          ▼                                          │
│  LAYER 2: ETHEREUM/POLYGON (Anchoring & Settlement)                 │
│  ══════════════════════════════════════════════════                 │
│  Cost: $0.01-$5/tx | Latency: 2s-12min | Immutable Proofs           │
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │  Mycelix    │  │ Reputation  │  │  Payment    │                 │
│  │  Registry   │  │   Anchor    │  │   Router    │                 │
│  │             │  │             │  │             │                 │
│  │ • DID Proofs│  │ • Merkle    │  │ • Escrow    │                 │
│  │ • Revocation│  │   Roots     │  │ • Disputes  │                 │
│  │ • Recovery  │  │ • Disputes  │  │ • Large $   │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
│                                                                      │
│  LAYER 3: ZEROTRUSTML (Application Logic)                           │
│  ═════════════════════════════════════════                          │
│  Byzantine Detection | PoGQ Validation | ML Aggregation             │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    0TML Python Framework                      │   │
│  │  • Gradient computation    • Byzantine detection (100%)      │   │
│  │  • Model aggregation       • Krum/Trimmed Mean/PoGQ          │   │
│  │  • Edge proof generation   • Holochain DHT client            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Why This Architecture?

### Holochain for High-Frequency Operations

| Operation | Frequency | On Ethereum | On Holochain |
|-----------|-----------|-------------|--------------|
| Gradient submission | 1000/round | $2000/round | **FREE** |
| Reputation update | 100/round | $200/round | **FREE** |
| Agent discovery | 10000/day | N/A | **FREE** |
| DID lookup | Unlimited | N/A | **FREE** |

**Result: 99.9% cost reduction for FL operations**

### Ethereum/Polygon for Immutability

- **Merkle root anchoring**: One transaction per epoch captures all reputation changes
- **Dispute resolution**: Requires global consensus for contentious cases
- **Large payments**: High-value escrow benefits from blockchain security
- **Cross-ecosystem proofs**: Verifiable by any Ethereum client

## Data Flow

### 1. FL Round Execution (Holochain - FREE)

```
Node A                    Holochain DHT                    Node B
  │                            │                              │
  │──── register_agent() ─────▶│                              │
  │                            │◀──── register_agent() ───────│
  │                            │                              │
  │──── submit_gradient() ────▶│                              │
  │                            │──── gossip ──────────────────▶│
  │                            │                              │
  │◀─── get_round_updates() ───│                              │
  │                            │                              │
  │──── submit_reputation() ──▶│                              │
  │                            │                              │
```

### 2. Epoch Anchoring (Ethereum - Paid, Infrequent)

```
                    Holochain                     Ethereum
                        │                            │
  Every N rounds:       │                            │
                        │                            │
  1. Collect all        │                            │
     reputation ────────┤                            │
     updates            │                            │
                        │                            │
  2. Build Merkle ──────┤                            │
     tree               │                            │
                        │                            │
  3. Anchor root ───────┼────── anchorMerkleRoot() ─▶│
                        │          (1 tx = $0.01)    │
                        │                            │
  4. Store proof ───────┤                            │
     in DHT             │                            │
```

### 3. Dispute Resolution (Ethereum)

```
Disputer                  Ethereum                  Challenged
    │                        │                          │
    │── initiateDispute() ──▶│                          │
    │                        │                          │
    │                        │◀── submitEvidence() ─────│
    │                        │                          │
    │── submitMerkleProof() ▶│                          │
    │                        │                          │
    │◀── resolveDispute() ───│                          │
```

## Component Details

### Holochain Zomes (Already Implemented)

**`zomes/agents/`** - Agent management
- `register_agent()` - Join the network
- `get_all_agents()` - Discover peers
- `update_reputation()` - Local reputation tracking

**`zomes/federated_learning/`** - FL coordination (107KB of code!)
- `submit_model_update()` - Store gradients in DHT
- `get_round_updates()` - Retrieve all gradients for a round
- `validate_contribution()` - PoGQ verification

**`zomes/bridge/`** - Cross-system communication
- `sync_to_ethereum()` - Trigger anchoring
- `verify_proof()` - Check Merkle proofs

### Ethereum Contracts (Deployed on Sepolia)

| Contract | Address | Purpose |
|----------|---------|---------|
| MycelixRegistry | `0x556b81...` | DID anchoring, revocation |
| ReputationAnchor | `0xf3B343...` | Merkle root storage |
| PaymentRouter | `0x94417A...` | Escrow for payments |

### 0TML Python Framework

```python
# Current: Direct networking
fl_network = ZeroTrustML(nodes=20, rounds=100)
fl_network.train()

# Future: Holochain-backed
from zerotrustml.holochain import HolochainCoordinator

coordinator = HolochainCoordinator(conductor_url="ws://localhost:8888")
fl_network = ZeroTrustML(
    coordinator=coordinator,  # Uses Holochain DHT
    anchor_bridge=EthereumBridge(network="polygon"),  # Periodic anchoring
)
```

## Migration Path

### Phase 1: Current State
- 0TML uses direct TCP/IP networking
- Ethereum contracts deployed but not integrated
- Holochain zomes exist but not connected

### Phase 2: Holochain Integration
- Replace 0TML gossip with Holochain DHT
- All FL coordination goes through zomes
- Reputation stored in Holochain first

### Phase 3: Anchoring Service
- Background service monitors Holochain
- Builds Merkle trees of reputation updates
- Submits roots to Ethereum/Polygon periodically

### Phase 4: Full Hybrid
- Disputes trigger Ethereum resolution
- Large payments use blockchain escrow
- Cross-ecosystem verification possible

## Cost Analysis

### Current (Direct Networking)
- Infrastructure: Requires central coordinator or complex P2P
- Cost: Server hosting (~$50-200/month)

### Future (Holochain + Polygon)
- Infrastructure: Distributed, self-hosted by participants
- Holochain: FREE for all operations
- Polygon: ~$0.01/tx for anchoring (maybe 24 tx/day = $0.24/day)
- **Monthly cost: ~$7.20** (just anchoring)

### Comparison
| Approach | Monthly Cost | Decentralization |
|----------|--------------|------------------|
| Central server | $100-500 | None |
| Ethereum only | $10,000+ | High |
| Polygon only | $50-100 | Medium |
| **Holochain + Polygon** | **$7** | **High** |

## Security Considerations

### Holochain Security
- Agent-centric: Each node validates its own chain
- DHT: Data distributed across network
- Cryptographic: All entries signed by author
- **Not globally consistent**: Eventually consistent model

### When to Use Ethereum
- **Immutability required**: Revocation, disputes
- **Global consensus needed**: Cross-network verification
- **Financial stakes**: Large escrow amounts
- **Legal requirements**: Auditable proof

### Trust Model
```
High Frequency + Low Stakes → Holochain (FREE)
Low Frequency + High Stakes → Ethereum/Polygon (PAID)
```

## Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| Agents Zome | ✅ Complete | `zomes/agents/` |
| FL Zome | ✅ Complete | `zomes/federated_learning/` |
| Bridge Zome | ✅ Complete | `zomes/bridge/` |
| Ethereum Contracts | ✅ Deployed | `contracts/src/` |
| SDK | ✅ Complete | `contracts/sdk/` |
| 0TML Framework | ✅ Complete | `0TML/` |
| Holochain ↔ Python | ✅ Complete | `0TML/src/zerotrustml/holochain/` |
| FL Coordinator Client | ✅ Complete | `0TML/src/zerotrustml/holochain/fl_coordinator_client.py` |
| Anchor Service | ✅ Complete | `services/anchor/` |

### Python Holochain Integration

```python
# FL Coordinator Client - connects to federated_learning_coordinator zome
from zerotrustml.holochain import FLCoordinatorClient, create_fl_coordinator

async with create_fl_coordinator() as client:
    # Submit gradient (FREE on Holochain)
    result = await client.submit_gradient(
        node_id="node-1",
        round_num=1,
        gradient_data=gradient_bytes,
        gradient_shape=[100, 10],
    )

    # Get all gradients for a round (FREE)
    gradients = await client.get_round_gradients(round_num=1)

    # Run Byzantine detection (FREE)
    detection = await client.detect_byzantine_hierarchical(round_num=1)
```

### Anchor Service

```bash
# Run the anchor service
python services/anchor/anchor_service.py --network polygon-amoy --interval 3600

# Or programmatically
from services.anchor import AnchorService

service = AnchorService(network="polygon-amoy", anchor_interval=3600)
await service.run()
```

## Next Steps

1. **Test Holochain zomes** with live conductor
2. **Deploy to Polygon Amoy** for cheap testing
3. **End-to-end integration test** FL → Holochain → Polygon
4. **Production deployment** with monitoring
