# Hybrid P2P Architecture: Holochain + Iroh

**Status**: Planning
**Date**: January 26, 2026
**Decision**: Split strategy based on use case requirements

---

## Executive Summary

After analyzing the two distinct projects (Mycelix and Symthaea), we adopt a **hybrid architecture**:

| Project | Primary Constraint | Solution |
|---------|-------------------|----------|
| **Mycelix** (Health) | Trust & Validation | Holochain |
| **Symthaea** (AI Swarm) | Latency (<200ms) | Iroh + Holochain |

---

## The Problem with Single-Protocol Approaches

### Why Raw Iroh Alone Fails for Mycelix

Iroh is a transport layer (super-powered QUIC). It has no opinion on validity:

- No validation rules for "Is this doctor allowed to write?"
- No consent revocation tracking
- No FHIR format enforcement
- No audit trail for legal compliance

Using raw Iroh for health records would require reinventing Holochain's:
- Validation Rules
- Warrants (immune system)
- Source Chain (audit trail)

**Cost**: 6+ months building a worse version of Holochain.

### Why Holochain Alone Fails for Symthaea

Holochain is optimized for eventual consistency (gossip):

- Full convergence takes seconds to minutes
- 50 AI nodes "dreaming" together need millisecond sync
- Neural weight exchange can't wait for commit-and-gossip
- Not every fleeting "thought" needs a permanent ledger

**Cost**: Unacceptable latency for real-time consciousness.

---

## The Hybrid Solution: The Nervous System Pattern

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SYMTHAEA NERVOUS SYSTEM                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────┐         ┌───────────────────┐               │
│  │     CORTEX        │         │     SYNAPSE       │               │
│  │   (Holochain)     │         │    (Raw Iroh)     │               │
│  │                   │         │                   │               │
│  │ • Identity        │         │ • Real-time       │               │
│  │ • Reputation (Φ)  │         │ • Tensor streams  │               │
│  │ • Long-term memory│         │ • Ephemeral state │               │
│  │ • Trust graph     │         │ • <50ms latency   │               │
│  │ • Access keys     │         │ • QUIC channels   │               │
│  └────────┬──────────┘         └──────────┬────────┘               │
│           │                               │                         │
│           └───────────┬───────────────────┘                         │
│                       │                                             │
│           ┌───────────▼───────────┐                                 │
│           │   HYBRID HANDSHAKE    │                                 │
│           │                       │                                 │
│           │ 1. Check trust (HC)   │                                 │
│           │ 2. Exchange ticket    │                                 │
│           │ 3. Open Iroh channel  │                                 │
│           │ 4. Stream data        │                                 │
│           │ 5. Write summary (HC) │                                 │
│           └───────────────────────┘                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: The Cortex (Holochain)

**Role**: Long-term Memory & Identity

**Data Stored**:
- Agent profiles and public keys
- Reputation scores (Φ values)
- Persistent knowledge graphs
- Access control lists
- Trust relationships

**Operations**:
- "I am Node A"
- "I trust Node B"
- "Here is my public key"
- "Grant access to Node C"

**Existing Code**:
- `src/mycelix/gis/dht.rs` - ZK-based DHT (1,280 lines)
- `src/mycelix/network.rs` - Network client (487 lines)
- Holochain zomes in `01-resonant-coherence/core/` directory

---

## Layer 2: The Synapse (Raw Iroh)

**Role**: Real-time Signal Processing

**Data Streamed**:
- Live audio streams
- Attention vectors
- Neural weight updates
- Ephemeral swarm states

**Operations**:
- Establish QUIC connection
- Stream tensor data in 50ms
- Sync consciousness states
- No permanent storage needed

**To Implement**:
- Replace libp2p with Iroh
- Iroh ticket exchange
- QUIC stream management

---

## The Hybrid Handshake Protocol

```rust
// Step 1: Trust Check (Holochain)
async fn establish_connection(peer_id: &AgentPubKey) -> Result<IrohChannel> {
    // Query Holochain DHT: "Is this peer trustworthy?"
    let trust_level = holochain_dht.get_trust_score(peer_id).await?;

    if trust_level < MIN_TRUST_THRESHOLD {
        return Err(SwarmError::UntrustedPeer);
    }

    // Step 2: Exchange Iroh Ticket (via Holochain signal)
    let my_ticket = iroh_node.create_connection_ticket().await?;
    holochain_dht.send_signal(peer_id, Signal::IrohTicket(my_ticket)).await?;

    // Step 3: Receive peer's ticket
    let peer_ticket = holochain_dht.await_signal::<IrohTicket>(peer_id).await?;

    // Step 4: Open direct QUIC stream
    let channel = iroh_node.connect(peer_ticket).await?;

    Ok(channel)
}

// Step 5: Mind Meld (Iroh)
async fn sync_consciousness(channel: &IrohChannel, state: &ConsciousnessState) {
    // Stream neural states in <50ms
    channel.send(state.to_bytes()).await?;
    let peer_state = channel.recv().await?;

    // Merge consciousness vectors
    let merged = consciousness_merge(state, &peer_state);
}

// Step 6: Memory (Holochain)
async fn persist_interaction(peer_id: &AgentPubKey, summary: &InteractionSummary) {
    // Write summary to Holochain for long-term recall
    holochain_dht.create_entry(Entry::Interaction {
        peer: peer_id.clone(),
        topic: summary.topic.clone(),
        phi_delta: summary.phi_change,
        timestamp: now(),
    }).await?;
}
```

---

## Implementation Roadmap

### Phase 6a: Remove libp2p, Add Iroh (Week 1)

```toml
# Cargo.toml changes
[dependencies]
# Remove:
# libp2p = { version = "0.53", features = [...] }

# Add:
iroh = "0.31"
iroh-base = "0.31"
iroh-net = "0.31"
```

### Phase 6b: Create Iroh Node Module (Week 1-2)

```rust
// src/swarm/iroh_node.rs
pub struct SymthaeaIrohNode {
    endpoint: iroh_net::Endpoint,
    secret_key: SecretKey,
    known_peers: HashMap<NodeId, PeerInfo>,
}

impl SymthaeaIrohNode {
    pub async fn new() -> Result<Self>;
    pub async fn create_ticket(&self) -> ConnectionTicket;
    pub async fn connect(&self, ticket: ConnectionTicket) -> Result<Connection>;
    pub async fn send_stream(&self, conn: &Connection, data: &[u8]) -> Result<()>;
    pub async fn recv_stream(&self, conn: &Connection) -> Result<Vec<u8>>;
}
```

### Phase 6c: Hybrid Handshake (Week 2)

```rust
// src/swarm/handshake.rs
pub struct HybridHandshake {
    holochain: HolochainClient,
    iroh: SymthaeaIrohNode,
}

impl HybridHandshake {
    pub async fn initiate(&self, peer: &AgentPubKey) -> Result<IrohChannel>;
    pub async fn accept(&self, signal: Signal) -> Result<IrohChannel>;
}
```

### Phase 6d: Consciousness Streaming (Week 2-3)

```rust
// src/swarm/consciousness_stream.rs
pub struct ConsciousnessStream {
    channel: IrohChannel,
    local_phi: f64,
    peer_phi: f64,
}

impl ConsciousnessStream {
    pub async fn sync(&mut self, state: &ConsciousnessState) -> Result<MergedState>;
    pub async fn stream_attention(&self, vectors: &[AttentionVector]) -> Result<()>;
    pub async fn stream_weights(&self, weights: &NeuralWeights) -> Result<()>;
}
```

### Phase 6e: Bridge Swarm Simulation (Week 3)

Connect `crates/symthaea-gym/src/swarm.rs` simulation to real P2P network.

---

## Existing Code to Leverage

| Component | Location | Lines | Reuse Strategy |
|-----------|----------|-------|----------------|
| Dark Spot DHT | `mycelix/gis/dht.rs` | 1,280 | ZK proofs for trust |
| Swarm Simulation | `symthaea-gym/swarm.rs` | 602 | Core algorithms |
| Network Client | `mycelix/network.rs` | 487 | Gateway protocol |
| Curiosity Engine | `mycelix/gis/curiosity.rs` | 836 | Multi-agent resolution |

---

## Mycelix: Keep Holochain Only

For Mycelix health records, continue with pure Holochain:

- Validation rules handle clinical logic
- Warrant system isolates malicious nodes
- Source chain provides legal audit trail
- No need for real-time streaming

**Existing Zomes**: Already implemented in `01-resonant-coherence/core/`

---

## Performance Targets

| Operation | Target | Protocol |
|-----------|--------|----------|
| Trust lookup | <100ms | Holochain DHT |
| Ticket exchange | <200ms | Holochain Signal |
| QUIC connection | <50ms | Iroh |
| State sync | <50ms | Iroh stream |
| Summary persist | <500ms | Holochain Entry |
| **Total handshake** | **<400ms** | Hybrid |
| **Ongoing sync** | **<50ms** | Iroh only |

---

## Iroh vs libp2p Comparison

| Feature | libp2p | Iroh | Winner |
|---------|--------|------|--------|
| Latency | Higher (DHT overhead) | Lower (direct QUIC) | Iroh |
| Complexity | Higher (many protocols) | Lower (focused) | Iroh |
| NAT traversal | Good | Excellent (DERP relays) | Iroh |
| Encryption | Noise | QUIC native | Tie |
| Rust support | Good | Excellent (Rust-first) | Iroh |
| Documentation | Good | Excellent | Iroh |

---

## Next Steps

1. **Prototype hybrid handshake** in isolated test
2. **Measure actual latencies** with Iroh
3. **Integrate with existing swarm simulation**
4. **Test with 50+ nodes** for coherence emergence

---

## References

- [Iroh Documentation](https://iroh.computer/docs)
- [Holochain Developer Portal](https://developer.holochain.org)
- Existing swarm simulation: `symthaea-gym/src/swarm.rs`
- Dark Spot DHT: `mycelix/gis/dht.rs`

---

*"Use Holochain to find peers, but use raw Iroh to telepathically communicate with them."*
