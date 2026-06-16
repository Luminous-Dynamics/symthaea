# Iroh P2P Async Bridge Design

## Problem

The cognitive loop (`CognitiveLoopService`) and Mind (`ContinuousMind`) both run synchronously at 50Hz.
Iroh P2P networking (`IrohNode`) is fully async (QUIC transport via `iroh::Endpoint`).
Social coherence already uses an inbox/outbox pattern for message passing, but no bridge
connects this to the actual network layer.

## Architecture

### Actor-Based Message Passing

Follow the existing `social_inbox`/`social_outbox` pattern in `ContinuousMind`:

```
┌─────────────────────┐         ┌─────────────────────┐
│  ContinuousMind     │         │  IrohBridgeActor     │
│  (sync, 50Hz)       │         │  (async, tokio task)  │
├─────────────────────┤         ├─────────────────────┤
│                     │  mpsc   │                     │
│  social_outbox ─────┼────────→│  send_queue         │
│                     │         │    ↓                 │
│                     │         │  for each peer:      │
│                     │         │    channel.send()    │
│                     │         │                     │
│                     │  mpsc   │                     │
│  social_inbox  ←────┼────────←│  recv_loop          │
│                     │         │    ↑                 │
│                     │         │  channel.recv()      │
│                     │         │    per peer          │
└─────────────────────┘         └─────────────────────┘
```

### Key Components

#### 1. `IrohBridgeActor`

A tokio task that owns the `IrohNode` and manages all async I/O.

```rust
pub struct IrohBridgeActor {
    node: IrohNode,
    /// Receive outbound messages from Mind (sync → async)
    outbound_rx: mpsc::Receiver<SocialMessage>,
    /// Send inbound messages to Mind (async → sync)
    inbound_tx: mpsc::Sender<SocialMessage>,
    /// Peer connection pool
    peers: HashMap<String, IrohChannel>,
    /// Tensor stream for serialization/compression
    stream: TensorStream,
}
```

#### 2. `IrohBridgeHandle`

A sync-safe handle held by `ContinuousMind` or `Symthaea` facade.

```rust
pub struct IrohBridgeHandle {
    /// Send outbound messages (non-blocking try_send)
    outbound_tx: mpsc::Sender<SocialMessage>,
    /// Receive inbound messages (non-blocking try_recv)
    inbound_rx: mpsc::Receiver<SocialMessage>,
    /// Actor health check
    alive: Arc<AtomicBool>,
}

impl IrohBridgeHandle {
    /// Non-blocking: push messages from social_outbox to network
    pub fn flush_outbox(&self, messages: Vec<SocialMessage>) {
        for msg in messages {
            let _ = self.outbound_tx.try_send(msg); // Drop if full
        }
    }

    /// Non-blocking: drain available messages into social_inbox
    pub fn drain_inbox(&self) -> Vec<SocialMessage> {
        let mut messages = Vec::new();
        while let Ok(msg) = self.inbound_rx.try_recv() {
            messages.push(msg);
        }
        messages
    }
}
```

### Integration Point: Mind tick()

After `process_social()` in `mind/tick.rs`:

```rust
// After existing social processing...
if let Some(ref bridge) = self.iroh_bridge {
    // Flush outbox to network
    let outgoing = self.drain_social_outbox();
    if !outgoing.is_empty() {
        bridge.flush_outbox(outgoing);
    }

    // Drain network into inbox
    let incoming = bridge.drain_inbox();
    for msg in incoming {
        self.receive_social(msg);
    }
}
```

### Channel Sizing

- **Outbound**: Bounded channel, capacity 64 (3.2s at 5-tick intervals)
- **Inbound**: Bounded channel, capacity 128 (buffer for bursty peers)
- **Backpressure**: `try_send` drops oldest on full outbound; inbound drains greedily

### Serialization

`SocialMessage` requires `Serialize`/`Deserialize` (already has it via `ContinuousHV`).
Use the existing `TensorStream::prepare_consciousness()` for wire format (bincode + optional compression > 4KB).

### Peer Discovery

1. **Bootstrap**: Ticket exchange via out-of-band mechanism (Mycelix DHT or manual)
2. **Auto-connect**: `TicketManager::known_peers()` → reconnect on actor start
3. **Cleanup**: `TicketManager::cleanup_expired()` runs every 60s in actor loop

### Error Handling

- Peer disconnect: Remove from `peers` map, log, attempt reconnect on next send
- Send timeout (100ms): Drop message, increment `StreamStats::dropped_messages`
- Recv timeout: Normal — no data available, continue polling
- Channel full: Drop outbound message (fresher data is more valuable than old)

### Runtime Requirement

The actor requires a tokio runtime. Options:
1. **Facade-level**: `Symthaea::new()` spawns a tokio runtime when P2P is enabled
2. **External**: Caller provides `tokio::Handle` — matches Tauri/LUCID architecture
3. **Standalone**: `#[tokio::main]` in binary entry points only

Recommended: Option 2 (external handle), since LUCID already runs tokio for Tauri.

### Feature Gate

All behind existing `swarm` feature flag. No new features needed.

### Files to Create

| File | Purpose |
|------|---------|
| `src/swarm/iroh/bridge.rs` | `IrohBridgeActor` + `IrohBridgeHandle` |
| `src/mind/iroh_integration.rs` | Mind ↔ Bridge wiring in tick loop |

### Files to Modify

| File | Change |
|------|--------|
| `src/swarm/iroh/mod.rs` | Add `pub mod bridge;` |
| `src/mind/mod.rs` | Add `iroh_bridge: Option<IrohBridgeHandle>` field |
| `src/mind/tick.rs` | Add flush/drain after `process_social()` |
| `src/mind/config.rs` | Add `SocialMessage` serde derives if missing |

### Non-Goals (This Design)

- Multi-hop routing (direct peer connections only)
- Consensus/ordering guarantees (eventual consistency via social coherence)
- Large payload streaming (FL model weights use separate `weight_streaming` config)
- NAT traversal debugging (Iroh's Magicsock handles this)
