# Mycelix Mail Architecture

This document provides a comprehensive overview of the Mycelix Mail system architecture, designed for developers integrating with or extending the platform.

## System Overview

Mycelix Mail is a decentralized email system built on Holochain, featuring:

- **P2P Email Delivery**: Direct agent-to-agent message delivery
- **Web of Trust (MATL)**: Mycelix Advanced Trust Logic for spam prevention
- **End-to-End Encryption**: Post-quantum cryptography support
- **Federation**: Cross-network email routing
- **Offline-First**: CRDT-based sync for mobile/desktop

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Applications                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  Web UI  │  │ Desktop  │  │  Mobile  │  │  Browser Ext.    │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘ │
└───────┼─────────────┼─────────────┼──────────────────┼──────────┘
        │             │             │                  │
        ▼             ▼             ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SDK Layer (TypeScript/Python)                │
│  ┌──────────────────────────┐  ┌──────────────────────────────┐ │
│  │   REST API Client        │  │   Holochain Direct Client    │ │
│  │   - Standard HTTP        │  │   - P2P via WebSocket        │ │
│  │   - Gateway required     │  │   - No intermediary          │ │
│  └────────────┬─────────────┘  └────────────┬─────────────────┘ │
└───────────────┼─────────────────────────────┼───────────────────┘
                │                             │
                ▼                             ▼
┌───────────────────────────┐    ┌────────────────────────────────┐
│     REST API Gateway      │    │      Holochain Conductor       │
│  ┌─────────────────────┐  │    │  ┌──────────────────────────┐  │
│  │ Authentication      │  │    │  │     Mycelix Mail hApp    │  │
│  │ Rate Limiting       │  │    │  │  ┌────────────────────┐  │  │
│  │ Request Validation  │  │    │  │  │   DNA (Zomes)      │  │  │
│  │ Response Formatting │  │    │  │  └────────────────────┘  │  │
│  └─────────────────────┘  │    │  └──────────────────────────┘  │
└─────────────┬─────────────┘    └──────────────┬─────────────────┘
              │                                  │
              └──────────────┬───────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Holochain DHT Network                         │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   Distributed Hash Table                     ││
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        ││
│  │  │ Agent A │◄─►│ Agent B │◄─►│ Agent C │◄─►│ Agent D │        ││
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘        ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Holochain Zome Architecture

The Mycelix Mail hApp consists of paired integrity and coordinator zomes:

### Zome Pairs

| Zome | Purpose |
|------|---------|
| `messages` | Email storage, delivery, folders, threads |
| `contacts` | Contact management, groups, blocking |
| `trust` | Web of trust, MATL algorithm, attestations |
| `keys` | Post-quantum key management, rotation |
| `capabilities` | Mailbox sharing, delegation, access control |
| `federation` | Cross-network routing, bridge protocols |
| `search` | Full-text indexing, query processing |
| `sync` | CRDT operations, offline/online sync |
| `backup` | Encrypted backup/restore |
| `scheduler` | Delayed/recurring email sends |
| `audit` | Compliance logging, retention policies |

### Integrity vs Coordinator Zomes

```
┌─────────────────────────────────────────────────────┐
│              Integrity Zome (Validation)            │
│  ┌───────────────────────────────────────────────┐  │
│  │ - Entry type definitions                      │  │
│  │ - Link type definitions                       │  │
│  │ - Validation callbacks                        │  │
│  │ - Cryptographic verification                  │  │
│  │ - Business rule enforcement                   │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              Coordinator Zome (Logic)               │
│  ┌───────────────────────────────────────────────┐  │
│  │ - CRUD operations                             │  │
│  │ - Query and aggregation                       │  │
│  │ - Real-time signals                           │  │
│  │ - Cross-zome calls                            │  │
│  │ - External integrations                       │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Data Model

### Email Entry

```rust
pub struct EncryptedEmail {
    pub sender: AgentPubKey,
    pub recipient: AgentPubKey,
    pub encrypted_subject: Vec<u8>,      // ChaCha20-Poly1305
    pub encrypted_body: Vec<u8>,
    pub encrypted_attachments: Vec<u8>,
    pub ephemeral_pubkey: Vec<u8>,       // X25519 or Kyber
    pub nonce: [u8; 24],
    pub signature: Vec<u8>,              // Ed25519 or Dilithium
    pub crypto_suite: CryptoSuite,
    pub message_id: String,
    pub in_reply_to: Option<String>,
    pub references: Vec<String>,
    pub timestamp: Timestamp,
    pub priority: EmailPriority,
    pub read_receipt_requested: bool,
    pub expires_at: Option<Timestamp>,
}
```

### Trust Attestation

```rust
pub struct TrustAttestation {
    pub truster: AgentPubKey,
    pub trustee: AgentPubKey,
    pub trust_level: f64,                // -1.0 to 1.0
    pub category: TrustCategory,
    pub evidence: Vec<TrustEvidence>,
    pub reason: Option<String>,
    pub created_at: Timestamp,
    pub expires_at: Option<Timestamp>,
    pub signature: Vec<u8>,
    pub revoked: bool,
    pub stake: Option<ReputationStake>,
}
```

### Link Graph

```
Agent ──────────────► Inbox Emails
   │                      ▲
   ├──────────────► Sent Emails
   │
   ├──────────────► Drafts
   │
   ├──────────────► Folders ──────► Folder Emails
   │
   ├──────────────► Contacts
   │
   ├──────────────► Given Attestations ──────► TrustAttestation
   │
   └──────────────► Received Attestations ──────► TrustAttestation

Email ──────────────► Attachments
   │
   ├──────────────► Read Receipts
   │
   ├──────────────► Email State (per agent)
   │
   └──────────────► Thread ──────► Thread Emails
```

## MATL Trust Algorithm

The Mycelix Advanced Trust Logic computes trust scores using:

### 1. Direct Trust

Trust from personal attestations weighted by:
- Evidence strength (in-person > video > phone > email)
- Temporal decay (recent attestations weighted higher)
- Reputation stake (staked trust carries more weight)

### 2. Transitive Trust

Trust propagated through network with:
- Configurable decay factor (default 0.7 per hop)
- Maximum depth limit (default 5 hops)
- Path redundancy bonus

### 3. Byzantine Detection

Flags for suspicious behavior:
- **Sybil Suspicion**: Many attestations from similar sources
- **Trust Volatility**: Rapid score changes
- **Inconsistent Attestations**: Same truster giving conflicting scores
- **Collusion Detection**: Groups with suspiciously similar patterns

### Trust Score Computation

```
TrustScore = aggregate(
    direct_trust * direct_weight * temporal_decay,
    transitive_trust * transitive_weight * path_decay
) - byzantine_penalty
```

## Encryption Architecture

### Key Hierarchy

```
┌─────────────────────────────────────────────────────┐
│                  Identity Keys                       │
│  ┌─────────────────────┐  ┌─────────────────────┐   │
│  │ Agent Signing Key   │  │ Post-Quantum Keys   │   │
│  │ (Ed25519)           │  │ (Dilithium/Kyber)   │   │
│  └─────────────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│               Per-Message Ephemeral Keys             │
│  ┌─────────────────────────────────────────────────┐│
│  │ Ephemeral X25519/Kyber keypair for each email   ││
│  │ Enables forward secrecy                          ││
│  └─────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                 Symmetric Keys                       │
│  ┌─────────────────────────────────────────────────┐│
│  │ ChaCha20-Poly1305 or AES-256-GCM                ││
│  │ Derived via ECDH/Kyber key agreement            ││
│  └─────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
```

### Email Encryption Flow

1. Sender generates ephemeral keypair
2. Performs key agreement with recipient's public key
3. Derives symmetric key via HKDF
4. Encrypts subject, body, attachments
5. Signs encrypted content
6. Creates DHT entry with encrypted payload

## Federation Protocol

### Cross-Network Routing

```
┌────────────────────┐     ┌────────────────────┐
│  Holochain         │     │  Other Network     │
│  Network A         │     │  (SMTP, Matrix)    │
│                    │     │                    │
│  ┌──────────────┐  │     │  ┌──────────────┐  │
│  │ Agent Alice  │──┼──┬──┼──│ User Bob     │  │
│  └──────────────┘  │  │  │  └──────────────┘  │
│                    │  │  │                    │
└────────────────────┘  │  └────────────────────┘
                        │
                        ▼
               ┌────────────────┐
               │ Federation     │
               │ Bridge Agent   │
               │                │
               │ - Route lookup │
               │ - Encryption   │
               │ - Delivery     │
               └────────────────┘
```

### Route Types

- **DirectHolochain**: P2P within same network
- **BridgeAgent**: Relay through bridge node
- **RelayServer**: HTTP relay service
- **Gateway**: Protocol translation gateway
- **Multipath**: Redundant delivery paths

## Real-Time Signals

Holochain signals enable instant notifications:

```typescript
client.onSignal((signal: MailSignal) => {
  switch (signal.type) {
    case 'EmailReceived':
      // New email arrived
      break;
    case 'ReadReceiptReceived':
      // Recipient read your email
      break;
    case 'TypingIndicator':
      // Someone is composing a reply
      break;
  }
});
```

## Offline-First Sync

CRDT-based synchronization enables:

- Queue operations while offline
- Conflict-free merging on reconnect
- Multi-device state synchronization
- Gossip-based eventual consistency

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Device A  │    │   Device B  │    │   Device C  │
│   (Online)  │◄──►│  (Offline)  │◄──►│   (Online)  │
│             │    │             │    │             │
│ ┌─────────┐ │    │ ┌─────────┐ │    │ ┌─────────┐ │
│ │ State   │ │    │ │ Pending │ │    │ │ State   │ │
│ │ Synced  │ │    │ │ Ops     │ │    │ │ Synced  │ │
│ └─────────┘ │    │ └─────────┘ │    │ └─────────┘ │
└─────────────┘    └─────────────┘    └─────────────┘
                         │
                         ▼ (reconnect)
                   ┌─────────────┐
                   │  Process    │
                   │  Pending    │
                   │  Operations │
                   └─────────────┘
```

## Security Model

### Threat Mitigations

| Threat | Mitigation |
|--------|------------|
| MITM Attack | End-to-end encryption, signature verification |
| Spam | Web of trust scoring, rate limiting |
| Sybil Attack | Byzantine detection, stake requirements |
| Key Compromise | Key rotation, forward secrecy |
| Quantum Threats | Post-quantum algorithms (Kyber, Dilithium) |
| Data Loss | DHT redundancy, encrypted backups |
| Censorship | Decentralized architecture, no central authority |

### Validation Rules

All DHT operations are validated:

1. **Sender verification**: Email sender matches action author
2. **Encryption validation**: Required fields present, valid algorithms
3. **Signature verification**: Content signed by claimed author
4. **Ownership checks**: Only owners can modify their data
5. **Rate limiting**: Max operations per time window

## Performance Considerations

### DHT Optimization

- Chunked attachments for large files
- Link-based indexing for queries
- Anchor patterns for global lookups
- Caching of trust scores (1-hour TTL)

### Scalability

- Horizontal scaling via additional agents
- Sharded data via DHT neighborhoods
- Lazy loading of email bodies
- Background sync operations

## Next Steps

- [Deployment Guide](./deployment.md) - Production setup
- [API Reference](/docs/api/) - Complete API documentation
- [Contributing](/CONTRIBUTING.md) - Development guidelines
