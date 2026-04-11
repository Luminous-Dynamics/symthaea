# Federation Security Model

## Overview

Mycelix Mail's federation system enables secure cross-cell communication between different Holochain networks. This document describes the security architecture, threat model, and protective measures.

---

## 1. Architecture

### 1.1 Cell Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                      Primary Cell (Local)                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Agent Source Chain                     │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │    │
│  │  │ Inbox   │  │ Sent    │  │ Trust   │  │ Contacts│    │    │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                     Federation Zome                       │    │
│  │  • Envelope Signing      • Network Discovery              │    │
│  │  • Route Resolution      • Protocol Translation           │    │
│  └─────────────────────────────────────────────────────────┘    │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Bridge Protocol     │
                    │   (Encrypted Channel) │
                    └───────────┬───────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                      Remote Cell (External)                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                     Federation Zome                       │    │
│  │  • Envelope Verification  • Permission Check              │    │
│  │  • Rate Limiting          • Audit Logging                 │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Communication Flow

```
Sender Cell                                      Receiver Cell
     │                                                │
     │  1. Create Envelope                            │
     ├─────────────────────────────────────────────►  │
     │     - Sign with agent key                      │
     │     - Encrypt payload                          │
     │     - Add routing metadata                     │
     │                                                │
     │  2. Verify Envelope                            │
     │  ◄─────────────────────────────────────────────┤
     │     - Validate signature                       │
     │     - Check permissions                        │
     │     - Rate limit check                         │
     │                                                │
     │  3. Process & Acknowledge                      │
     ├─────────────────────────────────────────────►  │
     │     - Store in local chain                     │
     │     - Send confirmation                        │
     │                                                │
```

---

## 2. Security Layers

### 2.1 Transport Layer Security

All cross-cell communication uses:

- **Encrypted WebRTC channels** with DTLS 1.3
- **Certificate pinning** for known networks
- **Perfect forward secrecy** via ephemeral keys

```rust
pub struct TransportConfig {
    /// Minimum TLS version
    pub min_tls_version: TlsVersion::V1_3,

    /// Allowed cipher suites
    pub cipher_suites: vec![
        CipherSuite::TLS_AES_256_GCM_SHA384,
        CipherSuite::TLS_CHACHA20_POLY1305_SHA256,
    ],

    /// Certificate verification mode
    pub verify_mode: VerifyMode::Strict,
}
```

### 2.2 Envelope Security

Every federation envelope includes:

| Field | Purpose | Security Measure |
|-------|---------|------------------|
| `sender` | Origin identification | Ed25519 public key |
| `signature` | Authenticity proof | Sign(sender_key, payload) |
| `timestamp` | Replay prevention | Must be within 5 minute window |
| `nonce` | Uniqueness | Random 256-bit value |
| `cell_id` | Network identification | DNA hash + agent key |
| `payload` | Encrypted content | X25519 + AES-256-GCM |

```rust
#[derive(Clone, Serialize, Deserialize)]
pub struct FederationEnvelope {
    pub version: u8,
    pub sender: AgentPubKey,
    pub recipient: AgentPubKey,
    pub sender_cell: CellId,
    pub recipient_cell: CellId,
    pub timestamp: Timestamp,
    pub nonce: [u8; 32],
    pub signature: Signature,
    pub encrypted_payload: Vec<u8>,
    pub routing_hints: Vec<RoutingHint>,
}

impl FederationEnvelope {
    pub fn verify(&self) -> Result<(), FederationError> {
        // 1. Check timestamp (within acceptable window)
        let now = sys_time()?;
        let age = now.as_millis() - self.timestamp.as_millis();
        if age > MAX_ENVELOPE_AGE_MS || age < 0 {
            return Err(FederationError::TimestampOutOfRange);
        }

        // 2. Verify signature
        let signing_data = self.signing_payload();
        if !self.sender.verify_signature(&self.signature, &signing_data)? {
            return Err(FederationError::InvalidSignature);
        }

        // 3. Check nonce uniqueness (replay prevention)
        if seen_nonce(&self.nonce)? {
            return Err(FederationError::ReplayDetected);
        }

        Ok(())
    }
}
```

### 2.3 Permission Model

Federation permissions are capability-based:

```rust
#[derive(Clone, Serialize, Deserialize)]
pub enum FederationPermission {
    /// Can receive emails from this network
    ReceiveEmail,

    /// Can query trust information
    QueryTrust,

    /// Can sync contacts
    SyncContacts,

    /// Can relay messages through this node
    Relay,

    /// Full federation access
    Full,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct FederationGrant {
    pub cell_id: CellId,
    pub permissions: Vec<FederationPermission>,
    pub granted_at: Timestamp,
    pub expires_at: Option<Timestamp>,
    pub granted_by: AgentPubKey,
    pub signature: Signature,
}
```

---

## 3. Threat Model

### 3.1 Identified Threats

| Threat | Severity | Mitigation |
|--------|----------|------------|
| **Envelope Spoofing** | Critical | Ed25519 signatures on all envelopes |
| **Replay Attacks** | High | Timestamp validation + nonce tracking |
| **Man-in-the-Middle** | High | TLS 1.3 with certificate pinning |
| **Sybil Networks** | Medium | Stake requirements for federation |
| **Traffic Analysis** | Medium | Optional onion routing (future) |
| **DoS via Federation** | Medium | Rate limiting per cell |
| **Privacy Leakage** | Low | Metadata encryption |

### 3.2 Spoofing Prevention

Envelope spoofing is prevented through multiple layers:

1. **Sender Verification**
   ```rust
   fn verify_sender(envelope: &FederationEnvelope) -> Result<(), Error> {
       // Verify the envelope signature matches the claimed sender
       let valid = envelope.sender.verify_signature(
           &envelope.signature,
           &envelope.canonical_bytes()
       )?;

       if !valid {
           return Err(Error::InvalidSenderSignature);
       }

       Ok(())
   }
   ```

2. **Cell ID Verification**
   ```rust
   fn verify_cell_origin(envelope: &FederationEnvelope) -> Result<(), Error> {
       // Verify the sender is actually part of the claimed cell
       let cell_agents = query_cell_agents(&envelope.sender_cell)?;

       if !cell_agents.contains(&envelope.sender) {
           return Err(Error::SenderNotInCell);
       }

       Ok(())
   }
   ```

3. **Network Allowlisting**
   ```rust
   fn check_network_allowed(cell_id: &CellId) -> Result<bool, Error> {
       let config = get_federation_config()?;

       match config.network_policy {
           NetworkPolicy::AllowAll => Ok(true),
           NetworkPolicy::AllowList(allowed) => {
               Ok(allowed.contains(&cell_id.dna_hash()))
           }
           NetworkPolicy::DenyList(denied) => {
               Ok(!denied.contains(&cell_id.dna_hash()))
           }
       }
   }
   ```

### 3.3 Replay Prevention

```rust
pub struct ReplayGuard {
    /// Maximum nonces to track
    max_entries: usize,

    /// Time window for nonce validity (5 minutes)
    window_ms: u64,
}

impl ReplayGuard {
    pub fn check_and_record(&self, nonce: &[u8; 32]) -> ExternResult<bool> {
        // Query for existing nonce
        let filter = ChainQueryFilter::new()
            .entry_type(EntryType::App(NONCE_ENTRY_DEF));

        let records = query(filter)?;

        for record in records {
            if let Some(entry) = record.entry().as_option() {
                if let Ok(stored) = NonceEntry::try_from(entry.clone()) {
                    if stored.nonce == *nonce {
                        return Ok(false); // Replay detected
                    }
                }
            }
        }

        // Record new nonce
        create_entry(&Entry::App(NonceEntry {
            nonce: *nonce,
            seen_at: sys_time()?,
        }.try_into()?))?;

        Ok(true) // First time seen
    }
}
```

---

## 4. Rate Limiting

### 4.1 Per-Cell Limits

```rust
pub struct FederationRateLimits {
    /// Maximum envelopes per minute per cell
    pub envelopes_per_minute: u32,

    /// Maximum bytes per minute per cell
    pub bytes_per_minute: u64,

    /// Maximum concurrent connections per cell
    pub max_connections: u32,

    /// Burst allowance
    pub burst_multiplier: f64,
}

impl Default for FederationRateLimits {
    fn default() -> Self {
        Self {
            envelopes_per_minute: 100,
            bytes_per_minute: 10 * 1024 * 1024, // 10 MB
            max_connections: 10,
            burst_multiplier: 2.0,
        }
    }
}
```

### 4.2 Graduated Response

```rust
pub enum RateLimitAction {
    /// Allow the request
    Allow,

    /// Slow down (add delay)
    Throttle { delay_ms: u64 },

    /// Reject this request
    Reject,

    /// Block the cell temporarily
    Block { duration_secs: u64 },

    /// Permanently blacklist
    Blacklist,
}

pub fn determine_action(
    violations: u32,
    current_rate: f64,
    limit: f64,
) -> RateLimitAction {
    let ratio = current_rate / limit;

    match (violations, ratio) {
        (_, r) if r <= 1.0 => RateLimitAction::Allow,
        (0..=2, r) if r <= 1.5 => RateLimitAction::Throttle { delay_ms: 100 },
        (0..=2, _) => RateLimitAction::Reject,
        (3..=5, _) => RateLimitAction::Block { duration_secs: 60 },
        (6..=10, _) => RateLimitAction::Block { duration_secs: 3600 },
        _ => RateLimitAction::Blacklist,
    }
}
```

---

## 5. Audit Logging

### 5.1 Logged Events

All federation activities are logged for compliance:

```rust
#[derive(Clone, Serialize, Deserialize)]
pub enum FederationEvent {
    /// Envelope received from external cell
    EnvelopeReceived {
        from_cell: CellId,
        from_agent: AgentPubKey,
        envelope_type: String,
        size_bytes: usize,
    },

    /// Envelope sent to external cell
    EnvelopeSent {
        to_cell: CellId,
        to_agent: AgentPubKey,
        envelope_type: String,
        size_bytes: usize,
    },

    /// Verification failure
    VerificationFailed {
        from_cell: CellId,
        reason: String,
    },

    /// Rate limit exceeded
    RateLimitExceeded {
        cell: CellId,
        action: RateLimitAction,
    },

    /// Permission denied
    PermissionDenied {
        cell: CellId,
        permission: FederationPermission,
    },

    /// Network added to allowlist/denylist
    NetworkPolicyChanged {
        cell: CellId,
        action: PolicyAction,
    },
}
```

### 5.2 Log Entry Structure

```rust
#[hdk_entry_helper]
pub struct FederationAuditLog {
    pub timestamp: Timestamp,
    pub event: FederationEvent,
    pub local_agent: AgentPubKey,
    pub metadata: HashMap<String, String>,
}

pub fn log_federation_event(event: FederationEvent) -> ExternResult<ActionHash> {
    let log = FederationAuditLog {
        timestamp: sys_time()?,
        event,
        local_agent: agent_info()?.agent_initial_pubkey,
        metadata: HashMap::new(),
    };

    create_entry(&Entry::App(log.try_into()?))
}
```

---

## 6. Network Discovery

### 6.1 Bootstrap Process

```
1. Agent joins network with known bootstrap nodes
2. Query bootstrap for list of federated networks
3. Retrieve public keys and DNA hashes
4. Establish encrypted connections
5. Exchange capability tokens
6. Begin federation operations
```

### 6.2 Discovery Protocol

```rust
#[derive(Clone, Serialize, Deserialize)]
pub struct NetworkAnnouncement {
    /// DNA hash of the announcing network
    pub dna_hash: DnaHash,

    /// Display name
    pub name: String,

    /// Capabilities offered
    pub capabilities: Vec<FederationCapability>,

    /// Bootstrap nodes
    pub bootstrap_nodes: Vec<String>,

    /// Minimum protocol version
    pub min_version: u32,

    /// Network administrator
    pub admin_key: AgentPubKey,

    /// Signature proving ownership
    pub signature: Signature,
}
```

---

## 7. Configuration

### 7.1 Federation Config

```yaml
# federation-config.yaml
federation:
  enabled: true

  # Network policy
  network_policy:
    mode: allowlist  # allowlist | denylist | open
    networks:
      - dna_hash: "uhCkk..."
        name: "Trusted Network 1"
        trust_level: high
      - dna_hash: "uhCkk..."
        name: "Partner Network"
        trust_level: medium

  # Rate limits
  rate_limits:
    envelopes_per_minute: 100
    bytes_per_minute: 10485760
    max_connections: 10

  # Security settings
  security:
    require_signed_envelopes: true
    max_envelope_age_seconds: 300
    replay_window_seconds: 600
    min_tls_version: "1.3"

  # Logging
  audit:
    enabled: true
    log_level: info
    retention_days: 90
```

### 7.2 Runtime Configuration

```rust
pub fn configure_federation(config: FederationConfig) -> ExternResult<()> {
    // Validate configuration
    config.validate()?;

    // Store in source chain
    create_entry(&Entry::App(config.try_into()?))?;

    // Log configuration change
    log_federation_event(FederationEvent::NetworkPolicyChanged {
        cell: cell_id()?,
        action: PolicyAction::ConfigUpdated,
    })?;

    Ok(())
}
```

---

## 8. Incident Response

### 8.1 Detection

Automated detection for:
- Unusual traffic patterns
- Failed verification spikes
- Rate limit violations
- Anomalous network activity

### 8.2 Response Procedures

1. **Immediate**: Block offending cell
2. **Investigation**: Review audit logs
3. **Mitigation**: Update allowlist/denylist
4. **Recovery**: Restore normal operations
5. **Post-mortem**: Document and improve

### 8.3 Manual Intervention

```rust
// Emergency block a cell
pub fn emergency_block_cell(cell_id: CellId) -> ExternResult<()> {
    // Requires admin capability
    check_admin_capability()?;

    // Add to denylist
    add_to_denylist(cell_id.clone())?;

    // Close existing connections
    close_cell_connections(&cell_id)?;

    // Log event
    log_federation_event(FederationEvent::NetworkPolicyChanged {
        cell: cell_id,
        action: PolicyAction::EmergencyBlock,
    })?;

    Ok(())
}
```

---

## 9. Best Practices

### For Operators

1. **Regular audits**: Review federation logs weekly
2. **Keep allowlists small**: Only federate with trusted networks
3. **Monitor rate limits**: Watch for anomalies
4. **Update regularly**: Apply security patches promptly
5. **Backup configs**: Maintain configuration backups

### For Developers

1. **Validate all inputs**: Never trust external data
2. **Sign all envelopes**: Even internal ones
3. **Log everything**: But sanitize sensitive data
4. **Test failure modes**: Handle network issues gracefully
5. **Rate limit by default**: Opt-out, not opt-in

---

## 10. Future Enhancements

### Planned

- [ ] Onion routing for metadata protection
- [ ] Zero-knowledge proofs for selective disclosure
- [ ] Multi-party threshold signatures
- [ ] Federated identity verification
- [ ] Cross-network trust aggregation

### Under Consideration

- Quantum-resistant cryptography
- Hardware security module integration
- Decentralized PKI
- Automated threat response

---

*Document Version: 1.0.0*
*Last Updated: January 2026*
