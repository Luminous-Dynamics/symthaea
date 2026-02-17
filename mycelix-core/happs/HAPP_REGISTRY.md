# Mycelix-Core hApp Registry

This registry documents all Holochain hApps in the Mycelix-Core ecosystem, their DNA hashes, versions, and configuration requirements.

## Table of Contents

- [hApp Overview](#happ-overview)
- [Registry Table](#registry-table)
- [DNA Registry](#dna-registry)
- [Installation Instructions](#installation-instructions)
- [Conductor Configuration](#conductor-configuration)
- [Network Architecture](#network-architecture)
- [Inter-hApp Communication](#inter-happ-communication)

---

## hApp Overview

The Mycelix-Core ecosystem consists of five interconnected Holochain hApps that provide decentralized infrastructure for trustless ML agent coordination.

```
                    +---------------------------+
                    |   zerotrustml-identity    |
                    |   (Foundation Layer)      |
                    +-------------+-------------+
                                  |
              +-------------------+-------------------+
              |                                       |
    +---------v---------+                   +---------v---------+
    | zerotrustml-      |                   |   mycelix-mail    |
    | reputation        |                   | (Communication)   |
    +--------+----------+                   +-------------------+
             |
    +--------v----------+
    | zerotrustml-      |
    | governance        |
    +--------+----------+
             |
    +--------v----------+
    | mycelix-          |
    | marketplace       |
    +-------------------+
```

---

## Registry Table

| hApp Name | Version | Status | DNA Count | Description |
|-----------|---------|--------|-----------|-------------|
| **zerotrustml-identity** | 0.1.0 | Production | 2 | Decentralized identity management for ML agents |
| **zerotrustml-reputation** | 0.1.0 | Production | 3 | Distributed reputation system with stake-weighted scoring |
| **zerotrustml-governance** | 0.1.0 | Beta | 4 | Governance, policy management, and prediction markets |
| **mycelix-mail** | 0.1.0 | Production | 4 | End-to-end encrypted agent communication |
| **mycelix-marketplace** | 0.1.0 | Beta | 5 | ML model and service exchange marketplace |

---

## DNA Registry

### zerotrustml-identity

| DNA Name | Hash (placeholder) | Version | Zomes |
|----------|-------------------|---------|-------|
| zerotrustml-identity | `uhC0k_identity_placeholder_hash` | 0.1.0 | identity_core, identity_coordinator |
| capability-tokens | `uhC0k_capability_placeholder_hash` | 0.1.0 | capability_tokens, capability_coordinator |

**Properties:**
```yaml
identity_expiry_days: 365
max_delegation_depth: 3
require_attestation: true
signature_scheme: ed25519
hash_algorithm: blake3
```

### zerotrustml-reputation

| DNA Name | Hash (placeholder) | Version | Zomes |
|----------|-------------------|---------|-------|
| zerotrustml-reputation | `uhC0k_reputation_placeholder_hash` | 0.1.0 | reputation_ledger, reputation_coordinator |
| behavioral-proofs | `uhC0k_behavioral_placeholder_hash` | 0.1.0 | behavioral_proofs, proofs_coordinator |
| stake-registry | `uhC0k_stake_placeholder_hash` | 0.1.0 | stake_registry, stake_coordinator |

**Properties:**
```yaml
initial_reputation: 50
min_reputation: 0
max_reputation: 100
decay_rate_per_day: 0.01
min_attestations_required: 3
sybil_resistance_enabled: true
```

### zerotrustml-governance

| DNA Name | Hash (placeholder) | Version | Zomes |
|----------|-------------------|---------|-------|
| zerotrustml-governance | `uhC0k_governance_placeholder_hash` | 0.1.0 | governance_core, governance_coordinator |
| policy-registry | `uhC0k_policy_placeholder_hash` | 0.1.0 | policy_registry, policy_coordinator |
| prediction-markets | `uhC0k_prediction_placeholder_hash` | 0.1.0 | prediction_markets, markets_coordinator |
| model-validation | `uhC0k_validation_placeholder_hash` | 0.1.0 | model_validation, validation_coordinator |

**Properties:**
```yaml
proposal_threshold: 100
voting_period_days: 7
quorum_percentage: 33
voting_mechanism: conviction
conviction_decay: 0.9
```

### mycelix-mail

| DNA Name | Hash (placeholder) | Version | Zomes |
|----------|-------------------|---------|-------|
| mycelix-mail | `uhC0k_mail_placeholder_hash` | 0.1.0 | mail_core, mail_coordinator |
| mail-channels | `uhC0k_channels_placeholder_hash` | 0.1.0 | channels, channels_coordinator |
| mail-attachments | `uhC0k_attachments_placeholder_hash` | 0.1.0 | attachments, attachments_coordinator |
| mail-notifications | `uhC0k_notifications_placeholder_hash` | 0.1.0 | notifications, notifications_coordinator |

**Properties:**
```yaml
encryption_scheme: x25519-xsalsa20-poly1305
max_message_size_kb: 1024
message_retention_days: 90
delivery_receipt_required: true
```

### mycelix-marketplace

| DNA Name | Hash (placeholder) | Version | Zomes |
|----------|-------------------|---------|-------|
| mycelix-marketplace | `uhC0k_marketplace_placeholder_hash` | 0.1.0 | marketplace_core, marketplace_coordinator |
| marketplace-escrow | `uhC0k_escrow_placeholder_hash` | 0.1.0 | escrow, escrow_coordinator |
| model-registry | `uhC0k_modelreg_placeholder_hash` | 0.1.0 | model_registry, modelreg_coordinator |
| marketplace-reviews | `uhC0k_reviews_placeholder_hash` | 0.1.0 | reviews, reviews_coordinator |
| marketplace-analytics | `uhC0k_analytics_placeholder_hash` | 0.1.0 | analytics, analytics_coordinator |

**Properties:**
```yaml
listing_fee_percentage: 0.5
min_listing_stake: 50
escrow_timeout_hours: 168
dispute_resolution: decentralized
```

---

## Installation Instructions

### Prerequisites

1. Holochain Launcher or Conductor running (version 0.3.x)
2. Valid agent key pair generated
3. Network connectivity to Holochain bootstrap servers

### Installing via Holochain Launcher

```bash
# Install from local bundle
holochain-launcher install ./dist/happs/zerotrustml-identity.happ

# Or install from URL
holochain-launcher install https://happ.mycelix.network/zerotrustml-identity-0.1.0.happ
```

### Installing via hc CLI

```bash
# Install hApp to conductor
hc sandbox call install-app ./dist/happs/zerotrustml-identity.happ \
  --agent-key <your-agent-key> \
  --app-id zerotrustml-identity

# Enable the app
hc sandbox call enable-app zerotrustml-identity
```

### Recommended Installation Order

Install hApps in dependency order:

```bash
# 1. Foundation: Identity (required by all others)
hc sandbox call install-app ./dist/happs/zerotrustml-identity.happ

# 2. Reputation (depends on identity)
hc sandbox call install-app ./dist/happs/zerotrustml-reputation.happ

# 3. Communication: Mail (depends on identity)
hc sandbox call install-app ./dist/happs/mycelix-mail.happ

# 4. Governance (depends on identity, reputation)
hc sandbox call install-app ./dist/happs/zerotrustml-governance.happ

# 5. Marketplace (depends on identity, reputation)
hc sandbox call install-app ./dist/happs/mycelix-marketplace.happ
```

### Verifying Installation

```bash
# List installed apps
hc sandbox call list-apps

# Check app status
hc sandbox call app-info zerotrustml-identity

# Expected output:
# {
#   "app_id": "zerotrustml-identity",
#   "status": "running",
#   "cells": [
#     { "role_id": "identity_core", "dna_hash": "uhC0k..." },
#     { "role_id": "capability_tokens", "dna_hash": "uhC0k..." }
#   ]
# }
```

---

## Conductor Configuration

### Basic Configuration

Create a conductor configuration file (`conductor-config.yaml`):

```yaml
---
environment_path: ./holochain-data
keystore:
  type: lair_server_in_proc

admin_interfaces:
  - driver:
      type: websocket
      port: 9000

network:
  bootstrap_service: https://bootstrap.holo.host
  transport_pool:
    - type: webrtc
      signal_url: wss://signal.holo.host

tuning:
  gossip_strategy: sharded
  gossip_arc_clamping: full
```

### Full Mycelix Configuration

```yaml
---
environment_path: /var/lib/mycelix/holochain

keystore:
  type: lair_server
  connection_url: unix:///var/run/lair-keystore/socket

admin_interfaces:
  - driver:
      type: websocket
      port: 9000
    allowed_origins:
      - localhost
      - 127.0.0.1

app_interfaces:
  - driver:
      type: websocket
      port: 9001
    allowed_origins: "*"

network:
  bootstrap_service: https://bootstrap.mycelix.network
  transport_pool:
    - type: webrtc
      signal_url: wss://signal.mycelix.network
    - type: quic
      bind_to: 0.0.0.0:8888

dpki:
  instance_id: mycelix-dpki
  init_humans: []

tuning:
  # Gossip configuration
  gossip_strategy: sharded
  gossip_arc_clamping: full
  gossip_peer_on_success_next_gossip_delay_ms: 100
  gossip_peer_on_error_next_gossip_delay_ms: 1000

  # Performance tuning
  concurrent_limit_per_thread: 4096
  tx2_implicit_timeout_ms: 30000
  tx2_channel_count_per_connection: 3

  # Storage
  db_sync_level: full
```

### Systemd Service

Create `/etc/systemd/system/mycelix-conductor.service`:

```ini
[Unit]
Description=Mycelix Holochain Conductor
After=network.target

[Service]
Type=simple
User=holochain
Group=holochain
Environment=RUST_LOG=info
ExecStart=/usr/local/bin/holochain -c /etc/mycelix/conductor-config.yaml
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

---

## Network Architecture

### DHT Topology

Each DNA creates its own DHT (Distributed Hash Table):

```
┌─────────────────────────────────────────────────────────────────┐
│                      Holochain Conductor                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  Identity DHT   │  │  Reputation DHT │  │  Governance DHT │ │
│  │  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────┐  │ │
│  │  │ Agent A   │  │  │  │ Agent A   │  │  │  │ Agent A   │  │ │
│  │  │ Agent B   │  │  │  │ Agent B   │  │  │  │ Agent B   │  │ │
│  │  │ Agent C   │  │  │  │ Agent C   │  │  │  │ Agent C   │  │ │
│  │  └───────────┘  │  │  └───────────┘  │  │  └───────────┘  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐                      │
│  │    Mail DHT     │  │ Marketplace DHT │                      │
│  │  ┌───────────┐  │  │  ┌───────────┐  │                      │
│  │  │ Agent A   │  │  │  │ Agent A   │  │                      │
│  │  │ Agent B   │  │  │  │ Agent B   │  │                      │
│  │  │ Agent C   │  │  │  │ Agent C   │  │                      │
│  │  └───────────┘  │  │  └───────────┘  │                      │
│  └─────────────────┘  └─────────────────┘                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Network Segmentation

| Network | Purpose | Access Level |
|---------|---------|--------------|
| Identity Network | Agent identity management | Public |
| Reputation Network | Reputation tracking | Public |
| Governance Network | Policy and voting | Public |
| Mail Network | Encrypted messaging | Private (E2E) |
| Marketplace Network | Trading and exchange | Public |

---

## Inter-hApp Communication

### Bridge Calls

hApps communicate via bridge calls between cells:

```rust
// Example: Reputation calling Identity for verification
#[hdk_extern]
pub fn verify_before_attestation(agent_id: AgentPubKey) -> ExternResult<bool> {
    // Bridge call to identity hApp
    let identity_cell_id = get_identity_cell_id()?;

    let verification: IdentityVerification = call(
        CallTargetCell::OtherCell(identity_cell_id),
        "identity_core",
        "verify_agent_identity".into(),
        None,
        &agent_id,
    )?;

    Ok(verification.valid)
}
```

### Capability Grants

Cross-hApp calls require capability grants:

```rust
// Grant capability to reputation hApp
#[hdk_extern]
pub fn grant_reputation_access(reputation_agent: AgentPubKey) -> ExternResult<()> {
    create_cap_grant(CapGrantEntry {
        tag: "reputation_access".into(),
        access: CapAccess::Assigned {
            secret: generate_cap_secret()?,
            assignees: BTreeSet::from([reputation_agent]),
        },
        functions: GrantedFunctions::Listed(BTreeSet::from([
            ("identity_core".into(), "verify_agent_identity".into()),
            ("identity_core".into(), "get_agent_metadata".into()),
        ])),
    })?;

    Ok(())
}
```

### Signal Propagation

Cross-hApp events via signals:

```rust
// Emit signal when identity is created
#[hdk_extern]
pub fn create_identity(input: CreateIdentityInput) -> ExternResult<ActionHash> {
    let identity_hash = create_entry(&Entry::Identity(input.clone()))?;

    // Signal to other hApps
    emit_signal(&IdentityCreatedSignal {
        agent_id: agent_info()?.agent_initial_pubkey,
        identity_hash: identity_hash.clone(),
    })?;

    Ok(identity_hash)
}
```

---

## Ecosystem Interaction Flow

### Agent Registration Flow

```
1. Agent creates identity in zerotrustml-identity
   └── Returns: agent_id, identity_hash

2. Identity notifies zerotrustml-reputation
   └── Initializes reputation score (50)

3. Agent joins mycelix-mail network
   └── Creates encrypted mailbox

4. Agent can now participate in:
   ├── zerotrustml-governance (voting, proposals)
   └── mycelix-marketplace (listing, purchasing)
```

### Marketplace Transaction Flow

```
1. Seller lists model in mycelix-marketplace
   ├── Validates identity (zerotrustml-identity)
   ├── Checks reputation threshold (zerotrustml-reputation)
   └── Creates listing entry

2. Buyer initiates purchase
   ├── Creates escrow (marketplace-escrow DNA)
   └── Notifies seller via mycelix-mail

3. Seller delivers model
   ├── Uploads to content-addressable storage
   └── Provides access key

4. Buyer confirms receipt
   ├── Escrow releases funds
   ├── Both parties can submit reviews
   └── Reputation updated for both parties
```

---

## Version Compatibility Matrix

| Component | v0.1.0 | v0.2.0 (planned) |
|-----------|--------|------------------|
| Holochain | 0.3.x | 0.4.x |
| HDK | 0.3.x | 0.4.x |
| Identity DNA | Compatible | Breaking changes |
| Reputation DNA | Compatible | Compatible |
| Governance DNA | Compatible | Minor updates |
| Mail DNA | Compatible | Compatible |
| Marketplace DNA | Compatible | Feature additions |

---

## Support and Resources

- **Documentation**: https://docs.mycelix.network
- **GitHub Issues**: https://github.com/Luminous-Dynamics/Mycelix-Core/issues
- **Discord**: https://discord.gg/mycelix
- **Forum**: https://forum.mycelix.network

---

*Last updated: 2024-01-15*
*Registry version: 1.0.0*
*Maintained by the Mycelix-Core Team*
