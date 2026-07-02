# Mycelix Civilizational OS: Unified Architecture & Integration Blueprint

## Executive Summary

This document defines the complete integration architecture for the Mycelix Civilizational OS - a comprehensive 32-hApp ecosystem built on Holochain that provides all the infrastructure primitives necessary for human civilization to operate with sovereignty, transparency, and resilience.

The architecture follows the principle of **composable sovereignty**: each hApp is fully functional independently, yet gains emergent capabilities when integrated with the broader ecosystem.

---

## Table of Contents

1. [Architectural Foundations](#architectural-foundations)
2. [The Integration Spine](#the-integration-spine)
3. [Cross-Cutting Protocols](#cross-cutting-protocols)
4. [Data Flow Architecture](#data-flow-architecture)
5. [Trust Architecture](#trust-architecture)
6. [Event Bus Architecture](#event-bus-architecture)
7. [hApp Cluster Analysis](#happ-cluster-analysis)
8. [Integration Patterns](#integration-patterns)
9. [Deployment Architecture](#deployment-architecture)
10. [Security Architecture](#security-architecture)
11. [Performance & Scaling](#performance--scaling)
12. [Migration & Evolution](#migration--evolution)

---

## Architectural Foundations

### Core Design Principles

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MYCELIX ARCHITECTURAL PRINCIPLES                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. AGENT-CENTRIC SOVEREIGNTY                                          │
│     └─ Data lives with agents, not platforms                           │
│     └─ Every agent controls their own information                       │
│     └─ No central point of control or failure                          │
│                                                                         │
│  2. TRUST AS FIRST-CLASS INFRASTRUCTURE                                │
│     └─ MATL provides unified trust scoring                             │
│     └─ All interactions are trust-weighted                             │
│     └─ Reputation is context-aware and portable                        │
│                                                                         │
│  3. EPISTEMIC HUMILITY                                                 │
│     └─ All claims classified on 3D epistemic cube                      │
│     └─ Certainty gradients, not binary truth                           │
│     └─ Materiality determines retention                                │
│                                                                         │
│  4. BYZANTINE RESILIENCE                                               │
│     └─ Assume 45% adversarial capacity                                 │
│     └─ Defense in depth across all layers                              │
│     └─ Graceful degradation, not catastrophic failure                  │
│                                                                         │
│  5. COMPOSABLE SOVEREIGNTY                                             │
│     └─ hApps are independent but synergistic                           │
│     └─ Opt-in integration, never forced coupling                       │
│     └─ Emergent capabilities from composition                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TECHNOLOGY LAYERS                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      PRESENTATION LAYER                          │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐    │   │
│  │  │  SvelteKit  │ │    Tauri    │ │   Mobile (Capacitor)    │    │   │
│  │  │   Web UI    │ │  Desktop    │ │    iOS / Android        │    │   │
│  │  └─────────────┘ └─────────────┘ └─────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      APPLICATION LAYER                           │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │                    Mycelix SDK                            │   │   │
│  │  │  • Unified API across all hApps                          │   │   │
│  │  │  • Trust-weighted method calls                           │   │   │
│  │  │  • Epistemic classification utilities                    │   │   │
│  │  │  • Cross-hApp event subscription                         │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      COORDINATION LAYER                          │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐    │   │
│  │  │   Bridge    │ │    MATL     │ │      Sentinel           │    │   │
│  │  │  Protocol   │ │  Trust Net  │ │    Security Mesh        │    │   │
│  │  └─────────────┘ └─────────────┘ └─────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                       PROTOCOL LAYER                             │   │
│  │  ┌──────────────────────────────────────────────────────────┐   │   │
│  │  │              Holochain 0.6 (Wasmer 3)                    │   │   │
│  │  │  • Coordinator Zomes (hdk 0.6.0)                         │   │   │
│  │  │  • Integrity Zomes (hdi 0.7.0)                           │   │   │
│  │  │  • DHT with validation                                   │   │   │
│  │  └──────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                       STORAGE LAYER                              │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐    │   │
│  │  │  Holochain  │ │    IPFS     │ │       Arweave           │    │   │
│  │  │     DHT     │ │  (M1-M2)    │ │      (M3 Archive)       │    │   │
│  │  └─────────────┘ └─────────────┘ └─────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## The Integration Spine

The Integration Spine consists of 5 foundational services that all hApps depend upon:

### Spine Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INTEGRATION SPINE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                         ┌─────────────┐                                 │
│                         │   ATTEST    │                                 │
│                         │  (Identity) │                                 │
│                         └──────┬──────┘                                 │
│                                │                                        │
│         ┌──────────────────────┼──────────────────────┐                │
│         │                      │                      │                │
│         ▼                      ▼                      ▼                │
│  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐           │
│  │    MATL     │       │   BRIDGE    │       │  CHRONICLE  │           │
│  │   (Trust)   │◄─────►│  (Router)   │◄─────►│  (Archive)  │           │
│  └──────┬──────┘       └──────┬──────┘       └──────┬──────┘           │
│         │                     │                      │                  │
│         └──────────────────────┼──────────────────────┘                 │
│                                │                                        │
│                         ┌──────▼──────┐                                 │
│                         │  SENTINEL   │                                 │
│                         │ (Security)  │                                 │
│                         └─────────────┘                                 │
│                                                                         │
│  Every hApp MUST integrate with all 5 spine components                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Spine Component Contracts

#### 1. Attest Contract (Identity)
Every hApp must:
```rust
trait AttestIntegration {
    /// Get verified identity for an agent
    fn get_agent_identity(&self, agent: AgentPubKey) -> Option<VerifiedIdentity>;

    /// Check if agent meets credential requirements
    fn verify_credential(&self, agent: AgentPubKey, requirement: CredentialRequirement) -> bool;

    /// Issue credential upon completion of hApp-specific achievement
    fn issue_credential(&self, recipient: AgentPubKey, credential: Credential) -> ExternResult<ActionHash>;
}
```

#### 2. MATL Contract (Trust)
Every hApp must:
```rust
trait MATLIntegration {
    /// Get composite trust score for agent in this context
    fn get_trust_score(&self, agent: AgentPubKey, context: TrustContext) -> f64;

    /// Report positive interaction
    fn report_positive(&self, interaction: PositiveInteraction) -> ExternResult<()>;

    /// Report negative interaction (with evidence)
    fn report_negative(&self, interaction: NegativeInteraction, evidence: Evidence) -> ExternResult<()>;

    /// Weight action by trust (used in voting, market matching, etc.)
    fn trust_weight(&self, agent: AgentPubKey, base_weight: f64) -> f64;
}
```

#### 3. Bridge Contract (Router)
Every hApp must:
```rust
trait BridgeIntegration {
    /// Register this hApp with the Bridge
    fn register_with_bridge(&self, capabilities: Vec<Capability>) -> ExternResult<()>;

    /// Handle incoming cross-hApp request
    fn handle_bridge_request(&self, request: BridgeRequest) -> ExternResult<BridgeResponse>;

    /// Emit event for subscribers
    fn emit_event(&self, event: HAppEvent) -> ExternResult<()>;

    /// Query another hApp via Bridge
    fn query_happ(&self, target: HAppId, query: Query) -> ExternResult<QueryResult>;
}
```

#### 4. Chronicle Contract (Archive)
Every hApp must:
```rust
trait ChronicleIntegration {
    /// Classify entry for archival
    fn classify_entry(&self, entry: Entry) -> EpistemicClassification;

    /// Archive entry based on materiality
    fn archive_if_material(&self, entry: Entry, materiality: MaterialityLevel) -> ExternResult<Option<ArchiveHash>>;

    /// Cite previous work
    fn cite(&self, source: CitationSource) -> ExternResult<CitationHash>;

    /// Check if entry should be pruned
    fn should_prune(&self, entry: Entry, age: Duration) -> bool;
}
```

#### 5. Sentinel Contract (Security)
Every hApp must:
```rust
trait SentinelIntegration {
    /// Report security-relevant event
    fn report_security_event(&self, event: SecurityEvent) -> ExternResult<()>;

    /// Check if agent is quarantined
    fn is_quarantined(&self, agent: AgentPubKey) -> bool;

    /// Respond to security directive
    fn handle_security_directive(&self, directive: SecurityDirective) -> ExternResult<()>;

    /// Get current threat level for this hApp
    fn get_threat_level(&self) -> ThreatLevel;
}
```

---

## Cross-Cutting Protocols

### Protocol 1: Epistemic Classification Protocol

All claims, facts, and assertions in the system are classified on a 3D epistemic cube:

```
                            EPISTEMIC CUBE

                              N3 (Imperative)
                              │
                              │
                        N2────┼────────────────► M3 (Civilizational)
                       /│     │                 /
                      / │     │                /
                     /  │     │               /
               N1───/───┼─────┼──────────────/──► M2 (Institutional)
               │   /    │     │             /│
               │  /     │     │            / │
               │ /      │     │           /  │
          N0──┼/────────┼─────┼──────────/───┼──► M1 (Community)
              /         │     │         /    │
             /          │     │        /     │
            /           │     │       /      │
           /            │     │      /       │
          ▼             │     │     ▼        │
    E0 (Opinion)        │     │   M0         │
         │              │     │  (Personal)  │
         │              │     │              │
    E1 (Observation)    │     │              │
         │              │     │              │
         │              │     ▼              │
    E2 (Verification)   │    N0              │
         │              │  (Descriptive)     │
         │              │                    │
    E3 (Proof)          │                    │
         │              │                    │
         ▼              ▼                    ▼
    E4 (Axiomatic)

    E-axis: Empirical certainty (how verifiable)
    N-axis: Normative strength (how prescriptive)
    M-axis: Materiality scope (how many affected)
```

### Classification Examples

| Claim | E | N | M | Retention |
|-------|---|---|---|-----------|
| "I prefer blue" | E0 | N0 | M0 | Agent-local |
| "Temperature is 25°C" (sensor) | E2 | N0 | M1 | 1 year |
| "Contract terms agreed" | E3 | N2 | M2 | Permanent |
| "Constitutional amendment" | E4 | N3 | M3 | Arweave |
| "Best practice recommendation" | E1 | N1 | M1 | 5 years |

### Protocol 2: Cross-hApp Event Protocol

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EVENT FLOW                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Source hApp                                                            │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────┐                                                   │
│  │  emit_event()   │                                                   │
│  └────────┬────────┘                                                   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                       BRIDGE ROUTER                              │   │
│  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────────────┐  │   │
│  │  │ Event Filter  │ │ Subscription  │ │   Priority Queue      │  │   │
│  │  │               │►│    Match      │►│                       │  │   │
│  │  └───────────────┘ └───────────────┘ └───────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                             │
│           ├─────────────────┬──────────────────┬───────────────────┐   │
│           ▼                 ▼                  ▼                   ▼   │
│  ┌─────────────┐   ┌─────────────┐    ┌─────────────┐    ┌─────────┐ │
│  │ Subscriber  │   │ Subscriber  │    │ Subscriber  │    │ Archive │ │
│  │   hApp A    │   │   hApp B    │    │   hApp C    │    │Chronicle│ │
│  └─────────────┘   └─────────────┘    └─────────────┘    └─────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Event Schema

```rust
pub struct HAppEvent {
    pub event_id: String,
    pub source_happ: HAppId,
    pub event_type: EventType,
    pub payload: SerializedPayload,
    pub timestamp: Timestamp,
    pub epistemic_class: EpistemicClassification,
    pub agents_involved: Vec<AgentPubKey>,
    pub requires_ack: bool,
}

pub enum EventType {
    // Identity Events
    CredentialIssued { credential_type: String },
    CredentialRevoked { credential_hash: ActionHash },
    IdentityVerified { level: u8 },

    // Trust Events
    ReputationChanged { agent: AgentPubKey, delta: f64 },
    TrustRelationshipFormed { from: AgentPubKey, to: AgentPubKey },
    CartelDetected { members: Vec<AgentPubKey> },

    // Governance Events
    ProposalCreated { proposal_id: String },
    VoteCast { proposal_id: String, voter: AgentPubKey },
    ProposalResolved { proposal_id: String, outcome: Outcome },

    // Economic Events
    TransactionCompleted { tx_id: String, amount: u64 },
    ContractExecuted { contract_id: String },
    DisputeRaised { dispute_id: String },

    // Security Events
    ThreatDetected { threat_id: String, severity: u8 },
    AgentQuarantined { agent: AgentPubKey },
    IncidentResolved { incident_id: String },

    // Custom Events
    Custom { category: String, data: Value },
}
```

### Protocol 3: Trust-Weighted Operations

All significant operations in the ecosystem are weighted by MATL trust scores:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     TRUST-WEIGHTED OPERATIONS                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Operation                  Trust Weighting Formula                     │
│  ───────────────────────────────────────────────────────────────────── │
│                                                                         │
│  Voting Power:              vote_weight = base × trust_score^α         │
│                             where α ∈ [0.5, 2.0] per governance space  │
│                                                                         │
│  Marketplace Priority:      priority = trust_score × recency_decay     │
│                                                                         │
│  Oracle Weight:             weight = stake × trust_score × accuracy    │
│                                                                         │
│  Arbitrator Selection:      P(selected) ∝ expertise × trust_score      │
│                                                                         │
│  Content Visibility:        visibility = trust_score × engagement      │
│                                                                         │
│  Emergency Response:        authority = role_weight × trust_score      │
│                                                                         │
│  Credential Issuance:       min_trust_required = f(credential_level)   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Architecture

### Primary Data Flows

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MAJOR DATA FLOWS                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  IDENTITY FLOW                                                          │
│  Attest ─────► All hApps                                               │
│  │             (identity verification, credential checking)             │
│  └────► MATL   (identity → trust binding)                              │
│                                                                         │
│  TRUST FLOW                                                             │
│  All hApps ───► MATL ───► All hApps                                    │
│  (reputation events)     (trust scores)                                 │
│                                                                         │
│  GOVERNANCE FLOW                                                        │
│  Agora ◄───► Treasury (controlled spending)                            │
│  │     ◄───► Commons (resource governance)                             │
│  │     ◄───► Diplomat (treaty ratification)                            │
│  └────────► All hApps (governance directives)                          │
│                                                                         │
│  ECONOMIC FLOW                                                          │
│  Marketplace ◄───► Treasury (escrow, payments)                         │
│  │           ◄───► Oracle (price feeds)                                │
│  │           ◄───► Arbiter (dispute resolution)                        │
│  └───────────► SupplyChain (physical goods)                            │
│                                                                         │
│  KNOWLEDGE FLOW                                                         │
│  Praxis ◄───► Chronicle (knowledge archival)                           │
│  │      ◄───► Guild (professional credentials)                         │
│  │      ◄───► Collab (skill verification)                              │
│  └──────► Attest (credential issuance)                                 │
│                                                                         │
│  SECURITY FLOW                                                          │
│  Sentinel ◄───► All hApps (threat events)                              │
│  │        ───► MATL (reputation impacts)                               │
│  │        ───► Arbiter (violation cases)                               │
│  └────────► Chronicle (incident archive)                               │
│                                                                         │
│  ARCHIVAL FLOW                                                          │
│  All hApps (M2+) ───► Chronicle                                        │
│  │                    │                                                 │
│  │                    ├───► IPFS (M1-M2)                               │
│  │                    └───► Arweave (M3)                               │
│  └────────────────────► Loom (narrative extraction)                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Sovereignty Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DATA SOVEREIGNTY LAYERS                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LAYER 1: Agent-Local Data                                             │
│  └─ Stored: Agent's local source chain                                 │
│  └─ Access: Only agent (+ explicit grants)                             │
│  └─ Examples: Private keys, personal notes, health records (default)   │
│                                                                         │
│  LAYER 2: Shared DHT Data                                              │
│  └─ Stored: Holochain DHT (distributed)                                │
│  └─ Access: hApp-specific rules                                        │
│  └─ Examples: Marketplace listings, public credentials, votes          │
│                                                                         │
│  LAYER 3: Encrypted Shared Data                                        │
│  └─ Stored: DHT with encryption                                        │
│  └─ Access: Key holders only                                           │
│  └─ Examples: Private messages, health records (shared), contracts     │
│                                                                         │
│  LAYER 4: Public Archive Data                                          │
│  └─ Stored: Chronicle → IPFS/Arweave                                   │
│  └─ Access: Public (after classification)                              │
│  └─ Examples: Governance decisions, treaty texts, cultural archives    │
│                                                                         │
│  LAYER 5: Collective Intelligence Data                                 │
│  └─ Stored: Aggregated, privacy-preserved                              │
│  └─ Access: Statistical only (differential privacy)                    │
│  └─ Examples: Pulse sentiment, Metabolism metrics, ML models           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Trust Architecture

### MATL Deep Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MATL TRUST ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                      ┌─────────────────────┐                           │
│                      │  COMPOSITE TRUST    │                           │
│                      │     SCORE           │                           │
│                      └──────────┬──────────┘                           │
│                                 │                                       │
│          ┌──────────────────────┼──────────────────────┐               │
│          │                      │                      │               │
│          ▼                      ▼                      ▼               │
│  ┌───────────────┐     ┌───────────────┐     ┌───────────────┐        │
│  │ Transaction   │     │   Social      │     │   Credential  │        │
│  │ Reputation    │     │   Vouching    │     │   Validity    │        │
│  └───────┬───────┘     └───────┬───────┘     └───────┬───────┘        │
│          │                     │                     │                 │
│          ▼                     ▼                     ▼                 │
│  ┌───────────────┐     ┌───────────────┐     ┌───────────────┐        │
│  │ • Completion  │     │ • Vouch count │     │ • Credential  │        │
│  │   rate        │     │ • Vouch       │     │   level       │        │
│  │ • Dispute     │     │   quality     │     │ • Issuer      │        │
│  │   rate        │     │ • Network     │     │   reputation  │        │
│  │ • Value       │     │   position    │     │ • Age         │        │
│  │   transacted  │     │               │     │               │        │
│  └───────────────┘     └───────────────┘     └───────────────┘        │
│                                                                         │
│  ──────────────────────────────────────────────────────────────────────│
│                                                                         │
│  CONTEXT WEIGHTING                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Trust scores are context-specific:                             │   │
│  │                                                                  │   │
│  │  Context: Marketplace                                           │   │
│  │  └─ Weight: transaction_rep=0.6, social=0.2, credential=0.2    │   │
│  │                                                                  │   │
│  │  Context: Governance                                            │   │
│  │  └─ Weight: transaction_rep=0.2, social=0.4, credential=0.4    │   │
│  │                                                                  │   │
│  │  Context: Health                                                │   │
│  │  └─ Weight: transaction_rep=0.1, social=0.2, credential=0.7    │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ──────────────────────────────────────────────────────────────────────│
│                                                                         │
│  BYZANTINE RESISTANCE                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Cartel Detection:                                              │   │
│  │  • Graph analysis for unusual clustering                        │   │
│  │  • Temporal correlation of vouches                              │   │
│  │  • Value flow analysis                                          │   │
│  │                                                                  │   │
│  │  Sybil Resistance:                                              │   │
│  │  • Progressive identity verification                            │   │
│  │  • Stake-weighted operations                                    │   │
│  │  • Social graph constraints                                     │   │
│  │                                                                  │   │
│  │  Reputation Manipulation:                                       │   │
│  │  • Rate limiting on reputation changes                          │   │
│  │  • Diminishing returns from same source                         │   │
│  │  • Historical pattern analysis                                  │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Trust Propagation Rules

```rust
/// Trust decays over graph distance
fn propagated_trust(direct_trust: f64, hops: u32) -> f64 {
    direct_trust * DECAY_FACTOR.powf(hops as f64)
}

/// Maximum propagation depth
const MAX_TRUST_HOPS: u32 = 4;

/// Decay per hop
const DECAY_FACTOR: f64 = 0.7;

/// Example:
/// A trusts B at 0.9
/// B trusts C at 0.8
/// A's propagated trust in C = 0.9 * 0.8 * 0.7 = 0.504
```

---

## Event Bus Architecture

### Subscription Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    EVENT SUBSCRIPTION MODEL                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Standard Subscriptions (Pre-configured):                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Sentinel subscribes to:                                        │   │
│  │    • All SecurityEvent from all hApps                           │   │
│  │    • TrustEvent.CartelDetected from MATL                        │   │
│  │    • DisputeRaised from Arbiter                                 │   │
│  │                                                                  │   │
│  │  Chronicle subscribes to:                                       │   │
│  │    • All events with Materiality >= M2                          │   │
│  │    • ProposalResolved from Agora                                │   │
│  │    • ContractExecuted from Marketplace                          │   │
│  │                                                                  │   │
│  │  MATL subscribes to:                                            │   │
│  │    • TransactionCompleted from Marketplace                      │   │
│  │    • DisputeResolved from Arbiter                               │   │
│  │    • CredentialIssued from Attest                               │   │
│  │    • ProjectCompleted from Collab                               │   │
│  │                                                                  │   │
│  │  Treasury subscribes to:                                        │   │
│  │    • ProposalResolved.SpendingApproved from Agora               │   │
│  │    • MilestoneCompleted from Collab                             │   │
│  │    • ClaimApproved from Mutual                                  │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Custom Subscriptions (Agent-defined):                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Agents can subscribe to:                                       │   │
│  │    • Events involving them                                      │   │
│  │    • Events in their communities                                │   │
│  │    • Events matching custom filters                             │   │
│  │                                                                  │   │
│  │  Example filter:                                                │   │
│  │  {                                                              │   │
│  │    "source_happ": "marketplace",                                │   │
│  │    "event_type": "ListingCreated",                              │   │
│  │    "filter": {                                                  │   │
│  │      "category": "tools",                                       │   │
│  │      "location.within_km": 50                                   │   │
│  │    }                                                            │   │
│  │  }                                                              │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## hApp Cluster Analysis

The 32 hApps naturally form functional clusters:

### Cluster Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        hApp CLUSTERS                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │                    IDENTITY CLUSTER                           │     │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │     │
│  │  │ Attest  │──│  MATL   │──│  Weave  │──│ Resonance│          │     │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘          │     │
│  │  identity     trust        relationships  values              │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │                  GOVERNANCE CLUSTER                           │     │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │     │
│  │  │  Agora  │──│ Arbiter │──│ Diplomat│──│Membrane │          │     │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘          │     │
│  │  decisions    disputes     federation   boundaries            │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │                   ECONOMIC CLUSTER                            │     │
│  │  ┌──────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐         │     │
│  │  │Marketplace│─│Treasury │──│ Mutual  │──│  Cycle  │         │     │
│  │  └──────────┘  └─────────┘  └─────────┘  └─────────┘         │     │
│  │  exchange      collective   insurance    circulation          │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │                  KNOWLEDGE CLUSTER                            │     │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │     │
│  │  │ Praxis  │──│Chronicle│──│  Loom   │──│  Guild  │          │     │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘          │     │
│  │  learning     archive      narrative    profession            │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │                 COORDINATION CLUSTER                          │     │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │     │
│  │  │ Collab  │──│  Seed   │──│  Nexus  │──│ Beacon  │          │     │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘          │     │
│  │  projects     incubation   events       emergency             │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │                   COMMONS CLUSTER                             │     │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │     │
│  │  │ Commons │──│ Anchor  │──│HealthV  │──│SupplyCh │          │     │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘          │     │
│  │  shared res   physical     health       logistics             │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │                 INTELLIGENCE CLUSTER                          │     │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │     │
│  │  │ Sentinel│──│  Pulse  │──│Metabolism──│Emergence│          │     │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘          │     │
│  │  security     sentiment    health       patterns              │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │                INFRASTRUCTURE CLUSTER                         │     │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │     │
│  │  │ Bridge  │──│ Oracle  │──│  0TML   │──│ Forecast│          │     │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘          │     │
│  │  routing      truth        learning     prediction            │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │                  LIFECYCLE CLUSTER                            │     │
│  │  ┌─────────┐  ┌─────────┐                                     │     │
│  │  │ Legacy  │──│ Civitas │                                     │     │
│  │  └─────────┘  └─────────┘                                     │     │
│  │  succession   citizenship                                      │     │
│  └───────────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Cluster Integration Matrix

| From/To | Identity | Governance | Economic | Knowledge | Coordination | Commons | Intelligence | Infra |
|---------|----------|------------|----------|-----------|--------------|---------|--------------|-------|
| **Identity** | - | Auth, Trust | Trust | Credentials | Team auth | Access | Analysis | Trust |
| **Governance** | Membership | - | Budgets | Policy | Approval | Rules | Input | Ratify |
| **Economic** | Verification | Disputes | - | Assets | Funding | Resources | Metrics | Prices |
| **Knowledge** | Attribution | Evidence | IP | - | Skills | Docs | Data | Truth |
| **Coordination** | Teams | Decisions | Resources | Expertise | - | Assets | Status | External |
| **Commons** | Access | Governance | Value | Info | Coordinate | - | Monitor | Location |
| **Intelligence** | Anomalies | Alerts | Fraud | Patterns | Threats | Anomalies | - | Feeds |
| **Infrastructure** | Auth | Routing | Prices | Queries | Events | Tracking | Sources | - |

---

## Integration Patterns

### Pattern 1: Credential-Gated Operations

```rust
/// Many operations require specific credentials
/// This pattern shows how to integrate Attest with any hApp

pub fn credential_gated_operation<T>(
    agent: AgentPubKey,
    required_credential: CredentialType,
    operation: impl FnOnce() -> ExternResult<T>,
) -> ExternResult<T> {
    // Query Attest via Bridge
    let has_credential = bridge_call::<bool>(
        HAppId::Attest,
        "verify_credential",
        VerifyCredentialRequest {
            agent: agent.clone(),
            credential_type: required_credential,
        },
    )?;

    if !has_credential {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Missing required credential".into()
        )));
    }

    // Check trust score meets minimum
    let trust_score = bridge_call::<f64>(
        HAppId::MATL,
        "get_trust_score",
        TrustScoreRequest {
            agent: agent.clone(),
            context: TrustContext::current_happ(),
        },
    )?;

    if trust_score < MINIMUM_TRUST_THRESHOLD {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Insufficient trust score".into()
        )));
    }

    // Execute operation
    let result = operation()?;

    // Report to MATL
    bridge_call::<()>(
        HAppId::MATL,
        "report_interaction",
        InteractionReport {
            agent,
            interaction_type: InteractionType::CredentialedOperation,
            success: true,
        },
    )?;

    Ok(result)
}
```

### Pattern 2: Dispute-Enabled Transactions

```rust
/// Economic transactions that can be disputed
/// Integrates Marketplace/Treasury with Arbiter

pub struct DisputeEnabledTransaction {
    pub transaction_id: String,
    pub parties: Vec<AgentPubKey>,
    pub amount: u64,
    pub escrow_until: Timestamp,
    pub dispute_window: Duration,
    pub arbiter_pool: Option<ActionHash>,
}

impl DisputeEnabledTransaction {
    pub fn execute(&self) -> ExternResult<TransactionResult> {
        // Hold funds in escrow via Treasury
        let escrow_hash = bridge_call::<ActionHash>(
            HAppId::Treasury,
            "create_escrow",
            EscrowRequest {
                amount: self.amount,
                parties: self.parties.clone(),
                release_conditions: vec![
                    ReleaseCondition::TimeElapsed(self.dispute_window),
                    ReleaseCondition::AllPartiesApprove,
                ],
            },
        )?;

        // Register with Arbiter for potential dispute
        bridge_call::<()>(
            HAppId::Arbiter,
            "register_disputable",
            DisputableRegistration {
                transaction_id: self.transaction_id.clone(),
                parties: self.parties.clone(),
                escrow_hash: escrow_hash.clone(),
                dispute_window: self.dispute_window,
                arbiter_pool: self.arbiter_pool.clone(),
            },
        )?;

        // Archive transaction
        bridge_call::<()>(
            HAppId::Chronicle,
            "archive",
            ArchiveRequest {
                content: Content::Transaction(self.clone()),
                materiality: MaterialityLevel::M1,
                epistemic: EpistemicClassification::new(E3, N2, M1),
            },
        )?;

        Ok(TransactionResult {
            transaction_id: self.transaction_id.clone(),
            escrow_hash,
            status: TransactionStatus::InEscrow,
        })
    }

    pub fn complete(&self) -> ExternResult<()> {
        // Release escrow
        bridge_call::<()>(
            HAppId::Treasury,
            "release_escrow",
            ReleaseRequest {
                transaction_id: self.transaction_id.clone(),
            },
        )?;

        // Report positive interaction to MATL
        for party in &self.parties {
            bridge_call::<()>(
                HAppId::MATL,
                "report_positive",
                PositiveInteraction {
                    agent: party.clone(),
                    interaction_type: InteractionType::SuccessfulTransaction,
                    value: self.amount,
                },
            )?;
        }

        Ok(())
    }
}
```

### Pattern 3: Governance-Controlled Resources

```rust
/// Resources controlled by governance decisions
/// Integrates Agora with Treasury/Commons

pub struct GovernanceControlledResource {
    pub resource_id: String,
    pub governance_space: ActionHash,
    pub resource_type: ResourceType,
    pub current_policy: Policy,
}

impl GovernanceControlledResource {
    pub fn propose_change(&self, proposer: AgentPubKey, new_policy: Policy) -> ExternResult<ActionHash> {
        // Create proposal in Agora
        let proposal_hash = bridge_call::<ActionHash>(
            HAppId::Agora,
            "create_proposal",
            ProposalRequest {
                governance_space: self.governance_space.clone(),
                proposer,
                proposal_type: ProposalType::PolicyChange {
                    resource_id: self.resource_id.clone(),
                    current_policy: self.current_policy.clone(),
                    new_policy: new_policy.clone(),
                },
                voting_mechanism: VotingMechanism::ConvictionVoting,
            },
        )?;

        Ok(proposal_hash)
    }

    pub fn on_proposal_passed(&self, new_policy: Policy) -> ExternResult<()> {
        // Update resource policy
        match self.resource_type {
            ResourceType::Treasury => {
                bridge_call::<()>(
                    HAppId::Treasury,
                    "update_policy",
                    PolicyUpdate {
                        resource_id: self.resource_id.clone(),
                        new_policy: new_policy.clone(),
                    },
                )?;
            }
            ResourceType::Commons => {
                bridge_call::<()>(
                    HAppId::Commons,
                    "update_policy",
                    PolicyUpdate {
                        resource_id: self.resource_id.clone(),
                        new_policy: new_policy.clone(),
                    },
                )?;
            }
            // ... other resource types
        }

        // Archive the change
        bridge_call::<()>(
            HAppId::Chronicle,
            "archive",
            ArchiveRequest {
                content: Content::PolicyChange {
                    resource_id: self.resource_id.clone(),
                    old_policy: self.current_policy.clone(),
                    new_policy,
                },
                materiality: MaterialityLevel::M2,
                epistemic: EpistemicClassification::new(E3, N3, M2),
            },
        )?;

        Ok(())
    }
}
```

### Pattern 4: Oracle-Verified Claims

```rust
/// Claims verified by Oracle network
/// Integrates Oracle with any claiming hApp

pub struct OracleVerifiedClaim {
    pub claim_id: String,
    pub claim_type: ClaimType,
    pub expected_value: Value,
    pub oracle_feed: ActionHash,
    pub tolerance: Option<f64>,
}

impl OracleVerifiedClaim {
    pub fn verify(&self) -> ExternResult<VerificationResult> {
        // Query Oracle for current value
        let oracle_response = bridge_call::<OracleResponse>(
            HAppId::Oracle,
            "query_feed",
            FeedQuery {
                feed_hash: self.oracle_feed.clone(),
                timestamp: None, // Current value
            },
        )?;

        // Verify consensus threshold
        if oracle_response.consensus_strength < MINIMUM_CONSENSUS {
            return Ok(VerificationResult::InsufficientConsensus);
        }

        // Compare values
        let matches = match (&self.expected_value, &oracle_response.value) {
            (Value::Numeric(expected), Value::Numeric(actual)) => {
                if let Some(tolerance) = self.tolerance {
                    (expected - actual).abs() <= tolerance
                } else {
                    expected == actual
                }
            }
            (expected, actual) => expected == actual,
        };

        // Classify epistemically
        let epistemic = EpistemicClassification::new(
            oracle_response.empirical_level,
            N0, // Descriptive
            self.claim_type.materiality_level(),
        );

        Ok(VerificationResult::Verified {
            matches,
            oracle_value: oracle_response.value,
            consensus: oracle_response.consensus_strength,
            epistemic,
        })
    }
}
```

### Pattern 5: Emergency Response Coordination

```rust
/// Emergency response across multiple hApps
/// Integrates Beacon with Sentinel, HealthVault, Treasury

pub struct EmergencyResponse {
    pub emergency_id: String,
    pub emergency_type: EmergencyType,
    pub severity: Severity,
    pub affected_area: GeographicScope,
}

impl EmergencyResponse {
    pub fn declare(&self, declarer: AgentPubKey) -> ExternResult<ActionHash> {
        // Verify declarer has authority
        let authorized = bridge_call::<bool>(
            HAppId::Attest,
            "verify_credential",
            VerifyCredentialRequest {
                agent: declarer.clone(),
                credential_type: CredentialType::EmergencyDeclarer,
            },
        )?;

        if !authorized {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Not authorized to declare emergency".into()
            )));
        }

        // Create emergency in Beacon
        let emergency_hash = bridge_call::<ActionHash>(
            HAppId::Beacon,
            "declare_emergency",
            EmergencyDeclaration {
                emergency_type: self.emergency_type.clone(),
                severity: self.severity.clone(),
                affected_area: self.affected_area.clone(),
                declarer,
            },
        )?;

        // Alert Sentinel
        bridge_call::<()>(
            HAppId::Sentinel,
            "emergency_alert",
            EmergencyAlert {
                emergency_hash: emergency_hash.clone(),
                severity: self.severity.clone(),
            },
        )?;

        // If health emergency, alert HealthVault
        if matches!(self.emergency_type, EmergencyType::Health { .. }) {
            bridge_call::<()>(
                HAppId::HealthVault,
                "emergency_mode",
                EmergencyMode {
                    emergency_hash: emergency_hash.clone(),
                    affected_area: self.affected_area.clone(),
                },
            )?;
        }

        // Unlock emergency funds
        if self.severity >= Severity::High {
            bridge_call::<()>(
                HAppId::Treasury,
                "unlock_emergency_funds",
                EmergencyFundRequest {
                    emergency_hash: emergency_hash.clone(),
                    max_amount: self.severity.max_emergency_funds(),
                },
            )?;
        }

        // Archive
        bridge_call::<()>(
            HAppId::Chronicle,
            "archive",
            ArchiveRequest {
                content: Content::EmergencyDeclaration(self.clone()),
                materiality: MaterialityLevel::M2,
                epistemic: EpistemicClassification::new(E2, N3, M2),
            },
        )?;

        Ok(emergency_hash)
    }
}
```

---

## Deployment Architecture

### Phase-Based Deployment

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT PHASES                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PHASE 0: FOUNDATION (Months 1-3)                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Deploy Infrastructure Spine:                                   │   │
│  │  • Bridge Protocol (cross-hApp routing)                         │   │
│  │  • MATL (trust layer)                                           │   │
│  │  • Attest (identity foundation)                                 │   │
│  │  • Sentinel (security monitoring)                               │   │
│  │  • Chronicle (archival)                                         │   │
│  │                                                                  │   │
│  │  Dependencies: None (foundation layer)                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  PHASE 1: CORE OPERATIONS (Months 4-6)                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Deploy Core Functionality:                                     │   │
│  │  • Agora (governance)                                           │   │
│  │  • Arbiter (disputes)                                           │   │
│  │  • Oracle (external data)                                       │   │
│  │  • Marketplace (exchange)                                       │   │
│  │  • Treasury (collective funds)                                  │   │
│  │                                                                  │   │
│  │  Dependencies: Phase 0 complete                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  PHASE 2: DOMAIN SPECIFIC (Months 7-12)                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Deploy Domain hApps:                                           │   │
│  │  • HealthVault, Praxis, SupplyChain                            │   │
│  │  • Collab, Seed, Guild                                         │   │
│  │  • Commons, Mutual, Beacon                                      │   │
│  │  • Nexus, Anchor, Cycle                                         │   │
│  │                                                                  │   │
│  │  Dependencies: Phase 1 complete                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  PHASE 3: ADVANCED (Months 13-18)                                      │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Deploy Advanced hApps:                                         │   │
│  │  • Diplomat, Membrane (federation)                              │   │
│  │  • Forecast (prediction markets)                                │   │
│  │  • Legacy, Civitas (lifecycle)                                  │   │
│  │  • Pulse, Weave, Loom (social)                                 │   │
│  │                                                                  │   │
│  │  Dependencies: Phase 2 complete                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  PHASE 4: COLLECTIVE INTELLIGENCE (Months 19-24)                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Deploy Meta-Intelligence:                                      │   │
│  │  • 0TML (federated learning)                                    │   │
│  │  • Metabolism (system health)                                   │   │
│  │  • Emergence (pattern detection)                                │   │
│  │  • Resonance (value alignment)                                  │   │
│  │                                                                  │   │
│  │  Dependencies: All prior phases + sufficient data              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Dependency Graph

```
                            Phase 0
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
         Bridge ───────────► MATL ◄────────── Attest
            │                  │                  │
            │                  │                  │
            ▼                  ▼                  ▼
        Sentinel ◄───────── Chronicle ◄──────────┘
                               │
                         Phase 1 │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
          Agora ◄──────────► Arbiter ◄───────► Oracle
            │                  │                  │
            ▼                  ▼                  ▼
        Treasury ◄────────► Marketplace ◄────────┘
                               │
                         Phase 2 │
     ┌──────────┬──────────┬───┴───┬──────────┬──────────┐
     ▼          ▼          ▼       ▼          ▼          ▼
 HealthVault  Praxis  SupplyChain Collab    Commons   Mutual
     │          │          │       │          │          │
     └──────────┴──────────┴───────┴──────────┴──────────┘
                               │
                         Phase 3 │
     ┌──────────┬──────────┬───┴───┬──────────┬──────────┐
     ▼          ▼          ▼       ▼          ▼          ▼
  Diplomat  Membrane   Forecast  Legacy    Civitas    Guild
                               │
                         Phase 4 │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
          0TML ────────► Metabolism ◄──────── Emergence
                               │
                               ▼
                          Resonance
```

---

## Security Architecture

### Defense in Depth

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SECURITY LAYERS                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LAYER 1: CRYPTOGRAPHIC FOUNDATION                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Ed25519 signatures for all entries                           │   │
│  │  • X25519 for encryption                                        │   │
│  │  • Hash chains for integrity                                    │   │
│  │  • Source chain = unforgeable history                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  LAYER 2: VALIDATION                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Integrity zomes define entry constraints                     │   │
│  │  • Every entry validated by multiple peers                      │   │
│  │  • Invalid entries rejected at DHT level                        │   │
│  │  • Custom validators for business logic                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  LAYER 3: TRUST WEIGHTING                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • All operations weighted by MATL trust                        │   │
│  │  • Low-trust agents have reduced influence                      │   │
│  │  • Progressive trust requirements for sensitive ops             │   │
│  │  • Cartel detection limits coordinated attacks                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  LAYER 4: MONITORING                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Sentinel monitors all hApps                                  │   │
│  │  • Anomaly detection (statistical + ML)                         │   │
│  │  • Pattern matching for known attacks                           │   │
│  │  • Cross-hApp correlation                                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  LAYER 5: RESPONSE                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Graduated sanctions (warning → restriction → quarantine)     │   │
│  │  • Coordinated response across hApps                            │   │
│  │  • Evidence preservation for Arbiter                            │   │
│  │  • Recovery procedures                                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  LAYER 6: GOVERNANCE                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Security policies set via Agora                              │   │
│  │  • Community-controlled threat response                         │   │
│  │  • Constitutional limits on security powers                     │   │
│  │  • Transparency requirements                                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Byzantine Resistance Target

The system is designed to maintain integrity with up to 45% adversarial nodes:

```rust
/// Byzantine Fault Tolerance Parameters

pub const BYZANTINE_THRESHOLD: f64 = 0.45;

/// Voting requires supermajority
pub const VOTING_THRESHOLD: f64 = 0.67;

/// Oracle consensus requires strong agreement
pub const ORACLE_CONSENSUS_THRESHOLD: f64 = 0.80;

/// Reputation changes require multiple confirmations
pub const REPUTATION_CONFIRMATION_COUNT: u32 = 5;

/// Quarantine requires evidence from multiple sources
pub const QUARANTINE_EVIDENCE_THRESHOLD: u32 = 3;
```

---

## Performance & Scaling

### Scaling Strategies

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SCALING ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  HORIZONTAL SCALING (Network Growth)                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • DHT automatically distributes load                           │   │
│  │  • More nodes = more storage + bandwidth                        │   │
│  │  • Sharding by DNA variant for mega-scale                       │   │
│  │  • Community-level DNA instances                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  VERTICAL SCALING (Node Performance)                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Local caching of frequently accessed data                    │   │
│  │  • Async processing for non-critical operations                 │   │
│  │  • Batch validation for high-throughput                         │   │
│  │  • Wasmer 3 WASM optimization                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  FEDERATION SCALING (Multi-Community)                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Each community runs own DNA instances                        │   │
│  │  • Diplomat enables cross-community interaction                 │   │
│  │  • Bridge aggregates across communities                         │   │
│  │  • Local-first, global-when-needed                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ARCHIVAL SCALING (Long-term Storage)                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • Chronicle prunes low-materiality data                        │   │
│  │  • IPFS for medium-term (M1-M2)                                 │   │
│  │  • Arweave for permanent (M3)                                   │   │
│  │  • Differential privacy for aggregates only                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Transaction latency | < 2s | End-to-end for simple operations |
| DHT propagation | < 10s | 95th percentile |
| Cross-hApp query | < 500ms | Via Bridge |
| Trust score calculation | < 100ms | Cached, async update |
| Validation throughput | > 1000 TPS | Per-hApp |
| Archive retrieval | < 5s | From IPFS/Arweave |

---

## Migration & Evolution

### Schema Evolution

```rust
/// All entries support schema versioning

pub struct VersionedEntry<T> {
    pub schema_version: u32,
    pub data: T,
    pub migrated_from: Option<ActionHash>,
}

/// Migration handlers registered per entry type
pub trait Migrate {
    fn migrate(old_version: u32, old_data: SerializedBytes) -> ExternResult<Self>
    where
        Self: Sized;
}

/// Example: Treasury entry migration
impl Migrate for Treasury {
    fn migrate(old_version: u32, old_data: SerializedBytes) -> ExternResult<Self> {
        match old_version {
            1 => {
                let v1: TreasuryV1 = old_data.try_into()?;
                Ok(Treasury {
                    treasury_id: v1.treasury_id,
                    governance_space: v1.governance_space,
                    balance: v1.balance,
                    signers: v1.signers,
                    threshold: v1.threshold,
                    budget_allocations: vec![], // New in v2
                })
            }
            _ => Err(wasm_error!(WasmErrorInner::Guest(
                format!("Unknown schema version: {}", old_version)
            ))),
        }
    }
}
```

### Upgrade Governance

All significant upgrades require governance approval:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    UPGRADE FLOW                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Proposal Submission                                                │
│     └─ Developer submits upgrade proposal to Agora                     │
│     └─ Includes: Code hash, changelog, migration plan, test results   │
│                                                                         │
│  2. Review Period                                                      │
│     └─ Community reviews proposed changes                              │
│     └─ Security audit by Guild-certified auditors                     │
│     └─ Testnet deployment for validation                               │
│                                                                         │
│  3. Voting                                                             │
│     └─ Conviction voting with 2-week minimum                           │
│     └─ Supermajority (67%) required for core changes                   │
│     └─ Simple majority for minor updates                               │
│                                                                         │
│  4. Staged Rollout                                                     │
│     └─ Canary deployment (5% of nodes)                                 │
│     └─ Gradual expansion over 1 week                                   │
│     └─ Automatic rollback on anomaly detection                         │
│                                                                         │
│  5. Migration                                                          │
│     └─ Data migration runs asynchronously                              │
│     └─ Old and new versions coexist during transition                  │
│     └─ Chronicle preserves all historical data                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Conclusion

The Mycelix Civilizational OS represents a complete infrastructure for human coordination. By following the architectural patterns in this blueprint, the 32 hApps work together as a coherent whole while maintaining individual sovereignty and composability.

### Key Takeaways

1. **Integration Spine**: All hApps integrate with the 5 core services (Attest, MATL, Bridge, Chronicle, Sentinel)

2. **Trust-First**: Every operation is trust-weighted, providing natural resistance to adversarial behavior

3. **Epistemic Humility**: All claims are classified, enabling appropriate retention and citation

4. **Phased Deployment**: Build foundations before advanced capabilities

5. **Federation-Ready**: Each community can run independently while interoperating via Diplomat

6. **Evolution-Capable**: Schema versioning and governance-controlled upgrades enable long-term adaptation

---

*"Infrastructure for civilization must be as robust as the social fabric it supports, as flexible as human creativity demands, and as trustworthy as the relationships it enables."*

---

## Appendix: Quick Reference

### hApp List by Tier

**Tier 0 - Infrastructure Spine (5)**
1. Bridge - Cross-hApp routing
2. MATL - Trust layer
3. Attest - Identity
4. Sentinel - Security
5. Chronicle - Archival

**Tier 1 - Core Operations (6)**
6. Agora - Governance
7. Arbiter - Disputes
8. Oracle - External data
9. Marketplace - Exchange
10. Treasury - Collective funds
11. 0TML - Federated learning

**Tier 2 - Domain (12)**
12. HealthVault - Health records
13. Praxis - Education
14. SupplyChain - Logistics
15. Collab - Projects
16. Seed - Incubation
17. Guild - Professional
18. Commons - Shared resources
19. Mutual - Insurance
20. Beacon - Emergency
21. Nexus - Events
22. Anchor - Physical
23. Cycle - Circulation

**Tier 3 - Advanced (7)**
24. Diplomat - Federation
25. Membrane - Boundaries
26. Forecast - Prediction
27. Legacy - Succession
28. Civitas - Citizenship
29. Pulse - Sentiment
30. Weave - Relationships

**Tier 4 - Meta-Intelligence (4)**
31. Loom - Narrative
32. Metabolism - System health
33. Emergence - Pattern detection
34. Resonance - Value alignment

---

*Document Version: 1.0*
*Last Updated: 2025*
*Status: Living Architecture Document*
