# Mycelix-Mail: Revolutionary Integration Architecture

## Vision

Transform email from a spam-ridden broadcast medium into **Epistemic Mail** - a trust-verified, consent-based, AI-enhanced communication system where claims are verifiable, trust is visible, and your attention is sovereign.

## Integration Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MYCELIX-MAIL UNIFIED STACK                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      React Frontend (UI Layer)                       │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────────┐  │   │
│  │  │ Trust Graph │ │ Claim       │ │ AI Insights │ │ Consent       │  │   │
│  │  │ Visualizer  │ │ Verifier    │ │ Panel       │ │ Manager       │  │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └───────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Axum Backend (API Gateway)                        │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────────┐  │   │
│  │  │ /api/claims │ │ /api/trust  │ │ /api/ai     │ │ /api/consent  │  │   │
│  │  │ (0TML)      │ │ (Graphs)    │ │ (Symthaea)  │ │ (Intros)      │  │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └───────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │                    │                    │                       │
│           ▼                    ▼                    ▼                       │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────────────────┐  │
│  │     0TML       │   │   Holochain    │   │       Symthaea             │  │
│  │  ZK Proofs &   │   │   DHT + DNA    │   │    Local-First AI         │  │
│  │  Credentials   │   │   Trust Graph  │   │   (Consciousness-Guided)  │  │
│  └────────────────┘   └────────────────┘   └────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. 0TML Integration: Verifiable Claims

### DID Upgrade Path

**Current**: `did:key:z6Mk...` (simple Ed25519 key)
**Target**: `did:mycelix:z6Mk...` (full identity ecosystem)

```rust
// Integration point in backend-rs
pub struct VerifiableIdentity {
    pub did: String,                           // did:mycelix:z6Mk...
    pub assurance_level: AssuranceLevel,       // E0-E4
    pub credentials: Vec<VerifiableCredential>,
    pub proof_of_humanity: Option<ProofOfHumanity>,
}

#[derive(Clone, Debug)]
pub enum AssuranceLevel {
    E0_Anonymous,        // No verification, $0 attack cost
    E1_VerifiedEmail,    // Email verified, $100 attack cost
    E2_GitcoinPassport,  // 10+ stamps, $1,000 attack cost
    E3_MultiFactor,      // Hardware key, $100,000 attack cost
    E4_Constitutional,   // Biometric recovery, $10M attack cost
}
```

### Epistemic Tiers → Verifiable Claims

```rust
/// Email claim that can be cryptographically verified
pub enum VerifiableClaim {
    /// Tier 1: "I am X" - Identity claim with optional proof
    Identity {
        claim: String,                    // "I am Dr. Smith"
        credential: Option<VerifiableCredential>,
        proof: Option<ZKProof>,
    },

    /// Tier 2: "I work at X" - Employment/affiliation
    Affiliation {
        organization: String,
        role: Option<String>,
        credential: Option<VerifiableCredential>,
        valid_until: Option<DateTime<Utc>>,
    },

    /// Tier 3: "I have credential X" - Education, certification
    Credential {
        credential_type: CredentialType,
        issuer_did: String,
        credential: VerifiableCredential,
        zk_proof: Option<ZKProof>,        // Prove without revealing details
    },

    /// Tier 4: "This statement is cryptographically proven"
    CryptographicProof {
        statement: String,
        proof_type: ProofType,            // RISC0, Winterfell, etc.
        proof: Vec<u8>,
        public_inputs: Vec<u8>,
    },
}

pub enum ProofType {
    Risc0ZkVm,           // RISC0 zkSTARK
    WinterfellAir,       // Winterfell AIR prover
    SimplifiedHash,      // SHA-256 commitment (fast)
    Ed25519Signature,    // Basic signature
}
```

### API Endpoints

```
POST /api/claims/verify
  - Verify a claim attached to an email
  - Returns verification result + proof details

POST /api/claims/attach
  - Attach a verifiable claim to outgoing email
  - Generates ZK proof if requested

GET /api/claims/credentials
  - List user's verifiable credentials
  - Filter by type, issuer, validity

POST /api/claims/request
  - Request a credential from another user
  - Initiates verification flow
```

### UI: Claim Verification Badge

```tsx
interface ClaimBadgeProps {
  claim: VerifiableClaim;
  verificationStatus: 'verified' | 'unverified' | 'failed' | 'pending';
}

// Visual indicators
// ✓ Verified (green) - Cryptographic proof valid
// ? Unverified (gray) - No proof attached
// ✗ Failed (red) - Proof invalid or expired
// ⏳ Pending (yellow) - Verification in progress
```

---

## 2. Symthaea Integration: Local-First AI

### Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    Email Processing Pipeline                    │
├────────────────────────────────────────────────────────────────┤
│  1. Email arrives                                               │
│  2. Symthaea processes locally (no cloud API calls)            │
│  3. Results cached in HDC vector space                         │
│  4. UI displays insights                                        │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                     Symthaea Capabilities                       │
├────────────────────────────────────────────────────────────────┤
│  Summarization     │ Thread → 3 bullet points                  │
│  Intent Detection  │ Meeting request, action item, FYI         │
│  Priority Scoring  │ Based on YOUR patterns, not ads           │
│  Draft Suggestions │ Learn your voice, suggest replies         │
│  Causal Reasoning  │ Why this sender? Trust path explanation   │
│  Safety Filtering  │ Algebraic guardrails, no neural network   │
└────────────────────────────────────────────────────────────────┘
```

### Rust Integration Layer

```rust
// In backend-rs/src/services/ai.rs
use symthaea_hlb::SophiaHLB;

pub struct LocalAIService {
    sophia: SophiaHLB,
}

impl LocalAIService {
    pub async fn new() -> Result<Self> {
        // 16,384D HDC space, 1,024 LTC neurons
        let sophia = SophiaHLB::new(16384, 1024).await?;
        Ok(Self { sophia })
    }

    /// Summarize an email thread
    pub async fn summarize_thread(&self, emails: &[Email]) -> Result<ThreadSummary> {
        let context = self.build_thread_context(emails);
        let response = self.sophia.process(&format!(
            "Summarize this email thread in 3 bullet points:\n{}",
            context
        )).await?;

        Ok(ThreadSummary {
            bullets: self.parse_bullets(&response.content),
            confidence: response.confidence,
            consciousness_level: self.sophia.introspect().consciousness_level,
        })
    }

    /// Detect intent of an email
    pub async fn detect_intent(&self, email: &Email) -> Result<EmailIntent> {
        let response = self.sophia.process(&format!(
            "What is the primary intent of this email? Categories: \
             MeetingRequest, ActionItem, FYI, Question, SocialGreeting, \
             Commercial, Spam. Email:\n{}",
            email.body
        )).await?;

        Ok(self.parse_intent(&response.content))
    }

    /// Generate reply suggestions
    pub async fn suggest_replies(&self, email: &Email) -> Result<Vec<ReplySuggestion>> {
        let response = self.sophia.process(&format!(
            "Suggest 3 possible replies to this email, ranging from brief to detailed:\n{}",
            email.body
        )).await?;

        Ok(self.parse_suggestions(&response.content))
    }

    /// Explain trust reasoning
    pub async fn explain_trust(&self,
        sender_did: &str,
        trust_path: &[TrustHop]
    ) -> Result<TrustExplanation> {
        // Use LTC causal reasoning
        let context = format!(
            "Explain why we trust sender {} via path: {:?}",
            sender_did, trust_path
        );
        let response = self.sophia.process(&context).await?;

        Ok(TrustExplanation {
            narrative: response.content,
            causal_factors: self.extract_causal_factors(&response),
        })
    }
}
```

### API Endpoints

```
POST /api/ai/summarize
  - Summarize email thread
  - Returns bullets + confidence

POST /api/ai/intent
  - Detect email intent
  - Returns category + confidence

POST /api/ai/suggest-reply
  - Generate reply suggestions
  - Returns 3 options (brief, medium, detailed)

POST /api/ai/explain-trust
  - Explain trust path in natural language
  - Uses causal reasoning

GET /api/ai/status
  - Symthaea consciousness state
  - Memory stats, safety stats
```

### Privacy Guarantees

```rust
/// All processing happens locally
/// No data leaves the device
/// No cloud API calls for AI features

pub struct PrivacyConfig {
    /// AI runs entirely on local CPU
    pub local_only: bool,  // Always true

    /// No telemetry or logging to external services
    pub no_telemetry: bool,  // Always true

    /// Memory cleared on session end
    pub ephemeral_memory: bool,  // Configurable

    /// Algebraic safety (not neural network)
    pub safety_mode: SafetyMode,  // Algebraic guardrails
}
```

---

## 3. Visual Trust Graphs

### Data Model

```rust
/// Trust relationship in the graph
pub struct TrustEdge {
    pub from_did: String,
    pub to_did: String,
    pub trust_score: f64,           // 0.0 - 1.0
    pub relationship_type: RelationType,
    pub established_at: DateTime<Utc>,
    pub decays_at: Option<DateTime<Utc>>,
    pub attestations: Vec<Attestation>,
}

pub enum RelationType {
    DirectTrust,          // I personally trust you
    Introduction,         // Someone I trust vouched for you
    OrganizationMember,   // Same organization
    CredentialIssuer,     // You issued me a credential
    TransitiveTrust,      // Trust through path
}

/// Path from you to a sender
pub struct TrustPath {
    pub hops: Vec<TrustHop>,
    pub total_decay: f64,           // Cumulative trust decay
    pub path_length: usize,
    pub strongest_link: TrustHop,
    pub weakest_link: TrustHop,
}

pub struct TrustHop {
    pub from: String,               // DID
    pub to: String,                 // DID
    pub from_name: Option<String>,  // Human-readable
    pub to_name: Option<String>,
    pub relationship: RelationType,
    pub trust_score: f64,
    pub reason: Option<String>,     // "Worked together 2019-2022"
}
```

### Visualization Component

```tsx
// TrustGraphVisualizer.tsx
interface TrustGraphProps {
  senderDid: string;
  trustPath: TrustPath;
  fullGraph?: boolean;  // Show entire trust network
}

export function TrustGraphVisualizer({ senderDid, trustPath }: TrustGraphProps) {
  // D3.js or visx force-directed graph
  // Nodes = people (DIDs)
  // Edges = trust relationships
  // Edge thickness = trust strength
  // Color = relationship type
  // Animation = trust flow from sender to you

  return (
    <div className="trust-graph">
      <svg>
        {/* Force-directed graph */}
        {trustPath.hops.map((hop, i) => (
          <TrustHopEdge key={i} hop={hop} />
        ))}
      </svg>

      <div className="trust-explanation">
        <h3>Why you trust {senderName}</h3>
        <ul>
          {trustPath.hops.map((hop, i) => (
            <li key={i}>
              <strong>{hop.from_name}</strong> trusts{' '}
              <strong>{hop.to_name}</strong>
              {hop.reason && <span> ({hop.reason})</span>}
            </li>
          ))}
        </ul>
        <p>
          Trust decays by {((1 - trustPath.total_decay) * 100).toFixed(0)}%
          over {trustPath.path_length} hops
        </p>
      </div>
    </div>
  );
}
```

### API Endpoints

```
GET /api/trust/graph/:did
  - Get trust graph for a DID
  - Returns nodes, edges, paths

GET /api/trust/path/:from_did/:to_did
  - Find trust path between two DIDs
  - Returns shortest path with scores

POST /api/trust/attest
  - Create trust attestation
  - "I trust X because Y"

DELETE /api/trust/revoke/:attestation_id
  - Revoke a trust attestation
```

---

## 4. Consent-Based Communication

### Introduction System

```rust
/// Request to introduce two parties
pub struct Introduction {
    pub id: Uuid,
    pub introducer_did: String,     // Person making intro
    pub introducee_did: String,     // Person being introduced
    pub target_did: String,         // Person receiving intro
    pub message: String,            // "I'd like to introduce..."
    pub trust_stake: f64,           // Trust capital risked
    pub status: IntroStatus,
    pub expires_at: DateTime<Utc>,
}

pub enum IntroStatus {
    Pending,
    Accepted,
    Declined,
    Expired,
}

/// When you introduce someone, you stake your reputation
impl Introduction {
    pub fn calculate_stake(&self, introducer_trust: f64) -> f64 {
        // Stake 10% of your trust capital on this introduction
        introducer_trust * 0.10
    }

    pub fn on_accepted(&self) -> TrustUpdate {
        // Introducee gains trust from target
        // Introducer's stake is returned + bonus
    }

    pub fn on_declined(&self) -> TrustUpdate {
        // Stake is returned (no penalty for declined)
    }

    pub fn on_negative_outcome(&self) -> TrustUpdate {
        // If introducee behaves badly, introducer loses stake
    }
}
```

### Attention Markets

```rust
/// Unknown senders can stake to reach you
pub struct AttentionBid {
    pub sender_did: String,
    pub recipient_did: String,
    pub stake_amount: u64,          // Credits staked
    pub message_preview: String,    // First 100 chars
    pub expires_at: DateTime<Utc>,
}

/// Recipient can set their attention price
pub struct AttentionPolicy {
    pub owner_did: String,
    pub min_stake: u64,              // Minimum to reach inbox
    pub trusted_free: bool,          // Trusted senders bypass stake
    pub introduction_discount: f64,  // Discount for introductions
    pub auto_refund_trusted: bool,   // Refund stake if trust established
}

impl AttentionBid {
    pub fn on_read(&self) -> StakeOutcome {
        // Recipient read the email
        // Stake goes to recipient as payment for attention
    }

    pub fn on_reply(&self) -> StakeOutcome {
        // Recipient replied - relationship established
        // Stake refunded, future emails free
    }

    pub fn on_spam_report(&self) -> StakeOutcome {
        // Recipient marked as spam
        // Stake burned, sender penalized
    }

    pub fn on_expire(&self) -> StakeOutcome {
        // Email never read
        // Stake refunded to sender
    }
}
```

### UI Components

```tsx
// IntroductionRequest.tsx
function IntroductionRequest({ intro }: { intro: Introduction }) {
  return (
    <div className="intro-request">
      <Avatar did={intro.introducer_did} />
      <div>
        <p>
          <strong>{introducerName}</strong> wants to introduce you to{' '}
          <strong>{introduceeName}</strong>
        </p>
        <p className="message">{intro.message}</p>
        <p className="stake">
          Staking {(intro.trust_stake * 100).toFixed(0)}% of their trust
        </p>
      </div>
      <div className="actions">
        <button onClick={accept}>Accept</button>
        <button onClick={decline}>Decline</button>
      </div>
    </div>
  );
}

// AttentionBidNotice.tsx
function AttentionBidNotice({ bid }: { bid: AttentionBid }) {
  return (
    <div className="attention-bid">
      <span className="stake-amount">{bid.stake_amount} credits</span>
      <p>Unknown sender is paying for your attention</p>
      <p className="preview">{bid.message_preview}</p>
      <div className="actions">
        <button onClick={viewEmail}>View (claim stake)</button>
        <button onClick={reject}>Reject (refund)</button>
        <button onClick={reportSpam}>Spam (burn stake)</button>
      </div>
    </div>
  );
}
```

---

## 5. Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
- [ ] Upgrade DID format to `did:mycelix:`
- [ ] Add 0TML crate dependency to backend-rs
- [ ] Create claims verification endpoint
- [ ] Basic UI for claim badges

### Phase 2: Local AI (Weeks 3-4)
- [ ] Add Symthaea crate dependency
- [ ] Create AI service in backend
- [ ] Email summarization endpoint
- [ ] Intent detection endpoint
- [ ] UI for AI insights panel

### Phase 3: Trust Graphs (Weeks 5-6)
- [ ] Trust graph data model in Holochain DNA
- [ ] Path finding algorithm
- [ ] D3.js visualization component
- [ ] Trust explanation with Symthaea

### Phase 4: Consent System (Weeks 7-8)
- [ ] Introduction data model
- [ ] Attention bidding system
- [ ] Credits integration
- [ ] UI for consent flows

### Phase 5: Integration (Weeks 9-10)
- [ ] End-to-end testing
- [ ] Performance optimization
- [ ] Documentation
- [ ] Beta launch

---

## 6. File Locations

### New Files to Create

```
happ/backend-rs/
├── src/
│   ├── services/
│   │   ├── claims.rs          # 0TML integration
│   │   ├── ai.rs              # Symthaea integration
│   │   ├── trust_graph.rs     # Graph algorithms
│   │   └── consent.rs         # Introduction/attention
│   ├── routes/
│   │   ├── claims.rs          # /api/claims/*
│   │   ├── ai.rs              # /api/ai/*
│   │   └── consent.rs         # /api/consent/*
│   └── types/
│       ├── claims.rs          # Verifiable claim types
│       ├── trust_graph.rs     # Graph types
│       └── consent.rs         # Introduction types

ui/frontend/
├── src/
│   ├── components/
│   │   ├── claims/
│   │   │   ├── ClaimBadge.tsx
│   │   │   ├── ClaimVerifier.tsx
│   │   │   └── CredentialPicker.tsx
│   │   ├── ai/
│   │   │   ├── AIInsightsPanel.tsx
│   │   │   ├── ThreadSummary.tsx
│   │   │   ├── IntentBadge.tsx
│   │   │   └── ReplySuggestions.tsx
│   │   ├── trust/
│   │   │   ├── TrustGraphVisualizer.tsx
│   │   │   ├── TrustPathExplainer.tsx
│   │   │   └── TrustAttestation.tsx
│   │   └── consent/
│   │       ├── IntroductionRequest.tsx
│   │       ├── AttentionBidNotice.tsx
│   │       └── ConsentSettings.tsx
│   └── services/
│       ├── claims.ts
│       ├── ai.ts
│       └── consent.ts
```

---

## 7. Dependencies

### Backend (Cargo.toml additions)

```toml
# 0TML ZK Proofs
zerotrustml = { path = "../../Mycelix-Core/0TML" }
risc0-zkvm = "3.0.3"

# Symthaea Local AI
symthaea-hlb = { path = "../../11-meta-consciousness/luminous-nix/symthaea-hlb" }

# Graph algorithms
petgraph = "0.6"
```

### Frontend (package.json additions)

```json
{
  "dependencies": {
    "d3": "^7.8.5",
    "@visx/network": "^3.5.0"
  }
}
```

---

## Summary

This integration transforms Mycelix-Mail into:

1. **Epistemic Mail**: Claims are verifiable, not just asserted
2. **Trust-Transparent**: See WHY you trust someone, not just a score
3. **Consent-First**: Your attention is sovereign and valued
4. **AI-Enhanced**: Local intelligence that respects privacy
5. **Byzantine-Resistant**: 45% fault tolerance from 0TML

No other email system has these capabilities.
