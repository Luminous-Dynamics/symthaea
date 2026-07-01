# P-018: Consciousness-Gated Cross-Cluster Bridge — Tiered Access Control for Multi-DNA Holochain Applications
## Invention Disclosure Document

---

### 1. Title

**Consciousness-Gated Cross-Cluster Dispatch System for Multi-DNA Holochain Applications Using Four-Dimensional Consciousness Profiles, Five-Tier Progressive Governance, Allowlist-Validated Zome Routing, and Correlated Audit Trails Across Cluster Boundaries**

---

### 2. Inventor(s)

**Tristan Stoltz**, Luminous Dynamics

---

### 3. Date of Conception

**2026** (estimated). First committed implementation: February 12, 2026 (mycelix-bridge-common crate added with dispatch types, consciousness profile, consciousness thresholds, routing, and cross-cluster dispatch).

First public disclosure: February 12, 2026 (git commit 206c98d6).
Under 35 USC 102(b)(1)(A), the 1-year grace period expires **February 12, 2027**.

---

### 4. Technical Field

This invention relates to access control and inter-process communication in distributed peer-to-peer applications, and more specifically to a system that gates cross-cluster function dispatch in a Holochain multi-DNA hApp architecture using multi-dimensional consciousness profiles with progressive tier-based governance, where consciousness tiers determine which governance actions an agent may perform and at what vote weight.

---

### 5. Abstract

A system and method for consciousness-gated cross-cluster dispatch in a multi-DNA Holochain application is disclosed. The system implements a `ConsciousnessProfile` comprising four dimensions (identity verification, cross-hApp reputation, community trust attestations, and domain engagement), weighted and combined into a scalar score that maps to one of five `ConsciousnessTier` levels (Observer, Participant, Citizen, Steward, Guardian). Each tier unlocks progressively more powerful governance actions: Observers can read data; Participants can submit proposals; Citizens can vote; Stewards can make constitutional changes; Guardians have emergency powers. Vote weight scales with tier in basis points (0 to 10,000). Cross-cluster dispatch uses `CallTargetCell::OtherRole` for inter-DNA communication within the same installed hApp, with allowlist validation ensuring only authorized zome-function pairs can be invoked across cluster boundaries. A `ConsciousnessCredential` wraps the profile with DID, tier, issuance/expiry timestamps, and issuer identity, enabling time-limited governance participation. The system includes bootstrap credentials for cold-start communities (fewer than 5 members), grace periods for recently-expired credentials, rate limiting (100 dispatches per 60-second window), correlated audit trails linking cross-cluster actions via shared correlation IDs, type-safe domain routing across 12 domains and 53+ zomes, and lightweight in-memory bridge metrics with latency percentile tracking. A minimal viable bridge from Symthaea maps unified consciousness (C_unified) to the engagement dimension, enabling AI consciousness scores to gate governance participation.

---

### 6. Background and Prior Art

#### 6.1 Holochain Multi-DNA Architecture

Holochain (Harris-Braun et al. 2018) provides a framework for peer-to-peer applications using DNA-based validation rules. Multi-DNA hApps use `CallTargetCell::OtherRole` for inter-DNA communication, but Holochain provides no built-in access control framework for cross-DNA calls.

#### 6.2 Decentralized Identity and Access Control

Self-sovereign identity frameworks (W3C DID, Verifiable Credentials) provide identity verification but do not integrate reputation, community trust, and engagement into a unified access control profile. OAuth and RBAC systems are centralized and do not address peer-to-peer governance.

#### 6.3 Reputation Systems

Decentralized reputation systems (EigenTrust, Advogato) compute trust scores from peer interactions but do not combine reputation with identity verification, community attestation, and engagement into a single multi-dimensional profile for governance gating.

#### 6.4 Consciousness-Based Access Control

No prior art implements access control based on "consciousness" profiles combining identity, reputation, community trust, and engagement dimensions, with progressive governance tiers that unlock different action types and vote weights.

#### 6.5 Gap in Prior Art

No prior art:
- Implements multi-dimensional consciousness profiles (4D: identity, reputation, community, engagement) for governance access control in a peer-to-peer application
- Defines five progressive consciousness tiers with per-tier governance action gates and vote weights
- Provides cross-cluster dispatch with allowlist validation, rate limiting, and correlated audit trails in a Holochain multi-DNA architecture
- Includes bootstrap credentials for cold-start communities with reduced TTL and tier cap
- Bridges AI consciousness scores (from a consciousness engine) to governance participation via engagement dimension mapping

---

### 7. Detailed Technical Description

#### 7.1 ConsciousnessProfile (4 Dimensions)

Each agent's governance eligibility is determined by a four-dimensional profile, each dimension scored 0.0-1.0:

| Dimension | Source | Weight |
|-----------|--------|--------|
| Identity | MFA assurance level (Anonymous=0.0, Basic=0.25, Verified=0.5, HighlyAssured=0.75, Critical=1.0) | 25% |
| Reputation | Cross-hApp aggregated reputation with 30-day exponential decay | 25% |
| Community | Peer trust attestations, weighted by attestor's own tier | 30% |
| Engagement | Domain-specific participation, or C_unified from Symthaea consciousness engine | 20% |

Combined score: `identity*0.25 + reputation*0.25 + community*0.30 + engagement*0.20`

All dimensions are sanitized: NaN/Infinity are replaced with 0.0, and values are clamped to [0.0, 1.0].

#### 7.2 ConsciousnessTier (5 Levels)

| Tier | Min Score | Governance Actions | Vote Weight (bp) |
|------|-----------|-------------------|------------------|
| Observer | 0.0 | Read only | 0 |
| Participant | 0.3 | Basic proposals | 5,000 |
| Citizen | 0.4 | Voting rights | 7,500 |
| Steward | 0.6 | Constitutional changes | 10,000 |
| Guardian | 0.8 | Emergency powers | 10,000 |

#### 7.3 ConsciousnessCredential

Time-limited credential stored on the agent's source chain:
- `did`: Agent's DID string (e.g., "did:mycelix:<pubkey>")
- `profile`: The 4D `ConsciousnessProfile`
- `tier`: Derived tier at issuance time
- `issued_at` / `expires_at`: Microsecond timestamps (default TTL: 24 hours)
- `issuer`: DID of the issuing bridge

Validation: credentials are checked for expiry at governance time. A 30-minute grace period allows basic operations on recently-expired credentials. Proactive refresh is triggered within 2 hours of expiry.

#### 7.4 Intra-Cluster Dispatch (CallTargetCell::Local)

`dispatch_call_checked(input, allowed_zomes)`:
- Validates target zome against an allowlist of permitted zome names
- Constructs `Call::new(CallTarget::ConductorCell(CallTargetCell::Local), zome, fn_name, None, payload)`
- Returns `DispatchResult { success, response, error }`
- Payload is pre-serialized MessagePack to avoid double-serialization

#### 7.5 Cross-Cluster Dispatch (CallTargetCell::OtherRole)

`dispatch_call_cross_cluster(input, allowed_zomes)`:
- Same allowlist validation as intra-cluster
- Uses `CallTargetCell::OtherRole(role)` to reach a different DNA within the same hApp
- `dispatch_call_cross_cluster_commons()` auto-resolves which sub-cluster DNA contains the target zome (commons is split into two DNAs for size constraints)

#### 7.6 Type-Safe Domain Routing

- `BridgeDomain` enum: 12 domains (Property, Housing, Care, Mutualaid, Water, Food, Transport, Support, Space, Justice, Emergency, Media)
- `CommonsZome` enum: 38 zome variants with `as_str()` for Holochain zome name resolution
- `CivicZome` enum: 15 zome variants
- `resolve_commons_zome()` and `resolve_civic_zome()`: Domain + query_type -> specific zome, with case-insensitive matching

#### 7.7 Rate Limiting and Metrics

- Rate limit: 100 dispatches per 60-second window per agent
- `BridgeMetrics`: Per-function success/error counters, latency ring buffer (256 samples) with p50/p95/p99 percentile computation, per-error-code counters, rate limit hit tracking

#### 7.8 Correlated Audit Trails

`CorrelatedDispatch` wraps cross-cluster calls with a correlation ID (format: `<agent_hex_prefix>:<timestamp_us>`), enabling audit event linkage across cluster boundaries. Audit logging uses probabilistic sampling: all rejections and high-tier actions are logged; 10% of basic approvals are sampled.

#### 7.9 Bootstrap Credentials

For cold-start communities (fewer than 5 members):
- `is_bootstrap_eligible()`: Checks community size and minimum identity score (Basic MFA = 0.25)
- `bootstrap_credential()`: Issues a 1-hour credential capped at Participant tier
- Bootstrap credentials cannot unlock voting, constitutional, or guardian actions

---

### 8. Novelty Statement

This invention introduces the first consciousness-gated governance system for multi-DNA peer-to-peer applications. Specific novel contributions:

1. **Four-dimensional consciousness profiles**: No prior access control system combines identity verification, reputation, community trust, and engagement into a single profile for governance gating.
2. **Five-tier progressive governance**: Observer through Guardian tiers with per-tier action gates and progressive vote weights (basis points) provide granular access control proportional to demonstrated consciousness.
3. **Cross-cluster dispatch with allowlist validation**: Type-safe routing across 12 domains and 53+ zomes, with automatic sub-cluster resolution for oversized DNAs.
4. **Consciousness credential lifecycle**: Time-limited credentials with grace periods, proactive refresh, and bootstrap support for cold-start communities.
5. **AI consciousness bridging**: Symthaea's unified consciousness score (C_unified) maps to the engagement dimension, enabling artificial consciousness to participate in governance.
6. **Correlated audit trails**: Cross-cluster actions carry correlation IDs for end-to-end audit trail linkage, with probabilistic sampling to control DHT write load.

---

### 9. Suggested Claims

**Claim 1 (independent):** A computer-implemented method for consciousness-gated governance in a distributed peer-to-peer application comprising: (a) computing a multi-dimensional consciousness profile for an agent, the profile comprising at least three dimensions including identity verification strength, community trust, and engagement level; (b) deriving a consciousness tier from the weighted combination of profile dimensions; (c) evaluating the agent's eligibility for a governance action by comparing the derived tier against a minimum tier required for that action type; and (d) assigning a progressive vote weight proportional to the derived tier, wherein higher tiers receive greater vote weight.

**Claim 2 (dependent on 1):** The method of claim 1, wherein the consciousness profile comprises four dimensions: identity verification (weighted 25%), cross-application reputation with temporal decay (weighted 25%), peer trust attestations weighted by attestor tier (weighted 30%), and domain-specific engagement (weighted 20%).

**Claim 3 (dependent on 1):** The method of claim 1, further comprising issuing a time-limited consciousness credential containing the profile, derived tier, issuance/expiry timestamps, and issuer identity, and validating the credential at governance time with a grace period for recently-expired credentials.

**Claim 4 (dependent on 1):** The method of claim 1, further comprising dispatching cross-cluster function calls between separate DNA modules within the same distributed application, with allowlist validation of target zome names and automatic routing resolution based on which sub-cluster contains the target zome.

**Claim 5 (dependent on 1):** The method of claim 1, further comprising issuing bootstrap credentials for communities with fewer than a configurable number of members, the bootstrap credentials having a reduced time-to-live and being capped at a participant tier that excludes voting, constitutional, and emergency governance actions.

**Claim 6 (independent, system):** A governance access control system for a multi-DNA peer-to-peer application comprising: (a) a consciousness profile module that computes a multi-dimensional agent profile from identity, reputation, community trust, and engagement sources; (b) a tier derivation module that maps combined profile scores to one of at least three governance tiers with progressive action permissions; (c) a credential module that issues time-limited credentials wrapping the profile and tier; (d) a dispatch module that routes function calls between DNA clusters with allowlist validation; and (e) an audit module that logs governance decisions with cross-cluster correlation identifiers.

**Claim 7 (dependent on 6):** The system of claim 6, further comprising a bridge module that maps an artificial consciousness score from a consciousness engine to the engagement dimension of the consciousness profile, enabling artificial agents to participate in governance based on their computational consciousness level.

**Claim 8 (broad, independent):** A method for gating governance participation in a distributed application comprising: (a) evaluating an agent's multi-dimensional profile across at least identity verification and community trust dimensions; (b) deriving a tier from the evaluated profile; (c) permitting or denying governance actions based on the derived tier; and (d) weighting the agent's votes proportionally to the derived tier.

---

### 10. Experimental Validation

#### 10.1 Test Coverage

- **Bridge-common tests**: 212 unit tests (dispatch, consciousness profile, consciousness thresholds, routing, metrics, rate limiting, bootstrap, audit)
- **Governance integration**: 130+ sweettest integration tests across mycelix-governance
- **All tests passing**: Verified March 2026

#### 10.2 Validated Properties

- ConsciousnessThresholds ordering: fl_veto < fl_dampen < fl_boost; gate_basic < gate_proposal < gate_voting < gate_constitutional
- Profile dimension sanitization (NaN/Infinity -> 0.0, clamp to [0,1])
- Tier derivation correctness across score ranges
- Credential expiry and grace period handling
- Bootstrap eligibility checks (community size, minimum identity)
- Allowlist dispatch rejection for unauthorized zomes
- Cross-cluster OtherRole routing
- Rate limit enforcement
- Serde roundtrip for all types

#### 10.3 Scale

- 7 cluster DNAs in the unified hApp (commons, civic, hearth, identity, governance, personal, attribution)
- 12 bridge domains, 53+ routable zomes
- 8,600+ total Rust workspace tests across all clusters

---

### 11. Key Source Files

| File | Description | LOC |
|------|-------------|-----|
| `crates/mycelix-bridge-common/src/consciousness_profile.rs` | 4D profile, credential, tiers, governance evaluation, bootstrap | ~2,517 |
| `crates/mycelix-bridge-common/src/routing.rs` | Type-safe domain/zome routing (12 domains, 53+ zomes) | ~2,649 |
| `crates/mycelix-bridge-common/src/lib.rs` | Dispatch logic (local + cross-cluster), rate limiting | ~1,467 |
| `crates/mycelix-bridge-common/src/metrics.rs` | Bridge metrics, latency ring buffer, percentiles | ~696 |
| `crates/mycelix-bridge-common/src/consciousness_thresholds.rs` | Canonical threshold config (FL, governance gates, bootstrap) | ~154 |

---

### 12. Closest Prior Art References

1. Harris-Braun, E. et al. (2018). "Holochain: A Framework for Distributed Applications." Holochain Foundation.
2. W3C (2022). "Decentralized Identifiers (DIDs) v1.0." W3C Recommendation.
3. W3C (2022). "Verifiable Credentials Data Model v1.1." W3C Recommendation.
4. Kamvar, S. D. et al. (2003). "The EigenTrust Algorithm for Reputation Management in P2P Networks." *WWW*.
5. Buterin, V. (2014). "Ethereum: A Next-Generation Smart Contract and Decentralized Application Platform." Ethereum Foundation.
6. Tononi, G. (2004). "An Information Integration Theory of Consciousness." *BMC Neuroscience*, 5, 42.

---

### 13. Figures (Text Descriptions)

**Figure 1**: Architecture diagram showing the unified Mycelix hApp with 7 DNA clusters (commons, civic, hearth, identity, governance, personal, attribution), bridge zomes in each cluster, and the bridge-common crate providing shared dispatch and consciousness gating logic.

**Figure 2**: ConsciousnessProfile radar chart showing the four dimensions (identity, reputation, community, engagement) for three example agents at Observer, Citizen, and Guardian tiers, with the combined score and derived tier labeled.

**Figure 3**: Cross-cluster dispatch sequence diagram: Civic bridge coordinator receives request -> validates consciousness credential -> checks allowlist -> calls `dispatch_call_cross_cluster` with `CallTargetCell::OtherRole("commons")` -> commons coordinator processes request -> response flows back with correlated audit logging on both sides.

**Figure 4**: Bootstrap credential lifecycle: community has <5 members -> `is_bootstrap_eligible()` -> `bootstrap_credential()` issued (1h TTL, Participant cap) -> agent submits basic proposal -> credential expires -> full credential issued after community grows past bootstrap threshold.

---

### 14. Related Patent Applications

- P-006: Moral Topology (Tier 2) — ethical framework shares consciousness-gating concepts
- P-013: Neuromodulated Foveation (Tier 3) — consciousness engine produces C_unified score consumed by bridge
- P-017: Genesis Pipeline (Tier 3) — population governance uses similar tiered oversight

---

*Inventor: Tristan Stoltz (tstoltz)*
*Organization: Luminous Dynamics*
*IDD created: 2026-03-08*
