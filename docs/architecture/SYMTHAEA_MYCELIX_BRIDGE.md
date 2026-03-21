# Symthaea <-> Mycelix Bridge Architecture

How a liquid-brain AI's consciousness metrics gate governance actions in a decentralized social OS.

## Overview

The Symthaea-Mycelix bridge connects two systems that operate at fundamentally different timescales and abstraction levels:

- **Symthaea** runs a ~31Hz cognitive loop producing real-time consciousness metrics (integrated information Phi, moral algebra scores, epistemic confidence, unified consciousness C_unified).
- **Mycelix** is a 16-cluster Holochain hApp (decentralized social OS) where governance actions -- voting, proposals, currency operations -- require proof that the acting agent meets consciousness-derived trust thresholds.

The bridge translates Symthaea's continuous neural signals into discrete governance credentials stored on each agent's Holochain source chain. These credentials gate what actions an agent can perform, with progressive permissions tied to a 5-tier consciousness hierarchy.

## Architecture Diagram

```
+---------------------------+              +---------------------------+
|        Symthaea           |              |         Mycelix           |
|    (Cognitive Loop)       |              |    (Holochain hApp)       |
|                           |              |                           |
|  CognitiveLoopService     |              |  mycelix-bridge-common    |
|  - unified_psi (C_unified)| -- profile ->|  - ConsciousnessProfile   |
|  - phi (IIT)              |              |  - ConsciousnessCredential|
|  - moral_score            |              |  - GovernanceRequirement  |
|  - epistemic_confidence   |              |  - ReputationState        |
|                           |              |                           |
|  GovernanceManager        |              |  16 Cluster DNAs          |
|  - interval 37            |<- events ----|  - Identity (gating src)  |
|  - neuromod coupling      |              |  - Governance (proposals) |
|  - learning rate modulate |              |  - Finance (gated ops)    |
|                           |              |  - Commons, Civic, ...    |
|  mycelix_bridge.rs        |              |                           |
|  - ConsciousnessSnapshot  |              |  consciousness_gating mod |
|  - value-aligned voting   |              |  - verify_participant_tier|
|  - Phi-gated proposals    |              |  - verify_citizen_tier    |
+---------------------------+              +---------------------------+
         |                                           |
         |          mycelix-bridge-common             |
         |       (shared Rust crate, no HDK)          |
         +-------------------------------------------+
         |  ConsciousnessProfile (4D scoring)         |
         |  ConsciousnessTier (5 levels)              |
         |  ConsciousnessCredential (24h TTL + DID)   |
         |  GovernanceRequirement (tier + dimensions) |
         |  ReputationState (decay, slash, blacklist) |
         |  evaluate_governance() (pure function)     |
         |  continuous_vote_weight() (sigmoid)        |
         |  CollectivePhiEngine (network-wide Phi)    |
         +-------------------------------------------+
```

## Consciousness Gates

Five tiers gate progressive governance participation. Tier is derived from the 4D combined score.

| Tier | Min Score | Vote Weight (bp) | Governance Actions |
|------|-----------|-------------------|--------------------|
| Observer | 0.0 | 0 | Read-only access |
| Participant | 0.3 | 5,000 | Deposits, payments, basic proposals |
| Citizen | 0.4 | 7,500 | Voting rights |
| Steward | 0.6 | 10,000 | Constitutional actions, parameter changes |
| Guardian | 0.8 | 10,000 | Emergency powers, full governance |

**Continuous vote weight**: In addition to the discrete tier-based weight, `continuous_vote_weight()` provides a sigmoid-smoothed weight centered at the Citizen threshold (0.4) with temperature 0.05. This avoids cliff effects at tier boundaries:

```
weight = max_bp / (1 + exp(-(score - 0.4) / 0.05))
```

## 4D Scoring

`ConsciousnessProfile` combines four independently-sourced dimensions into a weighted score:

| Dimension | Weight | Source | Range |
|-----------|--------|--------|-------|
| **identity** | 25% | MFA AssuranceLevel from identity cluster (Anonymous=0.0, Basic=0.25, Verified=0.5, HighlyAssured=0.75, Critical=1.0) | 0.0-1.0 |
| **reputation** | 25% | Cross-hApp reputation bridge, exponential decay with 30-day half-life, multi-source weighted average | 0.0-1.0 |
| **community** | 30% | Aggregated peer trust credentials, weighted by attestor's own tier | 0.0-1.0 |
| **engagement** | 20% | Domain-specific participation; maps to Symthaea's `C_unified` via `from_unified_consciousness()` | 0.0-1.0 |

Combined score formula:

```
combined = identity * 0.25 + reputation * 0.25 + community * 0.30 + engagement * 0.20
```

Community trust carries the highest weight (30%) because peer attestations from higher-consciousness agents are the strongest signal of genuine participation. Engagement is lowest (20%) because it is the dimension most susceptible to gaming.

## Data Flow

### Credential Issuance (Symthaea -> Mycelix)

1. **Symthaea computes consciousness metrics** each cycle (~31Hz): `unified_psi`, `phi`, `moral_score`, `epistemic_confidence`.
2. **`ConsciousnessProfile::from_unified_consciousness()`** maps `C_unified` to the engagement dimension. Identity, reputation, and community come from their respective Mycelix sources.
3. **`ConsciousnessCredential`** wraps the profile with:
   - Agent DID (`did:mycelix:<pubkey>`)
   - Derived `ConsciousnessTier` at issuance time
   - Issuance timestamp + 24-hour TTL (`DEFAULT_TTL_US = 86,400,000,000 us`)
   - Issuer DID (bridge zome)
   - Optional BLAKE3 trajectory commitment (behavioral trajectory hash)
   - Extensible key-value store (substrate type, moral score, freshness attestation, etc.)
4. **Credential stored** on the agent's Holochain source chain.
5. **Governance zomes validate locally** by checking issuer and expiry -- no cross-cluster call needed at governance time.

### Governance Evaluation (Mycelix-side)

1. **`evaluate_governance(credential, requirement, now_us)`** is a pure function (no HDK dependency).
2. Checks credential expiry, with 30-minute grace period for Participant-or-below operations.
3. Derives tier from profile, checks against `GovernanceRequirement.min_tier`.
4. Checks optional per-dimension minimums (`min_identity`, `min_community`).
5. Returns `GovernanceEligibility` with eligibility, vote weight, tier, and rejection reasons.

### Governance Feedback (Mycelix -> Symthaea)

1. **GovernanceManager** (interval 37, co-prime with other subsystem intervals) receives events via `inject_event()` / `inject_outcome()`.
2. Events are queued and drained during `process()` each 37th cycle.
3. Events become **embodied through neurochemistry**:
   - `EmergencyDeclared` -> NE baseline surge (Arnsten 2009)
   - `ReciprocityPledge` -> oxytocin injection (Zak 2012)
   - `JusticeDispute` -> cortisol-proxy NE+5-HT shift (Sapolsky 2004)
   - `TallyCompleted(passed=true, aligned)` -> DA phasic burst (Schultz 1997)
   - `TallyCompleted(passed=false, aligned)` -> DA baseline dip (Schultz 1997)
   - `ReputationChanged(negative)` -> 5-HT baseline dip (Crockett 2009)
   - High collective Phi -> ECB baseline nudge
4. `GovernanceOutcome` feeds value alignment learning (harmonic resonance scoring).

## Cross-Cluster Communication

All 16 Mycelix clusters run as roles within a single unified hApp (`mycelix-workspace/happs/mycelix-unified-happ.yaml`). Cross-cluster calls use `CallTargetCell::OtherRole`:

```rust
call(
    CallTargetCell::OtherRole("identity".into()),
    ZomeName::from("consciousness_gating"),
    FunctionName::from("check_participant_tier"),
    None,
    (),
)
```

### Finance Cluster Consciousness Gating

`mycelix-finance/zomes/shared/src/lib.rs` provides two shared gate functions:

- **`verify_participant_tier()`** -- required for deposits, payments, collateral registration (combined score >= 0.3)
- **`verify_citizen_tier()`** -- required for currency creation, parameter amendment (identity >= 0.25, reputation >= 0.10)

Both functions call the identity cluster via `OtherRole` and **fail closed**: if the identity cluster is unreachable, operations are suspended (not permitted).

### Circuit Breaker

A circuit breaker protects against identity cluster unavailability:

- **Failure threshold**: 5 consecutive failures opens the breaker
- **Cooldown**: 60 seconds before half-open retry
- Prevents cascading cross-cluster call storms during network partitions
- Records success/failure per target cluster

### Audit Trail

Gate decisions are selectively logged via `GateAuditInput`:
- All rejections: always logged
- Citizen+ actions: always logged
- Basic/Participant approvals: 10% sampled (reduces DHT write load)
- Correlation IDs (`<agent_hex_prefix>:<timestamp_us>`) link cross-cluster audit trails

## Hysteresis and Reputation

### Tier Hysteresis

`TIER_HYSTERESIS_MARGIN = 0.05` prevents oscillation at tier boundaries:

- **Promotion** requires `score >= threshold + 0.05`
- **Demotion** requires `score < threshold - 0.05`

Example: An agent at Citizen tier (threshold 0.4) must reach 0.45 to promote to Steward, but does not demote back to Participant until dropping below 0.35.

### Reputation System

`ReputationState` tracks long-term behavioral history:

| Parameter | Value | Effect |
|-----------|-------|--------|
| `REPUTATION_DECAY_PER_DAY` | 0.998 | Exponential decay: `score *= 0.998^elapsed_days` |
| `REPUTATION_SLASH_FACTOR` | 0.5 | Violation halves reputation: `score *= 0.5` |
| `REPUTATION_BLACKLIST_THRESHOLD` | 0.05 | Score below 0.05 triggers blacklist |
| `REPUTATION_MAX_SLASHES` | 5 | After 5 slashes, reputation is permanently capped |
| `REPUTATION_RESTORATION_INTERACTIONS` | 100 | 100 consecutive good interactions to restore from blacklist |

Reputation feeds into `evaluate_governance_with_reputation()`, which combines credential-based tier evaluation with reputation state. Blacklisted agents cannot participate in governance regardless of their consciousness profile.

### Credential Lifecycle

- **TTL**: 24 hours (`DEFAULT_TTL_US`)
- **Grace period**: 30 minutes after expiry for Participant-or-below operations (`GRACE_PERIOD_US`)
- **Proactive refresh**: Credentials within 2 hours of expiry trigger refresh (`REFRESH_WINDOW_US`)
- **Bootstrap**: `bootstrap_credential()` and `evaluate_bootstrap_governance()` handle initial credential issuance for new agents

## Symthaea-Side Bridge Module

`symthaea/src/consciousness/mycelix_bridge.rs` provides the Symthaea-facing bridge:

- **`ConsciousnessSnapshot`**: Captures phi, meta-awareness, self-model accuracy, coherence, affective valence, CARE activation at a point in time.
- **Consciousness-gated proposals**: Only submits governance proposals when Phi exceeds threshold (GOV_PROPOSAL = 0.3).
- **Value-aligned voting**: Evaluates proposals against Eight Harmonies via `UnifiedValueEvaluator`.
- **Phi thresholds**: Basic (0.2), Proposal (0.3), Voting (0.4), Constitutional (0.6) -- must match `mycelix_bridge_common::phi_thresholds`.
- **Reputation thresholds**: Higher than consciousness thresholds because they combine consciousness (60%) + hApp reputation (40%): Basic (0.3), Governance (0.5), Voting (0.6), Constitutional (0.8).

When the `mycelix_sdk` feature is enabled, the module connects to actual Mycelix SDK types (`BridgeMessage`, `LocalBridge`, `HyperFeelEncoder` for gradient compression, `EpistemicClaim` for truth classification).

## Testing

| Test Suite | Count | Location |
|------------|-------|----------|
| mycelix-bridge-common (unit + proptest) | 349+ | `crates/mycelix-bridge-common/` |
| Finance consciousness gating | 25 | `mycelix-finance/zomes/shared/` |
| GovernanceManager unit tests | 21 | `symthaea/src/cognitive_loop/managers/governance_manager.rs` |
| Mycelix consciousness integration | 17 | `symthaea/tests/mycelix_consciousness_integration.rs` |
| Consciousness profile proptests | 23 | `crates/mycelix-bridge-common/tests/proptest_gating_invariants.rs` |

Key proptest properties verified:
- Tier monotonicity: higher scores never produce lower tiers
- Vote weight bounds: always in [0, 10000] bp
- Hysteresis stability: tier does not oscillate under noise
- Reputation decay monotonicity: score never increases without positive interactions
- Combined score bounds: always in [0.0, 1.0] for valid inputs

## Key Source Files

| File | Purpose |
|------|---------|
| `crates/mycelix-bridge-common/src/consciousness_profile.rs` | ConsciousnessProfile, ConsciousnessTier, ConsciousnessCredential, GovernanceRequirement, ReputationState, evaluate_governance() |
| `crates/mycelix-bridge-common/src/lib.rs` | Module exports, DispatchInput/DispatchResult, cross-domain dispatch |
| `crates/mycelix-bridge-common/src/collective_phi.rs` | CollectivePhiEngine for network-wide Phi computation |
| `crates/mycelix-bridge-common/src/offline_credential.rs` | Offline credential issuance/validation |
| `crates/mycelix-bridge-common/src/sub_passport.rs` | Delegated sub-passport credentials |
| `crates/mycelix-bridge-common/src/routing.rs` | Cross-cluster routing (CommonsZome, CivicZome, CrossClusterRole) |
| `symthaea/src/consciousness/mycelix_bridge.rs` | Symthaea-side bridge: ConsciousnessSnapshot, Phi-gated proposals, value-aligned voting |
| `symthaea/src/cognitive_loop/managers/governance_manager.rs` | GovernanceManager: event processing, neuromod coupling, learning feedback |
| `mycelix-finance/zomes/shared/src/lib.rs` | consciousness_gating module (verify_participant_tier, verify_citizen_tier), circuit_breaker |
| `symthaea/tests/mycelix_consciousness_integration.rs` | Cross-project integration tests |
| `crates/mycelix-bridge-common/tests/proptest_gating_invariants.rs` | Property-based tests for gating invariants |
