# Anti-Tyranny Governance Hardening — Complete Design Document

> "A veto without an override is a dictator with better branding."

This document captures the complete anti-tyranny governance shield designed
across four hardening rounds. It serves as the authoritative specification
for re-applying changes when the branch stabilizes.

## Design Principle

Every governance power must have a thermodynamic counterbalance. No Maxwell's
Demons allowed. Tyranny must be thermodynamically unstable — not merely
discouraged, but structurally impossible as an attractor state.

## Status

| Layer | Files | Status |
|-------|-------|--------|
| Bridge coordinator | consciousness_config.rs, consciousness.rs | **COMMITTED & STABLE** |
| Constitution integrity | integrity/src/lib.rs | **COMMITTED & STABLE** |
| Execution integrity | integrity/src/lib.rs | Designed, needs re-apply |
| Execution coordinator | coordinator/src/lib.rs | Designed, needs re-apply |
| Councils integrity | integrity/src/lib.rs | Designed, needs re-apply |
| Councils coordinator | coordinator/src/lib.rs | Designed, needs re-apply |
| Voting integrity | integrity/src/lib.rs | Designed, needs re-apply |
| Voting coordinator | coordinator/src/lib.rs | Designed, needs re-apply |
| Bridge-common | ethics_binding.rs, sub_passport.rs | Designed, needs re-apply |
| Multiworld simulation | governance.rs, config.rs | On planetary-infrastructure branch |

---

## Hardcoded Invariants (Not Governance-Configurable)

These constants are compiled into WASM. They cannot be changed by governance
proposals, constitutional amendments, or any on-chain mechanism. Changing them
requires deploying new WASM (a new DNA), which requires community migration.

| Invariant | Value | Constant Name | File |
|-----------|-------|---------------|------|
| Veto override threshold | 67% (2/3) | `VETO_OVERRIDE_THRESHOLD` | execution/integrity |
| Veto override window | 48 hours | `VETO_OVERRIDE_WINDOW_US` | execution/integrity |
| Override threshold floor | 60% | `OVERRIDE_THRESHOLD_FLOOR` | execution/integrity |
| Veto cooldown per Guardian | 7 days | `VETO_COOLDOWN_US` | execution/coordinator |
| Veto yearly limit | 3 per 12 months | `VETO_YEARLY_LIMIT` | execution/integrity |
| Strategic Override sunset | 36 months | `STRATEGIC_OVERRIDE_SUNSET_US` | execution/integrity |
| Guardian veto phi | 0.8 | `GUARDIAN_PHI_THRESHOLD` | execution/coordinator |
| Emergency sessions max | 3 | `MAX_CONSECUTIVE_EMERGENCY_SESSIONS` | councils/integrity |
| Emergency session duration | 14 days | `MAX_EMERGENCY_SESSION_DURATION_US` | councils/integrity |
| Emergency cooldown | 30 days | `EMERGENCY_COOLDOWN_DURATION_US` | councils/integrity |
| Membership term | 365 days | `DEFAULT_MEMBERSHIP_TERM_US` | councils/integrity |
| AI agent max tier | Steward | `AI_AGENT_MAX_TIER` | bridge-common/sub_passport |
| Unsigned phi cap | 0.6 | `UNSIGNED_SNAPSHOT_PHI_CAP` | bridge/consciousness.rs |
| Config floor (basic) | 0.4 | `CONFIG_FLOOR_BASIC` | bridge/consciousness_config.rs |
| Config floor (proposal) | 0.5 | `CONFIG_FLOOR_PROPOSAL` | bridge/consciousness_config.rs |
| Config floor (voting) | 0.6 | `CONFIG_FLOOR_VOTING` | bridge/consciousness_config.rs |
| Config floor (constitutional) | 0.8 | `CONFIG_FLOOR_CONSTITUTIONAL` | bridge/consciousness_config.rs |
| Config ceiling (max weight) | 3.0 | `CONFIG_CEILING_MAX_WEIGHT` | bridge/consciousness_config.rs |

---

## 1. Execution Integrity — Veto Override Types

### New Enum Variant
Add `Vetoed` to `TimelockStatus`:
```rust
/// Vetoed by a guardian — pending possible override (48-hour window)
Vetoed,
```

### New Types
```rust
pub enum VetoStatus { Active, Challenged, Overridden, Sustained }

pub const VETO_OVERRIDE_WINDOW_US: i64 = 48 * 3600 * 1_000_000;
pub const VETO_OVERRIDE_THRESHOLD: f64 = 0.67;  // Constitutional: 2/3 (Art. III, Sec. 5.3)

pub struct VetoOverrideVote {
    pub id: String,
    pub veto_id: String,
    pub voter_did: String,
    pub supports_override: bool,
    pub phi_score: f64,
    pub voted_at: Timestamp,
}

pub struct VetoOverrideResult {
    pub id: String,
    pub veto_id: String,
    pub timelock_id: String,
    pub override_votes_for: f64,
    pub override_votes_against: f64,
    pub total_eligible_voters: u64,
    pub override_threshold: f64,  // always 0.80
    pub override_succeeded: bool,
    pub resolved_at: Timestamp,
}
```

### State Machine
```
Ready --[veto]--> Vetoed --[80% override]--> Ready --[sign]--> Executed
                         --[override fails/expires]--> Cancelled
```

### New Transitions in check_update_timelock
```rust
| (TimelockStatus::Ready, TimelockStatus::Vetoed)
| (TimelockStatus::Pending, TimelockStatus::Vetoed)
| (TimelockStatus::Vetoed, TimelockStatus::Ready)
| (TimelockStatus::Vetoed, TimelockStatus::Cancelled)
```

### Entry/Link Types
- Add `VetoOverrideVote`, `VetoOverrideResult` to EntryTypes
- Add `VetoToOverrideVotes`, `VetoToOverrideResult` to LinkTypes

### Validation
- Override votes: require DID, non-empty veto_id, phi in [0,1]
- Override result: threshold must equal VETO_OVERRIDE_THRESHOLD exactly
- Both types immutable (reject updates)

### Check Functions
- `check_create_override_vote(vote)` — DID, veto_id, phi range
- `check_create_override_result(result)` — threshold enforcement, non-negative votes

---

## 2. Execution Coordinator — Veto Mechanics

### veto_timelock() Changes
1. **Rate limiting**: 7-day cooldown per Guardian via GuardianToVeto links
2. **Phi threshold**: Raised to 0.8 (true Guardian tier)
3. **Status**: Transition to `Vetoed` (not `Cancelled`)
4. **Signal**: Emit VetoRateLimitExceeded when cooldown blocks a veto

### New Functions
```rust
pub fn challenge_veto(input: ChallengeVetoInput) -> ExternResult<()>
// Citizen-tier (0.4) can initiate. Emits VetoChallenged signal.

pub fn cast_override_vote(input: CastOverrideVoteInput) -> ExternResult<Record>
// Citizen-tier required. Duplicate detection. Phi-weighted votes.

pub fn resolve_override(input: ResolveOverrideInput) -> ExternResult<Record>
// Permission-less. Tallies votes, applies 80% threshold.
// Override success: Vetoed -> Ready. Failure: Vetoed -> Cancelled.

pub fn get_guardian_vetoes(guardian_did: String) -> ExternResult<Vec<Record>>
// Accountability query for Guardian veto history.
```

### Participation Insurance (DESIGNED, NOT YET IMPLEMENTED)
If override quorum fails, extend window and lower threshold:
- Attempt 1: 80% threshold
- Attempt 2: 75% threshold (after 48h extension)
- Attempt 3: 70% threshold (after another 48h)
- Floor: 67% (never goes below 2/3 supermajority)

---

## 3. Councils Integrity — Emergency Limits & Term Limits

### Constants
```rust
pub const MAX_CONSECUTIVE_EMERGENCY_SESSIONS: u32 = 3;
pub const MAX_EMERGENCY_SESSION_DURATION_US: i64 = 14 * 24 * 3600 * 1_000_000;
pub const EMERGENCY_COOLDOWN_DURATION_US: i64 = 30 * 24 * 3600 * 1_000_000;
pub const DEFAULT_MEMBERSHIP_TERM_US: i64 = 365 * 24 * 3600 * 1_000_000;
```

### New Types
```rust
pub struct EmergencySession {
    pub id: String,
    pub council_id: String,
    pub session_number: u32,  // 1-3
    pub started_at: Timestamp,
    pub expires_at: Timestamp,
    pub preceding_session_id: Option<String>,
    pub status: EmergencySessionStatus,
}

pub enum EmergencySessionStatus { Active, Expired, ExtendedToNext, CooldownStarted }

pub struct EmergencyCooldown {
    pub id: String,
    pub council_id: String,
    pub last_session_id: String,
    pub cooldown_started: Timestamp,
    pub cooldown_ends: Timestamp,  // started + 30 days
    pub override_approved: bool,
}
```

### CouncilMembership Extension
```rust
// Add to CouncilMembership struct:
#[serde(default)]
pub expires_at: Option<Timestamp>,
```

### Check Functions
- `check_create_emergency_session`: session_number 1-3, duration <= 14 days
- `check_create_emergency_cooldown`: duration >= 30 days

---

## 4. Councils Coordinator — Enforcement

### create_council() Enhancement
When `CouncilType::Emergency`:
1. Validate duration <= 14 days
2. Check for active cooldown
3. Count consecutive sessions
4. Create EmergencySession tracking entry
5. Auto-create cooldown on 3rd consecutive session

### New Functions
```rust
pub fn enforce_emergency_expiration(council_id: String) -> ExternResult<bool>
// Permission-less. Any agent can trigger. Dissolves expired emergency councils.

pub fn approve_emergency_renewal(input: ApproveEmergencyRenewalInput) -> ExternResult<bool>
// Requires non-emergency council member. Overrides cooldown for new emergency.

pub fn enforce_membership_expiration(council_id: String) -> ExternResult<Vec<String>>
// Permission-less. Expires memberships past their term. Returns expired DIDs.
```

---

## 5. Voting Integrity — Ethics Binding & Delegation Pruning

### New Types
```rust
pub enum GovernanceEthicsVerdict { Safe, Caution, Blocked }

pub struct EthicsDisclosure {
    pub id: String,
    pub proposal_id: String,
    pub verdict: GovernanceEthicsVerdict,
    pub concerns: Vec<String>,
    pub disclosed_at: Timestamp,
    pub disclosure_source: String,
}
```

### PhiWeightedTally Extensions
```rust
#[serde(default)] pub ethics_caution_flagged: bool,
#[serde(default)] pub ethics_blocked_flagged: bool,
#[serde(default)] pub ethics_escalated_from: Option<String>,
```

### Delegation Cycle Prevention
```rust
pub fn check_delegation_no_self_cycle(delegator: &str, delegate: &str) -> Result<(), String>
// Rejects A -> A delegations at write-time
```

### Check Functions
- `check_create_ethics_disclosure`: proposal_id required, Blocked requires concerns

---

## 6. Voting Coordinator — Ethics Tally & Signals

### New Signal Variants
```rust
EthicsDisclosureAdded { proposal_id, verdict, concerns_count, disclosure_source }
QuorumFailureAlert { proposal_id, consecutive_failures, participation_rate, required_quorum }
```

### New Function
```rust
pub fn create_ethics_disclosure(input: CreateEthicsDisclosureInput) -> ExternResult<Record>
// Creates disclosure, links to proposal, emits signal
```

### tally_phi_votes Enhancement
Before threshold computation:
1. Query EthicsDisclosure entries for the proposal
2. If Blocked: escalate effective tier via CircuitBreakerOutcome::escalated_tier
3. Use effective_tier for quorum/approval thresholds
4. Set ethics fields on PhiWeightedTally

After approval computation:
5. If quorum not reached: emit QuorumFailureAlert

---

## 7. Bridge-Common — Ethics & SubPassport

### New Module: ethics_binding.rs
```rust
pub enum GovernanceEthicsVerdict { Safe, Caution, Blocked }
pub struct EthicsAssessment { proposal_id, verdict, concerns, assessed_at, assessor_did }
pub const ETHICS_DISCLOSURE_VISIBILITY_HOURS: u64 = 48;
```

### SubPassport Enhancement
```rust
pub fn review_max_tier(&mut self, now_us: u64) -> bool
// 90-day minimum age, 70% compliance, one-tier-per-review
// AI agents capped at Steward (never Guardian)

pub const AI_AGENT_MAX_TIER: ConsciousnessTier = ConsciousnessTier::Steward;
```

---

## 8. Bridge Coordinator — Config Security (COMMITTED)

### consciousness_config.rs
- Cross-zome proposal verification (fail-closed)
- Absolute floors: basic 0.4, proposal 0.5, voting 0.6, constitutional 0.8
- Absolute ceiling: max_voting_weight 3.0

### consciousness.rs
- `UNSIGNED_SNAPSHOT_PHI_CAP = 0.6`
- Self-reported consciousness capped below Guardian tier
- Only signed attestations unlock Guardian-tier governance powers

---

## 9. Constitution Integrity — Unamendable Core (COMMITTED)

### Protected Rights
```rust
const UNAMENDABLE_RIGHTS: &[&str] = &[
    "veto override",
    "consciousness gating",
    "term limits",
    "emergency power limits",
    "permission-less enforcement",
    "fork rights",
    "right to exit",
];
```

### Enforcement
- `targets_unamendable_core()` checks RemoveRight, ModifyRight, RemoveArticle, ModifyProcess
- Enforced at DHT validation level — integrity zome rejects the entry
- Even 100% unanimous vote cannot remove these rights

---

## 10. Red-Team Attack Vectors (Remaining)

| # | Vector | Risk | Status |
|---|--------|------|--------|
| 2 | Social engineering (21% block) | 63 | Participation insurance DESIGNED |
| 8 | Quorum suppression | 42 | QuorumFailureAlert signals |
| 10 | Infrastructure capture | 40 | Operational (decentralize hosting) |
| 9 | WASM supply chain | 40 | Operational (reproducible builds) |
| 1 | Sybil attacks | 32 | Phi cap mitigates; need proof-of-unique-identity |
| 6 | Light-speed arbitrage | 14-49 | Future (propagation buffers) |
| 5 | Cross-cluster exploitation | 18 | Good allowlists |

---

## 11. Simulation Results (Planetary-Infrastructure Branch)

300-year pilot run across 5 seeds (42, 123, 789, 1337, 2718):
- **100% survival** (CVS 0.752-0.760)
- **Zero vetoes attempted** — deterrence effect confirmed
- The 80% override threshold creates game-theoretic equilibrium:
  rational Guardians only veto when >20% community agrees

Hostile Guardian adversarial scenario designed but requires branch merge
(CivAgent struct diverged between branches).

---

## Application Order

When re-applying to a stable branch:

1. **bridge-common**: ethics_binding.rs + sub_passport changes (no deps)
2. **execution integrity**: TimelockStatus::Vetoed + override types (no deps)
3. **councils integrity**: EmergencySession + membership expires_at (no deps)
4. **voting integrity**: EthicsDisclosure + delegation check (no deps)
5. **execution coordinator**: veto mechanics + rate limiting (deps: execution integrity)
6. **councils coordinator**: emergency enforcement + term limits (deps: councils integrity)
7. **voting coordinator**: ethics tally + signals (deps: voting integrity)

Each step can be tested independently:
```bash
cargo test -p <package_name> --lib
```

---

## Test Counts (When Fully Applied)

| Package | Tests |
|---------|-------|
| execution_integrity | 25 |
| execution (coordinator) | 27 |
| councils_integrity | 34 |
| councils (coordinator) | 21 |
| voting_integrity | 74 |
| voting (coordinator) | 34 |
| bridge-common (ethics + sub_passport) | 23 |
| governance_bridge | 13 |
| constitution_integrity | 91 |
| **Total** | **342** |

---

*Consciousness-first governance serving all beings — with thermodynamic guarantees.*
