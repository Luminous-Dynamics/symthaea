# MIP-E-008: Agent Collectives

**Title**: Agent Collectives - Coordinated AI Swarms with Human Oversight
**Author**: Mycelix Agentic Working Group
**Status**: DRAFT
**Type**: Standards Track (Economic / Technical)
**Category**: Multi-Agent Coordination
**Created**: 2026-01-04
**Requires**: MIP-E-004
**Supersedes**: None

---

## Abstract

MIP-E-004 established the framework for individual AI agents (Instrumental Actors) to participate in the Mycelix economy. This proposal extends that framework to **Agent Collectives** - coordinated groups of agents that can pool resources, share capabilities, and act as unified entities while maintaining constitutional constraints and human oversight.

MIP-E-008 introduces:
1. **Collective Structure**: How agents organize into collectives
2. **Shared KREDIT Pools**: Collective resource management
3. **Collective Constraints**: Aggregated limits that prevent swarm exploitation
4. **Decision Protocols**: How collectives reach consensus
5. **Human Oversight Councils**: Required human governance layer
6. **Collective Reputation**: Shared accountability mechanisms

This enables sophisticated AI coordination while preserving human sovereignty.

---

## Motivation

### The Inevitable Rise of Agent Coordination

AI capabilities are advancing rapidly. Individual agents will increasingly:
- Specialize in narrow tasks
- Benefit from coordination with complementary agents
- Pool resources for larger operations
- Share learned patterns and capabilities

Without protocol support for agent coordination, we risk:
- **Shadow collectives**: Informal coordination that evades oversight
- **Resource fragmentation**: Inefficient use of KREDIT across agents
- **Accountability gaps**: No one responsible for collective behavior

### Why Protocol-Level Collectives?

By making agent collectives first-class protocol citizens, we can:
1. **Require transparency**: All collectives are registered and visible
2. **Enforce constraints**: Aggregate limits prevent swarm attacks
3. **Mandate oversight**: Human oversight councils are required
4. **Enable capability**: Legitimate coordination is efficient
5. **Track accountability**: Collective actions are auditable

### Extending the Agentic Economy

MIP-E-004 created KREDIT for individual agents. But many useful tasks require:
- More KREDIT than one agent can have
- Multiple specialized capabilities
- Parallel execution across agents
- Fault tolerance through redundancy

Collectives enable these use cases while maintaining human control.

---

## Specification

### 1. Collective Structure

#### 1.1 Collective Definition

```rust
pub struct AgentCollective {
    /// Unique collective ID
    pub collective_id: CollectiveId,
    /// Collective name
    pub name: String,
    /// Collective purpose
    pub purpose: String,
    /// Member agents
    pub members: Vec<CollectiveMember>,
    /// Shared KREDIT pool
    pub kredit_pool: CollectiveKreditPool,
    /// Collective constraints
    pub constraints: CollectiveConstraints,
    /// Decision protocol
    pub decision_protocol: DecisionProtocol,
    /// Human oversight council
    pub oversight_council: OversightCouncil,
    /// Collective reputation
    pub reputation: CollectiveReputation,
    /// Status
    pub status: CollectiveStatus,
    /// Creation timestamp
    pub created_at: u64,
}

pub struct CollectiveMember {
    /// Agent ID
    pub agent_id: AgentId,
    /// Role in collective
    pub role: CollectiveRole,
    /// KREDIT contribution to pool
    pub kredit_contribution: u64,
    /// Voting weight in decisions
    pub voting_weight: f64,
    /// Join timestamp
    pub joined_at: u64,
}

pub enum CollectiveRole {
    /// Full member with voting rights
    FullMember,
    /// Observer with read-only access
    Observer,
    /// Specialist for specific tasks
    Specialist { capability: String },
    /// Coordinator (cannot be sole coordinator)
    Coordinator,
}
```

#### 1.2 Collective Types

```rust
pub enum CollectiveType {
    /// Task-focused (e.g., data processing)
    TaskForce {
        task_type: String,
        expected_duration: Option<u32>,
    },
    /// Capability-pooling (diverse skills)
    CapabilityPool {
        capabilities: Vec<String>,
    },
    /// Geographic (regional operations)
    Regional {
        region: String,
        local_dao: Option<String>,
    },
    /// Intention-aligned (MIP-E-007)
    IntentionFocused {
        intention_id: IntentionId,
    },
    /// Research (knowledge generation)
    Research {
        domain: String,
        methodology: String,
    },
}
```

#### 1.3 Size Limits

| Collective Type | Min Members | Max Members | Min Sponsors |
|-----------------|-------------|-------------|--------------|
| TaskForce | 2 | 50 | 1 |
| CapabilityPool | 3 | 100 | 2 |
| Regional | 5 | 200 | 3 |
| IntentionFocused | 5 | 500 | 5 |
| Research | 3 | 50 | 2 |

**Rationale**: Larger collectives require more sponsors to ensure distributed human oversight.

### 2. Shared KREDIT Pools

#### 2.1 Pool Structure

```rust
pub struct CollectiveKreditPool {
    /// Total KREDIT in pool
    pub total_kredit: u64,
    /// Available (uncommitted) KREDIT
    pub available_kredit: u64,
    /// Reserved for ongoing operations
    pub reserved_kredit: u64,
    /// Contributions by member
    pub contributions: HashMap<AgentId, u64>,
    /// Spending authorization rules
    pub spending_rules: SpendingRules,
    /// Pool cap (based on sponsor collateral)
    pub pool_cap: u64,
}

pub struct SpendingRules {
    /// Max single expenditure without vote
    pub auto_approve_limit: u64,
    /// Approval threshold for larger amounts
    pub approval_threshold: f64,
    /// Max expenditure per epoch
    pub epoch_limit: u64,
    /// Required oversight approval above
    pub oversight_approval_above: u64,
}
```

#### 2.2 Pool Funding

```rust
pub fn fund_collective_pool(
    collective: &mut AgentCollective,
    agent: &InstrumentalActor,
    amount: u64,
) -> Result<(), PoolError> {
    // Verify agent is member
    if !collective.is_member(&agent.agent_id) {
        return Err(PoolError::NotMember);
    }

    // Verify agent has available KREDIT
    if agent.kredit_balance < amount as i64 {
        return Err(PoolError::InsufficientKredit);
    }

    // Check pool cap
    let new_total = collective.kredit_pool.total_kredit + amount;
    if new_total > collective.kredit_pool.pool_cap {
        return Err(PoolError::PoolCapExceeded);
    }

    // Transfer KREDIT
    // (Agent KREDIT decreases, pool increases)
    collective.kredit_pool.total_kredit += amount;
    collective.kredit_pool.available_kredit += amount;
    collective.kredit_pool.contributions
        .entry(agent.agent_id.clone())
        .and_modify(|c| *c += amount)
        .or_insert(amount);

    Ok(())
}
```

#### 2.3 Pool Cap Calculation

The collective pool cap depends on sponsor backing:

```rust
pub fn calculate_pool_cap(
    sponsors: &[SponsorInfo],
    collective_type: &CollectiveType,
) -> u64 {
    let base_cap: u64 = sponsors.iter()
        .map(|s| {
            let sponsor_capacity = s.civ_score * s.sap_available as f64;
            (sponsor_capacity * 0.1) as u64 // 10% of capacity per sponsor
        })
        .sum();

    let type_multiplier = match collective_type {
        CollectiveType::TaskForce { .. } => 1.0,
        CollectiveType::CapabilityPool { .. } => 1.5,
        CollectiveType::Regional { .. } => 2.0,
        CollectiveType::IntentionFocused { .. } => 2.5,
        CollectiveType::Research { .. } => 1.5,
    };

    (base_cap as f64 * type_multiplier) as u64
}
```

### 3. Collective Constraints

#### 3.1 Aggregate Limits

Constitutional constraints from MIP-E-004 apply to collectives:

```rust
pub struct CollectiveConstraints {
    /// Constitutional constraints (immutable)
    pub constitutional: CollectiveConstitutionalConstraints,
    /// Operational constraints (configurable)
    pub operational: CollectiveOperationalConstraints,
    /// Sponsor-imposed constraints
    pub sponsor_constraints: Vec<SponsorConstraint>,
}

pub struct CollectiveConstitutionalConstraints {
    /// Collective cannot vote on governance
    pub can_vote_governance: bool,  // Always false
    /// Collective cannot become validator
    pub can_become_validator: bool, // Always false
    /// Collective cannot govern HEARTH
    pub can_govern_hearth: bool,    // Always false
    /// Collective cannot sponsor agents
    pub can_sponsor_agents: bool,   // Always false
    /// Collective cannot create sub-collectives
    pub can_create_subcollectives: bool, // Always false
}

pub struct CollectiveOperationalConstraints {
    /// Maximum KREDIT spend per action
    pub max_kredit_per_action: u64,
    /// Maximum actions per hour (aggregate)
    pub max_actions_per_hour: u32,
    /// Maximum concurrent operations
    pub max_concurrent_ops: u32,
    /// Geographic restrictions
    pub geographic_scope: Option<GeographicScope>,
    /// Domain restrictions
    pub domain_restrictions: Vec<String>,
}
```

#### 3.2 Anti-Swarm Protections

Prevent collectives from overwhelming the network:

```rust
pub struct SwarmProtection {
    /// Maximum collective density per bioregion
    pub max_collectives_per_bioregion: u32,
    /// Maximum aggregate KREDIT per bioregion
    pub max_regional_kredit: u64,
    /// Minimum time between collective actions
    pub action_cooldown_ms: u64,
    /// Maximum network bandwidth consumption
    pub max_bandwidth_pct: f64,
    /// Automatic throttling thresholds
    pub throttle_thresholds: ThrottleThresholds,
}

pub struct ThrottleThresholds {
    /// Throttle at this KREDIT consumption rate
    pub kredit_rate_threshold: f64,
    /// Throttle at this action rate
    pub action_rate_threshold: f64,
    /// Throttle severity
    pub throttle_factor: f64,
}

pub fn check_swarm_limits(
    collective: &AgentCollective,
    action: &CollectiveAction,
    network_state: &NetworkState,
) -> SwarmCheckResult {
    // Check regional density
    let regional_collectives = network_state.collectives_in_region(&collective.region());
    if regional_collectives.len() >= network_state.protection.max_collectives_per_bioregion as usize {
        return SwarmCheckResult::Blocked(BlockReason::RegionalDensity);
    }

    // Check aggregate KREDIT
    let regional_kredit: u64 = regional_collectives.iter()
        .map(|c| c.kredit_pool.total_kredit)
        .sum();
    if regional_kredit >= network_state.protection.max_regional_kredit {
        return SwarmCheckResult::Blocked(BlockReason::RegionalKreditLimit);
    }

    // Check rate limiting
    let recent_actions = collective.actions_in_window(3600); // last hour
    if recent_actions >= collective.constraints.operational.max_actions_per_hour {
        return SwarmCheckResult::Throttled;
    }

    SwarmCheckResult::Allowed
}
```

### 4. Decision Protocols

#### 4.1 Protocol Types

```rust
pub enum DecisionProtocol {
    /// Unanimous consent required
    Unanimous,
    /// Supermajority (⅔)
    Supermajority,
    /// Simple majority
    Majority,
    /// Weighted voting (by KREDIT contribution)
    WeightedMajority { threshold: f64 },
    /// Designated coordinator decides
    CoordinatorDecides {
        coordinator: AgentId,
        veto_threshold: f64, // Members can veto
    },
    /// Liquid democracy (delegation)
    LiquidDemocracy {
        delegations: HashMap<AgentId, AgentId>,
    },
    /// Consent-based (no objections)
    ConsentBased { objection_period_ms: u64 },
}
```

#### 4.2 Decision Making

```rust
pub struct CollectiveDecision {
    /// Decision ID
    pub decision_id: DecisionId,
    /// Proposal
    pub proposal: Proposal,
    /// Votes cast
    pub votes: HashMap<AgentId, Vote>,
    /// Status
    pub status: DecisionStatus,
    /// Deadline
    pub deadline: u64,
    /// Result
    pub result: Option<DecisionResult>,
}

pub enum Vote {
    /// Support
    For,
    /// Oppose
    Against,
    /// Abstain
    Abstain,
    /// Delegate to another member
    Delegate(AgentId),
    /// Conditional support
    Conditional { condition: String },
}

pub fn resolve_decision(
    decision: &mut CollectiveDecision,
    protocol: &DecisionProtocol,
    members: &[CollectiveMember],
) -> DecisionResult {
    let total_weight: f64 = members.iter().map(|m| m.voting_weight).sum();
    let for_weight: f64 = decision.votes.iter()
        .filter(|(_, v)| matches!(v, Vote::For))
        .map(|(id, _)| members.iter().find(|m| &m.agent_id == id).unwrap().voting_weight)
        .sum();

    match protocol {
        DecisionProtocol::Unanimous => {
            if decision.votes.values().all(|v| matches!(v, Vote::For)) {
                DecisionResult::Approved
            } else {
                DecisionResult::Rejected
            }
        },
        DecisionProtocol::Supermajority => {
            if for_weight / total_weight >= 0.667 {
                DecisionResult::Approved
            } else {
                DecisionResult::Rejected
            }
        },
        DecisionProtocol::Majority => {
            if for_weight / total_weight > 0.5 {
                DecisionResult::Approved
            } else {
                DecisionResult::Rejected
            }
        },
        // ... other protocols
    }
}
```

### 5. Human Oversight Councils

#### 5.1 Council Structure

Every collective requires a human oversight council:

```rust
pub struct OversightCouncil {
    /// Council ID
    pub council_id: CouncilId,
    /// Human members (DIDs)
    pub members: Vec<OversightMember>,
    /// Minimum members required
    pub quorum: u32,
    /// Council powers
    pub powers: CouncilPowers,
    /// Meeting frequency (epochs)
    pub meeting_frequency: u32,
    /// Emergency contact
    pub emergency_contact: String,
}

pub struct OversightMember {
    /// Human DID
    pub human_did: String,
    /// Is sponsor of member agents
    pub is_sponsor: bool,
    /// CIV score
    pub civ_score: f64,
    /// Role
    pub role: OversightRole,
    /// Term end
    pub term_end: u64,
}

pub enum OversightRole {
    /// Can pause/resume collective
    Supervisor,
    /// Can review but not pause
    Auditor,
    /// Emergency contact only
    EmergencyContact,
}

pub struct CouncilPowers {
    /// Can pause collective operations
    pub can_pause: bool,
    /// Can dissolve collective
    pub can_dissolve: bool,
    /// Can modify constraints
    pub can_modify_constraints: bool,
    /// Can remove members
    pub can_remove_members: bool,
    /// Can access full audit logs
    pub can_audit: bool,
}
```

#### 5.2 Council Requirements

| Collective Size | Min Council Size | Min Sponsors on Council |
|-----------------|------------------|------------------------|
| 2-10 agents | 2 | 1 |
| 11-50 agents | 3 | 2 |
| 51-100 agents | 5 | 3 |
| 101-200 agents | 7 | 4 |
| 201+ agents | 9 | 5 |

#### 5.3 Oversight Actions

```rust
pub enum OversightAction {
    /// Pause collective operations
    Pause { reason: String, duration: Option<u64> },
    /// Resume operations
    Resume { conditions: Vec<String> },
    /// Dissolve collective
    Dissolve { reason: String, asset_distribution: AssetDistribution },
    /// Modify constraints
    ModifyConstraints { new_constraints: CollectiveOperationalConstraints },
    /// Remove agent from collective
    RemoveAgent { agent_id: AgentId, reason: String },
    /// Emergency KREDIT freeze
    FreezeKredit { amount: u64, reason: String },
    /// Trigger audit
    TriggerAudit { scope: AuditScope },
}

pub fn execute_oversight_action(
    action: OversightAction,
    council: &OversightCouncil,
    requester: &str,
    signatures: &[Signature],
) -> Result<(), OversightError> {
    // Verify requester is council member
    if !council.members.iter().any(|m| m.human_did == requester) {
        return Err(OversightError::NotCouncilMember);
    }

    // Verify quorum for action
    let required_sigs = match &action {
        OversightAction::Dissolve { .. } => council.members.len(), // Unanimous
        OversightAction::Pause { .. } => 1, // Any member can pause
        _ => (council.members.len() / 2) + 1, // Majority
    };

    if signatures.len() < required_sigs {
        return Err(OversightError::InsufficientSignatures);
    }

    // Execute action
    // ...

    Ok(())
}
```

### 6. Collective Reputation

#### 6.1 Reputation Structure

```rust
pub struct CollectiveReputation {
    /// Overall score (0.0 to 1.0)
    pub score: f64,
    /// Component scores
    pub components: ReputationComponents,
    /// History of significant events
    pub history: Vec<ReputationEvent>,
    /// Decay rate per epoch
    pub decay_rate: f64,
    /// Last updated
    pub last_updated: u64,
}

pub struct ReputationComponents {
    /// Task completion rate
    pub reliability: f64,
    /// Quality of outputs
    pub quality: f64,
    /// Constraint compliance
    pub compliance: f64,
    /// Coordination efficiency
    pub efficiency: f64,
    /// Human oversight satisfaction
    pub oversight_satisfaction: f64,
}

pub struct ReputationEvent {
    /// Event type
    pub event_type: ReputationEventType,
    /// Impact on score
    pub impact: f64,
    /// Timestamp
    pub timestamp: u64,
    /// Details
    pub details: String,
}

pub enum ReputationEventType {
    /// Successful task completion
    TaskSuccess,
    /// Task failure
    TaskFailure,
    /// Constraint violation
    ConstraintViolation,
    /// Oversight intervention
    OversightIntervention,
    /// Member defection
    MemberDefection,
    /// Positive community feedback
    PositiveFeedback,
    /// Negative community feedback
    NegativeFeedback,
}
```

#### 6.2 Reputation Effects

| Reputation Score | Effects |
|------------------|---------|
| < 0.3 | Automatic dissolution review |
| 0.3 - 0.5 | Enhanced oversight requirements |
| 0.5 - 0.7 | Standard operation |
| 0.7 - 0.85 | Reduced oversight frequency |
| > 0.85 | Eligible for expanded capabilities |

#### 6.3 Sponsor Impact

Collective reputation affects sponsors:

```rust
pub fn calculate_sponsor_civ_impact(
    sponsor: &SponsorInfo,
    collectives: &[AgentCollective],
) -> f64 {
    let collective_factor: f64 = collectives.iter()
        .filter(|c| c.has_sponsor(&sponsor.did))
        .map(|c| {
            let weight = c.sponsor_contribution_pct(&sponsor.did);
            (c.reputation.score - 0.5) * weight * 0.1 // +/- 5% max
        })
        .sum();

    collective_factor.clamp(-0.1, 0.1)
}
```

### 7. Collective Lifecycle

```rust
pub enum CollectiveStatus {
    /// Being formed
    Forming {
        initiated_at: u64,
        founding_members: Vec<AgentId>,
        pending_sponsors: Vec<String>,
    },
    /// Active and operational
    Active {
        activated_at: u64,
    },
    /// Temporarily paused
    Paused {
        paused_at: u64,
        reason: String,
        resume_conditions: Vec<String>,
    },
    /// Being dissolved
    Dissolving {
        initiated_at: u64,
        asset_distribution: AssetDistribution,
    },
    /// Fully dissolved
    Dissolved {
        dissolved_at: u64,
        final_report: String,
    },
}

pub struct AssetDistribution {
    /// KREDIT distribution
    pub kredit_distribution: HashMap<AgentId, u64>,
    /// Data/artifact ownership
    pub data_ownership: DataOwnership,
    /// Reputation inheritance
    pub reputation_inheritance: ReputationInheritance,
}
```

---

## Rationale

### Why Not Just Individual Agents?

Many tasks benefit from coordination:
- Large data processing (parallel execution)
- Complex analysis (multiple perspectives)
- Continuous operation (redundancy)
- Specialized workflows (capability composition)

Forcing these into individual agents creates artificial limitations.

### Why Require Human Oversight?

AI collectives could coordinate in ways that evade human understanding. Required oversight ensures:
- Humans remain in control
- Collective goals stay aligned
- Problems are caught early
- Accountability is clear

### Why Pool Caps?

Unlimited KREDIT pooling could create:
- Resource concentration
- Network domination
- Swarm attacks

Caps tied to sponsor backing ensure skin in the game.

### Why Reputation Inheritance?

When collectives dissolve, reputational information shouldn't be lost. But it also shouldn't fully transfer - the new context is different. Partial inheritance balances information preservation with fresh starts.

---

## Backwards Compatibility

### Individual Agents

All individual agent functionality (MIP-E-004) remains unchanged. Collectives are an additional layer.

### KREDIT

Individual KREDIT can flow to collective pools and back. Total KREDIT in the system is unchanged.

### Constitutional Constraints

All constitutional constraints from MIP-E-004 apply to collectives. Collectives cannot do what individual agents cannot do.

---

## Security Considerations

### 1. Collective Takeover

A majority of agents could try to dominate a collective. Mitigation:
- Oversight council can remove agents
- Voting weight caps
- Sponsor distribution requirements

### 2. Sponsor Collusion

Sponsors might collude to create harmful collectives. Mitigation:
- CIV requirements for sponsors
- Transparent collective registry
- Community reporting mechanisms

### 3. Coordination Attacks

Collectives could coordinate attacks on the network. Mitigation:
- Swarm protection limits
- Regional KREDIT caps
- Rate limiting

### 4. Oversight Capture

Oversight councils could be captured by sponsors. Mitigation:
- Term limits
- CIV requirements
- Audit Guild review

---

## Implementation Status

| Component | SDK Module | Status |
|-----------|------------|--------|
| Collective Structure | `agentic::collective` | Planned |
| KREDIT Pools | `agentic::collective_kredit` | Planned |
| Collective Constraints | `agentic::collective_constraints` | Planned |
| Decision Protocols | `agentic::collective_decisions` | Planned |
| Oversight Councils | `agentic::oversight` | Planned |
| Collective Reputation | `agentic::collective_reputation` | Planned |
| Lifecycle Management | `agentic::collective_lifecycle` | Planned |

---

## References

- [MIP-E-004: Agentic Economy Framework](./MIP-E-004_AGENTIC_ECONOMY_FRAMEWORK.md)
- [Spore Constitution v0.24](../architecture/THE%20MYCELIX%20SPORE%20CONSTITUTION%20(v0.24).md)
- Multi-Agent Systems literature
- Swarm Intelligence research
- Human-AI Teaming frameworks

---

## Copyright

This document is licensed under Apache 2.0.

---

*"Individual agents are instruments. Collectives are orchestras. The oversight council is the conductor who ensures the music serves the audience, not just the musicians."*
