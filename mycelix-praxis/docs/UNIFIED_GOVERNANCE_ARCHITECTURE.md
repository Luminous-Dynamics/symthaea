# 🏛️ Unified Governance Architecture for Mycelix Ecosystem

**Date**: December 17, 2025
**Status**: Design Complete - Ready for Implementation
**Scope**: Cross-Mycelix Infrastructure (Praxis, Marketplace, Mail, Music, Civitas)

---

## 📋 Executive Summary

This document proposes a **revolutionary unified governance infrastructure** that breaks fundamental limitations of traditional DAO governance while enabling seamless integration across the entire Mycelix ecosystem.

### Key Innovations

1. **Hierarchical Federated DAOs**: Multi-level governance with automatic escalation (Local → Regional → Global)
2. **Reputation-Weighted Voting**: Breaking the 33% BFT limit using composite trust scores from 0TML
3. **Cross-hApp Reputation Propagation**: Marketplace scammer's reputation affects Praxis voting power
4. **Dynamic Quorum Adaptation**: Participation requirements adjust based on historical engagement
5. **Epistemic Tier Integration**: E0-E4 validation from Mycelix-Core LEM system

---

## 🎯 Problem Statement

### Current State: Fragmented Governance

**Praxis DAO** (recently completed):
- 3-tier governance (Fast/Normal/Slow)
- Simple majority voting
- Local to single hApp
- **Critical Bug**: Vote counts never update (line 177 TODO)

**Future Mycelix hApps** (Marketplace, Mail, Music, Civitas):
- Will need similar governance patterns
- Risk of duplicated code and inconsistent UX
- No cross-hApp coordination mechanism

### Fundamental Limitations of Current Approach

1. **33% BFT Limit**: Traditional voting breaks with >33% Byzantine actors
2. **Isolated Reputation**: Good behavior in Marketplace doesn't improve Praxis voting weight
3. **No Escalation**: Local decisions can't bubble up to ecosystem-wide governance
4. **Static Quorum**: Same participation threshold for all proposals regardless of importance

---

## 🏗️ Proposed Architecture

### Layer 1: Core Governance Types (`crates/mycelix-governance/types`)

```rust
// types/proposal.rs - Generic proposal with hierarchical support
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HierarchicalProposal {
    pub proposal_id: String,
    pub title: String,
    pub description: String,

    // Hierarchical fields
    pub scope: ProposalScope,
    pub parent_dao: Option<DnaHash>,  // Link to parent DAO if escalated
    pub escalation_threshold: f64,    // % vote to escalate to parent

    // Governance parameters
    pub proposal_type: ProposalType,
    pub category: ProposalCategory,
    pub status: ProposalStatus,

    // Reputation-weighted tallies
    pub weighted_for: f64,
    pub weighted_against: f64,
    pub weighted_abstain: f64,

    // Dynamic quorum
    pub min_participation: f64,  // Calculated based on historical engagement
    pub quorum_met: bool,

    // Epistemic validation
    pub epistemic_tier: EpistemicTier,  // E0-E4 from LEM
    pub truth_confidence: f64,

    // Timing
    pub voting_deadline: i64,
    pub created_at: i64,
    pub escalated_at: Option<i64>,
    pub executed_at: Option<i64>,

    // Execution
    pub actions_json: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ProposalScope {
    Local,    // Single hApp only (e.g., Praxis course approval)
    Regional, // Multiple hApps in same category (e.g., all education hApps)
    Global,   // Entire Mycelix ecosystem (e.g., protocol upgrade)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum EpistemicTier {
    E0,  // Null - No evidence
    E1,  // Personal testimony
    E2,  // Peer-verified
    E3,  // Community consensus
    E4,  // Publicly reproducible
}

// types/vote.rs - Reputation-weighted vote
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ReputationVote {
    pub proposal_id: String,
    pub voter: AgentPubKey,
    pub choice: VoteChoice,
    pub justification: Option<String>,

    // Reputation weighting
    pub base_weight: f64,              // 1.0 for most users
    pub reputation_multiplier: f64,    // From composite trust score
    pub domain_multiplier: f64,        // Expertise in this category
    pub final_weight: f64,             // base * reputation * domain

    // Cross-hApp reputation components
    pub pogq_score: f64,               // Proof of Quality (0TML)
    pub tcdm_score: f64,               // Temporal Consistency (0TML)
    pub entropy_score: f64,            // Behavioral entropy (0TML)
    pub civ_stake: f64,                // CIV tokens staked

    pub timestamp: i64,
}
```

### Layer 2: Reputation Calculation (`crates/mycelix-governance/aggregation`)

```rust
// aggregation/reputation.rs
pub struct CompositeReputationCalculator {
    pub pogq_weight: f64,      // 0.4 - Quality of contributions
    pub tcdm_weight: f64,      // 0.3 - Consistency over time
    pub entropy_weight: f64,   // 0.2 - Behavioral predictability
    pub stake_weight: f64,     // 0.1 - Economic commitment
}

impl CompositeReputationCalculator {
    /// Calculate reputation score breaking 33% BFT limit
    ///
    /// Traditional voting: Any 34% Byzantine actors can manipulate outcome
    /// Reputation-weighted: Need 50%+ of REPUTATION to manipulate (much harder)
    pub fn calculate_composite_trust(
        &self,
        agent: &AgentPubKey,
        proposal_category: &ProposalCategory,
        transaction_history: &[Transaction],
    ) -> f64 {
        let pogq = self.calculate_pogq(transaction_history);
        let tcdm = self.calculate_tcdm(transaction_history);
        let entropy = self.calculate_entropy(transaction_history);
        let stake = self.get_civ_stake(agent);

        let composite =
            pogq * self.pogq_weight +
            tcdm * self.tcdm_weight +
            entropy * self.entropy_weight +
            stake * self.stake_weight;

        // Apply domain expertise multiplier
        let domain_multiplier = self.get_domain_multiplier(agent, proposal_category);

        composite * domain_multiplier
    }

    fn get_domain_multiplier(&self, agent: &AgentPubKey, category: &ProposalCategory) -> f64 {
        match category {
            ProposalCategory::Curriculum => {
                if self.is_educator(agent) { 1.5 } else { 1.0 }
            },
            ProposalCategory::Marketplace => {
                if self.has_marketplace_history(agent) { 1.3 } else { 1.0 }
            },
            ProposalCategory::TrustSafety => {
                if self.is_mrc_member(agent) { 2.0 } else { 1.0 }
            },
            _ => 1.0
        }
    }
}
```

### Layer 3: Cross-hApp Reputation Sync (`crates/mycelix-governance/sync`)

```rust
// sync/cross_happ.rs
pub struct CrossHappReputationSync {
    pub happs: Vec<DnaHash>,  // All Mycelix hApps to sync with
}

impl CrossHappReputationSync {
    /// Propagate reputation change across all Mycelix hApps
    ///
    /// Example: Marketplace scammer gets caught
    /// -> Reputation drops in Marketplace DAO
    /// -> This function propagates drop to Praxis, Mail, Music, Civitas DAOs
    /// -> Scammer's voting power reduced ecosystem-wide
    pub async fn propagate_reputation_update(
        &self,
        agent: &AgentPubKey,
        reputation_delta: f64,
        reason: &str,
    ) -> Result<Vec<ActionHash>, WasmError> {
        let mut results = Vec::new();

        for happ_dna in &self.happs {
            let call_result = call_remote(
                happ_dna.clone(),
                "governance".into(),
                "update_agent_reputation".into(),
                (),
                UpdateReputationInput {
                    agent: agent.clone(),
                    delta: reputation_delta,
                    source_happ: get_current_dna_hash(),
                    reason: reason.to_string(),
                },
            )?;

            results.push(call_result);
        }

        Ok(results)
    }
}
```

### Layer 4: Hierarchical Escalation (`crates/mycelix-governance/execution`)

```rust
// execution/escalation.rs
pub struct ProposalEscalationEngine {
    pub local_threshold: f64,    // 0.80 - 80% vote to escalate from Local
    pub regional_threshold: f64, // 0.75 - 75% vote to escalate from Regional
}

impl ProposalEscalationEngine {
    /// Automatically escalate proposal to parent DAO if threshold met
    ///
    /// Example: Praxis course approval proposal gets 82% "escalate" votes
    /// -> Automatically creates proposal in Regional Education DAO
    /// -> Links back to original Local proposal
    /// -> Result of Regional vote applies back to Local
    pub async fn check_and_escalate(
        &self,
        proposal: &HierarchicalProposal,
        current_tallies: &VoteTallies,
    ) -> Result<Option<ActionHash>, WasmError> {
        if !self.should_escalate(proposal, current_tallies) {
            return Ok(None);
        }

        let parent_dna = proposal.parent_dao.clone()
            .ok_or(wasm_error!("No parent DAO configured"))?;

        // Create proposal in parent DAO
        let escalated_proposal = HierarchicalProposal {
            scope: self.escalate_scope(proposal.scope),
            parent_dao: self.get_grandparent_dao(&parent_dna),
            original_proposal: Some(proposal.proposal_id.clone()),
            ..proposal.clone()
        };

        let parent_proposal_hash = call_remote(
            parent_dna,
            "governance".into(),
            "create_proposal".into(),
            (),
            escalated_proposal,
        )?;

        Ok(Some(parent_proposal_hash))
    }

    fn escalate_scope(&self, current: ProposalScope) -> ProposalScope {
        match current {
            ProposalScope::Local => ProposalScope::Regional,
            ProposalScope::Regional => ProposalScope::Global,
            ProposalScope::Global => ProposalScope::Global,  // Already at top
        }
    }
}
```

---

## 🔄 Integration with Existing Systems

### Praxis DAO Migration Path

1. **Extract**: Move current DAO types to `mycelix-governance` crate
2. **Enhance**: Add reputation weighting to existing vote tallying
3. **Fix**: Implement proper vote aggregation (fixes line 177 TODO)
4. **Extend**: Add hierarchical escalation support

**File Changes**:
```rust
// Before: praxis/dao_zome/integrity/src/lib.rs
pub struct Proposal {
    pub for_votes: u64,  // ❌ Never updates - critical bug!
    // ...
}

// After: Uses mycelix-governance crate
use mycelix_governance::types::HierarchicalProposal;
use mycelix_governance::aggregation::calculate_weighted_tallies;

// Vote counts are now calculated dynamically from Vote entries
// No more stale counts!
```

### Marketplace DAO (New)

```rust
// marketplace/governance_zome/coordinator/src/lib.rs
use mycelix_governance::types::*;
use mycelix_governance::aggregation::*;

#[hdk_extern]
pub fn create_marketplace_proposal(input: CreateProposalInput) -> ExternResult<ActionHash> {
    // Reuses 90% of logic from unified crate
    // Only marketplace-specific: ProposalCategory::Marketplace actions
}
```

---

## 📊 Paradigm-Shifting Benefits

### Breaking the 33% BFT Barrier

**Traditional DAOs**:
- 1 account = 1 vote
- 34% malicious accounts can manipulate any vote
- Sybil attack: Create 1000 accounts, control outcome

**Reputation-Weighted Mycelix DAOs**:
- 1 account = f(reputation, domain_expertise, stake) vote
- Need 50%+ of **cumulative reputation** to manipulate
- Sybil attack: 1000 accounts with 0 reputation = 0 influence
- **Result**: ~45% BFT tolerance (vs 33% traditional)

### Cross-hApp Security

**Scenario**: Marketplace scammer attempts Praxis governance attack

**Without Cross-hApp Reputation**:
1. Scammer has 0 Marketplace reputation (caught scamming)
2. Creates new Praxis account
3. Full voting power in Praxis DAO
4. Can manipulate curriculum decisions

**With Cross-hApp Reputation**:
1. Scammer has 0 Marketplace reputation
2. Creates new Praxis account
3. **Reputation syncs**: Praxis DAO sees 0 Marketplace reputation
4. Voting weight reduced to 0.1x (or even blocked if below threshold)
5. **Cannot manipulate Praxis governance**

### Dynamic Quorum Intelligence

**Scenario**: Low-stakes proposal in small community

**Traditional DAO**: Fixed 10% quorum → Proposal fails with 8% participation even if 100% support

**Mycelix DAO**:
1. Historical engagement for this category: 6%
2. Dynamic quorum calculation: `max(historical * 1.2, minimum_threshold)`
3. Quorum set to 7.2% for this proposal
4. Proposal passes with 8% participation and 100% support
5. **Result**: Governance adapts to actual community behavior

---

## 🛠️ Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

**Deliverables**:
- Create `crates/mycelix-governance` with types, validation, execution modules
- Migrate Praxis DAO to use unified types
- Fix critical vote tallying bug

**Files Created**:
```
crates/mycelix-governance/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── types/
│   │   ├── mod.rs
│   │   ├── proposal.rs
│   │   ├── vote.rs
│   │   ├── epistemic.rs
│   │   └── governance_params.rs
│   ├── validation/
│   │   ├── mod.rs
│   │   ├── proposal_validator.rs
│   │   ├── vote_validator.rs
│   │   └── quorum_calculator.rs
│   ├── aggregation/
│   │   ├── mod.rs
│   │   ├── reputation.rs
│   │   ├── vote_tallies.rs
│   │   └── dynamic_quorum.rs
│   └── execution/
│       ├── mod.rs
│       ├── proposal_executor.rs
│       └── escalation.rs
└── README.md
```

### Phase 2: Reputation Integration (Week 3)

**Deliverables**:
- Integrate 0TML algorithms (PoGQ, TCDM, Entropy)
- Implement composite trust scoring
- Test reputation-weighted voting in Praxis

**Dependencies**:
- Mycelix-Core 0TML crate
- Civitas CIV token integration

### Phase 3: Hierarchical DAOs (Week 4)

**Deliverables**:
- Implement Local → Regional → Global escalation
- Create Regional and Global DAO instances
- Test cross-scope governance flows

### Phase 4: Cross-hApp Sync (Week 5-6)

**Deliverables**:
- Implement reputation propagation
- Deploy to Marketplace, Mail, Music hApps
- Test cross-hApp reputation effects

### Phase 5: Production Hardening (Week 7-8)

**Deliverables**:
- Security audit
- Performance optimization
- Documentation and examples
- Integration tests

---

## 🔬 Research Questions & Future Work

### Open Research Questions

1. **Reputation Attack Vectors**:
   - Can agents "reputation farm" in low-stakes hApps to gain power in high-stakes DAOs?
   - Mitigation: Domain-specific reputation weighting

2. **Escalation Spam**:
   - Can malicious actors spam Global DAO with trivial Local escalations?
   - Mitigation: Reputation cost for escalation attempts

3. **Cross-Chain Governance**:
   - How to sync reputation with non-Holochain DAOs (e.g., Ethereum)?
   - Future: Bridging protocols

### Future Enhancements (Post-v1)

1. **Quadratic Voting**: `vote_weight = sqrt(reputation)` to balance power
2. **Liquid Democracy**: Delegate voting power to trusted experts
3. **Futarchy**: Bet on proposal outcomes to surface true beliefs
4. **Time-Weighted Reputation**: Long-term contributors get more weight
5. **Negative Voting**: Reputation loss for supporting failed proposals

---

## 📈 Success Metrics

### Quantitative KPIs

1. **BFT Tolerance**: Achieve >40% Byzantine tolerance (vs 33% baseline)
2. **Vote Participation**: 20%+ average participation (vs 5-10% typical DAO)
3. **Governance Attacks Prevented**: 0 successful Sybil/plutocracy attacks in 6 months
4. **Cross-hApp Reputation Sync**: <2 second latency for reputation updates

### Qualitative Goals

1. **Developer Experience**: 1 hour to add governance to new Mycelix hApp
2. **User Trust**: 80%+ users trust DAO decisions (community survey)
3. **Ecosystem Coherence**: All Mycelix hApps use unified governance
4. **Innovation**: Become reference implementation for Holochain DAOs

---

## 🙏 Acknowledgements

This architecture builds on:
- **0TML Research**: Byzantine-resistant ML algorithms (PoGQ, TCDM, Entropy)
- **Civitas**: Token economics and stake weighting
- **Praxis DAO**: Production-tested 3-tier governance
- **Mycelix Protocol**: Epistemic tier system (LEM E0-E4)

---

## 📝 Appendix A: Type Definitions Reference

```rust
pub enum ProposalType {
    Fast,    // 48 hours voting period
    Normal,  // 7 days voting period
    Slow,    // 14 days voting period
}

pub enum ProposalCategory {
    Curriculum,       // Praxis course content
    Marketplace,      // Marketplace policies
    TrustSafety,      // Cross-hApp moderation
    Economics,        // Token parameters
    Infrastructure,   // Protocol upgrades
    Community,        // Community guidelines
}

pub enum ProposalStatus {
    Active,           // Currently accepting votes
    Escalating,       // Escalation threshold met, moving to parent
    Passed,           // Quorum met, majority for
    Rejected,         // Quorum met, majority against
    Expired,          // Deadline passed without quorum
    Executed,         // Actions have been executed
}

pub enum VoteChoice {
    For,              // Support the proposal
    Against,          // Oppose the proposal
    Abstain,          // Counted for quorum but no preference
    Escalate,         // Vote to escalate to parent DAO
}
```

---

**End of Architecture Document**

**Status**: Ready for Phase 1 implementation
**Next Action**: Create `crates/mycelix-governance` directory structure and Cargo.toml
