# Mycelix-Agora: Decentralized Governance & Polling

## Vision & Design Document

**Version**: 1.0.0
**Created**: December 30, 2025
**Status**: Design Phase
**Priority**: Tier 1 - Ecosystem Expansion

---

## Executive Summary

Mycelix-Agora is the governance layer for the entire Mycelix ecosystem and any community that builds on it. It provides flexible, Sybil-resistant voting mechanisms with support for multiple governance models—from simple polls to complex constitutional amendments. Integrated with Attest for identity verification and MATL for reputation-weighted voting, Agora enables legitimate collective decision-making at any scale.

### Why Agora?

Governance is where coordination becomes conscious:
- **DAOs need voting**: Token-based voting has plutocracy problems
- **Communities need consensus**: Not everything can be automated
- **Policies need legitimacy**: Decisions must feel fair to be accepted
- **Experiments need iteration**: Governance itself must be evolvable

Agora provides the primitives for any governance model, from direct democracy to liquid democracy to futarchy.

---

## Core Principles

### 1. Legitimacy Through Transparency
Every vote is auditable. Every proposal has clear history. Every decision can be traced.

### 2. Sybil Resistance
One person, one voice (weighted by reputation, not wealth).

### 3. Governance Pluralism
No one-size-fits-all. Communities choose their own governance models.

### 4. Evolutionary Governance
Governance rules themselves can be changed through governance.

### 5. Subsidiarity
Decisions should be made at the most local level possible.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Governance Contexts                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │   Mycelix    │  │  Community   │  │   Project/hApp           │   │
│  │ Ecosystem    │  │   DAO        │  │   Governance             │   │
│  │ Governance   │  │              │  │                          │   │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                          Agora hApp                                  │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                     Coordinator Zomes                            ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           ││
│  │  │ Proposal │ │ Voting   │ │Delegation│ │  Result  │           ││
│  │  │ Manager  │ │ Engine   │ │ System   │ │ Executor │           ││
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           ││
│  │  │  Space   │ │Constitution│ │ Signal  │ │Analytics │           ││
│  │  │ Manager  │ │  Engine   │ │ Polling │ │ & Trends │           ││
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           ││
│  └─────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────┤
│                      Voting Mechanisms                               │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      │
│  │Simple   │ │Quadratic│ │Conviction│ │ Ranked  │ │Futarchy │      │
│  │Majority │ │ Voting  │ │ Voting   │ │ Choice  │ │         │      │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Model

### Core Entry Types

```rust
/// A governance space (DAO, community, project)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GovernanceSpace {
    /// Unique space identifier
    pub space_id: String,
    /// Human-readable name
    pub name: String,
    /// Description
    pub description: String,
    /// Space type
    pub space_type: SpaceType,
    /// Governance configuration
    pub config: GovernanceConfig,
    /// Membership rules
    pub membership: MembershipRules,
    /// Constitution reference (if applicable)
    pub constitution: Option<ActionHash>,
    /// Created by
    pub creator: AgentPubKey,
    /// Created at
    pub created_at: Timestamp,
    /// Is this space active?
    pub active: bool,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum SpaceType {
    /// Ecosystem-wide governance
    Ecosystem,
    /// Community DAO
    CommunityDAO,
    /// Project-specific
    Project { project_id: String },
    /// hApp governance
    HAppGovernance { happ_id: String },
    /// Geographic community
    Geographic { location: String },
    /// Topical (interest-based)
    Topical { topic: String },
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct GovernanceConfig {
    /// Default voting mechanism
    pub default_voting: VotingMechanism,
    /// Quorum requirements
    pub quorum: QuorumConfig,
    /// Proposal requirements
    pub proposal_requirements: ProposalRequirements,
    /// Voting weight calculation
    pub weight_calculation: WeightCalculation,
    /// Decision thresholds
    pub thresholds: DecisionThresholds,
    /// Delegation allowed?
    pub delegation_enabled: bool,
    /// Can governance config itself be changed?
    pub meta_governance_enabled: bool,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum VotingMechanism {
    /// Simple majority (>50%)
    SimpleMajority,
    /// Supermajority (configurable threshold)
    SuperMajority { threshold: f64 },
    /// Quadratic voting (cost = votes^2)
    Quadratic { credits_per_period: u64 },
    /// Conviction voting (time-weighted)
    Conviction { decay_rate: f64, max_conviction: f64 },
    /// Ranked choice / instant runoff
    RankedChoice { rounds: u32 },
    /// Approval voting (vote for multiple)
    Approval { max_approvals: u32 },
    /// Futarchy (prediction market based)
    Futarchy { market_duration: u64, success_metric: String },
    /// Consensus (everyone must agree)
    Consensus,
    /// Lazy consensus (passes unless objection)
    LazyConsensus { objection_period: u64 },
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct QuorumConfig {
    /// Minimum participation (percentage of eligible)
    pub min_participation: f64,
    /// Minimum absolute votes
    pub min_absolute_votes: u64,
    /// Quorum can be different for different proposal types
    pub type_overrides: HashMap<String, f64>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct ProposalRequirements {
    /// Minimum identity level to propose
    pub min_identity_level: IdentityLevel,
    /// Minimum reputation to propose
    pub min_reputation: f64,
    /// Deposit required (refundable if passes threshold)
    pub deposit: Option<DepositConfig>,
    /// Sponsors required
    pub sponsors_required: u32,
    /// Discussion period before voting
    pub discussion_period: u64,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum WeightCalculation {
    /// One person, one vote
    EqualWeight,
    /// Weighted by MATL reputation
    ReputationWeighted { max_weight: f64 },
    /// Weighted by stake
    StakeWeighted,
    /// Weighted by membership duration
    TenureWeighted { max_multiplier: f64 },
    /// Quadratic (square root of stake/reputation)
    QuadraticWeight,
    /// Custom formula
    Custom { formula: String },
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct MembershipRules {
    /// Open to all?
    pub open_membership: bool,
    /// Minimum identity level
    pub min_identity_level: IdentityLevel,
    /// Minimum reputation
    pub min_reputation: Option<f64>,
    /// Required credentials
    pub required_credentials: Vec<String>,
    /// Vouches from existing members required
    pub vouches_required: u32,
    /// Membership can be revoked?
    pub revocable: bool,
}

/// A governance proposal
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Proposal {
    /// Unique proposal ID
    pub proposal_id: String,
    /// Space this proposal belongs to
    pub space: ActionHash,
    /// Proposal title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Proposal type
    pub proposal_type: ProposalType,
    /// Voting mechanism for this proposal
    pub voting_mechanism: VotingMechanism,
    /// Proposed actions (if executable)
    pub actions: Vec<ProposedAction>,
    /// Proposer
    pub proposer: AgentPubKey,
    /// Sponsors
    pub sponsors: Vec<AgentPubKey>,
    /// Current status
    pub status: ProposalStatus,
    /// Timeline
    pub timeline: ProposalTimeline,
    /// Evidence/supporting documents
    pub evidence: Vec<EvidenceRef>,
    /// Epistemic classification
    pub epistemic: EpistemicClaim,
    /// Discussion thread
    pub discussion: Option<ActionHash>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum ProposalType {
    /// Simple poll (non-binding)
    Poll,
    /// Signal vote (temperature check)
    Signal,
    /// Standard proposal (binding if passes)
    Standard,
    /// Constitutional amendment
    Constitutional,
    /// Budget allocation
    Budget { amount: u64, currency: String },
    /// Parameter change
    ParameterChange { parameter: String, new_value: String },
    /// Membership action
    Membership { action: MembershipAction },
    /// Emergency action
    Emergency { justification: String },
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum MembershipAction {
    Add(AgentPubKey),
    Remove(AgentPubKey),
    Suspend { agent: AgentPubKey, duration: u64 },
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum ProposalStatus {
    Draft,
    Discussion,
    Sponsored,
    Voting,
    Passed,
    Rejected,
    Executed,
    Vetoed,
    Expired,
    Withdrawn,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct ProposalTimeline {
    pub created_at: Timestamp,
    pub discussion_ends: Timestamp,
    pub voting_starts: Timestamp,
    pub voting_ends: Timestamp,
    pub execution_deadline: Option<Timestamp>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct ProposedAction {
    /// Target hApp/zome
    pub target: String,
    /// Function to call
    pub function: String,
    /// Arguments
    pub arguments: String, // JSON
    /// Conditions for execution
    pub conditions: Vec<ExecutionCondition>,
}

/// A vote on a proposal
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Vote {
    /// Proposal being voted on
    pub proposal: ActionHash,
    /// Voter
    pub voter: AgentPubKey,
    /// Vote choice
    pub choice: VoteChoice,
    /// Vote weight (calculated at time of voting)
    pub weight: f64,
    /// Delegation chain (if delegated)
    pub delegation_chain: Vec<AgentPubKey>,
    /// Rationale (optional)
    pub rationale: Option<String>,
    /// Voted at
    pub voted_at: Timestamp,
    /// Signature
    pub signature: Signature,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum VoteChoice {
    /// Simple yes/no
    Binary(bool),
    /// Multiple choice
    MultipleChoice(Vec<u32>),
    /// Ranked preferences
    Ranked(Vec<u32>),
    /// Approval set
    Approval(Vec<u32>),
    /// Quadratic allocation
    Quadratic(Vec<(u32, u64)>), // (option, credits)
    /// Conviction allocation
    Conviction { option: u32, amount: u64 },
    /// Abstain
    Abstain,
}

/// Delegation of voting power
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Delegation {
    /// Who is delegating
    pub delegator: AgentPubKey,
    /// Who receives delegation
    pub delegate: AgentPubKey,
    /// Scope of delegation
    pub scope: DelegationScope,
    /// Weight of delegation (for partial)
    pub weight: f64,
    /// Can delegate re-delegate?
    pub transitive: bool,
    /// Created at
    pub created_at: Timestamp,
    /// Expires at
    pub expires_at: Option<Timestamp>,
    /// Is active?
    pub active: bool,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum DelegationScope {
    /// All proposals in space
    AllProposals { space: ActionHash },
    /// Specific proposal types
    ProposalTypes { space: ActionHash, types: Vec<String> },
    /// Specific topics
    Topics { tags: Vec<String> },
    /// Single proposal
    SingleProposal { proposal: ActionHash },
}

/// Proposal result
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ProposalResult {
    /// Proposal this result is for
    pub proposal: ActionHash,
    /// Outcome
    pub outcome: Outcome,
    /// Vote tally
    pub tally: VoteTally,
    /// Participation stats
    pub participation: ParticipationStats,
    /// Calculated at
    pub calculated_at: Timestamp,
    /// Finalized at
    pub finalized_at: Option<Timestamp>,
    /// Executed at (if applicable)
    pub executed_at: Option<Timestamp>,
    /// Execution result
    pub execution_result: Option<ExecutionResult>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum Outcome {
    Passed,
    Rejected,
    NoQuorum,
    Tied,
    Vetoed { by: AgentPubKey, reason: String },
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct VoteTally {
    /// Votes per choice
    pub choices: HashMap<String, TallyEntry>,
    /// Total votes cast
    pub total_votes: u64,
    /// Total weight cast
    pub total_weight: f64,
    /// Abstentions
    pub abstentions: u64,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct TallyEntry {
    pub votes: u64,
    pub weight: f64,
    pub percentage: f64,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct ParticipationStats {
    /// Eligible voters
    pub eligible: u64,
    /// Voters who participated
    pub participated: u64,
    /// Participation rate
    pub rate: f64,
    /// Unique voters (excluding delegations)
    pub unique_voters: u64,
    /// Delegated votes
    pub delegated_votes: u64,
}
```

---

## Voting Mechanisms Deep Dive

### 1. Simple Majority

```rust
fn calculate_simple_majority(votes: &[Vote]) -> Outcome {
    let yes_weight: f64 = votes.iter()
        .filter(|v| matches!(v.choice, VoteChoice::Binary(true)))
        .map(|v| v.weight)
        .sum();

    let no_weight: f64 = votes.iter()
        .filter(|v| matches!(v.choice, VoteChoice::Binary(false)))
        .map(|v| v.weight)
        .sum();

    let total = yes_weight + no_weight;

    if total == 0.0 {
        Outcome::NoQuorum
    } else if yes_weight > total / 2.0 {
        Outcome::Passed
    } else if no_weight > total / 2.0 {
        Outcome::Rejected
    } else {
        Outcome::Tied
    }
}
```

### 2. Quadratic Voting

```rust
fn calculate_quadratic_voting(votes: &[Vote], credits_per_voter: u64) -> VoteTally {
    let mut option_scores: HashMap<u32, f64> = HashMap::new();

    for vote in votes {
        if let VoteChoice::Quadratic(allocations) = &vote.choice {
            // Verify credit budget
            let total_credits: u64 = allocations.iter()
                .map(|(_, credits)| credits)
                .sum();

            if total_credits > credits_per_voter {
                continue; // Invalid vote
            }

            // Calculate votes (sqrt of credits)
            for (option, credits) in allocations {
                let votes = (*credits as f64).sqrt();
                *option_scores.entry(*option).or_insert(0.0) += votes;
            }
        }
    }

    // Convert to tally
    build_tally(option_scores)
}
```

### 3. Conviction Voting

```rust
fn calculate_conviction(
    votes: &[ConvictionVote],
    current_time: Timestamp,
    decay_rate: f64,
    max_conviction: f64,
) -> VoteTally {
    let mut option_conviction: HashMap<u32, f64> = HashMap::new();

    for vote in votes {
        // Conviction grows over time, approaches max asymptotically
        let time_held = (current_time - vote.started_at).as_secs() as f64;

        // Conviction = amount * (1 - e^(-decay * time)) * max
        let conviction = vote.amount as f64
            * (1.0 - (-decay_rate * time_held / 86400.0).exp())
            * max_conviction;

        *option_conviction.entry(vote.option).or_insert(0.0) += conviction;
    }

    build_tally(option_conviction)
}

// Conviction can be moved between proposals
fn reallocate_conviction(
    from_proposal: ActionHash,
    to_proposal: ActionHash,
    amount: u64,
) -> ExternResult<()> {
    let voter = agent_info()?.agent_latest_pubkey;

    // Get current conviction allocation
    let mut allocation = get_conviction_allocation(&voter)?;

    // Move conviction
    allocation.remove(&from_proposal, amount);
    allocation.add(&to_proposal, amount);

    update_conviction_allocation(&voter, allocation)?;

    Ok(())
}
```

### 4. Ranked Choice

```rust
fn calculate_ranked_choice(votes: &[Vote], options: &[u32]) -> Outcome {
    let mut remaining_options: HashSet<u32> = options.iter().cloned().collect();
    let mut vote_counts: Vec<Vec<VoteChoice>> = votes.iter()
        .map(|v| v.choice.clone())
        .collect();

    loop {
        // Count first preferences
        let mut counts: HashMap<u32, u64> = HashMap::new();

        for vote in &vote_counts {
            if let VoteChoice::Ranked(prefs) = vote {
                // Find first preference still in running
                for pref in prefs {
                    if remaining_options.contains(pref) {
                        *counts.entry(*pref).or_insert(0) += 1;
                        break;
                    }
                }
            }
        }

        let total: u64 = counts.values().sum();

        // Check for majority
        for (option, count) in &counts {
            if *count > total / 2 {
                return Outcome::Passed; // Winner found
            }
        }

        // Eliminate lowest
        if let Some((lowest, _)) = counts.iter().min_by_key(|(_, c)| *c) {
            remaining_options.remove(lowest);
        }

        // Check if only one remains
        if remaining_options.len() == 1 {
            return Outcome::Passed;
        }

        if remaining_options.is_empty() {
            return Outcome::Tied;
        }
    }
}
```

### 5. Futarchy

```rust
fn calculate_futarchy(
    proposal: &Proposal,
    pass_market: &PredictionMarket,
    fail_market: &PredictionMarket,
) -> Outcome {
    // Compare prediction market outcomes

    // pass_market: "If proposal passes, what will success_metric be?"
    // fail_market: "If proposal fails, what will success_metric be?"

    let pass_prediction = pass_market.current_price();
    let fail_prediction = fail_market.current_price();

    // Proposal passes if predicted outcome is better with passage
    if pass_prediction > fail_prediction {
        Outcome::Passed
    } else {
        Outcome::Rejected
    }
}
```

---

## Zome Specifications

### 1. Proposal Manager Zome

```rust
// proposal_manager/src/lib.rs

/// Create a new proposal
#[hdk_extern]
pub fn create_proposal(input: CreateProposalInput) -> ExternResult<ActionHash> {
    let proposer = agent_info()?.agent_latest_pubkey;

    // Get space
    let space = get_space(&input.space)?;

    // Verify proposer meets requirements
    verify_can_propose(&proposer, &space.config.proposal_requirements)?;

    // Calculate timeline
    let now = sys_time()?;
    let timeline = ProposalTimeline {
        created_at: now,
        discussion_ends: now.add_seconds(space.config.proposal_requirements.discussion_period),
        voting_starts: now.add_seconds(space.config.proposal_requirements.discussion_period),
        voting_ends: now.add_seconds(
            space.config.proposal_requirements.discussion_period +
            space.config.quorum.type_overrides
                .get(&input.proposal_type.to_string())
                .copied()
                .unwrap_or(604800) as u64 // Default 7 days
        ),
        execution_deadline: input.execution_deadline,
    };

    // Determine voting mechanism
    let voting = input.voting_mechanism.unwrap_or(space.config.default_voting.clone());

    // Create epistemic claim
    let epistemic = EpistemicClaim::new(
        format!("Proposal: {}", input.title),
        match input.proposal_type {
            ProposalType::Constitutional => EmpiricalLevel::E4PublicRepro,
            ProposalType::Standard => EmpiricalLevel::E2PrivateVerify,
            ProposalType::Poll => EmpiricalLevel::E1Testimonial,
            _ => EmpiricalLevel::E2PrivateVerify,
        },
        NormativeLevel::N1Communal,
        MaterialityLevel::M2Persistent,
    );

    let proposal = Proposal {
        proposal_id: generate_proposal_id(),
        space: input.space,
        title: input.title,
        description: input.description,
        proposal_type: input.proposal_type,
        voting_mechanism: voting,
        actions: input.actions,
        proposer: proposer.clone(),
        sponsors: vec![proposer.clone()],
        status: ProposalStatus::Draft,
        timeline,
        evidence: input.evidence,
        epistemic,
        discussion: None,
    };

    let hash = create_entry(&EntryTypes::Proposal(proposal.clone()))?;

    // Handle deposit if required
    if let Some(deposit) = &space.config.proposal_requirements.deposit {
        lock_deposit(&proposer, deposit, &hash)?;
    }

    // Create discussion thread
    let discussion = create_discussion_thread(&hash)?;

    // Update proposal with discussion
    let mut updated = proposal;
    updated.discussion = Some(discussion);
    update_entry(hash.clone(), &updated)?;

    // Notify space members
    notify_space_members(&input.space, Signal::NewProposal(hash.clone()))?;

    Ok(hash)
}

/// Sponsor a proposal
#[hdk_extern]
pub fn sponsor_proposal(proposal_hash: ActionHash) -> ExternResult<()> {
    let sponsor = agent_info()?.agent_latest_pubkey;

    let mut proposal = get_proposal(&proposal_hash)?;
    let space = get_space(&proposal.space)?;

    // Verify sponsor is eligible
    verify_can_sponsor(&sponsor, &space)?;

    // Add sponsor
    if !proposal.sponsors.contains(&sponsor) {
        proposal.sponsors.push(sponsor);
    }

    // Check if enough sponsors
    if proposal.sponsors.len() >= space.config.proposal_requirements.sponsors_required as usize {
        proposal.status = ProposalStatus::Sponsored;
    }

    update_proposal(&proposal_hash, &proposal)?;

    Ok(())
}

/// Move proposal to voting
#[hdk_extern]
pub fn start_voting(proposal_hash: ActionHash) -> ExternResult<()> {
    let mut proposal = get_proposal(&proposal_hash)?;

    // Verify discussion period has ended
    if sys_time()? < proposal.timeline.discussion_ends {
        return Err(WasmError::Guest("Discussion period not ended".into()));
    }

    // Verify proposal is sponsored
    if proposal.status != ProposalStatus::Sponsored {
        return Err(WasmError::Guest("Proposal not sponsored".into()));
    }

    proposal.status = ProposalStatus::Voting;
    proposal.timeline.voting_starts = sys_time()?;

    update_proposal(&proposal_hash, &proposal)?;

    // Notify eligible voters
    let space = get_space(&proposal.space)?;
    notify_eligible_voters(&space, Signal::VotingStarted(proposal_hash))?;

    Ok(())
}
```

### 2. Voting Engine Zome

```rust
// voting_engine/src/lib.rs

/// Cast a vote
#[hdk_extern]
pub fn cast_vote(input: CastVoteInput) -> ExternResult<ActionHash> {
    let voter = agent_info()?.agent_latest_pubkey;

    // Get proposal
    let proposal = get_proposal(&input.proposal)?;

    // Verify voting is open
    if proposal.status != ProposalStatus::Voting {
        return Err(WasmError::Guest("Voting is not open".into()));
    }

    // Verify within voting period
    let now = sys_time()?;
    if now < proposal.timeline.voting_starts || now > proposal.timeline.voting_ends {
        return Err(WasmError::Guest("Outside voting period".into()));
    }

    // Get space
    let space = get_space(&proposal.space)?;

    // Verify voter is eligible
    verify_can_vote(&voter, &space)?;

    // Check for existing vote
    if has_voted(&voter, &input.proposal)? {
        return Err(WasmError::Guest("Already voted".into()));
    }

    // Calculate vote weight
    let weight = calculate_vote_weight(&voter, &space.config.weight_calculation)?;

    // Get delegation chain (if voting on behalf of delegators)
    let delegation_chain = get_delegation_chain(&voter, &proposal)?;

    // Validate vote choice for mechanism
    validate_vote_choice(&input.choice, &proposal.voting_mechanism)?;

    // Create vote
    let vote = Vote {
        proposal: input.proposal,
        voter: voter.clone(),
        choice: input.choice,
        weight,
        delegation_chain,
        rationale: input.rationale,
        voted_at: sys_time()?,
        signature: sign_vote(&input)?,
    };

    let hash = create_entry(&EntryTypes::Vote(vote))?;

    // Update live tally
    update_live_tally(&input.proposal)?;

    Ok(hash)
}

/// Calculate vote weight based on configuration
fn calculate_vote_weight(voter: &AgentPubKey, calc: &WeightCalculation) -> ExternResult<f64> {
    match calc {
        WeightCalculation::EqualWeight => Ok(1.0),

        WeightCalculation::ReputationWeighted { max_weight } => {
            let reputation = bridge_call::<f64>(
                "bridge",
                "get_cross_happ_reputation",
                voter.clone(),
            )?;
            Ok((reputation * max_weight).min(*max_weight))
        }

        WeightCalculation::TenureWeighted { max_multiplier } => {
            let membership = get_membership(voter)?;
            let tenure_days = (sys_time()? - membership.joined_at).as_secs() as f64 / 86400.0;
            let multiplier = (tenure_days / 365.0).min(*max_multiplier);
            Ok(1.0 + multiplier)
        }

        WeightCalculation::QuadraticWeight => {
            let reputation = bridge_call::<f64>(
                "bridge",
                "get_cross_happ_reputation",
                voter.clone(),
            )?;
            Ok(reputation.sqrt())
        }

        WeightCalculation::StakeWeighted => {
            let stake = get_staked_amount(voter)?;
            Ok(stake as f64)
        }

        WeightCalculation::Custom { formula } => {
            // Execute custom weight formula
            execute_weight_formula(formula, voter)
        }
    }
}

/// Finalize voting and calculate result
#[hdk_extern]
pub fn finalize_proposal(proposal_hash: ActionHash) -> ExternResult<ActionHash> {
    let mut proposal = get_proposal(&proposal_hash)?;

    // Verify voting period has ended
    if sys_time()? < proposal.timeline.voting_ends {
        return Err(WasmError::Guest("Voting period not ended".into()));
    }

    // Get all votes
    let votes = get_proposal_votes(&proposal_hash)?;

    // Get space for quorum
    let space = get_space(&proposal.space)?;

    // Calculate participation
    let eligible_voters = count_eligible_voters(&space)?;
    let participation = ParticipationStats {
        eligible: eligible_voters,
        participated: votes.len() as u64,
        rate: votes.len() as f64 / eligible_voters as f64,
        unique_voters: count_unique_voters(&votes),
        delegated_votes: count_delegated_votes(&votes),
    };

    // Check quorum
    if participation.rate < space.config.quorum.min_participation {
        proposal.status = ProposalStatus::Rejected;
        let result = ProposalResult {
            proposal: proposal_hash.clone(),
            outcome: Outcome::NoQuorum,
            tally: empty_tally(),
            participation,
            calculated_at: sys_time()?,
            finalized_at: Some(sys_time()?),
            executed_at: None,
            execution_result: None,
        };
        return create_entry(&EntryTypes::ProposalResult(result));
    }

    // Calculate tally based on voting mechanism
    let tally = match &proposal.voting_mechanism {
        VotingMechanism::SimpleMajority => calculate_simple_majority_tally(&votes),
        VotingMechanism::SuperMajority { threshold } => calculate_supermajority_tally(&votes, *threshold),
        VotingMechanism::Quadratic { credits_per_period } => calculate_quadratic_voting(&votes, *credits_per_period),
        VotingMechanism::Conviction { decay_rate, max_conviction } => {
            calculate_conviction_tally(&votes, sys_time()?, *decay_rate, *max_conviction)
        }
        VotingMechanism::RankedChoice { rounds } => calculate_ranked_choice_tally(&votes, *rounds),
        VotingMechanism::Approval { max_approvals } => calculate_approval_tally(&votes, *max_approvals),
        _ => calculate_simple_majority_tally(&votes),
    };

    // Determine outcome
    let outcome = determine_outcome(&tally, &proposal.voting_mechanism, &space.config.thresholds);

    // Update proposal status
    proposal.status = match outcome {
        Outcome::Passed => ProposalStatus::Passed,
        _ => ProposalStatus::Rejected,
    };

    update_proposal(&proposal_hash, &proposal)?;

    // Create result
    let result = ProposalResult {
        proposal: proposal_hash.clone(),
        outcome: outcome.clone(),
        tally,
        participation,
        calculated_at: sys_time()?,
        finalized_at: Some(sys_time()?),
        executed_at: None,
        execution_result: None,
    };

    let result_hash = create_entry(&EntryTypes::ProposalResult(result))?;

    // Handle deposit return
    if matches!(outcome, Outcome::Passed) {
        return_deposit(&proposal.proposer, &proposal_hash)?;
    }

    // Notify
    notify_space_members(&proposal.space, Signal::ProposalFinalized(proposal_hash, outcome))?;

    Ok(result_hash)
}
```

### 3. Delegation System Zome

```rust
// delegation_system/src/lib.rs

/// Create a delegation
#[hdk_extern]
pub fn delegate_vote(input: DelegateInput) -> ExternResult<ActionHash> {
    let delegator = agent_info()?.agent_latest_pubkey;

    // Can't delegate to yourself
    if delegator == input.delegate {
        return Err(WasmError::Guest("Cannot delegate to yourself".into()));
    }

    // Check for circular delegation
    if would_create_cycle(&delegator, &input.delegate, &input.scope)? {
        return Err(WasmError::Guest("Would create delegation cycle".into()));
    }

    let delegation = Delegation {
        delegator: delegator.clone(),
        delegate: input.delegate.clone(),
        scope: input.scope,
        weight: input.weight.unwrap_or(1.0),
        transitive: input.transitive,
        created_at: sys_time()?,
        expires_at: input.expires_at,
        active: true,
    };

    let hash = create_entry(&EntryTypes::Delegation(delegation))?;

    // Notify delegate
    send_signal_to_agent(&input.delegate, Signal::DelegationReceived(hash.clone()))?;

    Ok(hash)
}

/// Revoke a delegation
#[hdk_extern]
pub fn revoke_delegation(delegation_hash: ActionHash) -> ExternResult<()> {
    let delegator = agent_info()?.agent_latest_pubkey;

    let mut delegation = get_delegation(&delegation_hash)?;

    if delegation.delegator != delegator {
        return Err(WasmError::Guest("Only delegator can revoke".into()));
    }

    delegation.active = false;

    update_delegation(&delegation_hash, &delegation)?;

    // Notify delegate
    send_signal_to_agent(&delegation.delegate, Signal::DelegationRevoked(delegation_hash))?;

    Ok(())
}

/// Get my delegated voting power
#[hdk_extern]
pub fn get_my_delegated_power(space: ActionHash) -> ExternResult<DelegatedPower> {
    let agent = agent_info()?.agent_latest_pubkey;

    // Get direct delegations to me
    let direct = get_delegations_to(&agent, &space)?;

    // Calculate transitive delegations
    let mut total_weight = 0.0;
    let mut delegators: Vec<AgentPubKey> = vec![];

    for delegation in direct {
        if delegation.active {
            let delegator_weight = get_base_weight(&delegation.delegator)?;
            total_weight += delegator_weight * delegation.weight;
            delegators.push(delegation.delegator.clone());

            // Handle transitive
            if delegation.transitive {
                let transitive = get_transitive_delegations(&delegation.delegator, &space)?;
                for t in transitive {
                    let t_weight = get_base_weight(&t.delegator)?;
                    total_weight += t_weight * t.weight * delegation.weight;
                    delegators.push(t.delegator);
                }
            }
        }
    }

    Ok(DelegatedPower {
        agent,
        space,
        total_weight,
        delegators,
    })
}
```

---

## Cross-hApp Integration

### Attest Integration
```rust
// Verify voter identity level
let level = bridge_call::<IdentityLevel>(
    "attest",
    "get_identity_level",
    voter,
)?;

if level < space.membership.min_identity_level {
    return Err(WasmError::Guest("Identity level too low".into()));
}
```

### MATL Integration
```rust
// Get reputation for weighted voting
let reputation = bridge_call::<f64>(
    "bridge",
    "get_cross_happ_reputation",
    voter,
)?;
```

### Arbiter Integration
```rust
// Disputed election -> Arbiter
bridge_call::<ActionHash>(
    "arbiter",
    "file_dispute",
    DisputeInput {
        source_happ: "agora".to_string(),
        source_reference: proposal_hash,
        dispute_type: DisputeType::ElectionDispute,
        ..
    },
)?;
```

---

## Constitutional Framework

### Meta-Governance

```rust
/// Propose constitutional amendment
#[hdk_extern]
pub fn propose_constitutional_amendment(input: AmendmentInput) -> ExternResult<ActionHash> {
    // Constitutional amendments require:
    // - Higher quorum (e.g., 67%)
    // - Supermajority (e.g., 75%)
    // - Longer discussion period
    // - Multiple readings

    let proposal = create_proposal(CreateProposalInput {
        proposal_type: ProposalType::Constitutional,
        voting_mechanism: Some(VotingMechanism::SuperMajority { threshold: 0.75 }),
        ..input.into()
    })?;

    // Schedule first reading
    schedule_reading(&proposal, 1)?;

    Ok(proposal)
}

/// Constitutional readings (multiple rounds)
pub fn process_reading(proposal: ActionHash, reading: u32) -> ExternResult<()> {
    // First reading: Discussion only
    // Second reading: Amendments allowed
    // Third reading: Final vote

    match reading {
        1 => {
            // Just discussion, no vote
        }
        2 => {
            // Allow amendments
            open_amendment_period(&proposal)?;
        }
        3 => {
            // Final vote
            start_voting(proposal)?;
        }
        _ => {}
    }

    Ok(())
}
```

---

## Success Metrics

| Metric | 1 Year | 3 Years |
|--------|--------|---------|
| Governance spaces | 50 | 500 |
| Proposals created | 500 | 10,000 |
| Votes cast | 10,000 | 500,000 |
| Average participation | 40% | 60% |
| Delegations active | 1,000 | 50,000 |
| Constitutional amendments | 5 | 20 |

---

*"Governance is too important to leave to governments alone."*
