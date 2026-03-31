// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! WASM-safe mirror types for the Mycelix Governance cluster.
//!
//! These types mirror the governance zome entry types but use only `serde` —
//! no HDI/HDK dependencies. Holochain `AgentPubKey` → `String`,
//! `ActionHash` → `String`, `Timestamp` → `i64` (microseconds since epoch).

use serde::{Deserialize, Serialize};

// ============================================================================
// Proposal Types
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalType {
    Standard,
    Emergency,
    Constitutional,
    Parameter,
    Funding,
}

impl ProposalType {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Standard => "Standard",
            Self::Emergency => "Emergency",
            Self::Constitutional => "Constitutional",
            Self::Parameter => "Parameter",
            Self::Funding => "Funding",
        }
    }

    pub fn css_class(&self) -> &'static str {
        match self {
            Self::Standard => "proposal-standard",
            Self::Emergency => "proposal-emergency",
            Self::Constitutional => "proposal-constitutional",
            Self::Parameter => "proposal-parameter",
            Self::Funding => "proposal-funding",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalStatus {
    Draft,
    Active,
    Ended,
    Approved,
    Signed,
    Rejected,
    Executed,
    Cancelled,
    Failed,
}

impl ProposalStatus {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Draft => "Growing",
            Self::Active => "Deciding",
            Self::Ended => "Concluded",
            Self::Approved => "Approved",
            Self::Signed => "Sealed",
            Self::Rejected => "Composted",
            Self::Executed => "Released",
            Self::Cancelled => "Withdrawn",
            Self::Failed => "Failed",
        }
    }

    /// Which lifecycle phase this status belongs to (for CSS theming).
    pub fn phase(&self) -> &'static str {
        match self {
            Self::Draft => "discussion",
            Self::Active => "voting",
            Self::Ended => "voting",
            Self::Approved | Self::Signed => "timelock",
            Self::Executed => "executed",
            Self::Rejected | Self::Cancelled | Self::Failed => "rejected",
        }
    }

    pub fn is_active(&self) -> bool {
        matches!(self, Self::Draft | Self::Active | Self::Approved | Self::Signed)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProposalView {
    pub hash: String,
    pub id: String,
    pub title: String,
    pub description: String,
    pub proposal_type: ProposalType,
    pub author: String,
    pub status: ProposalStatus,
    pub actions: String,
    pub discussion_url: Option<String>,
    pub voting_starts: i64,
    pub voting_ends: i64,
    pub created: i64,
    pub updated: i64,
    pub version: u32,
}

// ============================================================================
// Voting Types
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoteChoice {
    For,
    Against,
    Abstain,
}

impl VoteChoice {
    pub fn label(&self) -> &'static str {
        match self {
            Self::For => "For",
            Self::Against => "Against",
            Self::Abstain => "Abstain",
        }
    }

    pub fn css_class(&self) -> &'static str {
        match self {
            Self::For => "vote-for",
            Self::Against => "vote-against",
            Self::Abstain => "vote-abstain",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalTier {
    Basic,
    Major,
    Constitutional,
}

impl ProposalTier {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Basic => "Basic",
            Self::Major => "Major",
            Self::Constitutional => "Constitutional",
        }
    }

    pub fn quorum_pct(&self) -> f64 {
        match self {
            Self::Basic => 15.0,
            Self::Major => 25.0,
            Self::Constitutional => 40.0,
        }
    }

    pub fn approval_pct(&self) -> f64 {
        match self {
            Self::Basic => 50.0,
            Self::Major => 60.0,
            Self::Constitutional => 67.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhiVoteView {
    pub hash: String,
    pub proposal_id: String,
    pub voter_did: String,
    pub tier: ProposalTier,
    pub choice: VoteChoice,
    pub effective_weight: f64,
    pub phi_score: f64,
    pub reasoning: Option<String>,
    pub delegated: bool,
    pub created: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoteTallyView {
    pub proposal_id: String,
    pub votes_for: f64,
    pub votes_against: f64,
    pub votes_abstain: f64,
    pub total_voters: u32,
    pub quorum_met: bool,
    pub approved: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DelegationView {
    pub hash: String,
    pub delegate_did: String,
    pub domain: Option<String>,
    pub decay_half_life_days: Option<f64>,
    pub expires_at: Option<i64>,
    pub created: i64,
}

// ============================================================================
// Contribution (Discussion) Types
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContributionStance {
    Support,
    Oppose,
    Concern,
    Question,
    Amendment,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContributionView {
    pub hash: String,
    pub proposal_id: String,
    pub author: String,
    pub content: String,
    pub stance: Option<ContributionStance>,
    pub parent_id: Option<String>,
    pub created: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscussionReadinessView {
    pub unique_contributors: u32,
    pub stances_represented: u32,
    pub total_contributions: u32,
    pub is_ready: bool,
}

// ============================================================================
// Council Types
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CouncilType {
    Root,
    Domain { domain: String },
    Regional { region: String },
    WorkingGroup { focus: String },
    Advisory,
    Emergency,
}

impl CouncilType {
    pub fn label(&self) -> String {
        match self {
            Self::Root => "Root Council".to_string(),
            Self::Domain { domain } => format!("{domain} Council"),
            Self::Regional { region } => format!("{region} Region"),
            Self::WorkingGroup { focus } => format!("{focus} Working Group"),
            Self::Advisory => "Advisory Council".to_string(),
            Self::Emergency => "Emergency Council".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CouncilView {
    pub hash: String,
    pub id: String,
    pub name: String,
    pub purpose: String,
    pub council_type: CouncilType,
    pub parent_council_id: Option<String>,
    pub phi_threshold: f64,
    pub quorum: f64,
    pub member_count: u32,
    pub status: String,
    pub created: i64,
}

// ============================================================================
// Constitution Types
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CharterView {
    pub hash: String,
    pub preamble: String,
    pub articles: Vec<ArticleView>,
    pub rights: Vec<String>,
    pub version: u32,
    pub adopted: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArticleView {
    pub number: u32,
    pub title: String,
    pub content: String,
}

// ============================================================================
// Execution Types
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimelockStatus {
    Pending,
    Ready,
    Executed,
    Vetoed,
    Failed,
    Cancelled,
}

impl TimelockStatus {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Pending => "Gestating",
            Self::Ready => "Ready",
            Self::Executed => "Released",
            Self::Vetoed => "Challenged",
            Self::Failed => "Failed",
            Self::Cancelled => "Withdrawn",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimelockView {
    pub hash: String,
    pub proposal_id: String,
    pub duration_hours: u64,
    pub status: TimelockStatus,
    pub created: i64,
    pub ready_at: Option<i64>,
}

// ============================================================================
// Budgeting Types
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BudgetPhase {
    Proposal,
    Deliberation,
    Voting,
    Execution,
    Complete,
}

impl BudgetPhase {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Proposal => "Proposals Open",
            Self::Deliberation => "Deliberation",
            Self::Voting => "Allocation Voting",
            Self::Execution => "Executing",
            Self::Complete => "Complete",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BudgetCycleView {
    pub hash: String,
    pub id: String,
    pub name: String,
    pub total_budget: u64,
    pub currency: String,
    pub phase: BudgetPhase,
    pub created: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BudgetProjectView {
    pub hash: String,
    pub id: String,
    pub cycle_id: String,
    pub title: String,
    pub description: String,
    pub requested_amount: u64,
    pub votes_for: f64,
    pub votes_against: f64,
}

// ============================================================================
// Consciousness Gating Types
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConsciousnessThresholds {
    pub basic: f64,
    pub proposal_submission: f64,
    pub voting: f64,
    pub constitutional: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateVerificationView {
    pub passed: bool,
    pub consciousness_level: f64,
    pub required_consciousness: f64,
    pub failure_reason: Option<String>,
}
