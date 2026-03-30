// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Governance types shared between portal and zomes.
//!
//! These mirror the integrity zome definitions in `mycelix-governance/zomes/*/integrity/`
//! but use plain `serde` instead of `#[hdk_entry_helper]`, making them safe for
//! browser WASM frontends.
//!
//! When the governance zome types change, update these to match. The compiler
//! will catch deserialization mismatches at the MessagePack boundary.

use serde::{Deserialize, Serialize};

// ── Proposals ──

/// A governance proposal (MIP — Mycelix Improvement Proposal).
/// Mirrors: `mycelix-governance/zomes/proposals/integrity/src/lib.rs`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Proposal {
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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
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
            Self::Parameter => "Parameter Change",
            Self::Funding => "Funding Request",
        }
    }

    pub fn voting_days(&self) -> u32 {
        match self {
            Self::Standard => 7,
            Self::Emergency => 1,
            Self::Constitutional => 30,
            Self::Parameter => 14,
            Self::Funding => 14,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
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
            Self::Draft => "Draft",
            Self::Active => "Active",
            Self::Ended => "Ended",
            Self::Approved => "Approved",
            Self::Signed => "Signed",
            Self::Rejected => "Rejected",
            Self::Executed => "Executed",
            Self::Cancelled => "Cancelled",
            Self::Failed => "Failed",
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Executed | Self::Rejected | Self::Cancelled | Self::Failed)
    }
}

// ── Voting ──

/// A Φ-weighted governance vote.
/// Mirrors: `mycelix-governance/zomes/voting/integrity/src/lib.rs`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vote {
    pub id: String,
    pub proposal_id: String,
    pub voter: String,
    pub choice: VoteChoice,
    pub weight: f64,
    pub reason: Option<String>,
    pub delegated: bool,
    pub delegator: Option<String>,
    pub phi_weight: PhiWeight,
    pub timestamp: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum VoteChoice {
    For,
    Against,
    Abstain,
}

/// Consciousness-weighted voting power.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PhiWeight {
    pub phi_score: f64,
    pub k_trust: f64,
    pub stake_weight: f64,
    pub participation_score: f64,
    pub domain_reputation: f64,
}

impl PhiWeight {
    /// Combined weight: phi × trust × (1 + stake + participation + reputation) / 4
    pub fn combined(&self) -> f64 {
        self.phi_score
            * self.k_trust
            * (1.0 + self.stake_weight + self.participation_score + self.domain_reputation)
            / 4.0
    }
}

// ── Councils ──

/// A nested governance council.
/// Mirrors: `mycelix-governance/zomes/councils/integrity/src/lib.rs`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Council {
    pub id: String,
    pub name: String,
    pub purpose: String,
    pub council_type: CouncilType,
    pub parent_council_id: Option<String>,
    pub phi_threshold: f64,
    pub quorum: f64,
    pub supermajority: f64,
    pub status: CouncilStatus,
    pub created_at: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum CouncilType {
    Root,
    Domain { domain: String },
    Regional { region: String },
    WorkingGroup { focus: String, expires: Option<i64> },
    Advisory,
    Emergency { expires: i64 },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum CouncilStatus {
    Active,
    Dormant,
    Dissolved,
    Suspended,
}

// ── Constitution ──

/// The constitutional charter.
/// Mirrors: `mycelix-governance/zomes/constitution/integrity/src/lib.rs`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Charter {
    pub id: String,
    pub version: u32,
    pub preamble: String,
    pub articles: String, // JSON
    pub rights: String,   // JSON
    pub amendment_process: String,
    pub adopted: i64,
    pub last_amended: i64,
}

// ── Budget ──

/// A time-bounded funding cycle.
/// Mirrors: `mycelix-governance/zomes/budgeting/integrity/src/lib.rs`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BudgetCycle {
    pub cycle_id: String,
    pub name: String,
    pub total_budget: u64,
    pub currency: String,
    pub phase: BudgetPhase,
    pub proposal_deadline: i64,
    pub voting_deadline: i64,
    pub created_at: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum BudgetPhase {
    Proposal,
    Deliberation,
    Voting,
    Execution,
    Complete,
    Cancelled,
}

/// A project within a budget cycle.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BudgetProject {
    pub project_id: String,
    pub cycle_id: String,
    pub title: String,
    pub description: String,
    pub requested_amount: u64,
    pub proposer_did: String,
    pub status: ProjectStatus,
    pub allocated_amount: u64,
    pub created_at: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ProjectStatus {
    Proposed,
    UnderReview,
    Approved,
    PartiallyFunded,
    FullyFunded,
    InExecution,
    Completed,
    Rejected,
    Withdrawn,
}

// ── Tally / Summary types for display ──

/// Summary of voting results for a proposal.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VoteTally {
    pub votes_for: u32,
    pub votes_against: u32,
    pub votes_abstain: u32,
    pub total_weight_for: f64,
    pub total_weight_against: f64,
    pub quorum_reached: bool,
    pub approved: bool,
}

/// Summary view of a proposal with tally.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProposalSummary {
    pub proposal: Proposal,
    pub tally: VoteTally,
    pub my_vote: Option<VoteChoice>,
}
