// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Participatory Budgeting Integrity Zome
//!
//! Defines entry types for community budget cycles, project proposals,
//! allocation votes, and fund disbursements.

use hdi::prelude::*;
use serde::{Deserialize, Serialize};

// ============================================================================
// Entry Types
// ============================================================================

/// A budget cycle defines a time-bounded funding round.
///
/// Communities create cycles (e.g., quarterly), set a total pot,
/// and members propose projects to be funded from that pot.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BudgetCycle {
    /// Unique cycle identifier (e.g., "Q2-2026")
    pub cycle_id: String,
    /// Human-readable name
    pub name: String,
    /// Total available funds in the cycle (credits)
    pub total_budget: u64,
    /// Currency denomination (e.g., "SAP", "TEND", "MYCEL")
    pub currency: String,
    /// Council responsible for oversight
    pub oversight_council_id: Option<String>,
    /// Phase: Proposal → Deliberation → Voting → Execution → Complete
    pub phase: BudgetPhase,
    /// When each phase ends (microseconds since epoch)
    pub proposal_deadline: u64,
    pub deliberation_deadline: u64,
    pub voting_deadline: u64,
    /// Minimum consciousness tier to submit a project
    pub min_proposal_tier: u8,
    /// Minimum consciousness tier to vote
    pub min_voting_tier: u8,
    /// Voice credits allocated to each voter for this cycle
    pub voice_credits_per_voter: u64,
    /// Created timestamp
    pub created_at: Timestamp,
    /// Who created this cycle
    pub creator_did: String,
}

/// Phase of a budget cycle.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetPhase {
    /// Accepting project proposals
    Proposal,
    /// Community deliberation on proposals
    Deliberation,
    /// Quadratic voting on allocation
    Voting,
    /// Approved projects being executed
    Execution,
    /// Cycle complete, all funds disbursed or returned
    Complete,
    /// Cycle cancelled
    Cancelled,
}

/// A project proposed for funding within a budget cycle.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BudgetProject {
    /// Unique project ID
    pub project_id: String,
    /// Which cycle this belongs to
    pub cycle_id: String,
    /// Project title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Requested funding amount (credits)
    pub requested_amount: u64,
    /// Minimum viable funding (project can proceed with less)
    pub minimum_amount: u64,
    /// Who submitted this project
    pub proposer_did: String,
    /// Which cluster/domain benefits (e.g., "commons:housing", "civic:emergency")
    pub beneficiary_domain: String,
    /// Milestones for staged disbursement
    pub milestones: Vec<ProjectMilestone>,
    /// Current status
    pub status: ProjectStatus,
    /// Quadratic votes received (updated during tally)
    pub votes_received: u64,
    /// Effective vote weight (after quadratic calculation)
    pub effective_weight: f64,
    /// Allocated amount (set during execution phase)
    pub allocated_amount: Option<u64>,
    /// Created timestamp
    pub created_at: Timestamp,
}

/// A milestone within a budget project for staged disbursement.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ProjectMilestone {
    /// Milestone description
    pub description: String,
    /// Percentage of total allocation (0-100)
    pub percentage: u8,
    /// Whether this milestone has been verified
    pub verified: bool,
    /// Who verified (DID)
    pub verifier_did: Option<String>,
}

/// Status of a budget project.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectStatus {
    /// Proposed, awaiting deliberation
    Proposed,
    /// Under community deliberation
    UnderReview,
    /// Approved for funding
    Approved,
    /// Partially funded
    PartiallyFunded,
    /// Fully funded
    FullyFunded,
    /// In execution (milestones being completed)
    InExecution,
    /// All milestones complete
    Completed,
    /// Rejected by community vote
    Rejected,
    /// Withdrawn by proposer
    Withdrawn,
}

/// A quadratic vote allocation on a budget project.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BudgetVote {
    /// Which cycle
    pub cycle_id: String,
    /// Which project
    pub project_id: String,
    /// Voter DID
    pub voter_did: String,
    /// Voice credits spent on this project (cost = credits², effective votes = √credits)
    pub credits_spent: u64,
    /// Direction: positive = support, negative = oppose
    pub direction: VoteDirection,
    /// Voter's consciousness tier at time of voting
    pub voter_tier: u8,
    /// Timestamp
    pub created_at: Timestamp,
}

/// Direction of a budget vote.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoteDirection {
    Support,
    Oppose,
}

/// A fund disbursement record for a completed milestone.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Disbursement {
    /// Which project
    pub project_id: String,
    /// Which milestone index
    pub milestone_index: u32,
    /// Amount disbursed (credits)
    pub amount: u64,
    /// Recipient DID
    pub recipient_did: String,
    /// Who authorized the disbursement
    pub authorizer_did: String,
    /// Timestamp
    pub disbursed_at: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// ============================================================================
// Entry & Link Types
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    BudgetCycle(BudgetCycle),
    BudgetProject(BudgetProject),
    BudgetVote(BudgetVote),
    Disbursement(Disbursement),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllCycles,
    CycleToProject,
    ProjectToVote,
    ProjectToDisbursement,
    AgentToVote,
    AgentToProject,
    CyclePhaseIndex,
}

// ============================================================================
// Validation
// ============================================================================

#[hdk_extern]
pub fn validate(_op: Op) -> ExternResult<ValidateCallbackResult> {
    // TODO: Validate budget constraints, milestone percentages sum to 100, etc.
    Ok(ValidateCallbackResult::Valid)
}
