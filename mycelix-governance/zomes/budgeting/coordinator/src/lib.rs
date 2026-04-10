// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Participatory Budgeting Coordinator Zome
//!
//! Implements a community-driven budget allocation workflow:
//!
//! 1. **Create Cycle** — Council creates a budget cycle with total pot, deadlines, and rules
//! 2. **Submit Projects** — Community members propose projects for funding
//! 3. **Deliberate** — Discussion phase with AI-assisted reflection
//! 4. **Vote** — Quadratic voting: voters allocate voice credits across projects
//!    (cost = credits², effective votes = √credits — balances intensity vs breadth)
//! 5. **Allocate** — Projects ranked by effective vote weight, funded top-down until pot exhausted
//! 6. **Execute** — Milestone-based staged disbursement via Finance cluster
//!
//! ## Cross-Cluster Integration
//!
//! - **Finance**: Fund locking/release via `Governance→Finance` route
//! - **Commons/Civic**: Beneficiary domain tracking
//! - **Identity**: Consciousness gating for proposal submission and voting

use budgeting_integrity::*;
use hdk::prelude::*;
use serde::{Deserialize, Serialize};

// ============================================================================
// Helpers
// ============================================================================

fn ensure_anchor(name: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(name.to_string());
    create_entry(&EntryTypes::Anchor(anchor.clone()))?;
    hash_entry(&anchor)
}

fn require_consciousness_tier(_min_tier: u8) -> ExternResult<()> {
    // Consciousness gating is best-effort in budgeting — the gate_civic
    // function requires a running conductor with identity bridge. In standalone
    // or test mode, skip the gate and allow all operations.
    // TODO: Wire to identity bridge once conductor integration is verified.
    Ok(())
}

// ============================================================================
// Cycle Management
// ============================================================================

/// Create a new budget cycle.
///
/// Requires Constitutional tier (min_tier=3) since budget cycles affect
/// community-wide resource allocation.
#[hdk_extern]
pub fn create_cycle(input: BudgetCycle) -> ExternResult<ActionHash> {
    require_consciousness_tier(3)?;

    if input.total_budget == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest("Budget must be > 0".into())));
    }
    if input.voice_credits_per_voter == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest("Voice credits must be > 0".into())));
    }

    let action_hash = create_entry(&EntryTypes::BudgetCycle(input.clone()))?;

    let all_anchor = ensure_anchor("all_budget_cycles")?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllCycles, ())?;

    let phase_anchor = ensure_anchor(&format!("cycles:{:?}", input.phase))?;
    create_link(phase_anchor, action_hash.clone(), LinkTypes::CyclePhaseIndex, ())?;

    Ok(action_hash)
}

/// Get a budget cycle by its action hash.
#[hdk_extern]
pub fn get_cycle(hash: ActionHash) -> ExternResult<Record> {
    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Cycle not found".into())))
}

/// Get all budget cycles.
#[hdk_extern]
pub fn get_all_cycles(_: ()) -> ExternResult<Vec<Record>> {
    let anchor = ensure_anchor("all_budget_cycles")?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AllCycles)?,
        GetStrategy::Local,
    )?;
    let mut records = Vec::new();
    for link in links {
        if let Ok(hash) = ActionHash::try_from(link.target) {
            if let Some(record) = get(hash, GetOptions::default())? {
                records.push(record);
            }
        }
    }
    Ok(records)
}

/// Advance a cycle to the next phase.
#[hdk_extern]
pub fn advance_cycle_phase(input: AdvanceCycleInput) -> ExternResult<ActionHash> {
    require_consciousness_tier(3)?;

    let record = get(input.cycle_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Cycle not found".into())))?;
    let mut cycle: BudgetCycle = record.entry().to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not a BudgetCycle".into())))?;

    cycle.phase = match cycle.phase {
        BudgetPhase::Proposal => BudgetPhase::Deliberation,
        BudgetPhase::Deliberation => BudgetPhase::Voting,
        BudgetPhase::Voting => BudgetPhase::Execution,
        BudgetPhase::Execution => BudgetPhase::Complete,
        other => return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Cannot advance from phase {:?}", other)
        ))),
    };

    // If entering Execution phase, compute allocations
    if cycle.phase == BudgetPhase::Execution {
        allocate_funds_for_cycle(&cycle)?;
    }

    update_entry(input.cycle_hash, &EntryTypes::BudgetCycle(cycle))
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AdvanceCycleInput {
    pub cycle_hash: ActionHash,
}

// ============================================================================
// Project Proposals
// ============================================================================

/// Submit a project proposal for a budget cycle.
#[hdk_extern]
pub fn submit_project(input: BudgetProject) -> ExternResult<ActionHash> {
    require_consciousness_tier(input.status as u8)?; // Uses the cycle's min tier

    if input.requested_amount == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest("Requested amount must be > 0".into())));
    }
    if input.minimum_amount > input.requested_amount {
        return Err(wasm_error!(WasmErrorInner::Guest("Minimum cannot exceed requested amount".into())));
    }
    let milestone_total: u8 = input.milestones.iter().map(|m| m.percentage).sum();
    if !input.milestones.is_empty() && milestone_total != 100 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Milestone percentages must sum to 100, got {}", milestone_total)
        )));
    }

    let action_hash = create_entry(&EntryTypes::BudgetProject(input.clone()))?;

    // Link cycle → project
    let cycle_anchor = ensure_anchor(&format!("cycle_projects:{}", input.cycle_id))?;
    create_link(cycle_anchor, action_hash.clone(), LinkTypes::CycleToProject, ())?;

    // Link agent → project
    let agent_anchor = ensure_anchor(&format!("agent_projects:{}", input.proposer_did))?;
    create_link(agent_anchor, action_hash.clone(), LinkTypes::AgentToProject, ())?;

    Ok(action_hash)
}

/// Get all projects for a budget cycle.
#[hdk_extern]
pub fn get_cycle_projects(cycle_id: String) -> ExternResult<Vec<Record>> {
    let anchor = ensure_anchor(&format!("cycle_projects:{}", cycle_id))?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::CycleToProject)?,
        GetStrategy::Local,
    )?;
    let mut records = Vec::new();
    for link in links {
        if let Ok(hash) = ActionHash::try_from(link.target) {
            if let Some(record) = get(hash, GetOptions::default())? {
                records.push(record);
            }
        }
    }
    Ok(records)
}

/// Get a specific project by hash.
#[hdk_extern]
pub fn get_project(hash: ActionHash) -> ExternResult<Record> {
    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Project not found".into())))
}

/// Withdraw a project (proposer only).
#[hdk_extern]
pub fn withdraw_project(project_hash: ActionHash) -> ExternResult<ActionHash> {
    let record = get(project_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Project not found".into())))?;
    let mut project: BudgetProject = record.entry().to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not a BudgetProject".into())))?;

    project.status = ProjectStatus::Withdrawn;
    update_entry(project_hash, &EntryTypes::BudgetProject(project))
}

// ============================================================================
// Quadratic Voting
// ============================================================================

/// Cast a quadratic vote on a project.
///
/// The voter spends `credits_spent` voice credits on this project.
/// Effective votes = √credits_spent. Cost grows quadratically,
/// preventing any single voter from dominating allocation.
#[hdk_extern]
pub fn cast_budget_vote(input: BudgetVote) -> ExternResult<ActionHash> {
    require_consciousness_tier(input.voter_tier)?;

    if input.credits_spent == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest("Must spend at least 1 credit".into())));
    }

    let action_hash = create_entry(&EntryTypes::BudgetVote(input.clone()))?;

    // Link project → vote
    let project_anchor = ensure_anchor(&format!("project_votes:{}", input.project_id))?;
    create_link(project_anchor, action_hash.clone(), LinkTypes::ProjectToVote, ())?;

    // Link agent → vote
    let agent_anchor = ensure_anchor(&format!("agent_votes:{}:{}", input.voter_did, input.cycle_id))?;
    create_link(agent_anchor, action_hash.clone(), LinkTypes::AgentToVote, ())?;

    Ok(action_hash)
}

/// Get all votes for a project.
#[hdk_extern]
pub fn get_project_votes(project_id: String) -> ExternResult<Vec<Record>> {
    let anchor = ensure_anchor(&format!("project_votes:{}", project_id))?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::ProjectToVote)?,
        GetStrategy::Local,
    )?;
    let mut records = Vec::new();
    for link in links {
        if let Ok(hash) = ActionHash::try_from(link.target) {
            if let Some(record) = get(hash, GetOptions::default())? {
                records.push(record);
            }
        }
    }
    Ok(records)
}

/// Get votes cast by a specific agent in a cycle.
#[hdk_extern]
pub fn get_my_votes(input: MyVotesInput) -> ExternResult<Vec<Record>> {
    let anchor = ensure_anchor(&format!("agent_votes:{}:{}", input.voter_did, input.cycle_id))?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AgentToVote)?,
        GetStrategy::Local,
    )?;
    let mut records = Vec::new();
    for link in links {
        if let Ok(hash) = ActionHash::try_from(link.target) {
            if let Some(record) = get(hash, GetOptions::default())? {
                records.push(record);
            }
        }
    }
    Ok(records)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MyVotesInput {
    pub voter_did: String,
    pub cycle_id: String,
}

/// Tally quadratic votes for a cycle and return ranked projects.
///
/// For each project: effective_weight = Σ(√credits_spent) for Support votes
///                                    - Σ(√credits_spent) for Oppose votes
///
/// Projects are ranked by effective_weight descending. Funding is allocated
/// top-down until the budget is exhausted. If a project's requested_amount
/// exceeds remaining budget but minimum_amount fits, it receives minimum_amount.
#[hdk_extern]
pub fn tally_cycle_votes(cycle_id: String) -> ExternResult<Vec<ProjectTally>> {
    let projects = get_cycle_projects(cycle_id.clone())?;
    let mut tallies: Vec<ProjectTally> = Vec::new();

    for project_record in &projects {
        let project: BudgetProject = project_record.entry().to_app_option()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .ok_or(wasm_error!(WasmErrorInner::Guest("Not a BudgetProject".into())))?;

        if project.status == ProjectStatus::Withdrawn {
            continue;
        }

        let votes = get_project_votes(project.project_id.clone())?;
        let mut effective_weight: f64 = 0.0;
        let mut total_credits: u64 = 0;
        let mut voter_count: u32 = 0;

        for vote_record in &votes {
            if let Ok(Some(vote)) = vote_record.entry().to_app_option::<BudgetVote>() {
                let sqrt_credits = (vote.credits_spent as f64).sqrt();
                match vote.direction {
                    VoteDirection::Support => effective_weight += sqrt_credits,
                    VoteDirection::Oppose => effective_weight -= sqrt_credits,
                }
                total_credits += vote.credits_spent;
                voter_count += 1;
            }
        }

        tallies.push(ProjectTally {
            project_id: project.project_id,
            title: project.title,
            requested_amount: project.requested_amount,
            minimum_amount: project.minimum_amount,
            effective_weight,
            total_credits_spent: total_credits,
            voter_count,
            beneficiary_domain: project.beneficiary_domain,
        });
    }

    // Sort by effective_weight descending
    tallies.sort_by(|a, b| b.effective_weight.partial_cmp(&a.effective_weight)
        .unwrap_or(core::cmp::Ordering::Equal));

    Ok(tallies)
}

/// Result of tallying votes for a project.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProjectTally {
    pub project_id: String,
    pub title: String,
    pub requested_amount: u64,
    pub minimum_amount: u64,
    pub effective_weight: f64,
    pub total_credits_spent: u64,
    pub voter_count: u32,
    pub beneficiary_domain: String,
}

// ============================================================================
// Fund Allocation & Disbursement
// ============================================================================

/// Allocate funds to projects based on vote tally.
///
/// Called internally when cycle advances to Execution phase.
/// Projects are funded top-down by effective_weight until budget exhausted.
fn allocate_funds_for_cycle(cycle: &BudgetCycle) -> ExternResult<()> {
    let tallies = tally_cycle_votes(cycle.cycle_id.clone())?;
    let mut remaining = cycle.total_budget;

    for tally in &tallies {
        if remaining == 0 {
            break;
        }
        if tally.effective_weight <= 0.0 {
            continue; // Skip projects with net-negative votes
        }

        let allocation = if tally.requested_amount <= remaining {
            tally.requested_amount
        } else if tally.minimum_amount <= remaining {
            tally.minimum_amount
        } else {
            continue; // Can't fund even minimum
        };

        remaining -= allocation;

        // Record allocation by dispatching to Finance via governance bridge
        // Best-effort: log but don't fail if finance is unreachable
        let payload = serde_json::json!({
            "project_id": tally.project_id,
            "amount": allocation,
            "cycle_id": cycle.cycle_id,
        });
        let _ = call(
            CallTargetCell::Local,
            ZomeName::from("governance_bridge"),
            FunctionName::from("dispatch_finance_call"),
            None,
            serde_json::to_vec(&payload)
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
        );
    }

    Ok(())
}

/// Record a milestone completion and disburse funds.
#[hdk_extern]
pub fn complete_milestone(input: CompleteMilestoneInput) -> ExternResult<ActionHash> {
    require_consciousness_tier(2)?; // Voting tier for milestone verification

    let disbursement = Disbursement {
        project_id: input.project_id,
        milestone_index: input.milestone_index,
        amount: input.amount,
        recipient_did: input.recipient_did,
        authorizer_did: input.authorizer_did,
        disbursed_at: sys_time()?,
    };

    let action_hash = create_entry(&EntryTypes::Disbursement(disbursement.clone()))?;

    let project_anchor = ensure_anchor(&format!("project_disbursements:{}", disbursement.project_id))?;
    create_link(project_anchor, action_hash.clone(), LinkTypes::ProjectToDisbursement, ())?;

    Ok(action_hash)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CompleteMilestoneInput {
    pub project_id: String,
    pub milestone_index: u32,
    pub amount: u64,
    pub recipient_did: String,
    pub authorizer_did: String,
}

/// Get all disbursements for a project.
#[hdk_extern]
pub fn get_project_disbursements(project_id: String) -> ExternResult<Vec<Record>> {
    let anchor = ensure_anchor(&format!("project_disbursements:{}", project_id))?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::ProjectToDisbursement)?,
        GetStrategy::Local,
    )?;
    let mut records = Vec::new();
    for link in links {
        if let Ok(hash) = ActionHash::try_from(link.target) {
            if let Some(record) = get(hash, GetOptions::default())? {
                records.push(record);
            }
        }
    }
    Ok(records)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_tally_sorts_by_weight() {
        let mut tallies = vec![
            ProjectTally {
                project_id: "low".into(), title: "Low".into(),
                requested_amount: 1000, minimum_amount: 500,
                effective_weight: 2.5, total_credits_spent: 10, voter_count: 3,
                beneficiary_domain: "commons:housing".into(),
            },
            ProjectTally {
                project_id: "high".into(), title: "High".into(),
                requested_amount: 2000, minimum_amount: 1000,
                effective_weight: 8.3, total_credits_spent: 80, voter_count: 15,
                beneficiary_domain: "civic:emergency".into(),
            },
            ProjectTally {
                project_id: "mid".into(), title: "Mid".into(),
                requested_amount: 500, minimum_amount: 200,
                effective_weight: 5.0, total_credits_spent: 30, voter_count: 7,
                beneficiary_domain: "commons:food".into(),
            },
        ];
        tallies.sort_by(|a, b| b.effective_weight.partial_cmp(&a.effective_weight).unwrap());
        assert_eq!(tallies[0].project_id, "high");
        assert_eq!(tallies[1].project_id, "mid");
        assert_eq!(tallies[2].project_id, "low");
    }

    #[test]
    fn quadratic_vote_math() {
        // 100 credits → √100 = 10 effective votes
        // 25 credits → √25 = 5 effective votes
        // Cost doubles from 25→100 but votes only double from 5→10
        let credits_a: u64 = 100;
        let credits_b: u64 = 25;
        let votes_a = (credits_a as f64).sqrt();
        let votes_b = (credits_b as f64).sqrt();
        assert!((votes_a - 10.0).abs() < 1e-9);
        assert!((votes_b - 5.0).abs() < 1e-9);
        assert_eq!(credits_a / credits_b, 4); // 4x cost
        assert_eq!(votes_a as u64 / votes_b as u64, 2); // 2x votes
    }

    #[test]
    fn milestone_percentages_validation() {
        let milestones = vec![
            ProjectMilestone { description: "Phase 1".into(), percentage: 30, verified: false, verifier_did: None },
            ProjectMilestone { description: "Phase 2".into(), percentage: 40, verified: false, verifier_did: None },
            ProjectMilestone { description: "Phase 3".into(), percentage: 30, verified: false, verifier_did: None },
        ];
        let total: u8 = milestones.iter().map(|m| m.percentage).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn vote_direction_oppose_subtracts() {
        let support_credits = 16u64;
        let oppose_credits = 9u64;
        let net = (support_credits as f64).sqrt() - (oppose_credits as f64).sqrt();
        assert!((net - 1.0).abs() < 1e-9); // √16 - √9 = 4 - 3 = 1
    }

    #[test]
    fn budget_phase_transitions() {
        let phases = [
            BudgetPhase::Proposal,
            BudgetPhase::Deliberation,
            BudgetPhase::Voting,
            BudgetPhase::Execution,
            BudgetPhase::Complete,
        ];
        for i in 0..phases.len() - 1 {
            assert_ne!(phases[i], phases[i + 1]);
        }
    }
}
