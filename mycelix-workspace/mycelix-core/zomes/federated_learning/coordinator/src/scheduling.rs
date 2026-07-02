// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Distributed round scheduling for decentralized FL.
//!
//! Manages round lifecycle: collection, commit, reveal, and finalization phases.

use hdk::prelude::*;
use federated_learning_integrity::*;

use crate::auth::*;
use crate::signals::Signal;
use super::ensure_path;
use crate::gradients::get_round_gradients;
use crate::consensus::{
    get_active_validators_internal,
    get_round_commitments_internal,
    get_round_reveals_internal,
    get_round_consensus_internal,
};

/// Schedule field values (used in both proposed changes and current values).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ScheduleValues {
    pub round_duration_secs: Option<u64>,
    pub commit_window_secs: Option<u64>,
    pub reveal_window_secs: Option<u64>,
    pub min_participants: Option<u32>,
    pub max_participants: Option<u32>,
}

/// Structured schedule change proposal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ScheduleProposal {
    pub proposer: String,
    pub proposed_at: i64,
    pub trust_score: f64,
    pub reason: String,
    pub changes: ScheduleValues,
    pub current_values: Option<ScheduleValues>,
}

/// Input for creating a round schedule
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateRoundScheduleInput {
    /// Duration of gradient collection phase (seconds)
    pub round_duration_secs: u64,
    /// Minimum participants to proceed with aggregation
    pub min_participants: u32,
    /// Maximum participants per round
    pub max_participants: u32,
    /// Duration of commit window (seconds)
    pub commit_window_secs: u64,
    /// Duration of reveal window (seconds)
    pub reveal_window_secs: u64,
}

/// Create a round schedule (requires coordinator role)
#[hdk_extern]
pub fn create_round_schedule(input: CreateRoundScheduleInput) -> ExternResult<ActionHash> {
    require_coordinator_role()?;

    let agent = agent_info()?.agent_initial_pubkey;
    let agent_str = agent.to_string();
    let now = sys_time()?.0 as i64 / 1_000_000;

    let schedule = RoundSchedule {
        round_duration_secs: input.round_duration_secs,
        min_participants: input.min_participants,
        max_participants: input.max_participants,
        current_round: 0,
        round_start_time: now,
        commit_window_secs: input.commit_window_secs,
        reveal_window_secs: input.reveal_window_secs,
        created_by: agent_str.clone(),
        approved_by_json: serde_json::to_string(&vec![agent_str])
            .unwrap_or_else(|_| "[]".to_string()),
        active: true,
    };

    let action_hash = create_entry(&EntryTypes::RoundSchedule(schedule))?;

    // Link to schedule path
    let schedule_path = Path::from("round_schedule");
    let schedule_hash = ensure_path(schedule_path, LinkTypes::RoundToSchedule)?;
    create_link(
        schedule_hash,
        action_hash.clone(),
        LinkTypes::RoundToSchedule,
        vec![],
    )?;

    let _ = log_coordinator_action(
        "create_round_schedule",
        "round_schedule",
        &format!("{{\"duration\":{},\"commit_window\":{},\"reveal_window\":{}}}",
            input.round_duration_secs, input.commit_window_secs, input.reveal_window_secs),
    );

    Ok(action_hash)
}

/// Get the active round schedule
#[hdk_extern]
pub fn get_round_schedule(_: ()) -> ExternResult<Option<RoundSchedule>> {
    get_active_round_schedule_internal()
}

pub(crate) fn get_active_round_schedule_internal() -> ExternResult<Option<RoundSchedule>> {
    let schedule_path = Path::from("round_schedule");
    let schedule_hash = match schedule_path.clone().typed(LinkTypes::RoundToSchedule) {
        Ok(typed) => {
            if !typed.exists()? {
                return Ok(None);
            }
            typed.path_entry_hash()?
        }
        Err(_) => return Ok(None),
    };

    let links = get_links(
        LinkQuery::new(
            schedule_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::RoundToSchedule as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    // Get the most recent active schedule
    for link in links.iter().rev() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(schedule) = RoundSchedule::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        if schedule.active {
                            return Ok(Some(schedule));
                        }
                    }
                }
            }
        }
    }

    Ok(None)
}

/// Round status information
#[derive(Serialize, Deserialize, Debug)]
pub struct RoundStatus {
    /// Current round number
    pub current_round: u64,
    /// Which phase we're in
    pub phase: String,
    /// Seconds remaining in current phase
    pub seconds_remaining: i64,
    /// Number of gradients submitted
    pub gradient_count: usize,
    /// Number of commitments received
    pub commitment_count: usize,
    /// Number of reveals received
    pub reveal_count: usize,
    /// Whether consensus has been reached
    pub consensus_reached: bool,
}

/// Check the status of the current round
#[hdk_extern]
pub fn check_round_status(_: ()) -> ExternResult<RoundStatus> {
    let now = sys_time()?.0 as i64 / 1_000_000;

    let schedule = match get_active_round_schedule_internal()? {
        Some(s) => s,
        None => {
            return Ok(RoundStatus {
                current_round: 0,
                phase: "no_schedule".to_string(),
                seconds_remaining: 0,
                gradient_count: 0,
                commitment_count: 0,
                reveal_count: 0,
                consensus_reached: false,
            });
        }
    };

    let round = schedule.current_round;
    let round_end = schedule.round_start_time + schedule.round_duration_secs as i64;
    let commit_end = round_end + schedule.commit_window_secs as i64;
    let reveal_end = commit_end + schedule.reveal_window_secs as i64;

    let (phase, seconds_remaining) = if now < round_end {
        ("collecting".to_string(), round_end - now)
    } else if now < commit_end {
        ("committing".to_string(), commit_end - now)
    } else if now < reveal_end {
        ("revealing".to_string(), reveal_end - now)
    } else {
        ("finalizing".to_string(), 0)
    };

    // Get counts
    let gradients = get_round_gradients(round as u32)?;
    let commitments = get_round_commitments_internal(round)?;
    let reveals = get_round_reveals_internal(round)?;
    let consensus = get_round_consensus_internal(round)?;

    Ok(RoundStatus {
        current_round: round,
        phase,
        seconds_remaining,
        gradient_count: gradients.len(),
        commitment_count: commitments.len(),
        reveal_count: reveals.len(),
        consensus_reached: consensus.is_some(),
    })
}

/// Advance to the next round
///
/// Any node can call this after the current round's finalization phase.
/// Creates a new round schedule entry with incremented round number.
#[hdk_extern]
pub fn advance_round(_: ()) -> ExternResult<u64> {
    let now = sys_time()?.0 as i64 / 1_000_000;

    let schedule = match get_active_round_schedule_internal()? {
        Some(s) => s,
        None => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "No active round schedule — create one first".to_string()
            )));
        }
    };

    // Check that current round's full cycle has elapsed
    let round_end = schedule.round_start_time + schedule.round_duration_secs as i64;
    let commit_end = round_end + schedule.commit_window_secs as i64;
    let reveal_end = commit_end + schedule.reveal_window_secs as i64;

    if now < reveal_end {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!(
                "Current round not yet complete. Reveal phase ends in {} seconds.",
                reveal_end - now
            )
        )));
    }

    // SECURITY [H-06]: Ensure consensus has been finalized before advancing.
    // A malicious node could advance rounds immediately after reveal_end,
    // before honest nodes finalize consensus, skipping accountability.
    let current_round = schedule.current_round;
    if get_round_consensus_internal(current_round)?.is_none() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!(
                "Cannot advance round: consensus has not been finalized for round {}. \
                 Call finalize_round_consensus first.",
                current_round
            )
        )));
    }

    let new_round = schedule.current_round + 1;
    let agent_str = agent_info()?.agent_initial_pubkey.to_string();

    // Create new schedule with incremented round
    let new_schedule = RoundSchedule {
        round_duration_secs: schedule.round_duration_secs,
        min_participants: schedule.min_participants,
        max_participants: schedule.max_participants,
        current_round: new_round,
        round_start_time: now,
        commit_window_secs: schedule.commit_window_secs,
        reveal_window_secs: schedule.reveal_window_secs,
        created_by: agent_str.clone(),
        approved_by_json: schedule.approved_by_json.clone(),
        active: true,
    };

    let action_hash = create_entry(&EntryTypes::RoundSchedule(new_schedule))?;

    // Link to schedule path
    let schedule_path = Path::from("round_schedule");
    let schedule_hash = ensure_path(schedule_path, LinkTypes::RoundToSchedule)?;
    create_link(
        schedule_hash,
        action_hash,
        LinkTypes::RoundToSchedule,
        vec![],
    )?;

    let _ = log_coordinator_action(
        "advance_round",
        &format!("round_{}", new_round),
        &format!("{{\"previous_round\":{},\"new_round\":{}}}", schedule.current_round, new_round),
    );

    // Emit gossip signal: new round has started
    emit_signal(Signal::RoundStarted {
        round: new_round,
        schedule_hash: None,
        source: Some(agent_str),
        signature: None,
    })?;

    Ok(new_round)
}

/// Input for proposing a schedule change
#[derive(Serialize, Deserialize, Debug)]
pub struct ProposeScheduleChangeInput {
    /// Proposed round duration (seconds), or None to keep current
    pub round_duration_secs: Option<u64>,
    /// Proposed commit window (seconds), or None to keep current
    pub commit_window_secs: Option<u64>,
    /// Proposed reveal window (seconds), or None to keep current
    pub reveal_window_secs: Option<u64>,
    /// Proposed minimum participants, or None to keep current
    pub min_participants: Option<u32>,
    /// Proposed maximum participants, or None to keep current
    pub max_participants: Option<u32>,
    /// Reason for the proposed change
    pub reason: String,
}

/// Propose a schedule change (requires minimum trust score of 0.5)
///
/// Creates a ScheduleProposal record serialized as JSON and links it to
/// the current round schedule path. Only registered validators with
/// trust_score >= 0.5 may propose changes.
#[hdk_extern]
pub fn propose_schedule_change(input: ProposeScheduleChangeInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_str = agent.to_string();
    let now = sys_time()?.0 as i64 / 1_000_000;

    // Verify caller is a registered validator with sufficient trust
    let validators = get_active_validators_internal()?;
    let validator = validators.iter().find(|v| v.agent_pubkey == agent_str);
    let trust_score = match validator {
        Some(v) => v.trust_score,
        None => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Must be a registered validator to propose schedule changes".to_string()
            )));
        }
    };

    if trust_score < 0.5 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!(
                "Insufficient trust score to propose schedule changes: {:.2} < 0.50",
                trust_score
            )
        )));
    }

    // Get current schedule to fill in defaults for None fields
    let current = get_active_round_schedule_internal()?;

    // Build proposal
    let proposal = ScheduleProposal {
        proposer: agent_str.clone(),
        proposed_at: now,
        trust_score: trust_score as f64,
        reason: input.reason.clone(),
        changes: ScheduleValues {
            round_duration_secs: input.round_duration_secs,
            commit_window_secs: input.commit_window_secs,
            reveal_window_secs: input.reveal_window_secs,
            min_participants: input.min_participants,
            max_participants: input.max_participants,
        },
        current_values: current.as_ref().map(|s| ScheduleValues {
            round_duration_secs: Some(s.round_duration_secs),
            commit_window_secs: Some(s.commit_window_secs),
            reveal_window_secs: Some(s.reveal_window_secs),
            min_participants: Some(s.min_participants),
            max_participants: Some(s.max_participants),
        }),
    };
    let proposal_json = serde_json::to_string(&proposal).unwrap_or_default();

    // Store proposal as a RoundSchedule entry with active=false (proposal, not enacted)
    let proposed_round_duration = input.round_duration_secs
        .unwrap_or_else(|| current.as_ref().map_or(300, |s| s.round_duration_secs));
    let proposed_commit_window = input.commit_window_secs
        .unwrap_or_else(|| current.as_ref().map_or(60, |s| s.commit_window_secs));
    let proposed_reveal_window = input.reveal_window_secs
        .unwrap_or_else(|| current.as_ref().map_or(60, |s| s.reveal_window_secs));
    let proposed_min = input.min_participants
        .unwrap_or_else(|| current.as_ref().map_or(2, |s| s.min_participants));
    let proposed_max = input.max_participants
        .unwrap_or_else(|| current.as_ref().map_or(100, |s| s.max_participants));

    let proposal_schedule = RoundSchedule {
        round_duration_secs: proposed_round_duration,
        min_participants: proposed_min,
        max_participants: proposed_max,
        current_round: current.as_ref().map_or(0, |s| s.current_round),
        round_start_time: now,
        commit_window_secs: proposed_commit_window,
        reveal_window_secs: proposed_reveal_window,
        created_by: agent_str.clone(),
        approved_by_json: serde_json::to_string(&Vec::<String>::new())
            .unwrap_or_else(|_| "[]".to_string()),
        active: false, // Proposal, not yet enacted
    };

    let action_hash = create_entry(&EntryTypes::RoundSchedule(proposal_schedule))?;

    // Link proposal to the schedule_proposals path for discovery
    let proposal_path = Path::from("schedule_proposals");
    let proposal_hash = ensure_path(proposal_path, LinkTypes::RoundToSchedule)?;
    create_link(
        proposal_hash,
        action_hash.clone(),
        LinkTypes::RoundToSchedule,
        proposal_json.as_bytes().to_vec(),
    )?;

    // Also link to the current schedule path so proposals can be found from there
    let schedule_path = Path::from("round_schedule");
    if let Ok(typed) = schedule_path.clone().typed(LinkTypes::RoundToSchedule) {
        if typed.exists()? {
            let schedule_hash = typed.path_entry_hash()?;
            create_link(
                schedule_hash,
                action_hash.clone(),
                LinkTypes::RoundToSchedule,
                format!("proposal:{}", agent_str).as_bytes().to_vec(),
            )?;
        }
    }

    let _ = log_coordinator_action(
        "propose_schedule_change",
        "round_schedule",
        &proposal_json,
    );

    Ok(action_hash)
}
