// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Wellbeing Coordinator Zome
//!
//! Self-sovereign wellbeing telemetry. Check-ins are private by default.
//! Opt-in aggregation feeds collective phi. Trend detection generates
//! gentle nudges (offers, never mandates).

use hdk::prelude::*;
use mycelix_bridge_common::civic_requirement_basic;
use mycelix_zome_helpers::records_from_links;
use wellbeing_integrity::*;

// =============================================================================
// Helpers
// =============================================================================


fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn ensure_anchor(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    create_entry(&EntryTypes::Anchor(anchor))?;
    anchor_hash(anchor_str)
}

/// Sustained high stress threshold for nudge generation.
const STRESS_NUDGE_THRESHOLD: u8 = 7;

/// Number of consecutive high-stress check-ins required to trigger a nudge.
const CONSECUTIVE_HIGH_STRESS: usize = 3;

/// Check-in regularity window (microseconds) — 30 days.
const CONSISTENCY_WINDOW_US: i64 = 30 * 24 * 60 * 60 * 1_000_000;

/// Expected check-ins per 30-day window for maximum consistency score.
const EXPECTED_CHECKINS_PER_MONTH: f32 = 15.0;

// =============================================================================
// Check-in CRUD
// =============================================================================

/// Create a new wellbeing check-in. Private by default.
/// After creation, evaluates trend detection for nudge generation.
#[hdk_extern]
pub fn create_checkin(input: WellbeingCheckIn) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "create_checkin")?;

    let action_hash = create_entry(&EntryTypes::WellbeingCheckIn(input))?;

    // Link agent -> check-in
    let agent_info = agent_info()?;
    let agent_anchor = ensure_anchor(&format!("agent_checkins:{}", agent_info.agent_initial_pubkey))?;
    create_link(
        agent_anchor,
        action_hash.clone(),
        LinkTypes::AgentToCheckIns,
        (),
    )?;

    // Auto-evaluate trend detection
    let _ = evaluate_trend_internal();

    let record = get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created check-in".into())))?;
    Ok(record)
}

/// Get the calling agent's check-in history.
#[hdk_extern]
pub fn get_my_checkins(_: ()) -> ExternResult<Vec<Record>> {
    let agent_info = agent_info()?;
    let agent_anchor = anchor_hash(&format!("agent_checkins:{}", agent_info.agent_initial_pubkey))?;
    let links = get_links(
        LinkQuery::try_new(agent_anchor, LinkTypes::AgentToCheckIns)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get recent check-ins (last N) for the calling agent.
#[hdk_extern]
pub fn get_recent_checkins(count: u32) -> ExternResult<Vec<Record>> {
    let mut checkins = get_my_checkins(())?;
    // Sort by timestamp descending (most recent first)
    checkins.sort_by(|a, b| {
        b.action().timestamp().cmp(&a.action().timestamp())
    });
    checkins.truncate(count as usize);
    Ok(checkins)
}

// =============================================================================
// Privacy / Sharing
// =============================================================================

/// Opt in to aggregate computation.
#[hdk_extern]
pub fn opt_in_aggregate(_: ()) -> ExternResult<Record> {
    let agent_info = agent_info()?;
    let now = sys_time()?;

    let opt_in = AggregateOptIn {
        agent: agent_info.agent_initial_pubkey.clone(),
        active: true,
        updated_at: now,
    };
    let action_hash = create_entry(&EntryTypes::AggregateOptIn(opt_in))?;

    // Link to opt-in anchor
    let opt_in_anchor = ensure_anchor("aggregate_opt_ins")?;
    create_link(
        opt_in_anchor,
        action_hash.clone(),
        LinkTypes::OptInToAgent,
        (),
    )?;

    // Link agent to their opt-in
    let agent_anchor = ensure_anchor(&format!("agent_optin:{}", agent_info.agent_initial_pubkey))?;
    create_link(
        agent_anchor,
        action_hash.clone(),
        LinkTypes::AgentToOptIn,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get opt-in record".into())))
}

/// Opt out of aggregate computation.
#[hdk_extern]
pub fn opt_out_aggregate(_: ()) -> ExternResult<Record> {
    let agent_info = agent_info()?;
    let now = sys_time()?;

    let opt_in = AggregateOptIn {
        agent: agent_info.agent_initial_pubkey.clone(),
        active: false,
        updated_at: now,
    };
    let action_hash = create_entry(&EntryTypes::AggregateOptIn(opt_in))?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get opt-out record".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ShareCheckInInput {
    pub checkin_hash: ActionHash,
    pub circle_hash: ActionHash,
}

/// Share a specific check-in with a care circle.
#[hdk_extern]
pub fn share_checkin_with_circle(input: ShareCheckInInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let share = CheckInShare {
        checkin_hash: input.checkin_hash.clone(),
        circle_hash: input.circle_hash.clone(),
        shared_at: now,
    };
    let action_hash = create_entry(&EntryTypes::CheckInShare(share))?;

    // Link check-in -> share
    create_link(
        input.checkin_hash,
        action_hash.clone(),
        LinkTypes::CheckInToShare,
        (),
    )?;

    // Link circle -> shared check-in
    let circle_anchor = ensure_anchor(&format!("circle_shared_checkins:{}", input.circle_hash))?;
    create_link(
        circle_anchor,
        action_hash.clone(),
        LinkTypes::CircleToSharedCheckIns,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get share record".into())))
}

/// Get check-ins shared with a specific circle.
#[hdk_extern]
pub fn get_circle_shared_checkins(circle_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let circle_anchor = anchor_hash(&format!("circle_shared_checkins:{}", circle_hash))?;
    let links = get_links(
        LinkQuery::try_new(circle_anchor, LinkTypes::CircleToSharedCheckIns)?, GetStrategy::default(),
    )?;
    records_from_links(links)
}

// =============================================================================
// Aggregate Computation
// =============================================================================

/// Compute and publish a world-level wellbeing aggregate.
/// Reads opted-in agents' most recent check-ins, computes means.
/// Does NOT reveal individual data — only aggregate statistics.
#[hdk_extern]
pub fn compute_aggregate(_: ()) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "compute_aggregate")?;

    let agent_info = agent_info()?;
    let now = sys_time()?;

    // Get opted-in agents
    let opt_in_anchor = anchor_hash("aggregate_opt_ins")?;
    let opt_in_links = get_links(
        LinkQuery::try_new(opt_in_anchor, LinkTypes::OptInToAgent)?, GetStrategy::default(),
    )?;
    let opt_in_records = records_from_links(opt_in_links)?;

    // Collect most recent check-in from each opted-in agent
    let mut total_stress = 0.0f32;
    let mut total_connection = 0.0f32;
    let mut total_engagement = 0.0f32;
    let mut sample_size = 0u32;

    for record in &opt_in_records {
        if let Some(opt_in) = record
            .entry()
            .to_app_option::<AggregateOptIn>()
            .ok()
            .flatten()
        {
            if !opt_in.active {
                continue;
            }

            // Get agent's most recent check-in
            let agent_anchor =
                anchor_hash(&format!("agent_checkins:{}", opt_in.agent))?;
            let checkin_links = get_links(
                LinkQuery::try_new(agent_anchor, LinkTypes::AgentToCheckIns)?, GetStrategy::default(),
            )?;
            let checkin_records = records_from_links(checkin_links)?;

            // Find the most recent
            if let Some(latest) = checkin_records
                .iter()
                .max_by_key(|r| r.action().timestamp())
            {
                if let Some(checkin) = latest
                    .entry()
                    .to_app_option::<WellbeingCheckIn>()
                    .ok()
                    .flatten()
                {
                    total_stress += checkin.stress_level as f32;
                    total_connection += checkin.connection_score as f32;
                    total_engagement += checkin.engagement_score as f32;
                    sample_size += 1;
                }
            }
        }
    }

    if sample_size == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "No opted-in agents with check-ins found".into()
        )));
    }

    let aggregate = WellbeingAggregate {
        mean_stress: total_stress / sample_size as f32,
        mean_connection: total_connection / sample_size as f32,
        mean_engagement: total_engagement / sample_size as f32,
        sample_size,
        computed_at: now,
        computed_by: agent_info.agent_initial_pubkey,
    };
    let action_hash = create_entry(&EntryTypes::WellbeingAggregate(aggregate))?;

    // Link to aggregates anchor
    let agg_anchor = ensure_anchor("all_aggregates")?;
    create_link(
        agg_anchor,
        action_hash.clone(),
        LinkTypes::AllAggregates,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get aggregate".into())))
}

/// Get the latest wellbeing aggregate.
#[hdk_extern]
pub fn get_latest_aggregate(_: ()) -> ExternResult<Option<Record>> {
    let agg_anchor = anchor_hash("all_aggregates")?;
    let links = get_links(
        LinkQuery::try_new(agg_anchor, LinkTypes::AllAggregates)?, GetStrategy::default(),
    )?;
    let records = records_from_links(links)?;

    Ok(records
        .into_iter()
        .max_by_key(|r| r.action().timestamp()))
}

/// Get aggregate history (for trend visualization).
#[hdk_extern]
pub fn get_aggregate_history(limit: u32) -> ExternResult<Vec<Record>> {
    let agg_anchor = anchor_hash("all_aggregates")?;
    let links = get_links(
        LinkQuery::try_new(agg_anchor, LinkTypes::AllAggregates)?, GetStrategy::default(),
    )?;
    let mut records = records_from_links(links)?;
    records.sort_by(|a, b| b.action().timestamp().cmp(&a.action().timestamp()));
    records.truncate(limit as usize);
    Ok(records)
}

// =============================================================================
// Trend Detection / Nudges
// =============================================================================

/// Check the calling agent's recent check-ins for sustained high stress.
/// If stress > 7 for 3+ consecutive check-ins, generates a WellbeingNudge.
#[hdk_extern]
pub fn evaluate_trend(_: ()) -> ExternResult<Option<Record>> {
    evaluate_trend_internal()
}

fn evaluate_trend_internal() -> ExternResult<Option<Record>> {
    let agent_info = agent_info()?;
    let agent_anchor = anchor_hash(&format!("agent_checkins:{}", agent_info.agent_initial_pubkey))?;
    let links = get_links(
        LinkQuery::try_new(agent_anchor, LinkTypes::AgentToCheckIns)?,
        GetStrategy::default(),
    )?;
    let mut records = records_from_links(links)?;
    records.sort_by(|a, b| b.action().timestamp().cmp(&a.action().timestamp()));

    // Check last N check-ins for sustained high stress
    let recent: Vec<(ActionHash, WellbeingCheckIn)> = records
        .iter()
        .take(CONSECUTIVE_HIGH_STRESS)
        .filter_map(|r| {
            r.entry()
                .to_app_option::<WellbeingCheckIn>()
                .ok()
                .flatten()
                .map(|c| (r.action_address().clone(), c))
        })
        .collect();

    if recent.len() < CONSECUTIVE_HIGH_STRESS {
        return Ok(None);
    }

    let all_high_stress = recent
        .iter()
        .all(|(_, c)| c.stress_level >= STRESS_NUDGE_THRESHOLD);

    if !all_high_stress {
        return Ok(None);
    }

    // Generate nudge
    let now = sys_time()?;
    let trigger_hashes: Vec<ActionHash> = recent.iter().map(|(h, _)| h.clone()).collect();

    let nudge = WellbeingNudge {
        agent: agent_info.agent_initial_pubkey.clone(),
        nudge_type: NudgeType::CareCircleConnection,
        message: format!(
            "Your last {} check-ins show sustained high stress (>{}). \
             Consider connecting with a care circle or scheduling a TEND care exchange.",
            CONSECUTIVE_HIGH_STRESS, STRESS_NUDGE_THRESHOLD
        ),
        trigger_checkins: trigger_hashes,
        created_at: now,
        acknowledged: false,
    };
    let action_hash = create_entry(&EntryTypes::WellbeingNudge(nudge))?;

    // Link agent -> nudge
    let nudge_anchor = ensure_anchor(&format!("agent_nudges:{}", agent_info.agent_initial_pubkey))?;
    create_link(
        nudge_anchor,
        action_hash.clone(),
        LinkTypes::AgentToNudges,
        (),
    )?;

    let record = get(action_hash, GetOptions::default())?;
    Ok(record)
}

/// Get the calling agent's active (unacknowledged) nudges.
#[hdk_extern]
pub fn get_my_nudges(_: ()) -> ExternResult<Vec<Record>> {
    let agent_info = agent_info()?;
    let nudge_anchor = anchor_hash(&format!("agent_nudges:{}", agent_info.agent_initial_pubkey))?;
    let links = get_links(
        LinkQuery::try_new(nudge_anchor, LinkTypes::AgentToNudges)?, GetStrategy::default(),
    )?;
    let records = records_from_links(links)?;

    // Filter to unacknowledged
    Ok(records
        .into_iter()
        .filter(|r| {
            r.entry()
                .to_app_option::<WellbeingNudge>()
                .ok()
                .flatten()
                .map(|n| !n.acknowledged)
                .unwrap_or(false)
        })
        .collect())
}

/// Acknowledge/dismiss a nudge.
#[hdk_extern]
pub fn acknowledge_nudge(nudge_hash: ActionHash) -> ExternResult<Record> {
    let record = get(nudge_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Nudge not found".into())))?;

    let mut nudge: WellbeingNudge = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {e}"))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not a nudge entry".into())))?;

    nudge.acknowledged = true;
    let new_hash = update_entry(nudge_hash, &EntryTypes::WellbeingNudge(nudge))?;

    get(new_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get updated nudge".into())))
}

// =============================================================================
// SubPassport Integration
// =============================================================================

/// Get the calling agent's check-in consistency score (0.0-1.0).
/// Feeds into the engagement dimension of the 4D trust profile.
/// Based on regularity of check-ins over the last 30 days.
#[hdk_extern]
pub fn get_checkin_consistency(_: ()) -> ExternResult<f32> {
    let agent_info = agent_info()?;
    let now = sys_time()?;
    let window_start = Timestamp::from_micros(
        now.as_micros() - CONSISTENCY_WINDOW_US,
    );

    let agent_anchor = anchor_hash(&format!("agent_checkins:{}", agent_info.agent_initial_pubkey))?;
    let links = get_links(
        LinkQuery::try_new(agent_anchor, LinkTypes::AgentToCheckIns)?,
        GetStrategy::default(),
    )?;
    let records = records_from_links(links)?;

    // Count check-ins within the window
    let recent_count = records
        .iter()
        .filter(|r| r.action().timestamp() >= window_start)
        .count() as f32;

    // Score: ratio of actual to expected, capped at 1.0
    let score = (recent_count / EXPECTED_CHECKINS_PER_MONTH).min(1.0);
    Ok(score)
}
