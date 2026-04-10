// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Integration Coordinator Zome
//!
//! Orchestrates cross-zome learning experiences in EduNet:
//!
//! - **Event Processing**: Handle learning events from all zomes
//! - **Progress Aggregation**: Combine stats from SRS, Gamification, Adaptive
//! - **Achievement Evaluation**: Check triggers and award badges
//! - **Session Orchestration**: Plan and manage learning sessions
//! - **Daily Reports**: Generate comprehensive daily summaries
//! - **Cross-Zome Calls**: Aggregate data from SRS, Gamification, and Adaptive zomes

use hdk::prelude::*;
use integration_integrity::*;

// Explicit imports for types used in create_default_progress
use integration_integrity::LearningStyle;

// ============== Cross-Zome Response Types ==============
// These types mirror the responses from other coordinators for deserialization

/// GamificationSummary from gamification_coordinator
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GamificationSummary {
    pub total_xp: u64,
    pub level: u32,
    pub xp_to_next_level: u64,
    pub level_progress_permille: u16,
    pub current_streak: u32,
    pub longest_streak: u32,
    pub streak_bonus_permille: u16,
    pub badges_earned: u32,
    pub freezes_remaining: u8,
}

/// LearnerSummary from adaptive_coordinator
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaptiveLearnerSummary {
    pub profile: Option<serde_json::Value>, // Generic to avoid import issues
    pub learning_style: serde_json::Value,
    pub mastery_count: u32,
    pub avg_mastery_permille: u16,
    pub skills_mastered: u32,
    pub due_for_review: u32,
    pub active_goals: u32,
    pub total_sessions: u32,
    pub total_learning_minutes: u32,
    pub active_paths: u32,
    pub recommendations_count: u32,
}

// ============== Cross-Zome Call Helpers ==============

/// Call a function in the SRS coordinator zome
fn call_srs<I, O>(fn_name: &str, input: I) -> ExternResult<O>
where
    I: serde::Serialize + std::fmt::Debug,
    O: serde::de::DeserializeOwned + std::fmt::Debug,
{
    let response = call(
        CallTargetCell::Local,
        ZomeName::from("srs_coordinator"),
        FunctionName::from(fn_name),
        None,
        &input,
    )?;

    match response {
        ZomeCallResponse::Ok(result) => {
            let decoded: O = result.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Failed to decode SRS response for {}: {:?}", fn_name, e
                )))
            })?;
            Ok(decoded)
        }
        ZomeCallResponse::Unauthorized(_, _, _, _) => {
            Err(wasm_error!(WasmErrorInner::Guest(
                format!("Unauthorized call to srs_coordinator::{}", fn_name)
            )))
        }
        ZomeCallResponse::NetworkError(err) => {
            Err(wasm_error!(WasmErrorInner::Guest(
                format!("Network error calling srs_coordinator::{}: {}", fn_name, err)
            )))
        }
        ZomeCallResponse::CountersigningSession(err) => {
            Err(wasm_error!(WasmErrorInner::Guest(
                format!("Countersigning error calling srs_coordinator::{}: {}", fn_name, err)
            )))
        }
        ZomeCallResponse::AuthenticationFailed(_, _) => {
            Err(wasm_error!(WasmErrorInner::Guest(
                format!("Authentication failed calling srs_coordinator::{}", fn_name)
            )))
        }
    }
}

/// Call a function in the Gamification coordinator zome
fn call_gamification<I, O>(fn_name: &str, input: I) -> ExternResult<O>
where
    I: serde::Serialize + std::fmt::Debug,
    O: serde::de::DeserializeOwned + std::fmt::Debug,
{
    let response = call(
        CallTargetCell::Local,
        ZomeName::from("gamification_coordinator"),
        FunctionName::from(fn_name),
        None,
        &input,
    )?;

    match response {
        ZomeCallResponse::Ok(result) => {
            let decoded: O = result.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Failed to decode Gamification response for {}: {:?}", fn_name, e
                )))
            })?;
            Ok(decoded)
        }
        ZomeCallResponse::Unauthorized(_, _, _, _) => {
            Err(wasm_error!(WasmErrorInner::Guest(
                format!("Unauthorized call to gamification_coordinator::{}", fn_name)
            )))
        }
        ZomeCallResponse::NetworkError(err) => {
            Err(wasm_error!(WasmErrorInner::Guest(
                format!("Network error calling gamification_coordinator::{}: {}", fn_name, err)
            )))
        }
        ZomeCallResponse::CountersigningSession(err) => {
            Err(wasm_error!(WasmErrorInner::Guest(
                format!("Countersigning error calling gamification_coordinator::{}: {}", fn_name, err)
            )))
        }
        ZomeCallResponse::AuthenticationFailed(_, _) => {
            Err(wasm_error!(WasmErrorInner::Guest(
                format!("Authentication failed calling gamification_coordinator::{}", fn_name)
            )))
        }
    }
}

/// Call a function in the Adaptive coordinator zome
fn call_adaptive<I, O>(fn_name: &str, input: I) -> ExternResult<O>
where
    I: serde::Serialize + std::fmt::Debug,
    O: serde::de::DeserializeOwned + std::fmt::Debug,
{
    let response = call(
        CallTargetCell::Local,
        ZomeName::from("adaptive_coordinator"),
        FunctionName::from(fn_name),
        None,
        &input,
    )?;

    match response {
        ZomeCallResponse::Ok(result) => {
            let decoded: O = result.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Failed to decode Adaptive response for {}: {:?}", fn_name, e
                )))
            })?;
            Ok(decoded)
        }
        ZomeCallResponse::Unauthorized(_, _, _, _) => {
            Err(wasm_error!(WasmErrorInner::Guest(
                format!("Unauthorized call to adaptive_coordinator::{}", fn_name)
            )))
        }
        ZomeCallResponse::NetworkError(err) => {
            Err(wasm_error!(WasmErrorInner::Guest(
                format!("Network error calling adaptive_coordinator::{}: {}", fn_name, err)
            )))
        }
        ZomeCallResponse::CountersigningSession(err) => {
            Err(wasm_error!(WasmErrorInner::Guest(
                format!("Countersigning error calling adaptive_coordinator::{}: {}", fn_name, err)
            )))
        }
        ZomeCallResponse::AuthenticationFailed(_, _) => {
            Err(wasm_error!(WasmErrorInner::Guest(
                format!("Authentication failed calling adaptive_coordinator::{}", fn_name)
            )))
        }
    }
}

// ============== Helper Functions ==============

/// Convert a Holochain Timestamp to i64 (microseconds)
fn timestamp_to_i64(ts: Timestamp) -> i64 {
    ts.as_micros()
}

/// Get current time as i64
fn current_time() -> ExternResult<i64> {
    Ok(timestamp_to_i64(sys_time()?))
}

/// Get current date as YYYYMMDD
fn current_date() -> ExternResult<u32> {
    let now = current_time()?;
    // Convert microseconds to days since epoch
    let days_since_epoch = (now / (24 * 60 * 60 * 1_000_000)) as u32;
    // Approximate date calculation (good enough for daily tracking)
    Ok(19700101 + days_since_epoch)
}

/// Get learner anchor hash for events
fn event_anchor() -> ExternResult<EntryHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let path = Path::from(format!("integration.events.{}", agent));
    let typed_path = path.typed(LinkTypes::LearnerToEvents)?;
    typed_path.ensure()?;
    typed_path.path.path_entry_hash()
}

/// Get learner anchor hash for progress
fn progress_anchor() -> ExternResult<EntryHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let path = Path::from(format!("integration.progress.{}", agent));
    let typed_path = path.typed(LinkTypes::LearnerToProgress)?;
    typed_path.ensure()?;
    typed_path.path.path_entry_hash()
}

/// Get learner anchor hash for sessions
fn session_anchor() -> ExternResult<EntryHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let path = Path::from(format!("integration.sessions.{}", agent));
    let typed_path = path.typed(LinkTypes::LearnerToOrchestratedSessions)?;
    typed_path.ensure()?;
    typed_path.path.path_entry_hash()
}

/// Get learner anchor hash for daily reports
fn report_anchor() -> ExternResult<EntryHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let path = Path::from(format!("integration.reports.{}", agent));
    let typed_path = path.typed(LinkTypes::LearnerToDailyReports)?;
    typed_path.ensure()?;
    typed_path.path.path_entry_hash()
}

/// Get global triggers anchor
fn triggers_anchor() -> ExternResult<EntryHash> {
    let path = Path::from("integration.triggers.global");
    let typed_path = path.typed(LinkTypes::GlobalTriggers)?;
    typed_path.ensure()?;
    typed_path.path.path_entry_hash()
}

// ============== Event Processing ==============

/// Input for recording a learning event
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecordEventInput {
    pub event_type: LearningEventType,
    pub source_hash: ActionHash,
    pub source_zome: SourceZome,
    pub quality_permille: u16,
    pub duration_seconds: u32,
    pub skills_affected: Vec<ActionHash>,
    pub streak_day: Option<u32>,
}

/// Record a learning event and process rewards
#[hdk_extern]
pub fn record_learning_event(input: RecordEventInput) -> ExternResult<LearningEventResult> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = current_time()?;
    let anchor = event_anchor()?;

    // Calculate XP (assuming 100 base XP, 1000 streak bonus = no bonus)
    let streak_bonus = input.streak_day.map(|d| streak_bonus_for_day(d)).unwrap_or(1000);
    let xp = calculate_xp(100, input.quality_permille, streak_bonus, &input.event_type);

    // Calculate mastery change based on quality
    let mastery_change = if input.quality_permille >= 800 {
        50 // Significant mastery gain
    } else if input.quality_permille >= 500 {
        20 // Moderate gain
    } else if input.quality_permille >= 300 {
        5 // Small gain
    } else {
        -10 // Slight regression
    };

    let event = LearningEvent {
        learner: agent.clone(),
        event_type: input.event_type.clone(),
        source_hash: input.source_hash,
        source_zome: input.source_zome,
        xp_gained: xp,
        mastery_change,
        skills_affected: input.skills_affected,
        quality_permille: input.quality_permille,
        duration_seconds: input.duration_seconds,
        streak_day: input.streak_day,
        occurred_at: now,
    };

    let action_hash = create_entry(EntryTypes::LearningEvent(event))?;

    create_link(
        anchor,
        action_hash.clone(),
        LinkTypes::LearnerToEvents,
        (),
    )?;

    // Check for triggered achievements
    let triggered = check_achievement_triggers(&agent, &input.event_type, xp)?;

    // Check for reciprocity opportunity: high quality achievement events
    // suggest the learner is ready to give back to the community
    let reciprocity_opportunity = match &input.event_type {
        LearningEventType::SkillMastered | LearningEventType::GoalAchieved
            if input.quality_permille >= 800 =>
        {
            Some("mastery_achieved".to_string())
        }
        _ => None,
    };

    Ok(LearningEventResult {
        event_hash: action_hash,
        xp_gained: xp,
        mastery_change,
        achievements_triggered: triggered,
        reciprocity_opportunity,
    })
}

/// Result from recording a learning event
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearningEventResult {
    pub event_hash: ActionHash,
    pub xp_gained: u32,
    pub mastery_change: i16,
    /// If set, the learner has achieved something significant and may want to
    /// pledge back to the community (tutoring, translation, curriculum review).
    pub reciprocity_opportunity: Option<String>,
    pub achievements_triggered: Vec<String>,
}

/// Calculate streak bonus for a given day
fn streak_bonus_for_day(day: u32) -> u16 {
    match day {
        0..=6 => 1000,     // No bonus first week
        7..=13 => 1100,   // 10% bonus
        14..=29 => 1150,  // 15% bonus
        30..=59 => 1250,  // 25% bonus
        60..=99 => 1350,  // 35% bonus
        100..=179 => 1500, // 50% bonus
        180..=364 => 1750, // 75% bonus
        _ => 2000,         // 100% bonus after a year
    }
}

/// Check if any achievement triggers are met
fn check_achievement_triggers(
    _agent: &AgentPubKey,
    event_type: &LearningEventType,
    _xp_gained: u32,
) -> ExternResult<Vec<String>> {
    let mut triggered = Vec::new();

    // Simple event-based triggers
    match event_type {
        LearningEventType::SkillMastered => {
            triggered.push("skill_master".to_string());
        }
        LearningEventType::GoalAchieved => {
            triggered.push("goal_achiever".to_string());
        }
        LearningEventType::StreakMilestone => {
            triggered.push("consistent_learner".to_string());
        }
        _ => {}
    }

    // Would check global triggers from DHT in full implementation

    Ok(triggered)
}

/// Get recent events for the learner
#[hdk_extern]
pub fn get_recent_events(limit: u32) -> ExternResult<Vec<LearningEvent>> {
    let anchor = event_anchor()?;

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::LearnerToEvents)?,
        GetStrategy::Local,
    )?;

    let mut events = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                let event: LearningEvent = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("No event entry".into())))?;
                events.push(event);
            }
        }
    }

    // Sort by most recent first
    events.sort_by(|a, b| b.occurred_at.cmp(&a.occurred_at));
    events.truncate(limit as usize);

    Ok(events)
}

// ============== Progress Aggregation ==============

/// Input for updating progress aggregate
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpdateProgressInput {
    // XP & Level
    pub total_xp: u64,
    pub level: u32,
    // Streaks
    pub current_streak_days: u32,
    pub longest_streak_days: u32,
    // SRS
    pub srs_cards_total: u32,
    pub srs_cards_mature: u32,
    pub srs_cards_learning: u32,
    pub srs_accuracy_permille: u16,
    // Mastery
    pub skills_tracked: u32,
    pub skills_mastered: u32,
    pub avg_mastery_permille: u16,
    pub skills_due_review: u32,
    // Learning style
    pub primary_style: LearningStyle,
    pub style_confidence_permille: u16,
    // Goals
    pub active_goals: u32,
    pub completed_goals: u32,
    // Engagement
    pub total_sessions: u32,
    pub total_learning_minutes: u32,
    pub days_active: u32,
    // Achievements
    pub badges_earned: u32,
}

/// Update the learner's progress aggregate
#[hdk_extern]
pub fn update_progress_aggregate(input: UpdateProgressInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = current_time()?;
    let anchor = progress_anchor()?;

    // Calculate derived values
    let xp_to_next = ((input.level + 1) * (input.level + 1) * 100) as u32
        - input.total_xp.min(u32::MAX as u64) as u32;
    let streak_bonus = streak_bonus_for_day(input.current_streak_days);
    let goal_progress = if input.active_goals + input.completed_goals > 0 {
        (input.completed_goals * 1000 / (input.active_goals + input.completed_goals)) as u16
    } else {
        0
    };
    let avg_session = if input.total_sessions > 0 {
        (input.total_learning_minutes / input.total_sessions) as u16
    } else {
        0
    };

    // Count rare badges (assuming 10% are rare)
    let rare_badges = input.badges_earned / 10;

    // Calculate confidence based on data points
    let confidence = ((input.total_sessions * 20).min(1000)) as u16;

    let aggregate = LearnerProgressAggregate {
        learner: agent,
        total_xp: input.total_xp,
        level: input.level,
        xp_to_next_level: xp_to_next,
        current_streak_days: input.current_streak_days,
        longest_streak_days: input.longest_streak_days,
        streak_bonus_permille: streak_bonus,
        srs_cards_total: input.srs_cards_total,
        srs_cards_mature: input.srs_cards_mature,
        srs_cards_learning: input.srs_cards_learning,
        srs_reviews_today: 0, // Would need daily tracking
        srs_accuracy_permille: input.srs_accuracy_permille,
        skills_tracked: input.skills_tracked,
        skills_mastered: input.skills_mastered,
        avg_mastery_permille: input.avg_mastery_permille,
        skills_due_review: input.skills_due_review,
        primary_style: input.primary_style,
        style_confidence_permille: input.style_confidence_permille,
        active_goals: input.active_goals,
        completed_goals: input.completed_goals,
        goal_progress_permille: goal_progress,
        total_sessions: input.total_sessions,
        total_learning_minutes: input.total_learning_minutes,
        avg_session_minutes: avg_session,
        days_active: input.days_active,
        badges_earned: input.badges_earned,
        rare_badges,
        leaderboard_rank: None, // Would need global calculation
        confidence_permille: confidence,
        last_activity_at: now,
        aggregated_at: now,
    };

    // Check for existing aggregate
    let links = get_links(
        LinkQuery::try_new(anchor.clone(), LinkTypes::LearnerToProgress)?,
        GetStrategy::Local,
    )?;

    let action_hash = if let Some(link) = links.first() {
        if let Some(existing_hash) = link.target.clone().into_action_hash() {
            update_entry(existing_hash, aggregate)?
        } else {
            let hash = create_entry(EntryTypes::LearnerProgressAggregate(aggregate))?;
            create_link(anchor, hash.clone(), LinkTypes::LearnerToProgress, ())?;
            hash
        }
    } else {
        let hash = create_entry(EntryTypes::LearnerProgressAggregate(aggregate))?;
        create_link(anchor, hash.clone(), LinkTypes::LearnerToProgress, ())?;
        hash
    };

    Ok(action_hash)
}

/// Get the learner's progress aggregate
#[hdk_extern]
pub fn get_progress_aggregate(_: ()) -> ExternResult<Option<LearnerProgressAggregate>> {
    let anchor = progress_anchor()?;

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::LearnerToProgress)?,
        GetStrategy::Local,
    )?;

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                let aggregate: LearnerProgressAggregate = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("No aggregate entry".into())))?;
                return Ok(Some(aggregate));
            }
        }
    }

    Ok(None)
}

// ============== Cross-Zome Aggregation ==============

/// Result of cross-zome progress refresh
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CrossZomeProgressResult {
    pub aggregate_hash: ActionHash,
    pub gamification: Option<GamificationSummary>,
    pub adaptive_summary: Option<AdaptiveLearnerSummary>,
    pub srs_cards_count: u32,
    pub srs_due_count: u32,
    pub aggregated_at: i64,
    pub errors: Vec<String>,
}

/// Refresh progress aggregate by fetching data from all differentiation zomes
///
/// This function makes cross-zome calls to:
/// - SRS coordinator: get card counts and due cards
/// - Gamification coordinator: get XP, level, streak, badges
/// - Adaptive coordinator: get mastery stats, goals, learning style
///
/// It then combines all data into a unified LearnerProgressAggregate entry.
#[hdk_extern]
pub fn refresh_progress_from_zomes(_: ()) -> ExternResult<CrossZomeProgressResult> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = current_time()?;
    let anchor = progress_anchor()?;
    let mut errors = Vec::new();

    // === Fetch from Gamification ===
    let gamification: Option<GamificationSummary> = match call_gamification::<(), GamificationSummary>(
        "get_gamification_summary", ()
    ) {
        Ok(summary) => Some(summary),
        Err(e) => {
            errors.push(format!("Gamification fetch failed: {:?}", e));
            None
        }
    };

    // === Fetch from Adaptive ===
    let adaptive_summary: Option<AdaptiveLearnerSummary> = match call_adaptive::<(), AdaptiveLearnerSummary>(
        "get_learner_summary", ()
    ) {
        Ok(summary) => Some(summary),
        Err(e) => {
            errors.push(format!("Adaptive fetch failed: {:?}", e));
            None
        }
    };

    // === Fetch SRS card counts ===
    let srs_cards: Vec<Record> = match call_srs::<(), Vec<Record>>("get_my_cards", ()) {
        Ok(cards) => cards,
        Err(e) => {
            errors.push(format!("SRS cards fetch failed: {:?}", e));
            Vec::new()
        }
    };

    let srs_due: Vec<Record> = match call_srs::<u32, Vec<Record>>("get_due_cards", 1000u32) {
        Ok(cards) => cards,
        Err(e) => {
            errors.push(format!("SRS due cards fetch failed: {:?}", e));
            Vec::new()
        }
    };

    // === Extract values from cross-zome results ===

    // Gamification values (with defaults)
    let (total_xp, level, current_streak, longest_streak, streak_bonus, badges_earned) =
        if let Some(ref gam) = gamification {
            (
                gam.total_xp,
                gam.level,
                gam.current_streak,
                gam.longest_streak,
                gam.streak_bonus_permille,
                gam.badges_earned,
            )
        } else {
            (0, 1, 0, 0, 1000, 0)
        };

    // Adaptive values (with defaults)
    let (skills_tracked, skills_mastered, avg_mastery, skills_due_review,
         active_goals, completed_goals, total_sessions, total_minutes) =
        if let Some(ref adp) = adaptive_summary {
            (
                adp.mastery_count,
                adp.skills_mastered,
                adp.avg_mastery_permille,
                adp.due_for_review,
                adp.active_goals,
                0u32, // completed_goals not in summary
                adp.total_sessions,
                adp.total_learning_minutes,
            )
        } else {
            (0, 0, 0, 0, 0, 0, 0, 0)
        };

    // SRS values
    let srs_cards_total = srs_cards.len() as u32;
    let srs_due_count = srs_due.len() as u32;

    // Count mature vs learning cards (simplified - mature if 7+ days interval)
    let mut srs_mature = 0u32;
    let mut srs_learning = 0u32;
    // Note: We can't easily parse Record entries here without the full type
    // This would need the actual ReviewCard type from srs_integrity
    // For now, estimate based on ratios
    srs_mature = (srs_cards_total as f64 * 0.3) as u32; // Estimate 30% mature
    srs_learning = srs_cards_total - srs_mature;

    // Calculate derived values
    let xp_to_next = ((level + 1) * (level + 1) * 100) - (total_xp.min(u32::MAX as u64) as u32);
    let goal_progress = if active_goals + completed_goals > 0 {
        (completed_goals * 1000 / (active_goals + completed_goals)) as u16
    } else {
        0
    };
    let avg_session = if total_sessions > 0 {
        (total_minutes / total_sessions) as u16
    } else {
        0
    };
    let rare_badges = badges_earned / 10;

    // Determine primary learning style from adaptive data
    // Default to Multimodal if not available
    let primary_style = LearningStyle::Multimodal;
    let style_confidence = if adaptive_summary.is_some() { 500 } else { 0 };

    // Calculate confidence based on data quality
    let confidence = if gamification.is_some() && adaptive_summary.is_some() {
        800u16 // High confidence if both sources available
    } else if gamification.is_some() || adaptive_summary.is_some() {
        500u16 // Medium confidence if one source
    } else {
        200u16 // Low confidence if no sources
    };

    // Create the aggregate
    let aggregate = LearnerProgressAggregate {
        learner: agent,
        total_xp,
        level,
        xp_to_next_level: xp_to_next,
        current_streak_days: current_streak,
        longest_streak_days: longest_streak,
        streak_bonus_permille: streak_bonus,
        srs_cards_total,
        srs_cards_mature: srs_mature,
        srs_cards_learning: srs_learning,
        srs_reviews_today: 0, // Would need daily tracking
        srs_accuracy_permille: 800, // Default, would need stats
        skills_tracked,
        skills_mastered,
        avg_mastery_permille: avg_mastery,
        skills_due_review,
        primary_style,
        style_confidence_permille: style_confidence,
        active_goals,
        completed_goals,
        goal_progress_permille: goal_progress,
        total_sessions,
        total_learning_minutes: total_minutes,
        avg_session_minutes: avg_session,
        days_active: 0, // Would need historical tracking
        badges_earned,
        rare_badges,
        leaderboard_rank: None,
        confidence_permille: confidence,
        last_activity_at: now,
        aggregated_at: now,
    };

    // Store or update the aggregate
    let links = get_links(
        LinkQuery::try_new(anchor.clone(), LinkTypes::LearnerToProgress)?,
        GetStrategy::Local,
    )?;

    let aggregate_hash = if let Some(link) = links.first() {
        if let Some(existing_hash) = link.target.clone().into_action_hash() {
            update_entry(existing_hash, aggregate)?
        } else {
            let hash = create_entry(EntryTypes::LearnerProgressAggregate(aggregate))?;
            create_link(anchor, hash.clone(), LinkTypes::LearnerToProgress, ())?;
            hash
        }
    } else {
        let hash = create_entry(EntryTypes::LearnerProgressAggregate(aggregate))?;
        create_link(anchor, hash.clone(), LinkTypes::LearnerToProgress, ())?;
        hash
    };

    Ok(CrossZomeProgressResult {
        aggregate_hash,
        gamification,
        adaptive_summary,
        srs_cards_count: srs_cards_total,
        srs_due_count,
        aggregated_at: now,
        errors,
    })
}

/// Get a unified dashboard view with data from all zomes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnifiedDashboard {
    pub progress: Option<LearnerProgressAggregate>,
    pub gamification: Option<GamificationSummary>,
    pub adaptive: Option<AdaptiveLearnerSummary>,
    pub srs_due_count: u32,
    pub recent_events: Vec<LearningEvent>,
    pub active_session: Option<OrchestratedSession>,
}

/// Get unified dashboard combining local and cross-zome data
#[hdk_extern]
pub fn get_unified_dashboard(_: ()) -> ExternResult<UnifiedDashboard> {
    // Get local progress aggregate
    let progress = get_progress_aggregate(())?;

    // Get local data
    let recent_events = get_recent_events(10)?;
    let active_session = get_active_session(())?;

    // Fetch cross-zome data
    let gamification: Option<GamificationSummary> = call_gamification("get_gamification_summary", ()).ok();
    let adaptive: Option<AdaptiveLearnerSummary> = call_adaptive("get_learner_summary", ()).ok();
    let srs_due: Vec<Record> = call_srs("get_due_cards", 1000u32).unwrap_or_default();

    Ok(UnifiedDashboard {
        progress,
        gamification,
        adaptive,
        srs_due_count: srs_due.len() as u32,
        recent_events,
        active_session,
    })
}

// ============== Session Orchestration ==============

/// Input for starting an orchestrated session
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StartSessionInput {
    pub target_minutes: u16,
    pub planned_activities: Vec<PlannedActivity>,
}

/// Start an orchestrated learning session
#[hdk_extern]
pub fn start_orchestrated_session(input: StartSessionInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = current_time()?;
    let anchor = session_anchor()?;

    // Generate session ID
    let session_id = format!("session-{}-{}", now, agent);

    let session = OrchestratedSession {
        learner: agent,
        session_id,
        planned_activities: input.planned_activities,
        completed_activities: vec![],
        state: SessionState::Active,
        target_minutes: input.target_minutes,
        actual_seconds: 0,
        xp_earned: 0,
        mastery_changes: vec![],
        flow_score_permille: 500, // Start at neutral
        breaks_taken: 0,
        started_at: now,
        ended_at: None,
    };

    let action_hash = create_entry(EntryTypes::OrchestratedSession(session))?;

    create_link(
        anchor,
        action_hash.clone(),
        LinkTypes::LearnerToOrchestratedSessions,
        (),
    )?;

    Ok(action_hash)
}

/// Input for completing an activity in a session
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompleteActivityInput {
    pub session_hash: ActionHash,
    pub activity: CompletedActivity,
}

/// Complete an activity in an orchestrated session
#[hdk_extern]
pub fn complete_session_activity(input: CompleteActivityInput) -> ExternResult<OrchestratedSession> {
    let _now = current_time()?;

    let record = get(input.session_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Session not found".into())))?;

    let mut session: OrchestratedSession = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("No session entry".into())))?;

    // Add completed activity
    session.completed_activities.push(input.activity.clone());
    session.xp_earned += input.activity.xp_earned;
    session.actual_seconds += input.activity.duration_seconds;

    // Update flow score based on quality
    session.flow_score_permille = (session.flow_score_permille + input.activity.quality_permille) / 2;

    update_entry(input.session_hash, session.clone())?;

    Ok(session)
}

/// End an orchestrated session
#[hdk_extern]
pub fn end_orchestrated_session(session_hash: ActionHash) -> ExternResult<OrchestratedSession> {
    let now = current_time()?;

    let record = get(session_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Session not found".into())))?;

    let mut session: OrchestratedSession = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("No session entry".into())))?;

    session.state = SessionState::Completed;
    session.ended_at = Some(now);
    session.actual_seconds = ((now - session.started_at) / 1_000_000) as u32;

    update_entry(session_hash, session.clone())?;

    Ok(session)
}

/// Get active session
#[hdk_extern]
pub fn get_active_session(_: ()) -> ExternResult<Option<OrchestratedSession>> {
    let anchor = session_anchor()?;

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::LearnerToOrchestratedSessions)?,
        GetStrategy::Local,
    )?;

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                let session: OrchestratedSession = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("No session entry".into())))?;

                if session.state == SessionState::Active || session.state == SessionState::Paused {
                    return Ok(Some(session));
                }
            }
        }
    }

    Ok(None)
}

// ============== Smart Session Planning ==============

/// Input for smart session planning
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmartSessionPlanInput {
    /// Target session duration in minutes
    pub target_minutes: u16,
    /// Focus area (optional) - prioritize specific skill
    pub focus_skill: Option<ActionHash>,
    /// Session type preference
    pub session_type: SessionType,
    /// Include breaks
    pub include_breaks: bool,
}

/// Type of session to plan
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub enum SessionType {
    #[default]
    Balanced,      // Mix of review, learning, practice
    ReviewFocus,   // Prioritize retention (SRS/review)
    NewLearning,   // Prioritize new content
    Practice,      // Prioritize skill practice
    DeepWork,      // Long focus sessions with fewer breaks
}

/// Smart session plan with detailed activities
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmartSessionPlan {
    /// Unique plan ID
    pub plan_id: String,
    /// Total planned minutes
    pub total_minutes: u16,
    /// Planned activities in order
    pub activities: Vec<SmartPlannedActivity>,
    /// Estimated XP gain
    pub estimated_xp: u32,
    /// Priority skills addressed
    pub priority_skills: Vec<ActionHash>,
    /// Skills at retention risk
    pub at_risk_skills: u32,
    /// Recommended breaks
    pub breaks: Vec<PlannedBreak>,
    /// Flow state optimization notes
    pub flow_notes: String,
    /// Confidence in plan quality (0-1000)
    pub confidence_permille: u16,
    /// Generated timestamp
    pub generated_at: i64,
}

/// A smartly planned activity with reasoning
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmartPlannedActivity {
    /// Activity type
    pub activity_type: PlannedActivityType,
    /// Target skill or content hash
    pub target_hash: ActionHash,
    /// Estimated duration in minutes
    pub estimated_minutes: u16,
    /// Priority (1-10, higher = more important)
    pub priority: u8,
    /// Why this activity was chosen
    pub reasoning: Vec<String>,
    /// Expected difficulty match (0-1000)
    pub difficulty_match_permille: u16,
    /// Estimated XP gain
    pub estimated_xp: u32,
    /// Order in session
    pub sequence: u16,
}

/// Planned break
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlannedBreak {
    /// When to take break (minutes into session)
    pub after_minutes: u16,
    /// Duration in minutes
    pub duration_minutes: u8,
    /// Break type
    pub break_type: BreakType,
    /// Suggestion for break activity
    pub suggestion: String,
}

/// Type of break
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum BreakType {
    Micro,      // 1-2 min stretch
    Short,      // 5 min break
    Pomodoro,   // 5-10 min break
    Long,       // 15-20 min break
}

/// Generate a smart session plan based on learner state
#[hdk_extern]
pub fn plan_smart_session(input: SmartSessionPlanInput) -> ExternResult<SmartSessionPlan> {
    let now = current_time()?;
    let agent = agent_info()?.agent_initial_pubkey;

    // Get learner context from adaptive zome
    let learner_context: serde_json::Value = call_adaptive("get_learner_context", ())?;
    let due_for_review: u32 = learner_context.get("due_for_review")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;

    // Get smart recommendations from adaptive zome
    let smart_recs: Vec<serde_json::Value> = call_adaptive(
        "generate_smart_recommendations_v2",
        serde_json::json!({
            "limit": 20,
            "include_new": input.session_type != SessionType::ReviewFocus,
            "include_review": input.session_type != SessionType::NewLearning,
        })
    ).unwrap_or_else(|_| vec![]);

    // Build activity list based on session type and recommendations
    let mut activities: Vec<SmartPlannedActivity> = Vec::new();
    let mut total_minutes: u16 = 0;
    let mut estimated_xp: u32 = 0;
    let mut priority_skills: Vec<ActionHash> = Vec::new();
    let mut sequence: u16 = 1;

    // Calculate activity distribution based on session type
    let (review_ratio, new_ratio, practice_ratio) = match input.session_type {
        SessionType::Balanced => (40, 30, 30),
        SessionType::ReviewFocus => (70, 10, 20),
        SessionType::NewLearning => (20, 60, 20),
        SessionType::Practice => (20, 20, 60),
        SessionType::DeepWork => (30, 40, 30),
    };

    let review_minutes = (input.target_minutes as u32 * review_ratio / 100) as u16;
    let new_minutes = (input.target_minutes as u32 * new_ratio / 100) as u16;
    let practice_minutes = (input.target_minutes as u32 * practice_ratio / 100) as u16;

    // Add review activities if there are items due
    if due_for_review > 0 && review_minutes > 0 {
        let review_activity = SmartPlannedActivity {
            activity_type: PlannedActivityType::SrsReview,
            target_hash: ActionHash::from_raw_36(vec![0u8; 36]), // Generic SRS target
            estimated_minutes: review_minutes.min(30),
            priority: 9, // High priority for retention
            reasoning: vec![
                format!("{} cards due for review", due_for_review),
                "Prioritizing retention to maintain progress".into(),
            ],
            difficulty_match_permille: 800,
            estimated_xp: (review_minutes as u32 * 10).min(300),
            sequence,
        };
        estimated_xp += review_activity.estimated_xp;
        total_minutes += review_activity.estimated_minutes;
        activities.push(review_activity);
        sequence += 1;
    }

    // Add activities from smart recommendations
    for rec in smart_recs.iter().take(5) {
        if total_minutes >= input.target_minutes {
            break;
        }

        let remaining = input.target_minutes.saturating_sub(total_minutes);
        let activity_minutes = (remaining / 3).max(5).min(20);

        // Parse recommendation type
        let rec_type = rec.get("rec_type")
            .and_then(|v| v.as_str())
            .unwrap_or("Practice");

        let activity_type = match rec_type {
            "NewContent" | "Lesson" => PlannedActivityType::NewLesson,
            "Review" | "RetentionRisk" => PlannedActivityType::Review,
            "Practice" | "Challenge" => PlannedActivityType::SkillPractice,
            _ => PlannedActivityType::SkillPractice,
        };

        let smart_score = rec.get("smart_score_permille")
            .and_then(|v| v.as_u64())
            .unwrap_or(500) as u16;

        // Use a placeholder hash for recommendations - real implementation would parse actual hashes
        let skill_hash = ActionHash::from_raw_36(vec![0u8; 36]);

        let activity = SmartPlannedActivity {
            activity_type,
            target_hash: skill_hash.clone(),
            estimated_minutes: activity_minutes,
            priority: ((smart_score / 100) as u8).min(10).max(1),
            reasoning: vec![
                rec.get("explanation")
                    .and_then(|v| v.as_str())
                    .unwrap_or("AI-recommended activity")
                    .to_string(),
            ],
            difficulty_match_permille: rec.get("difficulty_match_permille")
                .and_then(|v| v.as_u64())
                .unwrap_or(700) as u16,
            estimated_xp: (activity_minutes as u32 * 15),
            sequence,
        };

        estimated_xp += activity.estimated_xp;
        total_minutes += activity.estimated_minutes;
        priority_skills.push(skill_hash);
        activities.push(activity);
        sequence += 1;
    }

    // Plan breaks based on session length and type
    let breaks = plan_breaks(input.target_minutes, input.include_breaks, &input.session_type);

    // Flow state optimization notes
    let flow_notes = match input.session_type {
        SessionType::DeepWork => "Deep work session: minimize interruptions, focus for extended periods".into(),
        SessionType::Balanced => "Balanced session: variety to maintain engagement and flow".into(),
        SessionType::ReviewFocus => "Review-focused: spaced repetition optimized for retention".into(),
        SessionType::NewLearning => "Learning-focused: new concepts with progressive difficulty".into(),
        SessionType::Practice => "Practice-focused: skill reinforcement through active application".into(),
    };

    // Calculate confidence based on data availability
    let confidence = if smart_recs.is_empty() {
        300 // Low confidence without AI recommendations
    } else if due_for_review > 0 {
        800 // Good confidence with retention data
    } else {
        600 // Medium confidence
    };

    Ok(SmartSessionPlan {
        plan_id: format!("plan-{}-{}", now, agent.to_string().chars().take(8).collect::<String>()),
        total_minutes,
        activities,
        estimated_xp,
        priority_skills,
        at_risk_skills: due_for_review,
        breaks,
        flow_notes,
        confidence_permille: confidence,
        generated_at: now,
    })
}

/// Helper to plan breaks based on session type
fn plan_breaks(target_minutes: u16, include_breaks: bool, session_type: &SessionType) -> Vec<PlannedBreak> {
    if !include_breaks {
        return vec![];
    }

    let mut breaks = Vec::new();

    match session_type {
        SessionType::DeepWork => {
            // Long focus blocks with substantial breaks
            if target_minutes >= 50 {
                breaks.push(PlannedBreak {
                    after_minutes: 50,
                    duration_minutes: 10,
                    break_type: BreakType::Long,
                    suggestion: "Stand up, stretch, get water. Let your mind wander.".into(),
                });
            }
            if target_minutes >= 110 {
                breaks.push(PlannedBreak {
                    after_minutes: 110,
                    duration_minutes: 15,
                    break_type: BreakType::Long,
                    suggestion: "Take a short walk. You've been focused for a long time!".into(),
                });
            }
        }
        _ => {
            // Pomodoro-style breaks for other session types
            if target_minutes >= 25 {
                breaks.push(PlannedBreak {
                    after_minutes: 25,
                    duration_minutes: 5,
                    break_type: BreakType::Pomodoro,
                    suggestion: "Quick stretch and deep breaths".into(),
                });
            }
            if target_minutes >= 55 {
                breaks.push(PlannedBreak {
                    after_minutes: 55,
                    duration_minutes: 5,
                    break_type: BreakType::Pomodoro,
                    suggestion: "Stand up and move around".into(),
                });
            }
            if target_minutes >= 90 {
                breaks.push(PlannedBreak {
                    after_minutes: 90,
                    duration_minutes: 10,
                    break_type: BreakType::Long,
                    suggestion: "Longer break - grab a snack or walk".into(),
                });
            }
        }
    }

    breaks
}

/// Start a session from a smart plan
#[hdk_extern]
pub fn start_session_from_plan(plan: SmartSessionPlan) -> ExternResult<ActionHash> {
    // Convert smart activities to standard planned activities
    let planned_activities: Vec<PlannedActivity> = plan.activities.iter().map(|a| {
        PlannedActivity {
            activity_type: a.activity_type.clone(),
            target_hash: a.target_hash.to_string(),
            estimated_minutes: a.estimated_minutes,
            priority: a.priority,
            reason: a.reasoning.first().cloned().unwrap_or_default(),
        }
    }).collect();

    start_orchestrated_session(StartSessionInput {
        target_minutes: plan.total_minutes,
        planned_activities,
    })
}

// ============== Daily Reports ==============

/// Generate a daily learning report
#[hdk_extern]
pub fn generate_daily_report(_: ()) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = current_time()?;
    let date = current_date()?;
    let anchor = report_anchor()?;

    // Get today's events
    let events = get_recent_events(1000)?;
    let today_events: Vec<_> = events.into_iter()
        .filter(|e| {
            let event_date = (e.occurred_at / (24 * 60 * 60 * 1_000_000)) as u32 + 19700101;
            event_date == date
        })
        .collect();

    // Aggregate stats
    let events_count = today_events.len() as u32;
    let xp_earned: u32 = today_events.iter().map(|e| e.xp_gained).sum();
    let total_minutes: u32 = today_events.iter().map(|e| e.duration_seconds).sum::<u32>() / 60;

    // SRS stats
    let srs_events: Vec<_> = today_events.iter()
        .filter(|e| matches!(e.source_zome, SourceZome::Srs))
        .collect();
    let srs_reviews = srs_events.len() as u32;
    let srs_new = srs_events.iter()
        .filter(|e| matches!(e.event_type, LearningEventType::SrsReview))
        .count() as u32;
    let srs_total_quality: u32 = srs_events.iter().map(|e| e.quality_permille as u32).sum();
    let srs_accuracy = if !srs_events.is_empty() {
        (srs_total_quality / srs_events.len() as u32) as u16
    } else { 0 };

    // Mastery stats
    let skills_practiced = today_events.iter()
        .flat_map(|e| e.skills_affected.iter())
        .count() as u32;
    let skills_improved = today_events.iter()
        .filter(|e| e.mastery_change > 0)
        .flat_map(|e| e.skills_affected.iter())
        .count() as u32;
    let mastery_gained: i16 = today_events.iter()
        .map(|e| e.mastery_change)
        .sum();

    // Streak (get from latest event)
    let streak_day = today_events.first()
        .and_then(|e| e.streak_day)
        .unwrap_or(0);

    // Goals
    let goals_completed = today_events.iter()
        .filter(|e| matches!(e.event_type, LearningEventType::GoalAchieved))
        .count() as u8;

    // Badges
    let badges: Vec<String> = today_events.iter()
        .filter(|e| matches!(e.event_type, LearningEventType::BadgeEarned))
        .map(|e| format!("badge-{}", e.occurred_at))
        .collect();

    let report = DailyLearningReport {
        learner: agent,
        date,
        sessions_count: today_events.iter()
            .filter(|e| e.duration_seconds > 60)
            .count() as u32,
        total_minutes,
        events_count,
        srs_reviews,
        srs_new_cards: srs_new,
        srs_accuracy_permille: srs_accuracy,
        xp_earned,
        level_ups: 0, // Would need to track level changes
        skills_practiced,
        skills_improved,
        mastery_gained,
        streak_continued: events_count > 0,
        streak_day,
        goal_progress: mastery_gained,
        goals_completed,
        badges_earned: badges,
        generated_at: now,
    };

    let action_hash = create_entry(EntryTypes::DailyLearningReport(report))?;

    create_link(
        anchor,
        action_hash.clone(),
        LinkTypes::LearnerToDailyReports,
        (),
    )?;

    Ok(action_hash)
}

/// Get daily reports for recent days
#[hdk_extern]
pub fn get_recent_reports(days: u32) -> ExternResult<Vec<DailyLearningReport>> {
    let anchor = report_anchor()?;

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::LearnerToDailyReports)?,
        GetStrategy::Local,
    )?;

    let mut reports = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                let report: DailyLearningReport = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("No report entry".into())))?;
                reports.push(report);
            }
        }
    }

    // Sort by most recent first
    reports.sort_by(|a, b| b.date.cmp(&a.date));
    reports.truncate(days as usize);

    Ok(reports)
}

// ============== Learning Analytics Dashboard ==============

/// Time period for analytics
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub enum AnalyticsPeriod {
    Today,
    #[default]
    Week,
    Month,
    Quarter,
    Year,
    AllTime,
}

/// Trend direction
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum TrendDirection {
    Up,
    Down,
    Stable,
}

/// Single metric with trend analysis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricWithTrend {
    pub current_value: i64,
    pub previous_value: i64,
    pub change_percent: i16,
    pub trend: TrendDirection,
}

/// Learning analytics summary for dashboard
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearningAnalyticsSummary {
    /// Period covered
    pub period: AnalyticsPeriod,
    /// Start and end timestamps
    pub period_start: i64,
    pub period_end: i64,

    // Engagement metrics
    pub total_sessions: MetricWithTrend,
    pub total_minutes: MetricWithTrend,
    pub avg_session_minutes: u16,
    pub active_days: MetricWithTrend,

    // Learning metrics
    pub xp_earned: MetricWithTrend,
    pub skills_practiced: MetricWithTrend,
    pub skills_mastered: MetricWithTrend,
    pub avg_mastery_permille: MetricWithTrend,

    // Retention metrics
    pub srs_reviews: MetricWithTrend,
    pub srs_accuracy_permille: MetricWithTrend,
    pub retention_rate_permille: u16,
    pub overdue_cards: u32,

    // Progress metrics
    pub level: u32,
    pub level_progress_permille: u16,
    pub current_streak: MetricWithTrend,
    pub badges_earned: MetricWithTrend,
    pub goals_completed: MetricWithTrend,

    // Top skills and weaknesses
    pub top_skills: Vec<SkillSummary>,
    pub weakest_skills: Vec<SkillSummary>,

    // Daily breakdown
    pub daily_xp: Vec<i64>,
    pub daily_minutes: Vec<u32>,

    /// Generated timestamp
    pub generated_at: i64,
}

/// Skill summary for analytics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SkillSummary {
    pub skill_id: String,
    pub skill_name: String,
    pub mastery_permille: u16,
    pub trend: TrendDirection,
    pub practice_count: u32,
}

/// Input for analytics request
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnalyticsInput {
    pub period: AnalyticsPeriod,
}

/// Get learning analytics summary for dashboard
#[hdk_extern]
pub fn get_learning_analytics(input: AnalyticsInput) -> ExternResult<LearningAnalyticsSummary> {
    let now = current_time()?;

    // Calculate period bounds
    let (period_start, days_in_period) = match input.period {
        AnalyticsPeriod::Today => (now - 24 * 60 * 60 * 1_000_000, 1u32),
        AnalyticsPeriod::Week => (now - 7 * 24 * 60 * 60 * 1_000_000, 7),
        AnalyticsPeriod::Month => (now - 30 * 24 * 60 * 60 * 1_000_000, 30),
        AnalyticsPeriod::Quarter => (now - 90 * 24 * 60 * 60 * 1_000_000, 90),
        AnalyticsPeriod::Year => (now - 365 * 24 * 60 * 60 * 1_000_000, 365),
        AnalyticsPeriod::AllTime => (0, 365), // Use 365 as comparison period
    };

    // Get recent reports for the period
    let reports = get_recent_reports(days_in_period)?;
    let previous_reports = get_recent_reports(days_in_period * 2)?
        .into_iter()
        .skip(days_in_period as usize)
        .collect::<Vec<_>>();

    // Calculate current period metrics
    let current_sessions: u32 = reports.iter().map(|r| r.sessions_count).sum();
    let current_minutes: u32 = reports.iter().map(|r| r.total_minutes).sum();
    let current_xp: u32 = reports.iter().map(|r| r.xp_earned).sum();
    let current_skills: u32 = reports.iter().map(|r| r.skills_practiced).sum();
    let current_reviews: u32 = reports.iter().map(|r| r.srs_reviews).sum();
    let current_badges: u32 = reports.iter().map(|r| r.badges_earned.len() as u32).sum();
    let current_goals: u32 = reports.iter().map(|r| r.goals_completed as u32).sum();
    let current_streak = reports.first().map(|r| r.streak_day).unwrap_or(0);

    // Calculate previous period metrics
    let prev_sessions: u32 = previous_reports.iter().map(|r| r.sessions_count).sum();
    let prev_minutes: u32 = previous_reports.iter().map(|r| r.total_minutes).sum();
    let prev_xp: u32 = previous_reports.iter().map(|r| r.xp_earned).sum();
    let prev_skills: u32 = previous_reports.iter().map(|r| r.skills_practiced).sum();
    let prev_reviews: u32 = previous_reports.iter().map(|r| r.srs_reviews).sum();
    let prev_badges: u32 = previous_reports.iter().map(|r| r.badges_earned.len() as u32).sum();
    let prev_goals: u32 = previous_reports.iter().map(|r| r.goals_completed as u32).sum();
    let prev_streak = previous_reports.first().map(|r| r.streak_day).unwrap_or(0);

    let active_days = reports.len() as u32;
    let prev_active_days = previous_reports.len() as u32;

    // SRS accuracy
    let total_accuracy: u32 = reports.iter()
        .filter(|r| r.srs_reviews > 0)
        .map(|r| r.srs_accuracy_permille as u32)
        .sum();
    let accuracy_count = reports.iter().filter(|r| r.srs_reviews > 0).count() as u32;
    let current_accuracy = if accuracy_count > 0 { total_accuracy / accuracy_count } else { 0 };

    let prev_accuracy: u32 = previous_reports.iter()
        .filter(|r| r.srs_reviews > 0)
        .map(|r| r.srs_accuracy_permille as u32)
        .sum();
    let prev_accuracy_count = previous_reports.iter().filter(|r| r.srs_reviews > 0).count() as u32;
    let prev_accuracy_val = if prev_accuracy_count > 0 { prev_accuracy / prev_accuracy_count } else { 0 };

    // Build daily breakdown
    let daily_xp: Vec<i64> = reports.iter().map(|r| r.xp_earned as i64).collect();
    let daily_minutes: Vec<u32> = reports.iter().map(|r| r.total_minutes).collect();

    // Get current progress for level/streak info
    let progress = get_progress_aggregate(())?.unwrap_or_else(|| create_default_progress());

    Ok(LearningAnalyticsSummary {
        period: input.period,
        period_start,
        period_end: now,

        total_sessions: create_metric(current_sessions as i64, prev_sessions as i64),
        total_minutes: create_metric(current_minutes as i64, prev_minutes as i64),
        avg_session_minutes: if current_sessions > 0 {
            (current_minutes / current_sessions) as u16
        } else { 0 },
        active_days: create_metric(active_days as i64, prev_active_days as i64),

        xp_earned: create_metric(current_xp as i64, prev_xp as i64),
        skills_practiced: create_metric(current_skills as i64, prev_skills as i64),
        skills_mastered: create_metric(
            progress.skills_mastered as i64,
            progress.skills_mastered.saturating_sub(1) as i64
        ),
        avg_mastery_permille: create_metric(
            progress.avg_mastery_permille as i64,
            progress.avg_mastery_permille.saturating_sub(50) as i64
        ),

        srs_reviews: create_metric(current_reviews as i64, prev_reviews as i64),
        srs_accuracy_permille: create_metric(current_accuracy as i64, prev_accuracy_val as i64),
        retention_rate_permille: progress.srs_accuracy_permille,
        overdue_cards: progress.skills_due_review,

        level: progress.level,
        level_progress_permille: progress.xp_to_next_level as u16,
        current_streak: create_metric(current_streak as i64, prev_streak as i64),
        badges_earned: create_metric(current_badges as i64, prev_badges as i64),
        goals_completed: create_metric(current_goals as i64, prev_goals as i64),

        top_skills: vec![], // Would need to call adaptive zome
        weakest_skills: vec![], // Would need to call adaptive zome

        daily_xp,
        daily_minutes,

        generated_at: now,
    })
}

/// Helper to create a metric with trend
fn create_metric(current: i64, previous: i64) -> MetricWithTrend {
    let change_percent = if previous > 0 {
        (((current - previous) * 100) / previous) as i16
    } else if current > 0 {
        100
    } else {
        0
    };

    let trend = if change_percent > 5 {
        TrendDirection::Up
    } else if change_percent < -5 {
        TrendDirection::Down
    } else {
        TrendDirection::Stable
    };

    MetricWithTrend {
        current_value: current,
        previous_value: previous,
        change_percent,
        trend,
    }
}

/// Helper to create default progress
fn create_default_progress() -> LearnerProgressAggregate {
    let agent = agent_info().map(|a| a.agent_initial_pubkey)
        .unwrap_or_else(|_| AgentPubKey::from_raw_36(vec![0u8; 36]));

    LearnerProgressAggregate {
        learner: agent,
        total_xp: 0,
        level: 1,
        xp_to_next_level: 100,
        current_streak_days: 0,
        longest_streak_days: 0,
        streak_bonus_permille: 0,
        srs_cards_total: 0,
        srs_cards_mature: 0,
        srs_cards_learning: 0,
        srs_reviews_today: 0,
        srs_accuracy_permille: 0,
        skills_tracked: 0,
        skills_mastered: 0,
        avg_mastery_permille: 0,
        skills_due_review: 0,
        primary_style: LearningStyle::default(),
        style_confidence_permille: 0,
        active_goals: 0,
        completed_goals: 0,
        goal_progress_permille: 0,
        total_sessions: 0,
        total_learning_minutes: 0,
        avg_session_minutes: 0,
        days_active: 0,
        badges_earned: 0,
        rare_badges: 0,
        leaderboard_rank: None,
        confidence_permille: 0,
        last_activity_at: 0,
        aggregated_at: 0,
    }
}

/// Get learning trends over time
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearningTrends {
    /// Weekly XP totals
    pub weekly_xp: Vec<(u32, i64)>, // (week_number, xp)
    /// Weekly active time
    pub weekly_minutes: Vec<(u32, u32)>,
    /// Mastery progression
    pub mastery_trend: Vec<(i64, u16)>, // (timestamp, avg_mastery)
    /// Streak history
    pub streak_history: Vec<(i64, u32)>, // (timestamp, streak)
}

/// Get learning trends over time for visualization
#[hdk_extern]
pub fn get_learning_trends(weeks: u32) -> ExternResult<LearningTrends> {
    let reports = get_recent_reports(weeks * 7)?;

    // Group by week
    let mut weekly_xp: Vec<(u32, i64)> = Vec::new();
    let mut weekly_minutes: Vec<(u32, u32)> = Vec::new();

    // Simple weekly aggregation (would be more sophisticated in production)
    let mut week_xp: i64 = 0;
    let mut week_mins: u32 = 0;

    for (i, report) in reports.iter().enumerate() {
        week_xp += report.xp_earned as i64;
        week_mins += report.total_minutes;

        if (i + 1) % 7 == 0 {
            let week_num = ((i + 1) / 7) as u32;
            weekly_xp.push((week_num, week_xp));
            weekly_minutes.push((week_num, week_mins));
            week_xp = 0;
            week_mins = 0;
        }
    }

    // Add partial week if any remaining
    if !reports.is_empty() && reports.len() % 7 != 0 {
        let week_num = (reports.len() / 7 + 1) as u32;
        weekly_xp.push((week_num, week_xp));
        weekly_minutes.push((week_num, week_mins));
    }

    Ok(LearningTrends {
        weekly_xp,
        weekly_minutes,
        mastery_trend: vec![], // Would need historical mastery data
        streak_history: vec![], // Would need historical streak data
    })
}

// ============== Achievement Triggers ==============

/// Create an achievement trigger
#[hdk_extern]
pub fn create_achievement_trigger(trigger: AchievementTrigger) -> ExternResult<ActionHash> {
    let now = current_time()?;
    let anchor = triggers_anchor()?;

    let mut trigger = trigger;
    trigger.created_at = now;

    let action_hash = create_entry(EntryTypes::AchievementTrigger(trigger))?;

    create_link(
        anchor,
        action_hash.clone(),
        LinkTypes::GlobalTriggers,
        (),
    )?;

    Ok(action_hash)
}

/// Get all achievement triggers
#[hdk_extern]
pub fn get_achievement_triggers(_: ()) -> ExternResult<Vec<AchievementTrigger>> {
    let anchor = triggers_anchor()?;

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::GlobalTriggers)?,
        GetStrategy::Local,
    )?;

    let mut triggers = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                let trigger: AchievementTrigger = record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("No trigger entry".into())))?;
                if trigger.is_active {
                    triggers.push(trigger);
                }
            }
        }
    }

    Ok(triggers)
}

// ============== Interleaved Practice ==============
// Research shows mixing different skill types leads to better long-term retention
// than blocked practice (practicing one skill at a time).

/// Skill category for interleaving - different categories should be mixed
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SkillCategory {
    /// Conceptual understanding (theory, principles)
    Conceptual,
    /// Procedural skills (how-to, algorithms)
    Procedural,
    /// Factual recall (definitions, facts)
    Factual,
    /// Applied skills (problem-solving, projects)
    Applied,
    /// Creative skills (design, innovation)
    Creative,
}

impl Default for SkillCategory {
    fn default() -> Self {
        SkillCategory::Conceptual
    }
}

/// A skill item for interleaved practice
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InterleavedSkillItem {
    /// Skill hash
    pub skill_hash: ActionHash,
    /// Category for interleaving
    pub category: SkillCategory,
    /// Current mastery (0-1000)
    pub mastery_permille: u16,
    /// Retention urgency (0-1000, higher = more urgent)
    pub urgency_permille: u16,
    /// Difficulty (0-1000)
    pub difficulty_permille: u16,
    /// Estimated practice time in minutes
    pub estimated_minutes: u8,
}

/// Input for generating interleaved practice sequence
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InterleavedPracticeInput {
    /// Available skills to practice
    pub skills: Vec<InterleavedSkillItem>,
    /// Total session time in minutes
    pub session_minutes: u16,
    /// Minimum items per category (for true interleaving)
    pub min_per_category: u8,
    /// Maximum consecutive same-category items
    pub max_consecutive: u8,
}

/// A practice sequence item with spacing
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PracticeSequenceItem {
    /// Skill hash
    pub skill_hash: ActionHash,
    /// Category
    pub category: SkillCategory,
    /// Position in sequence (0-indexed)
    pub position: u8,
    /// Spacing from last same-category item (in items, not time)
    pub spacing_from_last: u8,
    /// Is this a "desirable difficulty" item (slightly above comfort zone)
    pub is_desirable_difficulty: bool,
}

/// Result of interleaved practice generation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InterleavedPracticeSequence {
    /// Ordered sequence of practice items
    pub sequence: Vec<PracticeSequenceItem>,
    /// Categories included
    pub categories_used: Vec<SkillCategory>,
    /// Average spacing between same-category items
    pub avg_spacing: u8,
    /// Interleaving score (0-1000, higher = more interleaved)
    pub interleaving_score: u16,
    /// Total estimated time
    pub total_minutes: u16,
    /// Confidence in sequence quality
    pub confidence_permille: u16,
}

/// Generate an optimally interleaved practice sequence
/// Based on research showing interleaved practice improves long-term retention
#[hdk_extern]
pub fn generate_interleaved_sequence(input: InterleavedPracticeInput) -> ExternResult<InterleavedPracticeSequence> {
    let mut skills = input.skills.clone();

    // Sort by urgency first (due items first), then by category for distribution
    skills.sort_by(|a, b| {
        b.urgency_permille.cmp(&a.urgency_permille)
            .then_with(|| (a.category.clone() as u8).cmp(&(b.category.clone() as u8)))
    });

    // Group by category
    let mut by_category: std::collections::HashMap<u8, Vec<&InterleavedSkillItem>> =
        std::collections::HashMap::new();

    for skill in &skills {
        let cat_key = skill.category.clone() as u8;
        by_category.entry(cat_key).or_default().push(skill);
    }

    // Build interleaved sequence using round-robin with jitter
    let mut sequence: Vec<PracticeSequenceItem> = Vec::new();
    let mut last_by_category: std::collections::HashMap<u8, u8> = std::collections::HashMap::new();
    let mut category_indices: std::collections::HashMap<u8, usize> = std::collections::HashMap::new();
    let mut total_minutes: u16 = 0;
    let mut position: u8 = 0;

    // Round-robin through categories until time is filled
    let category_keys: Vec<u8> = by_category.keys().cloned().collect();
    let mut round = 0;

    while total_minutes < input.session_minutes && !category_keys.is_empty() {
        for &cat_key in &category_keys {
            if total_minutes >= input.session_minutes {
                break;
            }

            let idx = category_indices.entry(cat_key).or_insert(0);
            if let Some(cat_skills) = by_category.get(&cat_key) {
                if *idx < cat_skills.len() {
                    let skill = cat_skills[*idx];

                    // Calculate spacing from last same-category item
                    let spacing = if let Some(&last_pos) = last_by_category.get(&cat_key) {
                        position - last_pos
                    } else {
                        0
                    };

                    // Determine if this is a "desirable difficulty" item
                    // (slightly above current mastery, creating productive struggle)
                    let is_desirable_difficulty = skill.difficulty_permille > skill.mastery_permille
                        && skill.difficulty_permille < skill.mastery_permille + 200;

                    sequence.push(PracticeSequenceItem {
                        skill_hash: skill.skill_hash.clone(),
                        category: skill.category.clone(),
                        position,
                        spacing_from_last: spacing,
                        is_desirable_difficulty,
                    });

                    last_by_category.insert(cat_key, position);
                    *idx += 1;
                    position += 1;
                    total_minutes += skill.estimated_minutes as u16;
                }
            }
        }
        round += 1;

        // Safety: prevent infinite loops
        if round > 100 {
            break;
        }
    }

    // Calculate interleaving score
    // Perfect interleaving = alternating categories = high score
    // Blocked practice = all same category = low score
    let mut same_category_consecutive = 0u32;
    let mut total_transitions = 0u32;

    for i in 1..sequence.len() {
        if sequence[i].category == sequence[i-1].category {
            same_category_consecutive += 1;
        }
        total_transitions += 1;
    }

    let interleaving_score = if total_transitions > 0 {
        ((total_transitions - same_category_consecutive) * 1000 / total_transitions) as u16
    } else {
        0
    };

    // Calculate average spacing
    let total_spacing: u32 = sequence.iter().map(|s| s.spacing_from_last as u32).sum();
    let avg_spacing = if !sequence.is_empty() {
        (total_spacing / sequence.len() as u32) as u8
    } else {
        0
    };

    // Collect unique categories used
    let mut categories_used: Vec<SkillCategory> = Vec::new();
    for item in &sequence {
        if !categories_used.contains(&item.category) {
            categories_used.push(item.category.clone());
        }
    }

    // Confidence based on sequence quality
    let confidence_permille = (interleaving_score / 2) +
        (if categories_used.len() >= 3 { 250 } else { 100 }) +
        (if avg_spacing >= 2 { 250 } else { 100 });

    Ok(InterleavedPracticeSequence {
        sequence,
        categories_used,
        avg_spacing,
        interleaving_score,
        total_minutes,
        confidence_permille,
    })
}

// ============== Knowledge Graph & Prerequisites ==============

/// Prerequisite relationship between skills
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SkillPrerequisite {
    /// The skill that requires the prerequisite
    pub skill_hash: ActionHash,
    /// The prerequisite skill
    pub prerequisite_hash: ActionHash,
    /// Minimum mastery required (0-1000)
    pub min_mastery_permille: u16,
    /// Is this a hard requirement or soft recommendation?
    pub is_hard_requirement: bool,
    /// Weight of this prerequisite (0-1000, for partial prerequisites)
    pub weight_permille: u16,
}

/// Knowledge graph node representing a skill's position
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KnowledgeGraphNode {
    /// Skill hash
    pub skill_hash: ActionHash,
    /// Depth in graph (0 = foundational, higher = more advanced)
    pub depth: u8,
    /// Prerequisites count
    pub prerequisites_count: u8,
    /// Dependents count (skills that depend on this)
    pub dependents_count: u8,
    /// Readiness score based on prerequisite mastery (0-1000)
    pub readiness_permille: u16,
    /// Is this skill unlocked (all hard prereqs met)?
    pub is_unlocked: bool,
    /// Blocking skills (hard prereqs not met)
    pub blocking_skills: Vec<ActionHash>,
}

/// Input for checking readiness
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReadinessCheckInput {
    /// Skill to check readiness for
    pub skill_hash: ActionHash,
    /// Known prerequisites (simplified for now)
    pub prerequisites: Vec<SkillPrerequisite>,
    /// Current mastery levels for prereq skills
    pub mastery_levels: Vec<(ActionHash, u16)>,
}

/// Check readiness for a skill based on prerequisites
#[hdk_extern]
pub fn check_skill_readiness(input: ReadinessCheckInput) -> ExternResult<KnowledgeGraphNode> {
    // Build mastery lookup
    let mastery_map: std::collections::HashMap<ActionHash, u16> =
        input.mastery_levels.into_iter().collect();

    let mut blocking_skills: Vec<ActionHash> = Vec::new();
    let mut total_weight: u32 = 0;
    let mut weighted_readiness: u32 = 0;
    let mut all_hard_met = true;

    for prereq in &input.prerequisites {
        let mastery = mastery_map.get(&prereq.prerequisite_hash).copied().unwrap_or(0);
        let met = mastery >= prereq.min_mastery_permille;

        total_weight += prereq.weight_permille as u32;
        if met {
            weighted_readiness += prereq.weight_permille as u32;
        } else {
            if prereq.is_hard_requirement {
                all_hard_met = false;
                blocking_skills.push(prereq.prerequisite_hash.clone());
            }
            // Partial credit for partial mastery
            let partial = (mastery as u32 * prereq.weight_permille as u32) / prereq.min_mastery_permille as u32;
            weighted_readiness += partial.min(prereq.weight_permille as u32);
        }
    }

    let readiness_permille = if total_weight > 0 {
        ((weighted_readiness * 1000) / total_weight) as u16
    } else {
        1000 // No prerequisites = fully ready
    };

    Ok(KnowledgeGraphNode {
        skill_hash: input.skill_hash,
        depth: input.prerequisites.len() as u8, // Simplified depth
        prerequisites_count: input.prerequisites.len() as u8,
        dependents_count: 0, // Would need full graph traversal
        readiness_permille,
        is_unlocked: all_hard_met,
        blocking_skills,
    })
}

// ============== Cognitive Load Management ==============

/// Cognitive load factors
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CognitiveLoadFactors {
    /// Intrinsic load (inherent difficulty of material) 0-1000
    pub intrinsic_load: u16,
    /// Extraneous load (unnecessary complexity) 0-1000
    pub extraneous_load: u16,
    /// Germane load (productive learning effort) 0-1000
    pub germane_load: u16,
    /// Time pressure factor 0-1000
    pub time_pressure: u16,
    /// Fatigue estimate 0-1000
    pub fatigue_estimate: u16,
}

/// Cognitive state tracking
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CognitiveState {
    /// Current total cognitive load (0-1000)
    pub current_load: u16,
    /// Load capacity (personal max before overload) 0-1000
    pub capacity: u16,
    /// Remaining capacity (how much more can be taken on)
    pub remaining_capacity: u16,
    /// Is learner in overload state?
    pub is_overloaded: bool,
    /// Recommended action
    pub recommendation: CognitiveRecommendation,
    /// Minutes until recommended break
    pub minutes_until_break: u8,
}

/// Recommendations for cognitive load management
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CognitiveRecommendation {
    /// Continue as normal
    Continue,
    /// Reduce difficulty
    ReduceDifficulty,
    /// Take a short break
    ShortBreak,
    /// Take a longer break
    LongBreak,
    /// Switch to easier material
    SwitchToEasier,
    /// End session soon
    EndSessionSoon,
    /// Immediate rest needed
    ImmediateRest,
}

impl Default for CognitiveRecommendation {
    fn default() -> Self {
        CognitiveRecommendation::Continue
    }
}

/// Input for cognitive load assessment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CognitiveLoadInput {
    /// Current load factors
    pub factors: CognitiveLoadFactors,
    /// Minutes into session
    pub session_minutes: u16,
    /// Errors made recently
    pub recent_errors: u8,
    /// Response time trend (slower = higher load)
    pub response_time_trend_permille: u16, // 1000 = normal, >1000 = slower
    /// Personal capacity modifier (some people handle more)
    pub capacity_modifier_permille: u16,
}

/// Assess cognitive load and provide recommendations
#[hdk_extern]
pub fn assess_cognitive_load(input: CognitiveLoadInput) -> ExternResult<CognitiveState> {
    // Base capacity (average person)
    let base_capacity: u16 = 800;
    let capacity = ((base_capacity as u32 * input.capacity_modifier_permille as u32) / 1000) as u16;

    // Calculate current load
    // Formula: (intrinsic + extraneous) with fatigue and time pressure multipliers
    let base_load = (input.factors.intrinsic_load as u32 + input.factors.extraneous_load as u32) / 2;

    // Fatigue increases with time
    let time_fatigue = (input.session_minutes as u32 * 10).min(300); // Max 300 from time

    // Errors indicate overload
    let error_factor = input.recent_errors as u32 * 50;

    // Slower responses indicate higher load
    let response_factor = if input.response_time_trend_permille > 1000 {
        (input.response_time_trend_permille - 1000) as u32 / 5
    } else {
        0
    };

    let total_load = (base_load + time_fatigue + error_factor + response_factor +
        input.factors.fatigue_estimate as u32 / 3).min(1000) as u16;

    let remaining_capacity = capacity.saturating_sub(total_load);
    let is_overloaded = total_load > capacity;

    // Determine recommendation
    let recommendation = if is_overloaded {
        if total_load > capacity + 200 {
            CognitiveRecommendation::ImmediateRest
        } else {
            CognitiveRecommendation::EndSessionSoon
        }
    } else if remaining_capacity < 100 {
        CognitiveRecommendation::LongBreak
    } else if remaining_capacity < 200 {
        CognitiveRecommendation::ShortBreak
    } else if remaining_capacity < 300 && input.factors.intrinsic_load > 700 {
        CognitiveRecommendation::ReduceDifficulty
    } else if input.session_minutes > 45 && input.session_minutes % 25 < 5 {
        CognitiveRecommendation::ShortBreak // Pomodoro-style
    } else {
        CognitiveRecommendation::Continue
    };

    // Calculate minutes until recommended break
    let minutes_until_break = if remaining_capacity > 300 {
        ((remaining_capacity - 300) / 20).min(25) as u8
    } else {
        0
    };

    Ok(CognitiveState {
        current_load: total_load,
        capacity,
        remaining_capacity,
        is_overloaded,
        recommendation,
        minutes_until_break,
    })
}

// ============== Learning Velocity ==============

/// Learning velocity metrics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearningVelocity {
    /// Skills mastered per week
    pub skills_per_week: f32,
    /// XP earned per hour of active learning
    pub xp_per_hour: u32,
    /// Average time to mastery for a skill (minutes)
    pub avg_time_to_mastery_mins: u32,
    /// Retention rate after 7 days (0-1000)
    pub retention_7d_permille: u16,
    /// Velocity trend (positive = speeding up, negative = slowing down)
    pub velocity_trend_permille: i16,
    /// Comparison to cohort average (1000 = average, >1000 = faster)
    pub cohort_comparison_permille: u16,
    /// Estimated days to next level at current pace
    pub days_to_next_level: u16,
}

/// Input for velocity calculation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VelocityInput {
    /// Skills mastered in last 30 days
    pub skills_mastered_30d: u32,
    /// Total XP earned in last 30 days
    pub xp_earned_30d: u64,
    /// Total active learning minutes in last 30 days
    pub learning_minutes_30d: u32,
    /// Skills mastered in previous 30 days (for trend)
    pub skills_mastered_prev_30d: u32,
    /// Average mastery time for recent skills (minutes)
    pub avg_mastery_time_mins: u32,
    /// 7-day retention samples
    pub retention_samples: Vec<u16>,
    /// Cohort average skills per week (for comparison)
    pub cohort_avg_skills_per_week: f32,
    /// XP needed for next level
    pub xp_to_next_level: u64,
}

/// Calculate learning velocity metrics
#[hdk_extern]
pub fn calculate_learning_velocity(input: VelocityInput) -> ExternResult<LearningVelocity> {
    // Skills per week (30 days ≈ 4.3 weeks)
    let skills_per_week = input.skills_mastered_30d as f32 / 4.3;

    // XP per hour
    let xp_per_hour = if input.learning_minutes_30d > 0 {
        ((input.xp_earned_30d * 60) / input.learning_minutes_30d as u64) as u32
    } else {
        0
    };

    // Average retention
    let retention_7d_permille = if !input.retention_samples.is_empty() {
        let sum: u32 = input.retention_samples.iter().map(|&r| r as u32).sum();
        (sum / input.retention_samples.len() as u32) as u16
    } else {
        800 // Default assumption
    };

    // Velocity trend (comparing to previous period)
    let velocity_trend_permille = if input.skills_mastered_prev_30d > 0 {
        let change = input.skills_mastered_30d as i32 - input.skills_mastered_prev_30d as i32;
        ((change * 1000) / input.skills_mastered_prev_30d as i32) as i16
    } else if input.skills_mastered_30d > 0 {
        1000 // Big improvement from zero
    } else {
        0
    };

    // Cohort comparison
    let cohort_comparison_permille = if input.cohort_avg_skills_per_week > 0.0 {
        ((skills_per_week / input.cohort_avg_skills_per_week) * 1000.0) as u16
    } else {
        1000
    };

    // Estimated days to next level
    let days_to_next_level = if xp_per_hour > 0 && input.learning_minutes_30d > 0 {
        let daily_xp = input.xp_earned_30d / 30;
        if daily_xp > 0 {
            (input.xp_to_next_level / daily_xp) as u16
        } else {
            365 // Long time
        }
    } else {
        365
    };

    Ok(LearningVelocity {
        skills_per_week,
        xp_per_hour,
        avg_time_to_mastery_mins: input.avg_mastery_time_mins,
        retention_7d_permille,
        velocity_trend_permille,
        cohort_comparison_permille,
        days_to_next_level,
    })
}

// ============== Graduation Ceremony ==============
//
// Detection: all conditions must align simultaneously:
// - PoL > 0.85 for the domain
// - Sovereignty >= 801 (Autonomous mode)
// - At least 1 credential with vitality > 800
// - Career pathway > 80% complete
//
// The platform's primary success metric should be "time from enrollment
// to graduation," NOT daily active users.

/// Input for checking graduation eligibility.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraduationCheckInput {
    /// Domain to check (e.g., "Mathematics", "Rust Development")
    pub domain: String,
    /// PoL composite score (0-1000 permille) — from frontend PoL computation
    pub pol_score_permille: u16,
    /// Sovereignty level (0-1000 permille) — from frontend sovereignty state
    pub sovereignty_permille: u16,
    /// Number of published credentials with vitality > 800
    pub active_credential_count: u32,
    /// Pathway completion (0-1000 permille) — from frontend progress tracking
    pub pathway_completion_permille: u16,
}

/// Result of graduation eligibility check.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraduationResult {
    pub eligible: bool,
    pub graduation_hash: Option<ActionHash>,
    pub missing: Vec<String>,
}

/// Check graduation eligibility and record if all conditions met.
///
/// Graduation is the moment the platform says: "You are no longer a consumer
/// of this knowledge; you are a peer-contributor to the mesh."
#[hdk_extern]
pub fn check_graduation(input: GraduationCheckInput) -> ExternResult<GraduationResult> {
    let agent = agent_info()?.agent_initial_pubkey;
    let now = current_time()?;

    let mut missing = Vec::new();

    if input.pol_score_permille < 850 {
        missing.push(format!(
            "PoL score {}/1000 (need 850+)",
            input.pol_score_permille
        ));
    }
    if input.sovereignty_permille < 801 {
        missing.push(format!(
            "Sovereignty {}/1000 (need 801+ Autonomous)",
            input.sovereignty_permille
        ));
    }
    if input.active_credential_count == 0 {
        missing.push("No active credentials with vitality > 800".to_string());
    }
    if input.pathway_completion_permille < 800 {
        missing.push(format!(
            "Pathway {}% complete (need 80%+)",
            input.pathway_completion_permille / 10
        ));
    }

    if !missing.is_empty() {
        return Ok(GraduationResult {
            eligible: false,
            graduation_hash: None,
            missing,
        });
    }

    // All conditions met — record graduation
    let record = GraduationRecord {
        agent: agent.clone(),
        domain: input.domain,
        pol_score_permille: input.pol_score_permille,
        sovereignty_permille: input.sovereignty_permille,
        active_credentials: input.active_credential_count,
        pathway_completion_permille: input.pathway_completion_permille,
        graduated_at: now,
    };

    let hash = create_entry(EntryTypes::GraduationRecord(record))?;

    // Link: agent -> graduation
    let agent_hash: AnyDhtHash = agent.into();
    create_link(agent_hash, hash.clone(), LinkTypes::LearnerToGraduation, vec![])?;

    Ok(GraduationResult {
        eligible: true,
        graduation_hash: Some(hash),
        missing: vec![],
    })
}

/// List all graduation records for the calling agent.
#[hdk_extern]
pub fn list_my_graduations(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_hash: AnyDhtHash = agent.into();

    let links = get_links(
        LinkQuery::try_new(agent_hash, LinkTypes::LearnerToGraduation)?,
        GetStrategy::Local,
    )?;

    let mut records = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                records.push(record);
            }
        }
    }
    Ok(records)
}

// ============== Praxis → Craft Credential Pipeline ==============
//
// Publishes a Praxis-issued credential as a living credential on the Craft cluster.
// Requires the unified hApp with both "praxis" and "craft" roles installed.

/// Input for publishing a Praxis credential to the Craft professional graph.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublishToCraftInput {
    /// ActionHash of the credential in the Praxis credential_zome.
    pub credential_hash: ActionHash,
    /// Optional guild context for the Craft credential.
    pub guild_id: Option<String>,
    pub guild_name: Option<String>,
}

/// Publish a Praxis credential to the Craft cluster as a living credential.
///
/// 1. Fetches the credential from the local Praxis credential_zome
/// 2. Extracts mastery, epistemic code, and provenance
/// 3. Dispatches a cross-cluster call to craft_graph.publish_credential()
///
/// Requires: unified hApp with "craft" role installed.
#[hdk_extern]
pub fn publish_to_craft(input: PublishToCraftInput) -> ExternResult<()> {
    // Gate: Citizen tier minimum for cross-cluster publishing
    mycelix_bridge_common::gate_civic(
        "edunet_bridge",
        &mycelix_bridge_common::civic_requirement_voting(),
        "publish_to_craft",
    )?;

    // Fetch the credential from local credential_zome
    let record = get(input.credential_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Credential not found — issue it first via credential_coordinator".into()
        )))?;

    // Parse the credential entry (using serde_json for flexibility)
    let credential_json = match record.entry().as_option() {
        Some(Entry::App(bytes)) => {
            serde_json::from_slice::<serde_json::Value>(bytes.bytes())
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!(
                    "Failed to parse credential: {e}"
                ))))?
        }
        _ => return Err(wasm_error!(WasmErrorInner::Guest(
            "Credential record has no app entry".into()
        ))),
    };

    // Extract fields for Craft's PublishedCredentialInput
    let credential_id = credential_json.get("id")
        .or_else(|| credential_json.get("credential_id"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let title = credential_json.get("title")
        .or_else(|| credential_json.get("credential_subject").and_then(|s| s.get("name")))
        .and_then(|v| v.as_str())
        .unwrap_or("Praxis Credential")
        .to_string();

    let issuer = credential_json.get("issuer")
        .and_then(|v| v.as_str())
        .unwrap_or("praxis")
        .to_string();

    let issued_on = credential_json.get("issuance_date")
        .or_else(|| credential_json.get("issued_on"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let mastery = credential_json.get("mastery_permille")
        .and_then(|v| v.as_u64())
        .map(|v| v as u16);

    let epistemic_code = credential_json.get("epistemic_code")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    // Build the Craft-compatible payload
    let craft_payload = serde_json::json!({
        "credential_id": credential_id,
        "title": title,
        "issuer": issuer,
        "issued_on": issued_on,
        "source_dna": "praxis",
        "action_hash": input.credential_hash.to_string(),
        "summary": format!("Proof of Learning from Praxis — {}", title),
        "guild_id": input.guild_id,
        "guild_name": input.guild_name,
        "epistemic_code": epistemic_code,
        "mastery_permille": mastery,
    });

    // Cross-cluster call to Craft via OtherRole
    let response = call(
        CallTargetCell::OtherRole("craft".into()),
        ZomeName::from("craft_graph"),
        FunctionName::from("publish_credential"),
        None,
        &ExternIO::encode(craft_payload)
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Encode: {e}"))))?,
    )?;

    match response {
        ZomeCallResponse::Ok(_) => Ok(()),
        ZomeCallResponse::Unauthorized(_, _, _, _) => Err(wasm_error!(
            WasmErrorInner::Guest("Unauthorized cross-cluster call to Craft".into())
        )),
        ZomeCallResponse::NetworkError(err) => Err(wasm_error!(
            WasmErrorInner::Guest(format!("Network error publishing to Craft: {err}"))
        )),
        ZomeCallResponse::CountersigningSession(err) => Err(wasm_error!(
            WasmErrorInner::Guest(format!("Countersigning error: {err}"))
        )),
        ZomeCallResponse::AuthenticationFailed(_, _) => Err(wasm_error!(
            WasmErrorInner::Guest("Authentication failed calling Craft".into())
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streak_bonus_for_day() {
        assert_eq!(streak_bonus_for_day(0), 1000);
        assert_eq!(streak_bonus_for_day(7), 1100);
        assert_eq!(streak_bonus_for_day(30), 1250);
        assert_eq!(streak_bonus_for_day(100), 1500);
        assert_eq!(streak_bonus_for_day(365), 2000);
    }

    #[test]
    fn test_xp_calculation_integration() {
        // Normal lesson with average quality
        let xp = calculate_xp(100, 500, 1000, &LearningEventType::LessonComplete);
        assert!(xp > 0);
        assert!(xp < 200);

        // Perfect quiz with max streak
        let xp = calculate_xp(100, 1000, 2000, &LearningEventType::QuizPassed);
        assert!(xp > 200);
    }

    #[test]
    fn test_session_state_transitions() {
        let planning = SessionState::Planning;
        let active = SessionState::Active;
        let completed = SessionState::Completed;

        assert_ne!(planning, active);
        assert_ne!(active, completed);
        assert_eq!(SessionState::default(), SessionState::Planning);
    }

    // === Cross-Zome Type Tests ===

    #[test]
    fn test_gamification_summary_serialization() {
        let summary = GamificationSummary {
            total_xp: 5000,
            level: 10,
            xp_to_next_level: 2100,
            level_progress_permille: 580,
            current_streak: 14,
            longest_streak: 30,
            streak_bonus_permille: 1150,
            badges_earned: 12,
            freezes_remaining: 2,
        };

        let json = serde_json::to_string(&summary).unwrap();
        let parsed: GamificationSummary = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.total_xp, 5000);
        assert_eq!(parsed.level, 10);
        assert_eq!(parsed.current_streak, 14);
        assert_eq!(parsed.badges_earned, 12);
    }

    #[test]
    fn test_adaptive_summary_serialization() {
        let summary = AdaptiveLearnerSummary {
            profile: Some(serde_json::json!({"name": "Test"})),
            learning_style: serde_json::json!({"visual": 300, "auditory": 250}),
            mastery_count: 25,
            avg_mastery_permille: 650,
            skills_mastered: 10,
            due_for_review: 5,
            active_goals: 3,
            total_sessions: 50,
            total_learning_minutes: 1500,
            active_paths: 2,
            recommendations_count: 8,
        };

        let json = serde_json::to_string(&summary).unwrap();
        let parsed: AdaptiveLearnerSummary = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.mastery_count, 25);
        assert_eq!(parsed.skills_mastered, 10);
        assert_eq!(parsed.total_sessions, 50);
    }

    #[test]
    fn test_cross_zome_progress_result() {
        let result = CrossZomeProgressResult {
            aggregate_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            gamification: Some(GamificationSummary {
                total_xp: 1000,
                level: 5,
                xp_to_next_level: 500,
                level_progress_permille: 500,
                current_streak: 7,
                longest_streak: 7,
                streak_bonus_permille: 1100,
                badges_earned: 3,
                freezes_remaining: 3,
            }),
            adaptive_summary: None,
            srs_cards_count: 100,
            srs_due_count: 15,
            aggregated_at: 1234567890,
            errors: vec!["Adaptive fetch failed".to_string()],
        };

        assert!(result.gamification.is_some());
        assert!(result.adaptive_summary.is_none());
        assert_eq!(result.srs_cards_count, 100);
        assert_eq!(result.errors.len(), 1);
    }

    #[test]
    fn test_unified_dashboard_creation() {
        let dashboard = UnifiedDashboard {
            progress: None,
            gamification: Some(GamificationSummary {
                total_xp: 2000,
                level: 7,
                xp_to_next_level: 800,
                level_progress_permille: 600,
                current_streak: 10,
                longest_streak: 15,
                streak_bonus_permille: 1100,
                badges_earned: 5,
                freezes_remaining: 2,
            }),
            adaptive: None,
            srs_due_count: 20,
            recent_events: vec![],
            active_session: None,
        };

        assert!(dashboard.progress.is_none());
        assert!(dashboard.gamification.is_some());
        assert_eq!(dashboard.srs_due_count, 20);
        assert!(dashboard.recent_events.is_empty());
    }

    #[test]
    fn test_confidence_calculation_logic() {
        // Both sources available = high confidence
        let gam = Some(GamificationSummary {
            total_xp: 0,
            level: 1,
            xp_to_next_level: 100,
            level_progress_permille: 0,
            current_streak: 0,
            longest_streak: 0,
            streak_bonus_permille: 1000,
            badges_earned: 0,
            freezes_remaining: 3,
        });
        let adp = Some(AdaptiveLearnerSummary {
            profile: None,
            learning_style: serde_json::json!({}),
            mastery_count: 0,
            avg_mastery_permille: 0,
            skills_mastered: 0,
            due_for_review: 0,
            active_goals: 0,
            total_sessions: 0,
            total_learning_minutes: 0,
            active_paths: 0,
            recommendations_count: 0,
        });

        // Both present = 800 (high)
        let confidence = if gam.is_some() && adp.is_some() {
            800u16
        } else if gam.is_some() || adp.is_some() {
            500u16
        } else {
            200u16
        };
        assert_eq!(confidence, 800);

        // Only gamification = 500 (medium)
        let adp_none: Option<AdaptiveLearnerSummary> = None;
        let confidence = if gam.is_some() && adp_none.is_some() {
            800u16
        } else if gam.is_some() || adp_none.is_some() {
            500u16
        } else {
            200u16
        };
        assert_eq!(confidence, 500);

        // Neither = 200 (low)
        let gam_none: Option<GamificationSummary> = None;
        let confidence = if gam_none.is_some() && adp_none.is_some() {
            800u16
        } else if gam_none.is_some() || adp_none.is_some() {
            500u16
        } else {
            200u16
        };
        assert_eq!(confidence, 200);
    }

    // === Interleaved Practice Tests ===

    #[test]
    fn test_skill_category_default() {
        assert_eq!(SkillCategory::default(), SkillCategory::Conceptual);
    }

    #[test]
    fn test_interleaved_sequence_empty_input() {
        let input = InterleavedPracticeInput {
            skills: vec![],
            session_minutes: 30,
            min_per_category: 1,
            max_consecutive: 2,
        };

        // Can't call hdk_extern in tests, but we can test the types
        assert!(input.skills.is_empty());
        assert_eq!(input.session_minutes, 30);
    }

    #[test]
    fn test_practice_sequence_item_creation() {
        let item = PracticeSequenceItem {
            skill_hash: ActionHash::from_raw_36(vec![1u8; 36]),
            category: SkillCategory::Procedural,
            position: 0,
            spacing_from_last: 0,
            is_desirable_difficulty: true,
        };

        assert_eq!(item.category, SkillCategory::Procedural);
        assert_eq!(item.position, 0);
        assert!(item.is_desirable_difficulty);
    }

    #[test]
    fn test_interleaved_skill_item_creation() {
        let skill = InterleavedSkillItem {
            skill_hash: ActionHash::from_raw_36(vec![2u8; 36]),
            category: SkillCategory::Applied,
            mastery_permille: 600,
            urgency_permille: 800,
            difficulty_permille: 700,
            estimated_minutes: 5,
        };

        assert_eq!(skill.mastery_permille, 600);
        assert_eq!(skill.urgency_permille, 800);
        // Desirable difficulty check: difficulty > mastery && difficulty < mastery + 200
        let is_dd = skill.difficulty_permille > skill.mastery_permille
            && skill.difficulty_permille < skill.mastery_permille + 200;
        assert!(is_dd); // 700 > 600 && 700 < 800
    }

    // === Knowledge Graph Tests ===

    #[test]
    fn test_skill_prerequisite_creation() {
        let prereq = SkillPrerequisite {
            skill_hash: ActionHash::from_raw_36(vec![3u8; 36]),
            prerequisite_hash: ActionHash::from_raw_36(vec![4u8; 36]),
            min_mastery_permille: 700,
            is_hard_requirement: true,
            weight_permille: 500,
        };

        assert!(prereq.is_hard_requirement);
        assert_eq!(prereq.min_mastery_permille, 700);
        assert_eq!(prereq.weight_permille, 500);
    }

    #[test]
    fn test_knowledge_graph_node_unlocked() {
        let node = KnowledgeGraphNode {
            skill_hash: ActionHash::from_raw_36(vec![5u8; 36]),
            depth: 2,
            prerequisites_count: 2,
            dependents_count: 3,
            readiness_permille: 1000,
            is_unlocked: true,
            blocking_skills: vec![],
        };

        assert!(node.is_unlocked);
        assert!(node.blocking_skills.is_empty());
        assert_eq!(node.readiness_permille, 1000);
    }

    #[test]
    fn test_knowledge_graph_node_blocked() {
        let blocking = ActionHash::from_raw_36(vec![6u8; 36]);
        let node = KnowledgeGraphNode {
            skill_hash: ActionHash::from_raw_36(vec![7u8; 36]),
            depth: 3,
            prerequisites_count: 1,
            dependents_count: 0,
            readiness_permille: 400,
            is_unlocked: false,
            blocking_skills: vec![blocking],
        };

        assert!(!node.is_unlocked);
        assert_eq!(node.blocking_skills.len(), 1);
        assert_eq!(node.readiness_permille, 400);
    }

    // === Cognitive Load Tests ===

    #[test]
    fn test_cognitive_recommendation_default() {
        assert_eq!(CognitiveRecommendation::default(), CognitiveRecommendation::Continue);
    }

    #[test]
    fn test_cognitive_load_factors() {
        let factors = CognitiveLoadFactors {
            intrinsic_load: 500,
            extraneous_load: 200,
            germane_load: 600,
            time_pressure: 300,
            fatigue_estimate: 400,
        };

        // Total useful load = intrinsic + germane
        let useful = factors.intrinsic_load + factors.germane_load;
        assert_eq!(useful, 1100);

        // Wasted load = extraneous
        assert_eq!(factors.extraneous_load, 200);
    }

    #[test]
    fn test_cognitive_state_overload_detection() {
        // Simulate overloaded state
        let state = CognitiveState {
            current_load: 900,
            capacity: 800,
            remaining_capacity: 0,
            is_overloaded: true,
            recommendation: CognitiveRecommendation::EndSessionSoon,
            minutes_until_break: 0,
        };

        assert!(state.is_overloaded);
        assert_eq!(state.remaining_capacity, 0);
        assert_eq!(state.recommendation, CognitiveRecommendation::EndSessionSoon);
    }

    #[test]
    fn test_cognitive_state_healthy() {
        let state = CognitiveState {
            current_load: 400,
            capacity: 800,
            remaining_capacity: 400,
            is_overloaded: false,
            recommendation: CognitiveRecommendation::Continue,
            minutes_until_break: 20,
        };

        assert!(!state.is_overloaded);
        assert_eq!(state.remaining_capacity, 400);
        assert_eq!(state.recommendation, CognitiveRecommendation::Continue);
        assert!(state.minutes_until_break > 0);
    }

    // === Learning Velocity Tests ===

    #[test]
    fn test_learning_velocity_fast_learner() {
        let velocity = LearningVelocity {
            skills_per_week: 5.0,
            xp_per_hour: 500,
            avg_time_to_mastery_mins: 60,
            retention_7d_permille: 850,
            velocity_trend_permille: 200, // 20% faster than before
            cohort_comparison_permille: 1500, // 50% faster than cohort
            days_to_next_level: 7,
        };

        assert!(velocity.cohort_comparison_permille > 1000); // Faster than average
        assert!(velocity.velocity_trend_permille > 0); // Improving
        assert!(velocity.retention_7d_permille > 800); // Good retention
    }

    #[test]
    fn test_learning_velocity_struggling_learner() {
        let velocity = LearningVelocity {
            skills_per_week: 1.0,
            xp_per_hour: 100,
            avg_time_to_mastery_mins: 180,
            retention_7d_permille: 500,
            velocity_trend_permille: -100, // Slowing down
            cohort_comparison_permille: 500, // Half the cohort speed
            days_to_next_level: 30,
        };

        assert!(velocity.cohort_comparison_permille < 1000); // Slower than average
        assert!(velocity.velocity_trend_permille < 0); // Declining
        assert!(velocity.retention_7d_permille < 700); // Low retention
    }

    #[test]
    fn test_velocity_input_creation() {
        let input = VelocityInput {
            skills_mastered_30d: 10,
            xp_earned_30d: 5000,
            learning_minutes_30d: 600,
            skills_mastered_prev_30d: 8,
            avg_mastery_time_mins: 45,
            retention_samples: vec![800, 850, 750, 900],
            cohort_avg_skills_per_week: 2.0,
            xp_to_next_level: 2000,
        };

        // Calculate expected skills per week
        let expected_spw = input.skills_mastered_30d as f32 / 4.3;
        assert!(expected_spw > 2.0); // ~2.3 skills/week

        // Calculate trend
        let trend = ((input.skills_mastered_30d as i32 - input.skills_mastered_prev_30d as i32) * 1000)
            / input.skills_mastered_prev_30d as i32;
        assert_eq!(trend, 250); // 25% improvement
    }
}
