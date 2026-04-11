// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hearth Gratitude Coordinator Zome
//! Business logic for expressing gratitude, managing appreciation circles,
//! and tracking gratitude statistics.

use hdk::prelude::*;
use hearth_coordinator_common::{get_latest_record, records_from_links, require_membership};
use hearth_gratitude_integrity::*;
use hearth_types::*;
use mycelix_bridge_common::{
    civic_requirement_basic, civic_requirement_proposal, GovernanceEligibility,
};

// ============================================================================
// Consciousness Gating
// ============================================================================


// ============================================================================
// Input Types
// ============================================================================

/// Input for expressing gratitude to another hearth member.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ExpressGratitudeInput {
    pub hearth_hash: ActionHash,
    pub to_agent: AgentPubKey,
    pub message: String,
    pub gratitude_type: GratitudeType,
    pub visibility: HearthVisibility,
}

/// Input for starting an appreciation circle.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StartCircleInput {
    pub hearth_hash: ActionHash,
    pub theme: String,
    pub participants: Vec<AgentPubKey>,
}

// ============================================================================
// Extern Functions
// ============================================================================

/// Express gratitude from the calling agent to another hearth member.
/// Creates a GratitudeExpression entry, links it to the hearth and both agents,
/// and emits a HearthSignal::GratitudeExpressed signal.
#[hdk_extern]
pub fn express_gratitude(input: ExpressGratitudeInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("hearth_bridge", &civic_requirement_basic(), "express_gratitude")?;
    require_membership(&input.hearth_hash)?;
    let caller = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let expression = GratitudeExpression {
        hearth_hash: input.hearth_hash.clone(),
        from_agent: caller.clone(),
        to_agent: input.to_agent.clone(),
        message: input.message.clone(),
        gratitude_type: input.gratitude_type.clone(),
        visibility: input.visibility,
        created_at: now,
    };

    let action_hash = create_entry(&EntryTypes::GratitudeExpression(expression))?;

    // Link hearth -> gratitude
    create_link(
        input.hearth_hash.clone(),
        action_hash.clone(),
        LinkTypes::HearthToGratitude,
        (),
    )?;

    // Link from_agent -> gratitude (given)
    create_link(
        caller.clone(),
        action_hash.clone(),
        LinkTypes::AgentToGratitudeGiven,
        (),
    )?;

    // Link to_agent -> gratitude (received)
    create_link(
        input.to_agent.clone(),
        action_hash.clone(),
        LinkTypes::AgentToGratitudeReceived,
        (),
    )?;

    // Emit real-time signal
    let signal = HearthSignal::GratitudeExpressed {
        from_agent: caller,
        to_agent: input.to_agent,
        message: input.message,
        gratitude_type: input.gratitude_type,
    };
    emit_signal(&signal)?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created gratitude expression".into()
    )))
}

/// Start a new appreciation circle with a theme and initial participants.
#[hdk_extern]
pub fn start_appreciation_circle(input: StartCircleInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("hearth_bridge", &civic_requirement_basic(), "start_appreciation_circle")?;
    require_membership(&input.hearth_hash)?;
    let now = sys_time()?;

    let circle = AppreciationCircle {
        hearth_hash: input.hearth_hash.clone(),
        theme: input.theme,
        participants: input.participants.clone(),
        started_at: now,
        completed_at: None,
        status: CircleStatus::Open,
    };

    let action_hash = create_entry(&EntryTypes::AppreciationCircle(circle))?;

    // Link hearth -> circle
    create_link(
        input.hearth_hash,
        action_hash.clone(),
        LinkTypes::HearthToCircles,
        (),
    )?;

    // Link each participant -> circle
    for participant in &input.participants {
        create_link(
            participant.clone(),
            action_hash.clone(),
            LinkTypes::AgentToCircles,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created appreciation circle".into()
    )))
}

/// Join an existing appreciation circle by adding the calling agent to participants.
#[hdk_extern]
pub fn join_circle(circle_hash: ActionHash) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("hearth_bridge", &civic_requirement_basic(), "join_circle")?;
    let caller = agent_info()?.agent_initial_pubkey;

    let record = get(circle_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Circle not found".into())
    ))?;

    let mut circle: AppreciationCircle = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid circle entry".into()
        )))?;

    require_membership(&circle.hearth_hash)?;

    if circle.status == CircleStatus::Completed {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot join a completed circle".into()
        )));
    }

    // Check if already a participant
    if circle.participants.contains(&caller) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Already a participant in this circle".into()
        )));
    }

    circle.participants.push(caller.clone());

    let updated_hash = update_entry(circle_hash, &EntryTypes::AppreciationCircle(circle))?;

    // Link caller -> circle
    create_link(caller, updated_hash.clone(), LinkTypes::AgentToCircles, ())?;

    get(updated_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated circle".into()
    )))
}

/// Complete an appreciation circle by setting its status to Completed
/// and recording the completion timestamp.
/// Only the circle creator (action author) can complete it.
#[hdk_extern]
pub fn complete_circle(circle_hash: ActionHash) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("hearth_bridge", &civic_requirement_proposal(), "complete_circle")?;
    let caller = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let record = get(circle_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Circle not found".into())
    ))?;

    // Auth: only the original creator (action author) can complete the circle
    let creator = record.action().author().clone();
    if !can_complete_circle(&caller, &creator) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the circle creator can complete it".into()
        )));
    }

    let mut circle: AppreciationCircle = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid circle entry".into()
        )))?;

    if !is_circle_open(&circle.status) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Circle is not open and cannot be completed".into()
        )));
    }

    circle.status = CircleStatus::Completed;
    circle.completed_at = Some(now);

    let updated_hash = update_entry(circle_hash, &EntryTypes::AppreciationCircle(circle))?;

    get(updated_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated circle".into()
    )))
}

/// Get all gratitude expressions for a given hearth.
#[hdk_extern]
pub fn get_gratitude_stream(hearth_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(hearth_hash, LinkTypes::HearthToGratitude)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get the gratitude balance (anchor) for a specific agent within a hearth.
#[hdk_extern]
pub fn get_gratitude_balance(agent: AgentPubKey) -> ExternResult<Option<Record>> {
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::AgentToGratitudeGiven)?,
        GetStrategy::default(),
    )?;
    // Return the first linked anchor record if it exists
    if let Some(link) = links.first() {
        let action_hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        return get_latest_record(action_hash);
    }
    Ok(None)
}

/// Get all appreciation circles for a given hearth.
#[hdk_extern]
pub fn get_hearth_circles(hearth_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(hearth_hash, LinkTypes::HearthToCircles)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Create a gratitude digest: query HearthToGratitude links, filter by
/// created_at within the epoch window, aggregate per (from, to) pair.
/// Returns Vec<GratitudeSummary> for inclusion in the WeeklyDigest.
#[hdk_extern]
pub fn create_gratitude_digest(input: DigestEpochInput) -> ExternResult<Vec<GratitudeSummary>> {
    if input.epoch_start >= input.epoch_end {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "epoch_start must be before epoch_end".into()
        )));
    }

    let links = get_links(
        LinkQuery::try_new(input.hearth_hash, LinkTypes::HearthToGratitude)?,
        GetStrategy::default(),
    )?;

    // Aggregate gratitude expressions per (from_agent, to_agent) pair
    let mut pair_counts: std::collections::HashMap<(AgentPubKey, AgentPubKey), u32> =
        std::collections::HashMap::new();

    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(action_hash)? {
            let expr: GratitudeExpression = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Invalid gratitude entry".into()
                )))?;

            // Filter by half-open epoch window [start, end)
            if expr.created_at >= input.epoch_start && expr.created_at < input.epoch_end {
                let key = (expr.from_agent, expr.to_agent);
                *pair_counts.entry(key).or_insert(0) += 1;
            }
        }
    }

    let summaries: Vec<GratitudeSummary> = pair_counts
        .into_iter()
        .map(|((from_agent, to_agent), count)| GratitudeSummary {
            from_agent,
            to_agent,
            count,
        })
        .collect();

    Ok(summaries)
}

// ============================================================================
// Helpers
// ============================================================================

/// Check whether a circle is in Open status (can still accept changes).
fn is_circle_open(status: &CircleStatus) -> bool {
    *status == CircleStatus::Open
}

/// Check whether the caller is authorized to complete the circle.
/// Only the original creator (action author) can complete it.
fn can_complete_circle(caller: &AgentPubKey, creator: &AgentPubKey) -> bool {
    caller == creator
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    fn fake_agent_b() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![2u8; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    // ── ExpressGratitudeInput serde roundtrip ─────────────────────────

    #[test]
    fn express_gratitude_input_serde_roundtrip() {
        let input = ExpressGratitudeInput {
            hearth_hash: fake_action_hash(),
            to_agent: fake_agent_b(),
            message: "Thank you!".to_string(),
            gratitude_type: GratitudeType::Appreciation,
            visibility: HearthVisibility::AllMembers,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: ExpressGratitudeInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.hearth_hash, input.hearth_hash);
        assert_eq!(decoded.to_agent, input.to_agent);
        assert_eq!(decoded.message, "Thank you!");
    }

    #[test]
    fn express_gratitude_input_all_gratitude_types() {
        let types = vec![
            GratitudeType::Appreciation,
            GratitudeType::Acknowledgment,
            GratitudeType::Celebration,
            GratitudeType::Blessing,
            GratitudeType::Custom("Warmth".to_string()),
        ];
        for gt in types {
            let input = ExpressGratitudeInput {
                hearth_hash: fake_action_hash(),
                to_agent: fake_agent_b(),
                message: "Thanks".to_string(),
                gratitude_type: gt.clone(),
                visibility: HearthVisibility::AllMembers,
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: ExpressGratitudeInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.gratitude_type, gt);
        }
    }

    #[test]
    fn express_gratitude_input_all_visibility_options() {
        let visibilities = vec![
            HearthVisibility::AllMembers,
            HearthVisibility::AdultsOnly,
            HearthVisibility::GuardiansOnly,
            HearthVisibility::Specified(vec![fake_agent()]),
        ];
        for vis in visibilities {
            let input = ExpressGratitudeInput {
                hearth_hash: fake_action_hash(),
                to_agent: fake_agent_b(),
                message: "Thanks".to_string(),
                gratitude_type: GratitudeType::Appreciation,
                visibility: vis.clone(),
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: ExpressGratitudeInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.visibility, vis);
        }
    }

    // ── StartCircleInput serde roundtrip ──────────────────────────────

    #[test]
    fn start_circle_input_serde_roundtrip() {
        let input = StartCircleInput {
            hearth_hash: fake_action_hash(),
            theme: "Evening gratitude".to_string(),
            participants: vec![fake_agent(), fake_agent_b()],
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: StartCircleInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.hearth_hash, input.hearth_hash);
        assert_eq!(decoded.theme, "Evening gratitude");
        assert_eq!(decoded.participants.len(), 2);
    }

    #[test]
    fn start_circle_input_many_participants() {
        let participants: Vec<AgentPubKey> = (0..50)
            .map(|i| AgentPubKey::from_raw_36(vec![i as u8; 36]))
            .collect();
        let input = StartCircleInput {
            hearth_hash: fake_action_hash(),
            theme: "Large circle".to_string(),
            participants: participants.clone(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: StartCircleInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.participants.len(), 50);
    }

    // ── GratitudeExpression serde roundtrip ───────────────────────────

    #[test]
    fn gratitude_expression_serde_roundtrip() {
        let expr = GratitudeExpression {
            hearth_hash: fake_action_hash(),
            from_agent: fake_agent(),
            to_agent: fake_agent_b(),
            message: "You are wonderful".to_string(),
            gratitude_type: GratitudeType::Celebration,
            visibility: HearthVisibility::AllMembers,
            created_at: Timestamp::from_micros(1000),
        };
        let json = serde_json::to_string(&expr).unwrap();
        let decoded: GratitudeExpression = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, expr);
    }

    // ── AppreciationCircle serde roundtrip ────────────────────────────

    #[test]
    fn appreciation_circle_serde_roundtrip() {
        let circle = AppreciationCircle {
            hearth_hash: fake_action_hash(),
            theme: "Weekly family gratitude".to_string(),
            participants: vec![fake_agent(), fake_agent_b()],
            started_at: Timestamp::from_micros(5000),
            completed_at: None,
            status: CircleStatus::Open,
        };
        let json = serde_json::to_string(&circle).unwrap();
        let decoded: AppreciationCircle = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, circle);
    }

    #[test]
    fn appreciation_circle_completed_serde_roundtrip() {
        let circle = AppreciationCircle {
            hearth_hash: fake_action_hash(),
            theme: "Done circle".to_string(),
            participants: vec![fake_agent(), fake_agent_b()],
            started_at: Timestamp::from_micros(1000),
            completed_at: Some(Timestamp::from_micros(2000)),
            status: CircleStatus::Completed,
        };
        let json = serde_json::to_string(&circle).unwrap();
        let decoded: AppreciationCircle = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.status, CircleStatus::Completed);
        assert!(decoded.completed_at.is_some());
    }

    // ── GratitudeAnchor serde roundtrip ──────────────────────────────

    #[test]
    fn gratitude_anchor_serde_roundtrip() {
        let anchor = GratitudeAnchor {
            agent: fake_agent(),
            hearth_hash: fake_action_hash(),
            total_given: 42,
            total_received: 17,
            current_streak_days: 7,
        };
        let json = serde_json::to_string(&anchor).unwrap();
        let decoded: GratitudeAnchor = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, anchor);
    }

    #[test]
    fn gratitude_anchor_zero_values() {
        let anchor = GratitudeAnchor {
            agent: fake_agent(),
            hearth_hash: fake_action_hash(),
            total_given: 0,
            total_received: 0,
            current_streak_days: 0,
        };
        let json = serde_json::to_string(&anchor).unwrap();
        let decoded: GratitudeAnchor = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.total_given, 0);
        assert_eq!(decoded.total_received, 0);
        assert_eq!(decoded.current_streak_days, 0);
    }

    // ---- is_circle_open helper ----

    #[test]
    fn circle_open_status_is_open() {
        assert!(is_circle_open(&CircleStatus::Open));
    }

    #[test]
    fn circle_in_progress_not_open() {
        assert!(!is_circle_open(&CircleStatus::InProgress));
    }

    #[test]
    fn circle_completed_not_open() {
        assert!(!is_circle_open(&CircleStatus::Completed));
    }

    // ---- can_complete_circle helper ----

    #[test]
    fn can_complete_circle_creator_matches() {
        let creator = fake_agent();
        assert!(can_complete_circle(&creator, &creator));
    }

    #[test]
    fn can_complete_circle_different_agent_rejected() {
        let caller = fake_agent();
        let creator = fake_agent_b();
        assert!(!can_complete_circle(&caller, &creator));
    }

    // ---- Signal serde roundtrips ----

    #[test]
    fn signal_gratitude_expressed_serde_roundtrip() {
        let sig = HearthSignal::GratitudeExpressed {
            from_agent: fake_agent(),
            to_agent: fake_agent_b(),
            message: "Thanks for dinner".to_string(),
            gratitude_type: GratitudeType::Appreciation,
        };
        let json = serde_json::to_string(&sig).unwrap();
        let back: HearthSignal = serde_json::from_str(&json).unwrap();
        match back {
            HearthSignal::GratitudeExpressed {
                message,
                gratitude_type,
                ..
            } => {
                assert_eq!(message, "Thanks for dinner");
                assert_eq!(gratitude_type, GratitudeType::Appreciation);
            }
            _ => panic!("Expected GratitudeExpressed signal"),
        }
    }

    #[test]
    fn signal_gratitude_expressed_all_types_serde() {
        let types = vec![
            GratitudeType::Appreciation,
            GratitudeType::Acknowledgment,
            GratitudeType::Celebration,
            GratitudeType::Blessing,
            GratitudeType::Custom("Heartfelt".to_string()),
        ];
        for gt in types {
            let sig = HearthSignal::GratitudeExpressed {
                from_agent: fake_agent(),
                to_agent: fake_agent_b(),
                message: "Test".to_string(),
                gratitude_type: gt.clone(),
            };
            let json = serde_json::to_string(&sig).unwrap();
            let back: HearthSignal = serde_json::from_str(&json).unwrap();
            match back {
                HearthSignal::GratitudeExpressed { gratitude_type, .. } => {
                    assert_eq!(gratitude_type, gt);
                }
                _ => panic!("Expected GratitudeExpressed signal"),
            }
        }
    }
}
