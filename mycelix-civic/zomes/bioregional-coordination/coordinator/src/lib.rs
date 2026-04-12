// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bioregional Coordination Coordinator Zome
//!
//! Orchestrates multi-robot, cross-platform missions within a bioregion.
//! Consciousness-gated at PROPOSAL tier (Φ ≥ 0.4) for mission creation.
//!
//! ## Example Flow
//!
//! 1. AUV detects arsenic contamination → triggers mission
//! 2. Phase 1 (AUV): Contamination tracking → produces handoff with coordinates
//! 3. Phase 2 (Helicopter): Aerial survey of contamination extent
//! 4. Phase 3 (Vehicle): Fleet rerouted away from contaminated zone
//!
//! Each phase transition is recorded on Holochain DHT with credential
//! verification (living credentials from robotics-dispatch).

use hdk::prelude::*;
use bioregional_coordination_integrity::*;

/// Anchor for all missions.
const MISSIONS_ANCHOR: &str = "bioregional_missions";

// ── Mission Creation ──────────────────────────────────────────────────

/// Input for creating a bioregional mission.
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateMissionInput {
    pub mission_id: String,
    pub trigger_event: TriggerEvent,
    pub phases: Vec<MissionPhase>,
    pub region_center_lat: f64,
    pub region_center_lon: f64,
    pub region_radius_km: f64,
}

/// Create a new bioregional mission.
///
/// Requires PROPOSAL tier (consciousness score ≥ 0.4).
#[hdk_extern]
pub fn create_mission(input: CreateMissionInput) -> ExternResult<Record> {
    let author = agent_info()?.agent_latest_pubkey;
    let now = sys_time()?;

    let mission = BioregionalMission {
        mission_id: input.mission_id,
        trigger_event: input.trigger_event,
        phases: input.phases,
        region_center_lat: input.region_center_lat,
        region_center_lon: input.region_center_lon,
        region_radius_km: input.region_radius_km,
        status: BioregionalMissionStatus::Planning,
        created_at: now,
        initiated_by: author,
    };

    let mission_hash = create_entry(&EntryTypes::BioregionalMission(mission))?;

    // Link from anchor
    let anchor_hash = hash_entry(&bioregional_coordination_integrity::EntryTypes::BioregionalMission(
        BioregionalMission {
            mission_id: MISSIONS_ANCHOR.to_string(),
            trigger_event: TriggerEvent::ManualDispatch {
                description: "anchor".to_string(),
            },
            phases: Vec::new(),
            region_center_lat: 0.0,
            region_center_lon: 0.0,
            region_radius_km: 0.0,
            status: BioregionalMissionStatus::Planning,
            created_at: now,
            initiated_by: agent_info()?.agent_latest_pubkey,
        },
    ))?;

    create_link(
        anchor_hash,
        mission_hash.clone(),
        LinkTypes::AnchorToMission,
        (),
    )?;

    let record = get(mission_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get mission".into())))?;
    Ok(record)
}

// ── Phase Transitions ──────────────────────────────────────────────────

/// Input for recording a phase handoff.
#[derive(Serialize, Deserialize, Debug)]
pub struct RecordHandoffInput {
    pub mission_hash: ActionHash,
    pub from_phase: String,
    pub to_phase: String,
    pub data: Vec<u8>,
    pub lat: f64,
    pub lon: f64,
    pub severity: f64,
}

/// Record a handoff between mission phases.
///
/// Called when one platform completes its phase and passes data
/// to the next platform in the sequence.
#[hdk_extern]
pub fn record_handoff(input: RecordHandoffInput) -> ExternResult<Record> {
    let now = sys_time()?;

    // Get mission to verify it exists and extract mission_id
    let mission_record = get(input.mission_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Mission not found".into())))?;

    let mission: BioregionalMission = mission_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize failed: {e}"))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not a BioregionalMission".into())))?;

    let handoff = PhaseHandoff {
        mission_id: mission.mission_id,
        from_phase: input.from_phase,
        to_phase: input.to_phase,
        data: input.data,
        lat: input.lat,
        lon: input.lon,
        severity: input.severity.clamp(0.0, 1.0),
        created_at: now,
    };

    let handoff_hash = create_entry(&EntryTypes::PhaseHandoff(handoff))?;

    // Link mission → handoff
    create_link(
        input.mission_hash,
        handoff_hash.clone(),
        LinkTypes::MissionToHandoff,
        (),
    )?;

    let record = get(handoff_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get handoff".into())))?;
    Ok(record)
}

// ── Queries ────────────────────────────────────────────────────────────

/// Get all handoffs for a mission (ordered by creation).
#[hdk_extern]
pub fn get_mission_handoffs(mission_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        GetLinksInputBuilder::try_new(mission_hash, LinkTypes::MissionToHandoff)?.build(),
    )?;

    let mut handoffs = Vec::new();
    for link in links {
        if let Some(target) = link.target.into_action_hash() {
            if let Some(record) = get(target, GetOptions::default())? {
                handoffs.push(record);
            }
        }
    }
    Ok(handoffs)
}

/// Get a mission by its action hash.
#[hdk_extern]
pub fn get_mission(mission_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(mission_hash, GetOptions::default())
}
