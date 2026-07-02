// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bioregional Coordination Integrity Zome
//!
//! Entry types for multi-robot, cross-platform mission orchestration.
//! A BioregionalMission is a sequence of phases that coordinate
//! heterogeneous robotic platforms (AUV → helicopter → vehicle fleet)
//! within a geographic region.
//!
//! Science: Ostrom (1990) governing the commons, bioregional governance.

use hdi::prelude::*;

// ── Entry Types ────────────────────────────────────────────────────────

/// A multi-phase bioregional mission coordinating multiple robot types.
///
/// Example: AUV detects water contamination → helicopter aerial survey →
/// vehicle fleet rerouted away from contaminated zone.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BioregionalMission {
    /// Unique mission identifier.
    pub mission_id: String,
    /// What triggered this mission.
    pub trigger_event: TriggerEvent,
    /// Ordered phases of the mission.
    pub phases: Vec<MissionPhase>,
    /// Geographic region (center lat/lon + radius km).
    pub region_center_lat: f64,
    pub region_center_lon: f64,
    pub region_radius_km: f64,
    /// Current status.
    pub status: BioregionalMissionStatus,
    /// When the mission was created.
    pub created_at: Timestamp,
    /// Agent who initiated the mission.
    pub initiated_by: AgentPubKey,
}

/// What triggered the bioregional mission.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TriggerEvent {
    /// AUV detected contamination above WHO threshold.
    ContaminationDetected { contaminant: String, level: f64 },
    /// Helicopter detected fire or hazard.
    HazardDetected { hazard_type: String },
    /// Swarm consensus on emerging threat.
    SwarmConsensus { coalition_size: usize, phi_swarm: f64 },
    /// Manual dispatch by governance agent.
    ManualDispatch { description: String },
    /// Sensor threshold exceeded.
    SensorThreshold { sensor: String, value: f64, threshold: f64 },
}

/// A single phase of a bioregional mission.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct MissionPhase {
    /// Phase identifier (unique within mission).
    pub phase_id: String,
    /// Which platform type executes this phase.
    pub platform_type: String,
    /// Required credential type (must have sufficient vitality).
    pub required_credential: Option<String>,
    /// Phases that must complete before this one starts.
    pub depends_on: Vec<String>,
    /// What triggers this phase to begin.
    pub trigger_condition: PhaseCondition,
    /// Current status of this phase.
    pub status: PhaseStatus,
    /// Target coordinates for this phase.
    pub target_lat: f64,
    pub target_lon: f64,
    /// Description of what this phase does.
    pub description: String,
}

/// Condition for a phase to begin.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PhaseCondition {
    /// Manual start by dispatcher.
    ManualStart,
    /// Previous phase completed.
    PreviousPhaseComplete { phase_id: String },
    /// Sensor reading exceeds threshold.
    SensorThreshold { sensor: String, threshold: f64 },
    /// Swarm consensus reached (N members agree).
    SwarmConsensus { min_coalition_size: usize },
    /// Timer elapsed since mission start.
    TimerElapsed { seconds: u64 },
}

/// Phase execution status.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PhaseStatus {
    /// Waiting for trigger condition.
    Pending,
    /// Dispatched to a robot, awaiting completion.
    Active,
    /// Successfully completed.
    Completed,
    /// Failed or timed out.
    Failed,
    /// Skipped (condition not met or superseded).
    Skipped,
}

/// Overall mission status.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum BioregionalMissionStatus {
    /// Mission created, phases pending.
    Planning,
    /// At least one phase is active.
    Active,
    /// All phases completed.
    Completed,
    /// Mission failed (critical phase failed).
    Failed,
    /// Mission cancelled by governance.
    Cancelled,
}

// ── Handoff Record ─────────────────────────────────────────────────────

/// Cross-platform handoff: data passed from one phase to the next.
///
/// When an AUV completes contamination detection, it produces a handoff
/// record with coordinates and severity data for the helicopter phase.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PhaseHandoff {
    /// Mission this handoff belongs to.
    pub mission_id: String,
    /// Phase that produced this handoff.
    pub from_phase: String,
    /// Phase that should consume this handoff.
    pub to_phase: String,
    /// Handoff data (JSON-serialized platform-specific data).
    pub data: Vec<u8>,
    /// Coordinates of interest.
    pub lat: f64,
    pub lon: f64,
    /// Severity/priority (0.0–1.0).
    pub severity: f64,
    /// When the handoff was created.
    pub created_at: Timestamp,
}

// ── Validation ─────────────────────────────────────────────────────────

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    BioregionalMission(BioregionalMission),
    PhaseHandoff(PhaseHandoff),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Anchor → BioregionalMission
    AnchorToMission,
    /// BioregionalMission → PhaseHandoff
    MissionToHandoff,
}

#[hdk_extern]
pub fn validate(_op: Op) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}
