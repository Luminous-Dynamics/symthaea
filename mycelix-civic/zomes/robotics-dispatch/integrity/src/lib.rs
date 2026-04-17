// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Robotics Dispatch Integrity Zome
//! Entry types and validation for consciousness-coupled robotic asset management.
//!
//! Defines three core entry types:
//! - `RoboticAsset`: registered robot with capabilities and consciousness level
//! - `DispatchOrder`: time-limited mission assignment (24-hour expiry per Essay 17)
//! - `TelemetryReport`: periodic status updates from deployed assets

use hdi::prelude::*;
use mycelix_bridge_entry_types::check_author_match;

/// Anchor entry for deterministic link bases.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// ── Entry Types ──────────────────────────────────────────────────────────────

/// A registered robotic asset in the Mycelix network.
///
/// Each robot is sponsored by a community member (owner) and registered
/// as an Instrumental Actor — no governance participation, no vote.
/// Consciousness level determines operational authority.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RoboticAsset {
    /// Unique asset identifier.
    pub asset_id: String,
    /// Platform type (Helicopter, AUV, Manipulator, CareBot, etc.).
    pub platform_type: PlatformType,
    /// Sponsoring agent (community member responsible for this robot).
    pub owner: AgentPubKey,
    /// Current measured consciousness level (Phi, 0.0–1.0).
    pub consciousness_level: f64,
    /// Maximum governance tier (capped below Participant for robots).
    pub max_tier: TierCap,
    /// Location (latitude, longitude).
    pub location_lat: f64,
    pub location_lon: f64,
    /// Operational status.
    pub status: AssetStatus,
    /// Platform-specific capabilities.
    pub capabilities: Vec<String>,
    /// Registration timestamp.
    pub registered_at: Timestamp,
}

/// A mission dispatch order for a robotic asset.
///
/// Authority automatically expires after 24 hours (Essay 17 constraint).
/// Emergency dispatches can use BASIC consciousness level but still expire.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DispatchOrder {
    /// Reference to the robotic asset being dispatched.
    pub asset_hash: ActionHash,
    /// Mission type identifier.
    pub mission_type: MissionType,
    /// Agent who authorized this dispatch.
    pub dispatched_by: AgentPubKey,
    /// Authority expiry timestamp (max 24 hours from creation).
    pub authority_expires: Timestamp,
    /// Target location.
    pub target_lat: f64,
    pub target_lon: f64,
    /// Dispatch priority.
    pub priority: DispatchPriority,
    /// Free-text mission description.
    pub description: String,
    /// Current order status.
    pub status: OrderStatus,
    /// Creation timestamp.
    pub created_at: Timestamp,
}

/// Periodic telemetry report from a deployed robotic asset.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TelemetryReport {
    /// Reference to the robotic asset.
    pub asset_hash: ActionHash,
    /// Reference to the active dispatch order.
    pub order_hash: ActionHash,
    /// Report timestamp.
    pub timestamp: Timestamp,
    /// Current position.
    pub lat: f64,
    pub lon: f64,
    pub alt: f64,
    /// Current consciousness level (Phi).
    pub consciousness_level: f64,
    /// Safety level (Green/Yellow/Orange/Red).
    pub safety_level: String,
    /// Mission progress (0.0–1.0).
    pub mission_progress: f64,
    /// Fuel/battery level (0.0–1.0).
    pub fuel_level: f64,
    /// Platform-specific telemetry (serialized bytes).
    pub platform_specific: Vec<u8>,
}

// ── Enums ────────────────────────────────────────────────────────────────────

/// Robotic platform type.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PlatformType {
    /// SAR helicopter.
    Helicopter,
    /// Autonomous underwater vehicle.
    Auv,
    /// Industrial manipulator arm.
    Manipulator,
    /// Care provider robot.
    CareBot,
    /// Quadrotor drone.
    Quadrotor,
    /// Autonomous vehicle.
    Vehicle,
    /// Custom/other platform.
    Custom(String),
}

/// Maximum governance tier cap for robotic assets.
///
/// Per the Sovereignty Papers, robots are Instrumental Actors
/// and cannot participate in governance above Observer tier.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TierCap {
    /// Read-only access, telemetry submission only.
    Observer,
    /// Can submit sensor readings and basic reports.
    /// Maximum for most robotic assets.
    Instrumental,
}

/// Operational status of a robotic asset.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AssetStatus {
    /// Available for dispatch.
    Available,
    /// Currently on a mission.
    Deployed,
    /// Under maintenance or repair.
    Maintenance,
    /// Emergency state (consciousness degraded, returning to base).
    Emergency,
    /// Decommissioned.
    Decommissioned,
}

/// Dispatch priority levels.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, PartialOrd)]
pub enum DispatchPriority {
    /// Routine operations (scheduled patrols, maintenance).
    Routine,
    /// Elevated priority (emerging situation).
    Priority,
    /// Urgent (active emergency, lives at risk).
    Urgent,
    /// Critical (mass casualty, environmental catastrophe).
    Critical,
}

/// Mission type identifier.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MissionType {
    /// Search and rescue grid search.
    SarGridSearch,
    /// SAR hover over target (winch deployment).
    SarHoverTarget,
    /// Water quality monitoring patrol.
    WaterQualityPatrol,
    /// Contamination source tracking.
    ContaminationTracking,
    /// Manufacturing task execution.
    ManufacturingTask,
    /// Care service delivery.
    CareService,
    /// Asset relocation.
    Relocation,
    /// Custom mission type.
    Custom(String),
}

/// Dispatch order status.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum OrderStatus {
    /// Order created, awaiting asset acknowledgement.
    Pending,
    /// Asset has acknowledged and is en route.
    Acknowledged,
    /// Mission in progress.
    InProgress,
    /// Mission completed successfully.
    Completed,
    /// Mission failed or aborted.
    Failed,
    /// Authority expired (24-hour limit reached).
    Expired,
    /// Order recalled by dispatcher.
    Recalled,
}

// ── Living Credentials ──────────────────────────────────────────────────────

/// A living credential for a robotic asset.
///
/// Follows the Ebbinghaus forgetting curve: vitality decays as R(t) = e^(-t/S)
/// where S = base_stability × mastery_factor × review_multiplier.
/// Credentials must be refreshed via periodic perturbation tests.
///
/// Science: Ebbinghaus (1885) forgetting curve, adapted from mycelix-craft.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RoboticCredential {
    /// The robotic asset this credential belongs to.
    pub asset_hash: ActionHash,
    /// Type of certification.
    pub credential_type: RoboticCredentialType,
    /// Current vitality (0–1000 permille). Decays over time.
    pub vitality_permille: u16,
    /// Mastery level at issuance (0–1000 permille). Drives stability.
    pub mastery_at_issue: u16,
    /// Timestamp of last perturbation test (ISO 8601).
    pub last_perturbation_test: Timestamp,
    /// Number of successful perturbation tests (drives review multiplier).
    pub successful_tests: u16,
    /// Issuing guild or authority (optional).
    pub issuing_authority: Option<String>,
    /// When this credential was first issued.
    pub issued_at: Timestamp,
}

/// Types of robotic certification.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RoboticCredentialType {
    /// SAR-certified helicopter (hover stability, wind robustness).
    SarCertified,
    /// Water quality sensor calibration (chemical accuracy).
    WaterQualityCalibrated,
    /// Manipulator precision (force control, trajectory tracking).
    ManipulatorPrecision,
    /// Navigation certified (GPS-denied nav, obstacle avoidance).
    NavigationCertified,
    /// Surgical precision (sub-mm accuracy, tremor rejection).
    SurgicalPrecision,
    /// Formation flight certified (swarm coordination).
    FormationFlightCertified,
    /// Custom certification type.
    Custom(String),
}

impl RoboticCredential {
    /// Base stability in minutes (drives decay rate).
    const BASE_STABILITY_MINUTES: f64 = 1440.0; // 1 day

    /// Compute current vitality using Ebbinghaus forgetting curve.
    ///
    /// R(t) = e^(-t/S) where:
    /// - t = minutes since last perturbation test
    /// - S = base × mastery_factor × review_multiplier
    pub fn compute_vitality(&self, now: Timestamp) -> u16 {
        let elapsed_us = now.as_micros().saturating_sub(self.last_perturbation_test.as_micros());
        let elapsed_minutes = elapsed_us as f64 / 60_000_000.0;

        let stability = self.stability_minutes();
        if stability <= 0.0 {
            return 0;
        }

        let retention = (-elapsed_minutes / stability).exp();
        (retention * 1000.0).clamp(0.0, 1000.0) as u16
    }

    /// Stability S in minutes (higher = slower decay).
    pub fn stability_minutes(&self) -> f64 {
        let mastery_normalized = self.mastery_at_issue as f64 / 1000.0;
        let mastery_factor = 0.5 + mastery_normalized * 2.0;
        let review_multiplier = 1.5_f64.powi(self.successful_tests.min(10) as i32);
        Self::BASE_STABILITY_MINUTES * mastery_factor * review_multiplier
    }

    /// Minutes until vitality drops to 80% (recommended review time).
    pub fn next_review_minutes(&self) -> f64 {
        let s = self.stability_minutes();
        -s * 0.8_f64.ln() // -S × ln(0.8) ≈ S × 0.223
    }
}

// ── Validation ───────────────────────────────────────────────────────────────

/// Entry types for this zome.
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    RoboticAsset(RoboticAsset),
    DispatchOrder(DispatchOrder),
    TelemetryReport(TelemetryReport),
    RoboticCredential(RoboticCredential),
}

/// Link types for this zome.
#[hdk_link_types]
pub enum LinkTypes {
    /// Anchor → RoboticAsset
    AnchorToAsset,
    /// RoboticAsset → DispatchOrder
    AssetToOrder,
    /// DispatchOrder → TelemetryReport
    OrderToTelemetry,
    /// Owner agent → RoboticAsset
    OwnerToAsset,
    /// Platform type anchor → RoboticAsset
    PlatformToAsset,
    /// RoboticAsset → RoboticCredential
    AssetToCredential,
}

/// Validate entries on creation.
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::RoboticAsset(asset) => validate_robotic_asset(&asset, &EntryCreationAction::Create(action)),
                EntryTypes::DispatchOrder(order) => validate_dispatch_order(&order, &EntryCreationAction::Create(action)),
                EntryTypes::TelemetryReport(report) => validate_telemetry_report(&report),
                EntryTypes::RoboticCredential(credential) => validate_robotic_credential(&credential),
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::RoboticAsset(asset) => {
                    let eca = EntryCreationAction::Update(action.clone());
                    if *eca.author() != action.author {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Only the original author can update this entry".into(),
                        ));
                    }
                    validate_robotic_asset(&asset, &eca)
                }
                EntryTypes::DispatchOrder(order) => validate_dispatch_order(&order, &EntryCreationAction::Update(action)),
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterCreateLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_robotic_asset(
    asset: &RoboticAsset,
    _action: &EntryCreationAction,
) -> ExternResult<ValidateCallbackResult> {
    // Consciousness level must be in [0, 1]
    if !asset.consciousness_level.is_finite()
        || asset.consciousness_level < 0.0
        || asset.consciousness_level > 1.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Consciousness level must be in [0.0, 1.0]".to_string(),
        ));
    }

    // Latitude must be in [-90, 90]
    if !asset.location_lat.is_finite()
        || asset.location_lat < -90.0
        || asset.location_lat > 90.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be in [-90, 90]".to_string(),
        ));
    }

    // Longitude must be in [-180, 180]
    if !asset.location_lon.is_finite()
        || asset.location_lon < -180.0
        || asset.location_lon > 180.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Longitude must be in [-180, 180]".to_string(),
        ));
    }

    // Asset ID must not be empty
    if asset.asset_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Asset ID must not be empty".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_dispatch_order(
    order: &DispatchOrder,
    _action: &EntryCreationAction,
) -> ExternResult<ValidateCallbackResult> {
    // Target coordinates must be valid
    if !order.target_lat.is_finite()
        || order.target_lat < -90.0
        || order.target_lat > 90.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Target latitude must be in [-90, 90]".to_string(),
        ));
    }

    if !order.target_lon.is_finite()
        || order.target_lon < -180.0
        || order.target_lon > 180.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Target longitude must be in [-180, 180]".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_telemetry_report(report: &TelemetryReport) -> ExternResult<ValidateCallbackResult> {
    if !report.consciousness_level.is_finite()
        || report.consciousness_level < 0.0
        || report.consciousness_level > 1.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Consciousness level must be in [0.0, 1.0]".to_string(),
        ));
    }

    if !report.mission_progress.is_finite()
        || report.mission_progress < 0.0
        || report.mission_progress > 1.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Mission progress must be in [0.0, 1.0]".to_string(),
        ));
    }

    if !report.fuel_level.is_finite()
        || report.fuel_level < 0.0
        || report.fuel_level > 1.0
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Fuel level must be in [0.0, 1.0]".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_robotic_credential(credential: &RoboticCredential) -> ExternResult<ValidateCallbackResult> {
    if credential.vitality_permille > 1000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Vitality must be in 0..=1000 permille".to_string(),
        ));
    }
    if credential.mastery_at_issue > 1000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Mastery must be in 0..=1000 permille".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
