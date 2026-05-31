// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Steward Integrity Zome
//! Watershed governance, water rights, transfers, and dispute resolution

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// ============================================================================
// WATER SOURCE TYPE (re-exported for cross-zome reference)
// ============================================================================

/// Type of water source (mirrored from flow for steward context)
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum WaterSourceType {
    Municipal,
    Well,
    Spring,
    Rainwater,
    Aquifer,
    River,
    Lake,
    Recycled,
    Desalinated,
}

// ============================================================================
// WATERSHED
// ============================================================================

/// Type of water stewardship governance
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum StewardshipType {
    /// Land-adjacent rights
    Riparian,
    /// First-in-time, first-in-right
    PriorAppropriation,
    /// Community-managed commons
    Commons,
    /// Mixed governance
    Hybrid,
}

/// A watershed under community stewardship
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Watershed {
    /// Unique identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Optional HUC (Hydrologic Unit Code) identifier
    pub huc_code: Option<String>,
    /// Boundary polygon as (lat, lon) pairs
    pub boundary: Vec<(f64, f64)>,
    /// Area in square kilometers
    pub area_sq_km: f32,
    /// Governance model for this watershed
    pub stewardship_type: StewardshipType,
    /// Optional link to governing body (e.g., governance hApp proposal)
    pub governing_body: Option<ActionHash>,
    /// Primary water source type in this watershed
    pub primary_source_type: WaterSourceType,
}

// ============================================================================
// WATER RIGHTS
// ============================================================================

/// Type of water right
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RightType {
    /// Based on land adjacency
    Riparian,
    /// Based on historical use
    Appropriative,
    /// Established through long use
    Prescriptive,
    /// Indigenous / first nations rights
    Aboriginal,
    /// Community commons right
    Commons,
}

/// Status of a water right
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RightStatus {
    Active,
    Suspended,
    Revoked,
    Transferred,
    Expired,
}

/// A legally recognized water right within a watershed
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct WaterRight {
    /// Watershed this right belongs to
    pub watershed_hash: ActionHash,
    /// Agent holding this right
    pub holder: AgentPubKey,
    /// Type of right
    pub right_type: RightType,
    /// Maximum authorized volume in liters
    pub volume_authorized_liters: u64,
    /// Priority date (for appropriative rights)
    pub priority_date: Option<Timestamp>,
    /// Conditions attached to this right
    pub conditions: Vec<String>,
    /// Current status of the right
    pub status: RightStatus,
    /// Whether this right can be transferred
    pub transferable: bool,
}

// ============================================================================
// RIGHT TRANSFERS
// ============================================================================

/// Type of water right transfer
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TransferType {
    Sale,
    Lease,
    Donation,
    Inheritance,
    Emergency,
}

/// A transfer of a water right from one holder to another
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RightTransfer {
    /// The water right being transferred
    pub right_hash: ActionHash,
    /// Current holder
    pub from_holder: AgentPubKey,
    /// New holder
    pub to_holder: AgentPubKey,
    /// Volume being transferred in liters
    pub volume_liters: u64,
    /// Type of transfer
    pub transfer_type: TransferType,
    /// Agent who approved the transfer (e.g., watershed governance)
    pub approved_by: Option<AgentPubKey>,
    /// When the transfer was executed
    pub transferred_at: Timestamp,
}

// ============================================================================
// WATER DISPUTES
// ============================================================================

/// Type of water dispute
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DisputeType {
    /// Disagreement over allocation amounts
    Allocation,
    /// Disagreement over water quality responsibility
    Quality,
    /// Access to water sources
    Access,
    /// Excessive water use
    Overuse,
    /// Physical or legal encroachment
    Encroachment,
}

/// Status of a water dispute
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DisputeStatus {
    Filed,
    UnderReview,
    Mediation,
    Arbitration,
    Resolved,
    Dismissed,
}

/// A water dispute between parties in a watershed
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct WaterDispute {
    /// Watershed where the dispute occurs
    pub watershed_hash: ActionHash,
    /// Agent filing the complaint
    pub complainant: AgentPubKey,
    /// Agent being complained about
    pub respondent: AgentPubKey,
    /// Type of dispute
    pub dispute_type: DisputeType,
    /// Description of the dispute
    pub description: String,
    /// Evidence action hashes
    pub evidence: Vec<ActionHash>,
    /// Current status
    pub status: DisputeStatus,
    /// Resolution text (when resolved)
    pub resolution: Option<String>,
}

// ============================================================================
// ENTRY & LINK TYPE REGISTRATION
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Watershed(Watershed),
    WaterRight(WaterRight),
    RightTransfer(RightTransfer),
    WaterDispute(WaterDispute),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Anchor to all watersheds
    AllWatersheds,
    /// Watershed to its water rights
    WatershedToRight,
    /// Holder to their water rights
    HolderToRight,
    /// Right to its transfers
    RightToTransfer,
    /// Watershed to disputes
    WatershedToDispute,
    /// Agent to disputes they filed
    AgentToDispute,
    /// Stewardship type to watersheds
    StewardshipTypeToWatershed,
}

// ============================================================================
// VALIDATION
// ============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Watershed(ws) => validate_create_watershed(action, ws),
                EntryTypes::WaterRight(right) => validate_create_water_right(action, right),
                EntryTypes::RightTransfer(transfer) => {
                    validate_create_right_transfer(action, transfer)
                }
                EntryTypes::WaterDispute(dispute) => validate_create_water_dispute(action, dispute),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Watershed(ws) => validate_update_watershed(ws, original_action_hash),
                EntryTypes::WaterRight(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::WaterDispute(_) => Ok(ValidateCallbackResult::Valid),
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::AllWatersheds => Ok(ValidateCallbackResult::Valid),
            LinkTypes::WatershedToRight => Ok(ValidateCallbackResult::Valid),
            LinkTypes::HolderToRight => Ok(ValidateCallbackResult::Valid),
            LinkTypes::RightToTransfer => Ok(ValidateCallbackResult::Valid),
            LinkTypes::WatershedToDispute => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AgentToDispute => Ok(ValidateCallbackResult::Valid),
            LinkTypes::StewardshipTypeToWatershed => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            link_type: _,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_watershed(
    _action: Create,
    ws: Watershed,
) -> ExternResult<ValidateCallbackResult> {
    if ws.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Watershed ID cannot be empty".into(),
        ));
    }
    if ws.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Watershed name cannot be empty".into(),
        ));
    }
    if ws.boundary.len() < 3 {
        return Ok(ValidateCallbackResult::Invalid(
            "Watershed boundary must have at least 3 points".into(),
        ));
    }
    if ws.area_sq_km <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Watershed area must be positive".into(),
        ));
    }
    // Validate boundary coordinates
    for (lat, lon) in &ws.boundary {
        if *lat < -90.0 || *lat > 90.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Boundary latitude must be between -90 and 90".into(),
            ));
        }
        if *lon < -180.0 || *lon > 180.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Boundary longitude must be between -180 and 180".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_watershed(
    ws: Watershed,
    original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash)?;
    let original_ws: Watershed = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original watershed not found".into()
        )))?;
    if ws.id != original_ws.id {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change watershed ID".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_water_right(
    _action: Create,
    right: WaterRight,
) -> ExternResult<ValidateCallbackResult> {
    if right.volume_authorized_liters == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Authorized volume must be greater than zero".into(),
        ));
    }
    if right.conditions.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 50 conditions".into(),
        ));
    }
    for condition in &right.conditions {
        if condition.is_empty() || condition.len() > 1024 {
            return Ok(ValidateCallbackResult::Invalid(
                "Each condition must be 1-1024 characters".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_right_transfer(
    _action: Create,
    transfer: RightTransfer,
) -> ExternResult<ValidateCallbackResult> {
    if transfer.volume_liters == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Transfer volume must be greater than zero".into(),
        ));
    }
    if transfer.from_holder == transfer.to_holder {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot transfer a right to the same holder".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_water_dispute(
    _action: Create,
    dispute: WaterDispute,
) -> ExternResult<ValidateCallbackResult> {
    if dispute.description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Dispute description cannot be empty".into(),
        ));
    }
    if dispute.description.len() > 8192 {
        return Ok(ValidateCallbackResult::Invalid(
            "Dispute description must be at most 8192 characters".into(),
        ));
    }
    if dispute.complainant == dispute.respondent {
        return Ok(ValidateCallbackResult::Invalid(
            "Complainant and respondent cannot be the same agent".into(),
        ));
    }
    if dispute.evidence.len() > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 100 evidence items".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
