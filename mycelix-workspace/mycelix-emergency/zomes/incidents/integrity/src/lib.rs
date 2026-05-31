// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Incidents Integrity Zome
//! Defines entry types and validation for disaster incidents

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// A declared disaster incident
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Disaster {
    pub id: String,
    pub disaster_type: DisasterType,
    pub title: String,
    pub description: String,
    pub severity: SeverityLevel,
    pub declared_by: AgentPubKey,
    pub declared_at: Timestamp,
    pub affected_area: AffectedArea,
    pub status: DisasterStatus,
    pub estimated_affected: u32,
    pub coordination_lead: Option<AgentPubKey>,
}

/// Types of disasters
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DisasterType {
    Hurricane,
    Earthquake,
    Wildfire,
    Flood,
    Tornado,
    Pandemic,
    Industrial,
    MassCasualty,
    CyberAttack,
    Infrastructure,
    Other(String),
}

/// Severity levels (FEMA-aligned)
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum SeverityLevel {
    Level1,
    Level2,
    Level3,
    Level4,
    Level5,
}

/// Status of a disaster
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DisasterStatus {
    Declared,
    Active,
    Recovery,
    Closed,
}

/// Geographic area affected by the disaster
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct AffectedArea {
    pub center_lat: f64,
    pub center_lon: f64,
    pub radius_km: f32,
    pub boundary: Option<Vec<(f64, f64)>>,
    pub zones: Vec<OperationalZone>,
}

/// An operational zone within the affected area
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct OperationalZone {
    pub id: String,
    pub name: String,
    pub boundary: Vec<(f64, f64)>,
    pub priority: ZonePriority,
    pub status: ZoneStatus,
}

/// Priority level for operational zones
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ZonePriority {
    Critical,
    High,
    Medium,
    Low,
}

/// Status of an operational zone
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ZoneStatus {
    Unassessed,
    Active,
    Cleared,
    Hazardous,
    Evacuated,
}

/// An update to an incident
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct IncidentUpdate {
    pub disaster_hash: ActionHash,
    pub author: AgentPubKey,
    pub timestamp: Timestamp,
    pub update_type: UpdateType,
    pub content: String,
}

/// Types of incident updates
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum UpdateType {
    StatusChange,
    SeverityChange,
    AreaExpansion,
    AreaContraction,
    CasualtyReport,
    ResourceUpdate,
    WeatherUpdate,
    InfrastructureUpdate,
    EvacuationOrder,
    AllClear,
    General,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Disaster(Disaster),
    IncidentUpdate(IncidentUpdate),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllDisasters,
    ActiveDisasters,
    DisasterByType,
    DisasterToUpdate,
    AgentToDisaster,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Disaster(disaster) => validate_create_disaster(action, disaster),
                EntryTypes::IncidentUpdate(update) => validate_create_update(action, update),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Disaster(disaster) => validate_update_disaster(disaster),
                EntryTypes::IncidentUpdate(_) => Ok(ValidateCallbackResult::Invalid(
                    "Incident updates are immutable".into(),
                )),
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
            LinkTypes::AllDisasters => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ActiveDisasters => Ok(ValidateCallbackResult::Valid),
            LinkTypes::DisasterByType => Ok(ValidateCallbackResult::Valid),
            LinkTypes::DisasterToUpdate => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AgentToDisaster => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            link_type,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::ActiveDisasters => Ok(ValidateCallbackResult::Valid),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_disaster(
    _action: Create,
    disaster: Disaster,
) -> ExternResult<ValidateCallbackResult> {
    if disaster.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Disaster ID cannot be empty".into(),
        ));
    }
    if disaster.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Disaster title cannot be empty".into(),
        ));
    }
    if disaster.description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Disaster description cannot be empty".into(),
        ));
    }
    if disaster.affected_area.radius_km <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Affected area radius must be positive".into(),
        ));
    }
    if disaster.affected_area.center_lat < -90.0 || disaster.affected_area.center_lat > 90.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be between -90 and 90".into(),
        ));
    }
    if disaster.affected_area.center_lon < -180.0 || disaster.affected_area.center_lon > 180.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Longitude must be between -180 and 180".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_disaster(disaster: Disaster) -> ExternResult<ValidateCallbackResult> {
    if disaster.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Disaster ID cannot be empty".into(),
        ));
    }
    if disaster.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Disaster title cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_update(
    _action: Create,
    update: IncidentUpdate,
) -> ExternResult<ValidateCallbackResult> {
    if update.content.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Update content cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
