//! Conjunctions Integrity Zome
//!
//! Defines entry types for conjunction events (close approaches)
//! and Conjunction Data Messages (CDMs).

use hdi::prelude::*;
use mycelix_space_shared::{SpaceTimestamp, ConjunctionDataMessage};

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    /// Conjunction event between two objects
    ConjunctionEvent(ConjunctionEvent),

    /// Conjunction Data Message (CDM)
    Cdm(CdmEntry),

    /// Collision avoidance maneuver
    AvoidanceManeuver(AvoidanceManeuver),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// CDMs for a conjunction event
    EventToCdms,
    /// Conjunctions involving an object
    ObjectConjunctions,
    /// All active conjunctions
    ActiveConjunctions,
    /// Maneuvers for a conjunction
    EventToManeuvers,
}

/// A conjunction event (potential collision)
#[hdk_entry_helper]
#[derive(Clone)]
pub struct ConjunctionEvent {
    /// Unique event ID
    pub event_id: String,

    /// Primary object NORAD ID
    pub primary_norad_id: u32,

    /// Secondary object NORAD ID
    pub secondary_norad_id: u32,

    /// Time of closest approach
    pub tca: SpaceTimestamp,

    /// Miss distance at TCA (km)
    pub miss_distance_km: f64,

    /// Peak collision probability
    pub max_pc: f64,

    /// Risk level
    pub risk_level: RiskLevel,

    /// Event status
    pub status: EventStatus,

    /// First detected
    pub created_at: SpaceTimestamp,

    /// Last updated
    pub updated_at: SpaceTimestamp,
}

/// Risk classification
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel {
    Negligible,
    Low,
    Medium,
    High,
    Emergency,
}

/// Event status
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum EventStatus {
    /// Newly detected, under analysis
    Screening,
    /// Being actively monitored
    Monitoring,
    /// Maneuver being planned
    Planning,
    /// Maneuver executed
    Mitigated,
    /// TCA passed without incident
    Passed,
    /// Collision occurred (hopefully rare!)
    Collision,
}

/// CDM entry wrapper
#[hdk_entry_helper]
#[derive(Clone)]
pub struct CdmEntry {
    /// The CDM data
    pub cdm: ConjunctionDataMessage,

    /// Version/update number
    pub version: u32,

    /// Supersedes previous CDM
    pub supersedes: Option<ActionHash>,
}

/// Collision avoidance maneuver record
#[hdk_entry_helper]
#[derive(Clone)]
pub struct AvoidanceManeuver {
    /// Related conjunction event
    pub event_id: String,

    /// Object performing maneuver
    pub norad_id: u32,

    /// Operator authorizing maneuver
    pub operator: AgentPubKey,

    /// Planned burn time
    pub burn_time: SpaceTimestamp,

    /// Delta-V (m/s)
    pub delta_v_ms: f64,

    /// Direction (unit vector in ECI)
    pub direction: [f64; 3],

    /// Status
    pub status: ManeuverStatus,

    /// Created at
    pub created_at: SpaceTimestamp,
}

/// Maneuver status
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ManeuverStatus {
    Planned,
    Announced,
    Executed,
    Confirmed,
    Cancelled,
}

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => {
                match app_entry {
                    EntryTypes::ConjunctionEvent(event) => validate_event(&event),
                    EntryTypes::Cdm(cdm) => validate_cdm(&cdm),
                    EntryTypes::AvoidanceManeuver(maneuver) => validate_maneuver(&maneuver),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_event(event: &ConjunctionEvent) -> ExternResult<ValidateCallbackResult> {
    // Both objects must have valid NORAD IDs
    if event.primary_norad_id == 0 || event.primary_norad_id > 999999 {
        return Ok(ValidateCallbackResult::Invalid(
            "Invalid primary NORAD ID".to_string()
        ));
    }
    if event.secondary_norad_id == 0 || event.secondary_norad_id > 999999 {
        return Ok(ValidateCallbackResult::Invalid(
            "Invalid secondary NORAD ID".to_string()
        ));
    }

    // Can't have conjunction with self
    if event.primary_norad_id == event.secondary_norad_id {
        return Ok(ValidateCallbackResult::Invalid(
            "Object cannot have conjunction with itself".to_string()
        ));
    }

    // Miss distance must be positive
    if event.miss_distance_km < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Miss distance cannot be negative".to_string()
        ));
    }

    // Pc must be 0-1
    if event.max_pc < 0.0 || event.max_pc > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Collision probability must be between 0 and 1".to_string()
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_cdm(cdm: &CdmEntry) -> ExternResult<ValidateCallbackResult> {
    // Basic CDM validation
    if cdm.cdm.collision_probability < 0.0 || cdm.cdm.collision_probability > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Invalid collision probability in CDM".to_string()
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_maneuver(maneuver: &AvoidanceManeuver) -> ExternResult<ValidateCallbackResult> {
    // Delta-V must be positive
    if maneuver.delta_v_ms <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Delta-V must be positive".to_string()
        ));
    }

    // Direction must be approximately unit vector
    let mag_sq = maneuver.direction.iter().map(|x| x * x).sum::<f64>();
    if (mag_sq - 1.0).abs() > 0.01 {
        return Ok(ValidateCallbackResult::Invalid(
            "Direction must be a unit vector".to_string()
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
