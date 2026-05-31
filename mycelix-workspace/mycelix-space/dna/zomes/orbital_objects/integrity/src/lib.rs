// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Orbital Objects Integrity Zome
//!
//! Defines the entry types and validation rules for the orbital object catalog.
//! This is the foundational data structure - a decentralized registry of all
//! tracked space objects (satellites, debris, rocket bodies).

use hdi::prelude::*;
use mycelix_space_shared::{DataSourceType, QualityScore, SpaceTimestamp};
use orbital_mechanics::tle::TwoLineElement;

/// Entry types for the orbital objects DNA
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    /// A tracked space object (satellite, debris, rocket body)
    OrbitalObject(OrbitalObject),

    /// Two-Line Element set for an object
    TleRecord(TleRecord),

    /// Operator claim for an object
    OperatorClaim(OperatorClaim),

    /// Object metadata (size, mass, type)
    ObjectMetadata(ObjectMetadata),
}

/// Link types for relationships
#[hdk_link_types]
pub enum LinkTypes {
    /// Link from object to its TLE history
    ObjectToTles,

    /// Link from object to its operator claims
    ObjectToOperator,

    /// Link from operator to their claimed objects
    OperatorToObjects,

    /// Link from object to its metadata
    ObjectToMetadata,

    /// Index: All objects anchor
    AllObjects,

    /// Index: Objects by orbit type (LEO, MEO, GEO, etc.)
    OrbitTypeIndex,

    /// Index: Objects by operator
    OperatorIndex,
}

/// A tracked orbital object
#[hdk_entry_helper]
#[derive(Clone)]
pub struct OrbitalObject {
    /// NORAD Catalog Number (unique identifier)
    pub norad_id: u32,

    /// International designator (e.g., "1998-067A" for ISS)
    pub intl_designator: String,

    /// Object name
    pub name: String,

    /// Object type
    pub object_type: ObjectType,

    /// Country/organization of origin
    pub country: Option<String>,

    /// Launch date
    pub launch_date: Option<SpaceTimestamp>,

    /// Decay date (if decayed)
    pub decay_date: Option<SpaceTimestamp>,

    /// Current operational status
    pub status: OperationalStatus,

    /// When this record was created
    pub created_at: SpaceTimestamp,

    /// Who created this record
    pub created_by: AgentPubKey,
}

/// Type of space object
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ObjectType {
    /// Active/inactive satellite
    Payload,
    /// Rocket body
    RocketBody,
    /// Debris (fragments, mission-related debris)
    Debris,
    /// Unknown/unclassified
    Unknown,
}

/// Operational status
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum OperationalStatus {
    /// Currently operational
    Operational,
    /// Non-operational but controlled (can maneuver)
    NonOperational,
    /// Decaying, expected to reenter
    Decaying,
    /// Has reentered (historical record)
    Decayed,
    /// Status unknown
    Unknown,
}

/// TLE record for an object
#[hdk_entry_helper]
#[derive(Clone)]
pub struct TleRecord {
    /// Object this TLE is for
    pub norad_id: u32,

    /// TLE Line 1 (69 characters)
    pub line1: String,

    /// TLE Line 2 (69 characters)
    pub line2: String,

    /// TLE epoch (extracted from line1)
    pub epoch: SpaceTimestamp,

    /// Source of this TLE
    pub source: DataSourceType,

    /// Quality assessment
    pub quality: QualityScore,

    /// When this was added to the network
    pub submitted_at: SpaceTimestamp,

    /// Who submitted this TLE
    pub submitted_by: AgentPubKey,
}

/// Operator claim for an object
#[hdk_entry_helper]
#[derive(Clone)]
pub struct OperatorClaim {
    /// Object being claimed
    pub norad_id: u32,

    /// Agent claiming to be the operator
    pub operator: AgentPubKey,

    /// Organization name
    pub organization: String,

    /// Contact information (optional, encrypted in practice)
    pub contact: Option<String>,

    /// When this claim was made
    pub claimed_at: SpaceTimestamp,

    /// Verification status
    pub verified: bool,

    /// Verification evidence (e.g., hash of signed document)
    pub verification_hash: Option<[u8; 32]>,
}

/// Object physical metadata
#[hdk_entry_helper]
#[derive(Clone)]
pub struct ObjectMetadata {
    /// Object this metadata is for
    pub norad_id: u32,

    /// Radar cross-section (m²)
    pub rcs_m2: Option<f64>,

    /// Estimated mass (kg)
    pub mass_kg: Option<f64>,

    /// Characteristic length (m)
    pub length_m: Option<f64>,

    /// Hard-body radius for conjunction (m)
    pub hard_body_radius_m: Option<f64>,

    /// Ballistic coefficient (kg/m²)
    pub ballistic_coefficient: Option<f64>,

    /// Area-to-mass ratio (m²/kg)
    pub area_to_mass: Option<f64>,

    /// Source of this metadata
    pub source: DataSourceType,

    /// Last updated
    pub updated_at: SpaceTimestamp,
}

/// Genesis validation
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation dispatcher
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => validate_create_entry(app_entry, action),
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => validate_update_entry(app_entry, action),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address,
            target_address,
            tag,
            action,
        } => validate_create_link(link_type, base_address, target_address, tag, action),
        FlatOp::RegisterDeleteLink {
            link_type,
            original_action,
            base_address,
            target_address,
            tag,
            action,
        } => validate_delete_link(
            link_type,
            original_action,
            base_address,
            target_address,
            tag,
            action,
        ),
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate entry creation
fn validate_create_entry(
    entry: EntryTypes,
    _action: Create,
) -> ExternResult<ValidateCallbackResult> {
    match entry {
        EntryTypes::OrbitalObject(obj) => validate_orbital_object(&obj),
        EntryTypes::TleRecord(tle) => validate_tle_record(&tle),
        EntryTypes::OperatorClaim(claim) => validate_operator_claim(&claim),
        EntryTypes::ObjectMetadata(meta) => validate_object_metadata(&meta),
    }
}

/// Validate entry update
fn validate_update_entry(
    entry: EntryTypes,
    _action: Update,
) -> ExternResult<ValidateCallbackResult> {
    match entry {
        EntryTypes::OrbitalObject(obj) => validate_orbital_object(&obj),
        EntryTypes::TleRecord(tle) => validate_tle_record(&tle),
        EntryTypes::OperatorClaim(claim) => validate_operator_claim(&claim),
        EntryTypes::ObjectMetadata(meta) => validate_object_metadata(&meta),
    }
}

/// Validate an orbital object entry
fn validate_orbital_object(obj: &OrbitalObject) -> ExternResult<ValidateCallbackResult> {
    // NORAD ID must be valid (1-999999)
    if obj.norad_id == 0 || obj.norad_id > 999999 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Invalid NORAD ID: {}. Must be 1-999999",
            obj.norad_id
        )));
    }

    // International designator format: YYYY-NNNA (year-launch_number-piece)
    if !obj.intl_designator.is_empty() {
        let parts: Vec<&str> = obj.intl_designator.split('-').collect();
        if parts.len() != 2 && parts.len() != 3 {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "Invalid international designator format: {}",
                obj.intl_designator
            )));
        }
    }

    // Name must not be empty
    if obj.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Object name cannot be empty".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a TLE record using the orbital-mechanics library parser
/// This validates line lengths, line numbers, checksums, and all field formats
fn validate_tle_record(tle: &TleRecord) -> ExternResult<ValidateCallbackResult> {
    // NORAD ID must be valid
    if tle.norad_id == 0 || tle.norad_id > 999999 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Invalid NORAD ID: {}",
            tle.norad_id
        )));
    }

    // Full TLE parse validates: line length, line numbers, checksums, epoch, all fields
    let parsed = match TwoLineElement::parse_lines(None, &tle.line1, &tle.line2) {
        Ok(p) => p,
        Err(e) => {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "TLE parse failed: {}",
                e
            )));
        }
    };

    // NORAD ID in TLE must match the record's norad_id
    if parsed.norad_id != tle.norad_id {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "NORAD ID mismatch: record says {} but TLE contains {}",
            tle.norad_id, parsed.norad_id
        )));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate operator claim
fn validate_operator_claim(claim: &OperatorClaim) -> ExternResult<ValidateCallbackResult> {
    // NORAD ID must be valid
    if claim.norad_id == 0 || claim.norad_id > 999999 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Invalid NORAD ID: {}",
            claim.norad_id
        )));
    }

    // Organization name must not be empty
    if claim.organization.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Organization name cannot be empty".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate object metadata
fn validate_object_metadata(meta: &ObjectMetadata) -> ExternResult<ValidateCallbackResult> {
    // NORAD ID must be valid
    if meta.norad_id == 0 || meta.norad_id > 999999 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Invalid NORAD ID: {}",
            meta.norad_id
        )));
    }

    // Physical values must be positive if present
    if let Some(rcs) = meta.rcs_m2 {
        if rcs <= 0.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "RCS must be positive".to_string(),
            ));
        }
    }

    if let Some(mass) = meta.mass_kg {
        if mass <= 0.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Mass must be positive".to_string(),
            ));
        }
    }

    if let Some(hbr) = meta.hard_body_radius_m {
        if hbr <= 0.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Hard body radius must be positive".to_string(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate link creation
fn validate_create_link(
    link_type: LinkTypes,
    _base: AnyLinkableHash,
    _target: AnyLinkableHash,
    _tag: LinkTag,
    _action: CreateLink,
) -> ExternResult<ValidateCallbackResult> {
    match link_type {
        LinkTypes::ObjectToTles => Ok(ValidateCallbackResult::Valid),
        LinkTypes::ObjectToOperator => Ok(ValidateCallbackResult::Valid),
        LinkTypes::OperatorToObjects => Ok(ValidateCallbackResult::Valid),
        LinkTypes::ObjectToMetadata => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AllObjects => Ok(ValidateCallbackResult::Valid),
        LinkTypes::OrbitTypeIndex => Ok(ValidateCallbackResult::Valid),
        LinkTypes::OperatorIndex => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate link deletion
fn validate_delete_link(
    _link_type: LinkTypes,
    _original_action: CreateLink,
    _base: AnyLinkableHash,
    _target: AnyLinkableHash,
    _tag: LinkTag,
    _action: DeleteLink,
) -> ExternResult<ValidateCallbackResult> {
    // Allow deletion for now - may add restrictions later
    Ok(ValidateCallbackResult::Valid)
}
