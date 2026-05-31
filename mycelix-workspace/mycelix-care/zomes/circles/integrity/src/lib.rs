// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Circles Integrity Zome
//! Defines entry types and validation for care circles and membership.

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Type of care circle
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CircleType {
    Neighborhood,
    Workplace,
    Faith,
    Family,
    School,
    Custom(String),
}

impl CircleType {
    pub fn anchor_key(&self) -> String {
        match self {
            CircleType::Neighborhood => "neighborhood".to_string(),
            CircleType::Workplace => "workplace".to_string(),
            CircleType::Faith => "faith".to_string(),
            CircleType::Family => "family".to_string(),
            CircleType::School => "school".to_string(),
            CircleType::Custom(s) => format!("custom_{}", s.to_lowercase().replace(' ', "_")),
        }
    }
}

/// Role within a care circle
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MemberRole {
    Organizer,
    Member,
    Observer,
}

/// A care circle - a group of people who coordinate mutual aid
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CareCircle {
    /// Circle name
    pub name: String,
    /// Description of the circle's purpose
    pub description: String,
    /// Location or area the circle serves
    pub location: String,
    /// Maximum number of members allowed
    pub max_members: u32,
    /// Agent who created the circle
    pub created_by: AgentPubKey,
    /// Type of circle
    pub circle_type: CircleType,
    /// Whether the circle is currently active
    pub active: bool,
    /// When the circle was created
    pub created_at: Timestamp,
}

/// Membership record linking an agent to a circle
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CircleMembership {
    /// Hash of the CareCircle entry
    pub circle_hash: ActionHash,
    /// The member agent
    pub member: AgentPubKey,
    /// Role in the circle
    pub role: MemberRole,
    /// When the member joined
    pub joined_at: Timestamp,
    /// Whether the membership is currently active
    pub active: bool,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    CareCircle(CareCircle),
    CircleMembership(CircleMembership),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// All circles anchor
    AllCircles,
    /// Circle type anchor to circles of that type
    TypeToCircle,
    /// Circle to its memberships
    CircleToMembership,
    /// Agent to their memberships
    AgentToMembership,
    /// Agent to circles they created
    AgentToCreatedCircle,
}

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::CareCircle(circle) => validate_create_circle(action, circle),
                EntryTypes::CircleMembership(membership) => {
                    validate_create_membership(action, membership)
                }
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::CareCircle(circle) => validate_update_circle(circle),
                EntryTypes::CircleMembership(_) => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => Ok(ValidateCallbackResult::Valid),
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

fn validate_create_circle(
    _action: Create,
    circle: CareCircle,
) -> ExternResult<ValidateCallbackResult> {
    if circle.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle name cannot be empty".into(),
        ));
    }
    if circle.name.len() > 128 {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle name must be 128 characters or fewer".into(),
        ));
    }
    if circle.description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle description cannot be empty".into(),
        ));
    }
    if circle.description.len() > 2048 {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle description must be 2048 characters or fewer".into(),
        ));
    }
    if circle.max_members < 2 {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle must allow at least 2 members".into(),
        ));
    }
    if circle.max_members > 500 {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle cannot have more than 500 members".into(),
        ));
    }
    if circle.location.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Location cannot be empty".into(),
        ));
    }
    if circle.location.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Location must be 512 characters or fewer".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_circle(circle: CareCircle) -> ExternResult<ValidateCallbackResult> {
    if circle.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle name cannot be empty".into(),
        ));
    }
    if circle.max_members < 2 {
        return Ok(ValidateCallbackResult::Invalid(
            "Circle must allow at least 2 members".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_membership(
    _action: Create,
    _membership: CircleMembership,
) -> ExternResult<ValidateCallbackResult> {
    // Membership validation is primarily handled at the coordinator level
    // (checking circle exists, member count, etc.)
    Ok(ValidateCallbackResult::Valid)
}
